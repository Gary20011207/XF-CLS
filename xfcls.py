"""Compact XF-CLS reference for the canonical v4 equations.

This is an illustrative component, not the benchmark runner. The caller supplies
NACLIP dense logits, multilayer agreement, and frozen DINO tokens on a common
patch grid. Sliding-window inference, text/class mapping, loaders, baseline ports,
and checkpoint/resume are deliberately outside this small supplement.
"""
from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Callable, Optional
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class XFCLSConfig:
    lr: float = 2e-5
    steps: int = 2
    recovery_rate: float = 0.001  # ACDC; use 0.002 for OnDA.
    cls_v4_slow_ema_rate: float = 0.99999
    cls_v4_drift_budget: float = 0.30
    cls_v4_drift_pullback: float = 0.50
    cls_anchor_blend: float = 0.50
    cls_snapshot_interval: int = 100  # Incoming images, not inner attempts.
    entropy_skip_factor: float = 1.30
    eval_topk: int = 11
    eval_dino_temperature: float = 6.0
    eval_graph_blend: float = 0.45
    eval_proto_blend: float = 0.35
    eval_proto_temperature: float = 8.0
    eval_uncertainty_power: float = 1.6
    eval_class_prior_power: float = 1.0
    eval_xfr_decay_steps: int = 5600
    eval_xfr_min_weight: float = 0.12
    eval_proto_iterations: int = 2
    eval_cross_disagree_gate: bool = True
    eval_disagree_weight: float = 0.5


def multilayer_agreement(layer_logits: torch.Tensor) -> float:
    """Logits [layers, batch, locations, classes], with fixed text entries."""
    with torch.no_grad():
        pred = layer_logits.argmax(dim=-1)
        mode = pred.mode(dim=0).values
        return (pred == mode.unsqueeze(0)).float().mean().item()


def _on_grid(logits, tokens, grid_hw):
    if logits.ndim != 4 or tokens.ndim != 3:
        raise ValueError('Expected logits [B,C,H,W] and DINO tokens [B,N,D].')
    if tokens.shape[1] != grid_hw[0] * grid_hw[1] or logits.shape[0] != tokens.shape[0]:
        raise ValueError('Frozen token grid does not match the supplied spatial grid.')
    z = F.interpolate(logits.float(), size=grid_hw, mode='bilinear', align_corners=False)
    f = tokens.detach().float()
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z.flatten(2), f


@torch.no_grad()
def structural_disagreement(logits, tokens, grid_hw) -> float:
    z, f = _on_grid(logits, tokens, grid_hw)
    p = z.softmax(dim=1).clamp_min(1e-8)
    proto = torch.bmm(p, f) / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    reconstructed = torch.bmm(f, proto.transpose(1, 2)).transpose(1, 2)
    return (p.argmax(1) != reconstructed.argmax(1)).float().mean().item()


@torch.no_grad()
def xfr_refine(logits, tokens, grid_hw, image_index: int, config: XFCLSConfig):
    """Eq. (XFR): conservative graph logits mixed with log probability mixture.

    Refinement is computed on the frozen observer's grid and interpolated back.
    image_index is one-based and must advance even for a matched static control.
    """
    c = config
    z, f = _on_grid(logits, tokens, grid_hw)
    p = z.softmax(dim=1).clamp_min(1e-8)
    sim = torch.bmm(f, f.transpose(1, 2))
    values, index = sim.topk(max(1, min(c.eval_topk, f.shape[1])), dim=-1)
    weights = (values * c.eval_dino_temperature).softmax(dim=-1)
    def smooth(v):
        out = torch.zeros_like(v)
        for b in range(v.shape[0]):
            out[b] = (v[b, :, index[b]] * weights[b].unsqueeze(0)).sum(dim=-1)
        return out
    graph_logits = (1-c.eval_graph_blend)*z + c.eval_graph_blend*smooth(z)
    graph_p = smooth(p)
    graph_p = graph_p / graph_p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assignment = p
    if c.eval_proto_iterations < 1:
        raise ValueError('At least one prototype iteration is required.')
    for iteration in range(c.eval_proto_iterations):
        proto = torch.bmm(assignment, f) / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proto = proto / proto.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        score = torch.bmm(f, proto.transpose(1, 2)).transpose(1, 2)
        proto_p = (score*c.eval_proto_temperature).softmax(dim=1)
        if iteration < c.eval_proto_iterations-1:
            assignment = 0.5*p + 0.5*proto_p
            assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-8)
    proto_p = proto_p * p.mean(dim=-1, keepdim=True).pow(c.eval_class_prior_power)
    proto_p = proto_p / proto_p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    gate = (-(p*p.log()).sum(dim=1) / math.log(max(z.shape[1],2))).pow(c.eval_uncertainty_power).unsqueeze(1)
    if c.eval_cross_disagree_gate:
        disagree = (p.argmax(1) != proto_p.argmax(1)).float().unsqueeze(1)
        gate = (1-c.eval_disagree_weight)*gate + c.eval_disagree_weight*disagree
    gg, pg = c.eval_graph_blend*gate, c.eval_proto_blend*gate
    mixture = (1-gg-pg).clamp_min(0.05)*p + gg*graph_p + pg*proto_p
    mixture = mixture / mixture.sum(dim=1, keepdim=True).clamp_min(1e-8)
    full = mixture.clamp_min(1e-8).log()
    shape = (logits.shape[0], logits.shape[1], *grid_hw)
    def up(x):
        return F.interpolate(x.reshape(shape).to(logits.dtype), size=logits.shape[-2:], mode='bilinear', align_corners=False)
    conservative, full = up(graph_logits), up(full)
    weight = min(1., max(c.eval_xfr_min_weight, math.exp(-(max(image_index,1)-1)/c.eval_xfr_decay_steps)))
    return conservative + weight*(full-conservative)


class XFCLS:
    """Persistent visual-LayerNorm state with explicit host-model callbacks.

    Pass a model whose visual tower is model.visual. Only its LayerNorm affine
    tensors are optimized. Call process_image once per incoming image. Frozen
    DINO tokens must refer to that same image; reuse is valid only if all logical
    observer calls have identical input, preprocessing and precision.
    """
    def __init__(self, model: nn.Module, config: Optional[XFCLSConfig] = None):
        self.model = model
        self.config = config or XFCLSConfig()
        if not hasattr(model, 'visual'):
            raise ValueError('Host model must expose its visual tower as model.visual.')
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.parameters = {}
        for name, module in model.visual.named_modules():
            if isinstance(module, nn.LayerNorm):
                for pname, parameter in module.named_parameters(recurse=False):
                    if pname in ('weight','bias'):
                        parameter.requires_grad_(True)
                        self.parameters[f'{name}.{pname}'] = parameter
        if not self.parameters:
            raise ValueError('No visual LayerNorm affine parameters found.')
        self.source = {k:p.detach().clone() for k,p in self.parameters.items()}
        self.slow = {k:v.clone() for k,v in self.source.items()}
        self.optimizer = torch.optim.Adam(self.parameters.values(), lr=self.config.lr,
                                          betas=(0.9,0.999), weight_decay=0.)
        self.image_index = 0
        self.inner_attempts = 0
        self.entropy_ema = 0.
        self.disagreement_ema = None
        self.agreements = deque(maxlen=20)

    def process_image(self, logits_fn: Callable[[], torch.Tensor],
                      agreement_fn: Callable[[], float], *,
                      dino_tokens=None, grid_hw=None, adapt=True, output_xfr=True,
                      entropy_class_count=None):
        """Adapt, recover, then predict. Callbacks receive no labels.

        logits_fn returns dense [B,C,H,W] logits with gradients when enabled.
        agreement_fn computes the pre-update multilayer modal agreement.
        entropy_class_count is the evaluation vocabulary size (19 for driving),
        which may differ from the number of extended text entries in the logits.
        Static controls use adapt=False: only image_index advances.
        """
        c = self.config
        self.image_index += 1
        observations = []
        if dino_tokens is not None and grid_hw is None:
            raise ValueError('grid_hw is required with frozen observer tokens.')
        if adapt:
            for _ in range(c.steps):
                self.inner_attempts += 1
                with torch.no_grad():
                    agreement = float(agreement_fn())
                if not 0 <= agreement <= 1:
                    raise ValueError('Agreement must lie in [0,1].')
                self.agreements.append(agreement)
                with torch.enable_grad():
                    z = logits_fn()
                    loss = -(z.softmax(dim=1)*z.log_softmax(dim=1)).sum(dim=1).mean()
                d = None if dino_tokens is None else structural_disagreement(z, dino_tokens, grid_hw)
                normalized = loss.detach().item()/math.log(max(entropy_class_count or z.shape[1],2))
                self.entropy_ema = .99*self.entropy_ema + .01*normalized
                skip = self.entropy_ema > 0 and normalized > c.entropy_skip_factor*self.entropy_ema
                self.optimizer.zero_grad(set_to_none=True)
                if not skip:
                    loss.backward()
                    self.optimizer.step()
                a = list(self.agreements)
                trend = statistics.mean(a[-10:])-statistics.mean(a[:10]) if len(a)==20 else 0.
                multiplier = min(5.,max(.1,math.exp(-50*trend)))
                gate = 1.
                if d is not None:
                    if self.disagreement_ema is None:self.disagreement_ema=d
                    self.disagreement_ema=.99*self.disagreement_ema+.01*d
                    gate=min(1.5,max(.5,1+d-self.disagreement_ema))
                rate=c.recovery_rate*(1-agreement)*multiplier*gate
                self._consolidate_and_recover(rate)
                observations.append({'accepted':not skip,'normalized_entropy':normalized,
                                     'agreement':agreement,'disagreement':d,'recovery_rate':rate})
        with torch.no_grad():
            # Fresh prediction from the recovered state, never stale pre-update logits.
            logits=logits_fn()
            if output_xfr and dino_tokens is not None:
                logits=xfr_refine(logits,dino_tokens,grid_hw,self.image_index,c)
        return logits.detach(), observations

    @torch.no_grad()
    def _consolidate_and_recover(self, rate):
        c=self.config
        # Canonical v4 has no 200-step consolidation warmup; that legacy flag
        # belongs to other historical modes and is ignored by v4.
        for k,p in self.parameters.items():
            self.slow[k].mul_(c.cls_v4_slow_ema_rate).add_(p,alpha=1-c.cls_v4_slow_ema_rate)
        if self.image_index % c.cls_snapshot_interval == 0:
            drift=sum((self.slow[k]-self.source[k]).square().sum().item() for k in self.slow)
            norm=sum(v.square().sum().item() for v in self.source.values())
            if norm>1e-8 and math.sqrt(drift/norm)>c.cls_v4_drift_budget:
                for k in self.slow:
                    self.slow[k].mul_(1-c.cls_v4_drift_pullback).add_(self.source[k],alpha=c.cls_v4_drift_pullback)
        for k,p in self.parameters.items():
            anchor=(1-c.cls_anchor_blend)*self.slow[k]+c.cls_anchor_blend*self.source[k]
            p.mul_(1-rate).add_(anchor,alpha=rate)
