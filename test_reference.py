"""CPU-only behavioral checks for this illustrative component; no downloads."""
import unittest
import torch
from torch import nn
from xfcls import XFCLS, XFCLSConfig, multilayer_agreement, xfr_refine

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(4))
        self.text_side=nn.LayerNorm(4)
        self.head=nn.Linear(4,3,bias=False)
    def forward(self,x):
        return self.head(self.visual(x)).transpose(1,2).reshape(1,3,2,3)

class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(26);torch.set_num_threads(1)
        self.model=Toy();self.x=torch.randn(1,6,4)
    def test_visual_subset_adam_and_static_clock(self):
        method=XFCLS(self.model)
        self.assertIs(type(method.optimizer),torch.optim.Adam)
        expected={id(p) for p in self.model.visual.parameters()}
        actual={id(p) for p in self.model.parameters() if p.requires_grad}
        self.assertEqual(expected,actual)
        before={k:p.clone() for k,p in self.model.state_dict().items()}
        for _ in range(3):method.process_image(lambda:self.model(self.x),lambda:.5,adapt=False)
        self.assertEqual(method.image_index,3);self.assertEqual(method.inner_attempts,0)
        self.assertEqual(len(method.optimizer.state),0)
        for k,p in self.model.state_dict().items():self.assertTrue(torch.equal(p,before[k]))
    def test_skip_startup_and_fresh_prediction(self):
        method=XFCLS(self.model);accepted=[]
        for _ in range(80):
            out,obs=method.process_image(lambda:self.model(self.x),lambda:.5,output_xfr=False)
            accepted.extend(v['accepted'] for v in obs)
            self.assertTrue(torch.equal(out,self.model(self.x).detach()))
        self.assertFalse(accepted[0]);self.assertTrue(any(accepted))
        self.assertEqual(method.image_index,80);self.assertEqual(method.inner_attempts,160)
        self.assertTrue(any(not torch.equal(p,method.source[k]) for k,p in method.parameters.items()))
        self.assertFalse(out.requires_grad)
    def test_slow_update_without_legacy_warmup_and_mixed_anchor(self):
        c=XFCLSConfig(cls_v4_slow_ema_rate=.8,cls_anchor_blend=.5)
        method=XFCLS(self.model,c);method.image_index=1
        with torch.no_grad():
            for p in method.parameters.values():p.add_(.2)
        old={k:p.clone() for k,p in method.parameters.items()}
        method._consolidate_and_recover(.1)
        for k,p in method.parameters.items():
            slow=.8*method.source[k]+.2*old[k]
            torch.testing.assert_close(method.slow[k],slow)
            torch.testing.assert_close(p,.9*old[k]+.1*(.5*slow+.5*method.source[k]))
    def test_global_halfway_source_pullback(self):
        c=XFCLSConfig(cls_v4_slow_ema_rate=1.,cls_snapshot_interval=100)
        method=XFCLS(self.model,c);method.image_index=100
        with torch.no_grad():
            for k in method.slow:method.slow[k].copy_(2*method.source[k])
        method._consolidate_and_recover(0.)
        for k in method.slow:torch.testing.assert_close(method.slow[k],1.5*method.source[k])
    def test_agreement_and_frozen_refinement(self):
        preds=torch.tensor([[[0,1]],[[0,2]],[[1,2]]])
        logits=torch.nn.functional.one_hot(preds,3).float()
        self.assertAlmostEqual(multilayer_agreement(logits),2/3,places=6)
        z=self.model(self.x);f=torch.randn(1,6,5,requires_grad=True);c=XFCLSConfig()
        out=xfr_refine(z,f,(2,3),1,c)
        self.assertEqual(out.shape,z.shape);self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(out.requires_grad);self.assertIsNone(f.grad)
        floor=xfr_refine(z,f,(2,3),100000,c)
        self.assertFalse(torch.allclose(out,floor))

if __name__=='__main__':unittest.main(verbosity=2)
