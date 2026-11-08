import argparse
import os
import subprocess
import shlex
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Example command builder for an external XF-CLS-compatible runner; this archive does not include that runner.")
    parser.add_argument("--rain_data_dir", required=True)
    parser.add_argument("--gt_data_dir", required=True)
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--runner", default="")
    parser.add_argument("--intensities", nargs="+", default=["25mm", "50mm", "75mm", "100mm", "200mm"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--init_resize", nargs=2, type=int, default=[1120, 630])
    parser.add_argument("--patch_size", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--patch_stride", type=int, default=112)
    parser.add_argument("--ovss_type", default="naclip")
    parser.add_argument("--ovss_backbone", default="ViT-L/14")
    parser.add_argument("--recovery_rate", type=float, default=0.002)
    parser.add_argument("--eval_topk", type=int, default=11)
    parser.add_argument("--eval_dino_temperature", type=float, default=6.0)
    parser.add_argument("--eval_graph_blend", type=float, default=0.45)
    parser.add_argument("--eval_proto_blend", type=float, default=0.35)
    parser.add_argument("--eval_proto_temperature", type=float, default=8.0)
    parser.add_argument("--eval_uncertainty_power", type=float, default=1.6)
    parser.add_argument("--eval_class_prior_power", type=float, default=1.0)
    parser.add_argument("--eval_xfr_decay_steps", type=int, default=5600)
    parser.add_argument("--eval_xfr_min_weight", type=float, default=0.12)
    parser.add_argument("--eval_proto_iterations", type=int, default=2)
    parser.add_argument("--eval_cross_disagree_gate", type=int, default=1)
    parser.add_argument("--eval_disagree_weight", type=float, default=0.5)
    parser.add_argument("--cls_v4_slow_ema_rate", type=float, default=0.99999)
    parser.add_argument("--cls_v4_drift_budget", type=float, default=0.30)
    parser.add_argument("--cls_v4_drift_pullback", type=float, default=0.50)
    parser.add_argument("--cls_anchor_blend", type=float, default=0.50)
    parser.add_argument("--cls_consolidation_warmup", type=int, default=200)
    parser.add_argument("--cls_snapshot_interval", type=int, default=100)
    parser.add_argument("--entropy_skip_factor", type=float, default=1.30)
    parser.add_argument("--dry_run", type=int, default=0)
    return parser.parse_args()


def find_runner(args):
    if args.runner:
        path = Path(args.runner)
        return path if path.exists() else None
    roots = []
    env_root = os.environ.get("MLMP_ROOT", "")
    if env_root:
        roots.append(Path(env_root))
    here = Path(__file__).resolve().parent
    roots.extend([Path.cwd(), here, here.parent])
    for root in roots:
        path = root / "run_ctta_onda.py"
        if path.exists():
            return path
    return None


def build_command(args, runner):
    save_dir = args.save_dir or "outputs_onda_xfcls"
    python_bin = os.environ.get("PYTHON", sys.executable)
    command = [
        python_bin,
        str(runner),
        "--save_dir",
        save_dir,
        "--rain_data_dir",
        args.rain_data_dir,
        "--gt_data_dir",
        args.gt_data_dir,
        "--intensities",
        *args.intensities,
        "--rounds",
        str(args.rounds),
        "--seed",
        str(args.seed),
        "--prompt_integration",
        "text",
        "--vision_outputs",
        "-1",
        "--alpha_cls",
        "0.0",
        "--agreement_threshold",
        "0.7",
        "--eval_refine_mode",
        "xfr",
        "--eval_topk",
        str(args.eval_topk),
        "--eval_dino_temperature",
        str(args.eval_dino_temperature),
        "--eval_graph_blend",
        str(args.eval_graph_blend),
        "--eval_proto_blend",
        str(args.eval_proto_blend),
        "--eval_proto_temperature",
        str(args.eval_proto_temperature),
        "--eval_uncertainty_power",
        str(args.eval_uncertainty_power),
        "--eval_class_prior_power",
        str(args.eval_class_prior_power),
        "--eval_xfr_decay_steps",
        str(args.eval_xfr_decay_steps),
        "--eval_xfr_min_weight",
        str(args.eval_xfr_min_weight),
        "--eval_use_agreement_gate",
        "0",
        "--eval_proto_iterations",
        str(args.eval_proto_iterations),
        "--eval_cross_disagree_gate",
        str(args.eval_cross_disagree_gate),
        "--eval_disagree_weight",
        str(args.eval_disagree_weight),
        "--entropy_skip_factor",
        str(args.entropy_skip_factor),
        "--disable_agreement_weighting",
        "1",
        "--init_resize",
        str(args.init_resize[0]),
        str(args.init_resize[1]),
        "--patch_size",
        str(args.patch_size[0]),
        str(args.patch_size[1]),
        "--patch_stride",
        str(args.patch_stride),
        "--batch_size",
        str(args.batch_size),
        "--class_extensions",
        "--ovss_type",
        args.ovss_type,
        "--ovss_backbone",
        args.ovss_backbone,
        "--method",
        "ours",
        "--adapt",
        "--lr",
        str(args.lr),
        "--steps",
        str(args.steps),
        "--cls_enabled",
        "1",
        "--cls_mode",
        "v4",
        "--cls_v4_slow_ema_rate",
        str(args.cls_v4_slow_ema_rate),
        "--cls_v4_drift_budget",
        str(args.cls_v4_drift_budget),
        "--cls_v4_drift_pullback",
        str(args.cls_v4_drift_pullback),
        "--cls_anchor_blend",
        str(args.cls_anchor_blend),
        "--cls_consolidation_warmup",
        str(args.cls_consolidation_warmup),
        "--cls_snapshot_interval",
        str(args.cls_snapshot_interval),
        "--recovery_rate",
        str(args.recovery_rate),
    ]
    return command


def main():
    args = parse_args()
    runner = find_runner(args)
    if args.dry_run:
        command = build_command(args, runner or "run_ctta_onda.py")
        print(shlex.join(command))
        return
    if runner is None:
        raise SystemExit("No compatible external runner found. Use --dry_run 1 to inspect the example command, or set --runner to a compatible local implementation.")
    command = build_command(args, runner)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
