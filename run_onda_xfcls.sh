#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ONDA_RAIN_ROOT="${ONDA_RAIN_ROOT:-}"
ONDA_GT_ROOT="${ONDA_GT_ROOT:-}"
SAVE_DIR="${SAVE_DIR:-}"

"${PYTHON:-python3}" "$SCRIPT_DIR/run_onda_xfcls.py" \
  --rain_data_dir "$ONDA_RAIN_ROOT" \
  --gt_data_dir "$ONDA_GT_ROOT" \
  --save_dir "$SAVE_DIR" \
  --intensities 25mm 50mm 75mm 100mm 200mm \
  --rounds 5 \
  --seed 0 \
  --lr 0.00002 \
  --steps 2 \
  --recovery_rate 0.002 \
  --eval_topk 11 \
  --eval_dino_temperature 6.0 \
  --eval_graph_blend 0.45 \
  --eval_proto_blend 0.35 \
  --eval_proto_temperature 8.0 \
  --eval_uncertainty_power 1.6 \
  --eval_class_prior_power 1.0 \
  --eval_xfr_decay_steps 5600 \
  --eval_xfr_min_weight 0.12 \
  --eval_proto_iterations 2 \
  --eval_cross_disagree_gate 1 \
  --eval_disagree_weight 0.5 \
  --cls_v4_slow_ema_rate 0.99999 \
  --cls_v4_drift_budget 0.30 \
  --cls_v4_drift_pullback 0.5 \
  --cls_anchor_blend 0.5 \
  --cls_consolidation_warmup 200 \
  --cls_snapshot_interval 100 \
  --entropy_skip_factor 1.3 \
  "$@"
