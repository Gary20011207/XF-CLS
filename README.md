# XF-CLS supplementary reference code

Cross-Foundation Complementary Learning Systems for Continual Test-Time Adaptation in Open-Vocabulary Semantic Segmentation — NeurIPS 2026, submission 26268. Camera-ready update: 7 November 2026.

This small archive illustrates the canonical XF-CLS update and refinement operations. It is not a complete benchmark reproduction package. The paper PDF contains the technical appendices and experimental protocols; they are not duplicated here. No dataset images or pretrained model weights are redistributed.

## Files

- `xfcls.py`: visual-LayerNorm Adam updates, entropy filtering, source-anchored slow consolidation and recovery, and frozen structural output refinement.
- `run_acdc_xfcls.py`, `run_onda_xfcls.py`: example command builders for an external, compatible CTTA runner.
- `run_acdc_xfcls.sh`, `run_onda_xfcls.sh`: corresponding environment-variable examples.
- `test_reference.py`: five small CPU behavioral checks; these are not benchmark experiments.

## Quick checks

The command builders use the Python standard library. The reference component and tests require PyTorch. They were checked with Python 3.12.13 and PyTorch 2.12.1+cu126 using CPU tensors only; no GPU, network, or model download is needed for these checks.

```bash
python test_reference.py
python run_acdc_xfcls.py --data_dir /path/to/ACDC --dry_run 1
python run_onda_xfcls.py --rain_data_dir /path/to/rain --gt_data_dir /path/to/labels --dry_run 1
```

## Reference component interface

Construct `XFCLS(model, XFCLSConfig(...))` with a host model exposing its visual tower as `model.visual`. Its visual LayerNorm affine tensors are the only optimized parameters; the text side and remaining model stay frozen. The class names Hippocampal Adaptation and Neocortical Consolidation denote the design analogy described in the paper.

Call `process_image(logits_fn, agreement_fn, ...)` once per incoming image. `logits_fn()` must return differentiable dense logits `[B,C,H,W]`; `agreement_fn()` must return pre-update multilayer modal agreement. Neither callback receives ground-truth labels. Pass frozen DINO tokens `[B,N,D]` and their `(height,width)` grid separately. `entropy_class_count=19` specifies the driving evaluation vocabulary when extended text entries make the logit channel count different. The returned prediction is recomputed after both inner updates and recovery.

The host is responsible for NACLIP feature extraction, text/class mapping, resizing, overlapping patch aggregation, and evaluation-only handling of ground truth. Standard CLIP pooled image embeddings are not dense segmentation logits. Frozen observer tokens can be reused within an image only when input, preprocessing and precision match all logical observer calls. This reference has no checkpoint/resume interface.

`adapt=False` provides a static control: parameters, optimizer and slow memory remain unchanged while the image-indexed XFR schedule advances. `output_xfr=False` disables output correction; supplied DINO tokens may still inform recovery. Omitting DINO tokens fixes the cross-foundation recovery gate to one.

## Example benchmark commands

The launchers need an external XF-CLS-compatible `run_ctta.py` / `run_ctta_onda.py` and method registration. Use `--runner /path/to/runner.py` or `MLMP_ROOT` to identify that installation. The upstream MLMP code alone is not guaranteed to implement these CTTA arguments. These wrappers do not automatically install or register `xfcls.py`. A missing runner causes an explicit error instead of a successful-looking no-op.

The default commands record the paper's operating points: two inner attempts per image, Adam learning rate `2e-5`, `r0=0.001/0.002` for ACDC/OnDA, and the same XFR settings. The legacy `cls_consolidation_warmup=200` flag is retained for host CLI compatibility but has no effect in canonical v4; its slow EMA begins at the first inner attempt. The benchmark-specific operating points were informed by labeled benchmark results during offline development; no labels enter the online update. Inspect the generated command before using a compatible host installation.

This component was checked against the canonical refinement arithmetic on synthetic tensors and against the stated update ordering. It has not been used to rerun the complete published benchmark suite; no new benchmark result is asserted by this archive.
