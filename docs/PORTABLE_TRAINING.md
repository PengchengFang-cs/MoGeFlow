# Portable Training Notes

This repository is prepared for code-only transfer. Large or generated files are intentionally ignored by git: datasets, evaluator checkpoints, model checkpoints, logs, run outputs, GloVe files, and videos.

## KIT Part-VQ Training

Prepare KIT-ML in the usual MoMask/HumanML3D layout:

```text
KIT-ML/
  Mean.npy
  Std.npy
  train.txt
  val.txt
  test.txt
  new_joint_vecs/
  texts/
```

Then run:

```bash
DATA_ROOT=/path/to/KIT-ML \
OUTPUT_DIR=/path/to/output/vqvae_kit_pscf \
bash scripts/launch/train_kit_part_vq.sh
```

The part-aware VQ implementation is vendored under `kvctrl/`, so an external KV-Control checkout is not required. If you want to compare against a separate KV-Control checkout, pass `--kv_root /path/to/KV-Control` to `tools/train_kv_part_vq.py`.

The trainer uses the KV-Control VQ configuration:

- 6 body parts
- 128 codes per part
- code dim 128
- EMA-reset quantizer with `mu=0.99`
- smooth-L1 reconstruction loss
- commitment weight `0.02`
- explicit local-position loss weight `0.5`
- 300k iterations

## Evaluation Assets

Full text-motion evaluation uses files that are not committed:

```text
checkpoints/kit/Comp_v6_KLD005/
checkpoints/kit/text_mot_match/model/finest.tar
glove/
```

Place those assets in the same relative paths before running with `--eval_iter > 0`. For train-only smoke tests, pass `--disable_eval`.

## Git Policy

Do not commit:

- `checkpoints/`
- `dataset/`
- `glove/`
- `log/`, `logs/`, `run_logs/`
- `*.pt`, `*.pth`, `*.tar`, `*.npy`, `*.pkl`, `*.zip`

The `.gitignore` already covers these patterns.
