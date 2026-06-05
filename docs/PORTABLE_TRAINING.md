# Portable Training Notes

This repository is prepared for code-only transfer. Large or generated files are
intentionally ignored by git: datasets, evaluator checkpoints, model
checkpoints, logs, run outputs, GloVe files, generated videos, and internal
reports.

## HumanML3D Layout

Prepare HumanML3D in the standard text-to-motion layout:

```text
HumanML3D/
  Mean.npy
  Std.npy
  train.txt
  val.txt
  test.txt
  new_joint_vecs/
  texts/
```

Full evaluation also needs:

```text
checkpoints/t2m/Comp_v6_KLD005/
checkpoints/t2m/text_mot_match/model/finest.tar
glove/
```

## Standard CodeFlow Training

The canonical public launch script is:

```bash
DATA_ROOT=/path/to/HumanML3D \
KV_ROOT=/path/to/KV-Control \
VQ_CHECKPOINT=/path/to/part_vq_best_top3.pth \
VQ_PARTITION=/path/to/skeleton_partition.json \
MEAN_PATH=/path/to/mean.npy \
STD_PATH=/path/to/std.npy \
CLIP_PATH=/path/to/ViT-B-32.pt \
bash scripts/launch/train_humanml3d_pscf_standard.sh
```

The default script uses:

- HumanML3D/T2M data.
- Part-Structured CodeFlow.
- 6 tokenizer groups with 128-dim code embeddings.
- `part_hidden_dim=128` and `hidden_size=768`.
- Dropout `0.05`.
- Batch size `64`.
- 600 epochs.
- Half-cosine LR schedule from `1e-4`.
- Test-split full evaluation every 10 epochs.
- Top-3 checkpoint retention by full-eval FID and Top3.

## Git Policy

Do not commit:

- `checkpoints/`
- `dataset/` or `datasets/`
- `glove/`
- `log/`, `logs/`, `run_logs/`
- internal narrative or handoff reports
- `*.pt`, `*.pth`, `*.tar`, `*.ckpt`, `*.npy`, `*.npz`, `*.pkl`, `*.zip`

The `.gitignore` covers these patterns.
