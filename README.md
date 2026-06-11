# CodeFlow

[![Hugging Face Weights](https://img.shields.io/badge/Download-Weights%20%28HF%29-f7c843.svg)](https://huggingface.co/AmberJar/CodeFlow-HumanML3D)
[![HumanML3D Data](https://img.shields.io/badge/Download-Data%20%28HumanML3D%29-2ea44f.svg)](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data)
[![arXiv](https://img.shields.io/badge/arXiv-2606.11656-b31b1b.svg)](https://arxiv.org/abs/2606.11656)

Part-Structured CodeFlow (PS-CF) is a text-to-motion generation codebase built
around continuous flow matching over frozen motion-code tokenizers. The canonical
release path uses HumanML3D features, a part-aware VQ tokenizer, and a
part-structured DiT prior with one frame token per RVQ timestep.

This repository is the public code release for our CodeFlow experiments. It is
adapted from the MoMask/HumanML3D codebase, but the main training and evaluation
entry points here are CodeFlow-specific.

## What Is Included

- Part-Structured CodeFlow training on HumanML3D.
- Full HumanML3D text-to-motion evaluation during training.
- Top-k best checkpoint tracking by full-eval FID and Top3.
- Frozen tokenizer support for KV-Control part VQ and a MoMask-compatible RVQ
  backend for HumanML3D.
- Portable launch script for the standard PS-CF setting.

The public release documents HumanML3D only. KIT experiment artifacts, logs,
checkpoints, and internal reports are intentionally not part of the GitHub
release.

## Environment

Create the environment and install CLIP through the dependency file:

```bash
conda env create -f environment.yml
conda activate codeflow
```

If you prefer pip in an existing environment:

```bash
pip install -r requirements.txt
```

The training runs used PyTorch with CUDA and mixed precision on H100-class GPUs.
The scripts keep paths configurable so the repository does not depend on a
specific cluster layout.

## Data And Assets

Download or reproduce HumanML3D from the official
[HumanML3D data instructions](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data),
then prepare it in the standard layout:

```text
dataset/HumanML3D/
  train.txt
  val.txt
  test.txt
  Mean.npy
  Std.npy
  new_joint_vecs/
  texts/
```

Full text-to-motion evaluation also needs the HumanML3D evaluator and GloVe:

```text
checkpoints/t2m/Comp_v6_KLD005/
checkpoints/t2m/text_mot_match/model/finest.tar
glove/
```

Large files are ignored by git: datasets, checkpoints, generated motions,
evaluation outputs, logs, and pretrained weights.

## Pretrained Weights

The released HumanML3D checkpoint bundle is hosted on Hugging Face:

```text
AmberJar/CodeFlow-HumanML3D
```

It contains:

- `codeflow/codeflow_hml3d_best_top3_ema.pt`: inference-only EMA CodeFlow checkpoint.
- `rvq/part_vq_hml3d_overlap_best_top3.pth`: frozen part-aware RVQ tokenizer.
- `rvq/skeleton_partition.json`: six-part overlap partition.
- `stats/mean.npy`, `stats/std.npy`: RVQ normalization statistics.

The released CodeFlow checkpoint is the training-time HumanML3D best-Top3
model: Top3 `0.873060`, FID `0.058190`, epoch `290`, step `111070`. Its
architecture is `part_hidden_dim=192`, `hidden_size=1152`, `dropout=0.05`.
This is the strongest released checkpoint; the standard training recipe below
keeps `part_hidden_dim=128` and `hidden_size=768` as the baseline setting.

Download once if you want a local copy:

```bash
huggingface-cli download AmberJar/CodeFlow-HumanML3D \
  --local-dir checkpoints/codeflow_hml3d_release
```

## Inference

Generate from a single prompt with automatic Hugging Face download:

```bash
python gen_codeflow_t2m.py \
  --text_prompt "A person walks forward and waves with the right hand." \
  --motion_length 196 \
  --output_dir generation/codeflow_hml3d \
  --gpu_id 0
```

Or use a downloaded local weight directory:

```bash
python gen_codeflow_t2m.py \
  --local_dir checkpoints/codeflow_hml3d_release \
  --text_prompt "A person walks forward and waves with the right hand." \
  --motion_length 196 \
  --output_dir generation/codeflow_hml3d \
  --gpu_id 0
```

The script saves HumanML3D features, normalized features, RVQ ids, recovered
joint arrays, and `results.json`. To render simple MP4 stick figures, add
`--save_mp4`.

## Standard PS-CF Training

The standard HumanML3D setting is:

```text
representation      part_structured
tokenizer backend   kv_part
code_dim            128
num_parts           6
num_codes           128
part_hidden_dim     128
hidden_size         768
depth               double=6, single=12
num_heads           12
dropout             0.05
batch_size          64
epochs              600
learning rate       1e-4
scheduler           half_cosine, eta_min_ratio=0.01
seed                42
terminal loss       0.0
full eval           test split, every 10 epochs, 96 steps, CFG=6.0
checkpoint select   top-3 by full-eval FID and Top3
```

Set the asset paths and launch:

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

The script writes checkpoints under:

```text
checkpoints/t2m/codeflow_part_structured_pscf_hml3d_standard/
```

You can override `RUN_NAME`, `OUT_DIR`, `CUDA_VISIBLE_DEVICES`,
`PART_HIDDEN_DIM`, and `HIDDEN_SIZE` through environment variables.

## Evaluation

Evaluate a saved CodeFlow checkpoint on HumanML3D test:

```bash
python eval_codeflow_part_structured_t2m.py \
  --checkpoint checkpoints/codeflow_hml3d_release/codeflow/codeflow_hml3d_best_top3_ema.pt \
  --dataset_opt_path checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --data_root dataset/HumanML3D \
  --vq_checkpoint checkpoints/codeflow_hml3d_release/rvq/part_vq_hml3d_overlap_best_top3.pth \
  --vq_partition checkpoints/codeflow_hml3d_release/rvq/skeleton_partition.json \
  --mean_path checkpoints/codeflow_hml3d_release/stats/mean.npy \
  --std_path checkpoints/codeflow_hml3d_release/stats/std.npy \
  --clip_path /path/to/ViT-B-32.pt \
  --repeat_times 20 \
  --steps 96 \
  --cond_scale 6.0 \
  --gpu_id 0
```

Evaluation results are saved as JSON under the checkpoint output directory
unless `--eval_dir` is provided.

## Repository Notes

- `gen_codeflow_t2m.py` is the public text-to-motion inference entry.
- `train_codeflow_part_structured.py` is the canonical PS-CF training entry.
- `train_codeflow.py` contains the shared training loop, checkpoint selection,
  full-eval scheduling, and optimizer logic.
- `models/codeflow/part_structured_motion_code_flow.py` contains the canonical
  part-structured model.
- `models/codeflow/momask_vq.py` is a compatibility wrapper for HumanML3D
  MoMask RVQ checkpoints. The project branding remains CodeFlow.

## License

The code is released under the MIT license. The original upstream copyright
notice is preserved in `LICENSE`; additional CodeFlow changes are released under
the same license.

## Acknowledgement

If you use this code or find it helpful for your research, please cite:

```bibtex
@article{fang2026mogeflow,
  title={MoGeFlow: Flowing Through Motion Codebook Geometry for Text-to-Motion Generation},
  author={Fang, Pengcheng and Sun, Tengjiao and Zhan, Xiaoyu and Cai, Xiaohao and Fu, Dongjie},
  journal={arXiv preprint arXiv:2606.11656},
  year={2026},
  eprint={2606.11656},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2606.11656}
}
```
