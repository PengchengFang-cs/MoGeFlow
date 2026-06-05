# CodeFlow

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

Prepare HumanML3D in the standard layout:

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
  --checkpoint checkpoints/t2m/codeflow_part_structured_pscf_hml3d_standard/model/best_top3.pt \
  --dataset_opt_path checkpoints/t2m/Comp_v6_KLD005/opt.txt \
  --data_root dataset/HumanML3D \
  --vq_checkpoint /path/to/part_vq_best_top3.pth \
  --vq_partition /path/to/skeleton_partition.json \
  --mean_path /path/to/mean.npy \
  --std_path /path/to/std.npy \
  --clip_path /path/to/ViT-B-32.pt \
  --repeat_times 20 \
  --steps 96 \
  --cond_scale 6.0 \
  --gpu_id 0
```

Evaluation results are saved as JSON under the checkpoint output directory
unless `--eval_dir` is provided.

## Repository Notes

- `train_codeflow_part_structured.py` is the canonical PS-CF training entry.
- `train_codeflow.py` contains the shared training loop, checkpoint selection,
  full-eval scheduling, and optimizer logic.
- `models/codeflow/part_structured_motion_code_flow.py` contains the canonical
  part-structured model.
- `models/codeflow/momask_vq.py` is a compatibility wrapper for HumanML3D
  MoMask RVQ checkpoints. The project branding remains CodeFlow.

## Acknowledgements

This repository builds on the open-source HumanML3D/MoMask ecosystem and its
evaluation stack. We also acknowledge the related open-source projects used by
the upstream codebase, including vector-quantize-pytorch, T2M-GPT, MDM, MLD,
Muse, and deep-motion-editing.

If you use the upstream MoMask components, please also cite the original work:

```bibtex
@inproceedings{guo2024momask,
  title={Momask: Generative masked modeling of 3d human motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1900--1910},
  year={2024}
}
```

## License

The code is released under the MIT license. The original upstream copyright
notice is preserved in `LICENSE`; additional CodeFlow changes are released under
the same license.
