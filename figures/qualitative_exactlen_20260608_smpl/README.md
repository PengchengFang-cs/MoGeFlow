# Qualitative SMPL figure, 2026-06-08

This redraw fixes the earlier qualitative samples by using exact HumanML3D test-set captions and rounded ground-truth motion lengths, and by sampling PS-CF from the EMA weights of the best-FID checkpoint.

## Final files

- `fig_qualitative_exactlen_smpl_panel_pscf_mtransformer_momask.png`
- `fig_qualitative_exactlen_smpl_panel_pscf_mtransformer_momask.pdf`
- `_contact_check_panel.png` is only a quick visual contact sheet.

## Rows

| Row | Test ID | Caption | Rounded length |
| --- | --- | --- | --- |
| 1 | `013846` | `a person is doing jumping jacks.` | 72 |
| 2 | `001632` | `a man sits down and then stays still.` | 196 |
| 3 | `002795` | `a person jogs in place.` | 92 |

## Columns

- `Ours (PS-CF)`: `generation/debug_test_caption_exact_length_20260608/w200_bestfid_ema/`
- `M-Transformer`: `generation/paper_qual_mtransformer_exactlen_20260608/`
- `MoMask`: `generation/paper_qual_momask_exactlen_20260608/`

## PS-CF sampling

- Checkpoint: `checkpoints/t2m/codeflow_part_structured_newvqtop3_w200_p256_h1536_d6s12_drop005_w0_b64_lr1e4_e600_eval10from0_seed42_test_h200_blossom_g0_20260606/model/best_fid.pt`
- Weight source: `ema`
- Prompt file: `generation/debug_test_caption_exact_length_20260608/prompts.txt`

## Rendering

- Route: HumanML3D joint sequence -> SMPLify -> SMPL mesh panel render.
- SMPLify output: `smpl_fit_iter08/`
- Keyframes per sequence: 7.
- Renderer source: sibling KVContext SMPL panel renderer.

The previous qualitative folders using hand-written prompts and raw model weights should be treated as deprecated for paper figures.
