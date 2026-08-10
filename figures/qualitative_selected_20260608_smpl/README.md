# Qualitative SMPL Figure, 2026-06-08

This folder contains the corrected qualitative comparison after rejecting the
previous three failed prompts.

Final figure:

- `fig_qualitative_selected_smpl_strip_pscf_mtransformer_momask.pdf`
- `fig_qualitative_selected_smpl_strip_pscf_mtransformer_momask.png`

Rows:

- Jumping jacks
- Sit down
- Jog in place

Columns:

- Ours (PS-CF)
- M-Transformer
- MoMask

Generation notes:

- Source motions were screened first with skeleton previews.
- SMPLify was run on 7 keyframes per method/action, producing 63 fitted PLY
  meshes in `smpl_fit_iter08/`.
- The final figure uses `tools/render_smpl_sequence_strip.py`, which lays the
  fitted SMPL meshes left-to-right to avoid unreadable overlap on in-place
  motions.
- The older folders `figures/qualitative_20260608_smpl/`,
  `figures/qualitative_20260608_retarget/`, and `figures/qualitative_20260607/`
  are deprecated for paper use.
