# Qualitative SMPL Mesh Figure

Generated on 2026-06-08 from the paper qualitative prompts in
`figures/qualitative_prompts_20260607.txt`.

Main figure:

- `fig_qualitative_smpl_pscf_mtransformer_momask_readable.pdf`
- `fig_qualitative_smpl_pscf_mtransformer_momask_readable.png`

Layout:

- Columns: PS-CF, M-Transformer, MoMask
- Rows: three text prompts
- Each cell: 7 key frames rendered as fitted SMPL meshes

Pipeline:

1. Select 7 key frames from generated HumanML3D joints into `fit_inputs/`.
2. Fit SMPL meshes with:
   `/scratch/pf2m24/projects/Umdd/momask-kvcontext-rl/third_party/HumanML3D-SMPL/joints2smpl/fit_seq.py`
3. Render each cell with:
   `../momask-codes-unifiedflow-base-trange-kvcontext/drawing_results/assets_plan/render_smpl_sequence_panel.py`
4. Compose the final 3x3 grid with PIL.

Notes:

- SMPL fitting used `--num_smplify_iters 8`.
- Blender binary: `/scratch/pf2m24/tools/bin/blender`.
- Full SMPL fit outputs are under `smpl_fit_iter08/`.
