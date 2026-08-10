# Qualitative Character Retarget Figure

Generated on 2026-06-08 as a skinned-character alternative to the SMPL mesh figure.

Main figure:

- `fig_qualitative_retarget_quaternius_pscf_mtransformer_momask.pdf`
- `fig_qualitative_retarget_quaternius_pscf_mtransformer_momask.png`

Layout:

- Columns: PS-CF, M-Transformer, MoMask
- Rows: three text prompts
- Each cell: 7 key frames rendered with a Quaternius humanoid character

Character asset:

- Quaternius RPG Characters, Humanoid Rig Version, `Warrior.blend`
- Local path: `assets/quaternius_rpg/RPG Characters - Nov 2020/Humanoid Rig Versions/Blends/Warrior.blend`
- License file in the downloaded package states CC0 1.0 Universal.

Pipeline:

1. Downloaded the Quaternius RPG character pack with `gdown`.
2. Rendered each cell with `tools/render_quaternius_retarget_panel.py`.
3. The renderer approximates retargeting by rotating the simple humanoid rig bones from HumanML3D joint segment directions, baking each posed mesh, and placing the baked characters along the root trajectory.
4. Composed the final 3x3 grid with PIL.

Retarget-ready BVH files:

- PS-CF BVH files: `bvh/pscf/pscf_p*.bvh`
- M-Transformer BVH files: `bvh/mtransformer/mtransformer_p*.bvh`
- MoMask BVH files are in `generation/paper_qual_momask_20260607/animations/`.

Notes:

- This is a fast automatic character-retarget approximation for paper figures.
- A Rain v3.3 Blender Studio asset was also downloaded, but its CloudRig control layer was not suitable for direct automatic joint-to-IK retargeting without manual rig setup.
