# Selected three qualitative motions, SMPL render version

Text-free figure:

- `fig_qual_selected_three_smpl_notext.png`
- `fig_qual_selected_three_smpl_notext.pdf`

Cell renders:

- `cells_panel/*.png`

Row order:

1. `A person crawls forward on hands and knees, then gets up and walks forward.`
2. `A person crouches down, picks something up from the floor, then stands up and turns left.`
3. `A person kneels on one knee, poses, then stands up and raises both arms.`

Column order:

1. `Ours (PS-CF)`
2. `M-Transformer`
3. `MoMask`

SMPL route:

- Source joints: `generation/qual_selected_three_20260609/`
- Keyframes: `[0, 32, 65, 98, 130, 162, 195]`
- SMPLify: `num_smplify_iters=8`
- Meshes: `smpl_fit_iter08/`
- Renderer: sibling KVContext Blender panel renderer.

This version intentionally contains no captions or method labels.
