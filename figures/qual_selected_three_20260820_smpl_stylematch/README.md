# Selected three qualitative motions — SMPL bodies, stick-figure composition

Drop-in replacement for the paper's qualitative figure
(`overleaf-paper/20260610-003631.png`, i.e.
`figures/qual_selected_three_20260609_stick/fig_qual_selected_three_stick_labeled_compact.png`).

- `fig_qual_selected_three_smpl_labeled_compact.png` / `.pdf`
- `cells_panel/*.png` (Blender cell renders), `cells_panel_cropped/*.png` (tight crops)

## What is kept from the stick version

Same 3 prompts (rows), same 3 methods (columns), same 7 keyframes
`[0, 32, 65, 98, 130, 162, 195]`, same rule for spreading the keyframes along the
normalised root path (scaled to 3.6, or a synthetic +-1.8 line when the motion
barely translates), same ground patch, same method-coloured trajectory arrow
(MoGeFlow `#2ca25f`, M-Transformer `#525252`, MoMask `#c2410c`), same view
direction (elev 18, azim -64), same column headers and row captions.

## What changed

Cells are rendered in **Blender 4.2 (Cycles, CPU)**, not matplotlib:
`tools/blender_qual_cell.py`, driven by
`scripts/run_qual_smpl_stylematch_blender_cells.sh`.

matplotlib can only flat-shade a mesh (one constant colour per triangle, no
vertex normals), has no shadows or ambient occlusion, and sorts faces with a
painter's algorithm that mis-orders a body against itself — that is what made
the first matplotlib attempt look faceted and blotchy. Blender gives smooth
shading, a real light rig and contact shadows, i.e. the usual SMPL look.

Bodies are a single opaque grey (`--body-color`, default `#a8a8a8`); there is no
temporal fade or colour ramp. The camera is orthographic and framed to the union
of bodies and ground patch, so proportions are true (the stick version's
`box_aspect=(1.7,1.0,0.85)` squash is deliberately not reproduced — it would
distort the bodies).

Ground contact: `--floor-mode per_frame_up` (default) lifts only those keyframes
whose lowest vertex would fall below the plane, so nothing clips through the
floor and no body is left hovering. `min` reproduces the stick version's single
global offset (no clipping, but most bodies hover); `median` is what produced the
visible floor penetration in the first pass.

Meshes come from the existing SMPLify fits in
`figures/qual_selected_three_20260609_smpl/smpl_fit_iter08/` (8 iterations);
nothing was regenerated or refitted.

## Reproduce

```
srun --jobid=<job> --overlap --ntasks=1 bash -lc '
  bash scripts/run_qual_smpl_stylematch_blender_cells.sh &&
  python tools/render_smpl_qual_grid_stylematch.py --stage compose'
```

`tools/render_smpl_qual_grid_stylematch.py --stage cells` is the superseded
matplotlib cell renderer; `--stage compose` is the panel assembly used above.

Row order:

1. `A person crawls forward on hands and knees, then gets up and walks forward.`
2. `A person crouches down, picks something up from the floor, then stands up and turns left.`
3. `A person kneels on one knee, poses, then stands up and raises both arms.`

Column order: `MoGeFlow`, `M-Transformer`, `MoMask`.

If this figure goes into the paper, the caption sentence
"Motions are visualized by overlaying temporal skeleton poses on the ground plane"
must say SMPL body meshes instead of skeleton poses.
