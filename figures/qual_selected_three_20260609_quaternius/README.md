# Selected Three Quaternius Retarget Trial

Generated on 2026-06-09 as a fast skinned-character visualization trial for
the three selected qualitative prompts.

Main files:

- `fig_qual_selected_three_quaternius_notext.png`
- `fig_qual_selected_three_quaternius_notext.pdf`
- `cells_panel/*.png`
- `previews/character_preview_sheet.png`

Layout:

- Columns: PS-CF / M-Transformer / MoMask
- Rows:
  1. crawl forward, get up, walk forward
  2. crouch, pick up from floor, stand up, turn left
  3. kneel on one knee, pose, stand up, raise both arms
- No text is baked into the figure.

Status:

- This is useful as a proof that the selected HumanML3D joints can be rendered
  as a skinned character.
- It is not the preferred paper-ready figure. The local Quaternius characters
  have stylized proportions and bulky accessories, which obscure the temporal
  overlays.

Reproduction:

- Render cells with `scripts/run_qual_selected_three_20260609_quaternius_cells.sh`.
- The renderer is `tools/render_quaternius_retarget_panel.py`.

Recommended final route:

- Use the BVH files under `../qual_selected_three_20260609_bvh/`.
- Retarget them in Blender to a clean Mixamo-style T-pose character with the
  MoMask/KeeMap mapping files in `assets/mapping.json` or `assets/mapping6.json`.
