# Selected Three Mixamo/KeeMap Retarget Figure

Generated on 2026-06-09 using the local Mixamo-style character:

- `../../Ch46_nonPBR.fbx`

Main outputs:

- `fig_qual_selected_three_mixamo_keemap_notext.png`
- `fig_qual_selected_three_mixamo_keemap_notext.pdf`
- `cells_panel/*.png`

Layout:

- Columns: PS-CF / M-Transformer / MoMask
- Rows:
  1. crawl forward, get up, walk forward
  2. crouch, pick up from floor, stand up, turn left
  3. kneel on one knee, pose, stand up, raise both arms
- No text is baked into the figure.

Character check:

- FBX imports into Blender.
- One mesh: `Ch46`.
- One armature: `Armature`.
- Bone names use the standard `mixamorig:*` prefix, matching `assets/mapping.json`.

Pipeline:

1. Use the BVH files from `../qual_selected_three_20260609_bvh/`.
2. Import each BVH and `Ch46_nonPBR.fbx` in Blender.
3. Load the KeeMap mapping from `assets/mapping.json`.
4. Transfer the 196-frame motion to the Mixamo armature.
5. Bake 7 displayed frames and render one cell.
6. Compose the 3x3 grid without text.

Scripts:

- `tools/render_mixamo_keemap_bvh_panel.py`
- `scripts/run_qual_selected_three_20260609_mixamo_keemap_cells.sh`

Notes:

- This is the correct MoMask-style route: BVH motion retargeted to a skinned
  Mixamo character.
- The character is clean and usable, but visually cartoon-like with a large
  head. For a more serious paper style, replace `Ch46_nonPBR.fbx` with another
  clean `mixamorig:*` Mixamo character and rerun the same script.
- The script clones KeeMap to `/tmp/Keemap-Blender-Rig-ReTargeting-Addon` if it
  is missing.
