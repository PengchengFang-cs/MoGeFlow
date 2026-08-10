# Quaternius Retarget Trial, 2026-06-08

This folder contains a trial retarget of the corrected qualitative motions to
the local Quaternius Warrior character.

Status: not paper-ready.

Reason:

- The local Quaternius humanoid asset is low-poly and visually unlike the clean
  SMPL-style qualitative figures used in motion-generation papers.
- The quick bone-orientation retargeting is not reliable enough. In the
  `pscf_p0_jacks` scale tests, the jumping-jacks motion largely collapses to a
  T-pose-like arm configuration.

Use the corrected SMPL strip figure instead:

- `../qualitative_selected_20260608_smpl/fig_qualitative_selected_smpl_strip_pscf_mtransformer_momask.pdf`

To make a publishable character-retarget version, use a proper production rig
with a verified HumanML3D/SMPL-to-rig bone mapping, or manually retarget the
selected motions in Blender/KeeMap and then render the same left-to-right strip
layout.
