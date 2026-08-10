# Balanced PCA Mixamo Render

This directory contains the recommended skinned-character qualitative render for the three selected prompts.

Outputs:
- `fig_qual_selected_three_mixamo_keemap_balanced_pca_light_notext.png`
- `fig_qual_selected_three_mixamo_keemap_balanced_pca_light_notext.pdf`
- `fig_qual_selected_three_mixamo_keemap_balanced_pca_notext.png`
- `fig_qual_selected_three_mixamo_keemap_balanced_pca_notext.pdf`
- `cells_panel/*.png`

Rendering choices:
- Retarget raw BVH files to `Ch46_nonPBR.fbx` with KeeMap.
- Select six key poses with adaptive sampling from the original joint sequences.
- Place poses using PCA-aligned root trajectories so the main motion direction is horizontally readable.
- Use lighter character/material settings and a post-processed light floor for the paper-ready version.

The `*_light_notext` files are the recommended versions for the paper figure.
