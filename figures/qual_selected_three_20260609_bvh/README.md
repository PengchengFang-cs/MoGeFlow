# Selected Three BVH Exports

Generated on 2026-06-09 for MoMask-style Blender character retargeting.

Contents:

- `pscf/*.bvh`
- `mtransformer/*.bvh`
- `momask/*.bvh`

Each method directory contains three motions:

- `p0_crawl`
- `p1_pickup`
- `p2_kneel`

Each motion has two versions:

- `*.bvh`: direct HumanML3D-to-BVH conversion
- `*_ik.bvh`: conversion with the MoMask naive foot IK pass

The complete set contains 18 BVH files:

- 3 methods
- 3 prompts
- 2 BVH variants per motion

Generation script:

- `tools/export_qual_selected_three_bvh.py`

Environment note:

- Run with `/scratch/pf2m24/miniconda3/envs/fudoki-momask/bin/python`.
- The default Python 3.12 environment cannot import MoMask's old BVH dependency
  `numpy.core.umath_tests`.

Recommended Blender route:

1. Import a generated `.bvh`.
2. Import a clean Mixamo-style character `.fbx` downloaded in T-pose with
   skeleton.
3. Use KeeMapRig with `assets/mapping.json`; try `assets/mapping6.json` if the
   first mapping is unstable.
4. Render the same 7 key-frame overlay layout for all methods.
