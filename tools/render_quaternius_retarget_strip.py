#!/usr/bin/env python3
"""Render a left-to-right Quaternius retarget strip from HumanML3D joints.

This is a paper-figure renderer for quick character-skinning previews. It
reuses the bone-orientation retargeting helpers from
`render_quaternius_retarget_panel.py`, but places selected poses in a clean
strip instead of stacking them at the root trajectory.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_quaternius_retarget_panel import (  # noqa: E402
    apply_pose,
    bake_current_mesh,
    character_height,
    configure_render,
    hide_source_objects,
    look_at,
    main_armature,
    make_mat,
    normalize_scene,
    restore_pose_state,
    save_pose_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--character-scale", type=float, default=0.68)
    parser.add_argument("--strip-spacing", type=float, default=0.82)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=6.5)
    parser.add_argument("--camera-height", type=float, default=1.1)
    parser.add_argument("--ortho-scale", type=float, default=3.0)
    parser.add_argument("--no-ground", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def strip_frames(
    coords: np.ndarray,
    frame_ids: list[int],
    body_scale: float,
    strip_spacing: float,
    yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = coords[frame_ids]
    frames = np.zeros_like(selected)
    offsets = np.zeros((len(selected), 3), dtype=np.float32)
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    center = (len(selected) - 1) / 2.0
    for i, frame in enumerate(selected):
        root = frame[0]
        local = (frame - root) * body_scale
        flat = local[:, :2].reshape(-1, 2)
        local[:, :2] = (flat @ rot.T).reshape(local[:, :2].shape)
        frames[i] = local
        offsets[i] = np.asarray([(i - center) * strip_spacing, 0.0, 0.0], dtype=np.float32)
    return frames, offsets


def add_ground(offsets: np.ndarray, char_h: float) -> None:
    ground_mat = make_mat("ground", (0.90, 0.90, 0.86, 1.0))
    width = float(max(abs(offsets[0, 0]), abs(offsets[-1, 0])) * 2.0 + char_h * 1.1)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.025))
    ground = bpy.context.object
    ground.name = "ground"
    ground.scale = (width / 2.0, char_h * 0.22, 0.010)
    ground.data.materials.append(ground_mat)


def configure_camera(char_h: float, camera_distance: float, camera_height: float, ortho_scale: float) -> None:
    center = mathutils.Vector((0.0, 0.0, char_h * 0.52))
    bpy.ops.object.camera_add(location=(0.0, -camera_distance, camera_height))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    look_at(camera, center)
    bpy.context.scene.camera = camera


def main() -> None:
    args = parse_args()
    coords = normalize_scene(np.load(args.joints))
    frame_ids = np.linspace(0, len(coords) - 1, args.frame_count).astype(int).tolist()
    source_height = max(float(coords[..., 2].max() - coords[..., 2].min()), 1.0)
    auto_body_scale = character_height() / source_height
    frames, offsets = strip_frames(
        coords,
        frame_ids,
        auto_body_scale * float(args.body_scale),
        float(args.strip_spacing),
        math.radians(float(args.yaw_deg)),
    )

    arm = main_armature()
    state = save_pose_state(arm)
    baked_collection = bpy.data.collections.new("character_baked_strip")
    bpy.context.scene.collection.children.link(baked_collection)

    for display_i, (frame, offset) in enumerate(zip(frames, offsets)):
        restore_pose_state(arm, state)
        apply_pose(arm, frame)
        bpy.context.view_layer.update()
        bake_current_mesh(display_i, baked_collection, offset, float(args.character_scale))

    hide_source_objects()
    configure_render(args.width, args.height)
    char_h = character_height() * float(args.character_scale)
    if not args.no_ground:
        add_ground(offsets, char_h)
    configure_camera(char_h, float(args.camera_distance), float(args.camera_height), float(args.ortho_scale))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
