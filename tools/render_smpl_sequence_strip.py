#!/usr/bin/env python3
"""Render a plain SMPL motion strip from fitted per-frame PLY meshes.

The renderer centers every selected pose at its pelvis and lays poses out
left-to-right. This is useful for qualitative paper figures where in-place
motions would otherwise collapse into an unreadable pile of overlapping bodies.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--spacing", type=float, default=0.62)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=6.8)
    parser.add_argument("--camera-height", type=float, default=1.1)
    parser.add_argument("--ortho-scale", type=float, default=2.75)
    parser.add_argument("--no-ground", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.68
    return mat


def import_ply(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    new_objects = [obj for obj in bpy.data.objects if obj not in before]
    if not new_objects:
        raise RuntimeError(f"No object imported from {path}")
    return new_objects[0]


def look_at(obj: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def scene_joints(joints: np.ndarray) -> tuple[np.ndarray, float]:
    coords = np.stack([joints[..., 0], joints[..., 2], joints[..., 1]], axis=-1).astype(np.float32)
    min_z = float(coords[..., 2].min())
    coords[..., 2] -= min_z
    return coords, min_z


def transform_mesh(
    mesh: bpy.types.Object,
    root: np.ndarray,
    min_z: float,
    offset_x: float,
    yaw_rad: float,
) -> None:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    for vertex in mesh.data.vertices:
        source = np.asarray(vertex.co[:], dtype=np.float32)
        point = np.asarray([source[0], source[2], source[1] - min_z], dtype=np.float32)
        local_xy = point[:2] - root[:2]
        local_xy = local_xy @ rot.T
        vertex.co = (float(local_xy[0] + offset_x), float(local_xy[1]), float(point[2]))
    mesh.matrix_world = mathutils.Matrix.Identity(4)
    mesh.data.update()


def main() -> None:
    args = parse_args()
    joints = np.load(args.joints)
    coords, min_z = scene_joints(joints)
    frame_ids = np.linspace(0, len(coords) - 1, args.frame_count).astype(int).tolist()
    offsets = (np.arange(len(frame_ids), dtype=np.float32) - (len(frame_ids) - 1) / 2.0) * args.spacing
    yaw_rad = math.radians(args.yaw_deg)

    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds.new("strip_world")
    scene.world.color = (0.985, 0.985, 0.975)

    body_mats = [
        make_mat("body_0", (0.70, 0.72, 0.73, 1.0)),
        make_mat("body_1", (0.62, 0.64, 0.65, 1.0)),
        make_mat("body_2", (0.54, 0.56, 0.57, 1.0)),
        make_mat("body_final", (0.34, 0.36, 0.37, 1.0)),
    ]
    ground_mat = make_mat("ground", (0.90, 0.90, 0.86, 1.0))

    ply_dir = Path(args.ply_dir)
    for display_i, (frame_id, offset_x) in enumerate(zip(frame_ids, offsets)):
        ply_path = ply_dir / f"{frame_id:04d}.ply"
        if not ply_path.exists():
            raise FileNotFoundError(ply_path)
        mesh = import_ply(ply_path)
        mesh.name = f"smpl_frame_{frame_id:04d}"
        transform_mesh(mesh, coords[frame_id, 0], min_z, float(offset_x), yaw_rad)
        mat = body_mats[-1] if display_i == len(frame_ids) - 1 else body_mats[display_i % (len(body_mats) - 1)]
        mesh.data.materials.append(mat)
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.shade_smooth()
        mesh.select_set(False)

    if not args.no_ground:
        plane_width = float(max(abs(offsets[0]), abs(offsets[-1])) * 2.0 + 1.2)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.035))
        ground = bpy.context.object
        ground.name = "ground"
        ground.scale = (plane_width / 2.0, 0.62, 0.012)
        ground.data.materials.append(ground_mat)

    center = mathutils.Vector((0.0, 0.0, 0.86))
    bpy.ops.object.camera_add(location=(0.0, -args.camera_distance, args.camera_height))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = args.ortho_scale
    look_at(camera, center)
    scene.camera = camera

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
