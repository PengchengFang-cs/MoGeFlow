#!/usr/bin/env python3
"""Render a static Rain-character panel from HumanML3D joints.

Run inside Blender:
  blender --enable-autoexec -b rain_v3.2.blend --python tools/render_rain_retarget_panel.py -- ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


LEFT = ".L"
RIGHT = ".R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--align-route-x", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--lens", type=float, default=58.0)
    parser.add_argument("--camera-side", type=float, default=0.62)
    parser.add_argument("--camera-distance", type=float, default=2.05)
    parser.add_argument("--camera-height", type=float, default=0.92)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def normalize_scene(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coords = np.stack([joints[..., 0], joints[..., 2], joints[..., 1]], axis=-1).astype(np.float32)
    root = coords[:, 0]
    center_xy = root[:, :2].mean(axis=0)
    min_z = float(coords[..., 2].min())
    coords[..., 0] -= center_xy[0]
    coords[..., 1] -= center_xy[1]
    coords[..., 2] -= min_z
    return coords, center_xy, min_z


def route_forward_direction(root_xy: np.ndarray) -> np.ndarray:
    centered = root_xy - root_xy.mean(axis=0, keepdims=True)
    if len(centered) < 2 or float(np.linalg.norm(centered)) < 1e-5:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    path_dir = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    displacement = root_xy[-1] - root_xy[0]
    if float(np.dot(path_dir, displacement)) < 0:
        path_dir = -path_dir
    return path_dir / max(float(np.linalg.norm(path_dir)), 1e-6)


def rotate_xy_to_route_x(points: np.ndarray, direction_xy: np.ndarray) -> np.ndarray:
    angle = math.atan2(float(direction_xy[1]), float(direction_xy[0]))
    c = math.cos(-angle)
    s = math.sin(-angle)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    out = points.copy()
    flat = out[..., :2].reshape(-1, 2)
    out[..., :2] = (flat @ rot.T).reshape(out[..., :2].shape)
    return out


def look_at(obj: bpy.types.Object, target: mathutils.Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_mat(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.62
    return mat


def add_curve(points: np.ndarray, bevel_depth: float, mat: bpy.types.Material, name: str) -> None:
    if len(points) < 2:
        return
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 3
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    polyline = curve.splines.new("POLY")
    polyline.points.add(len(points) - 1)
    for pnt, xyz in zip(polyline.points, points):
        pnt.co = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)


def main_armature() -> bpy.types.Object:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found in the Rain blend file")
    return armatures[0]


def rain_height() -> float:
    z_min = math.inf
    z_max = -math.inf
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.name.startswith("GEO-rain"):
            continue
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            z_min = min(z_min, float(point.z))
            z_max = max(z_max, float(point.z))
    return max(z_max - z_min, 1.0)


def save_pose_state(arm: bpy.types.Object) -> dict[str, tuple[mathutils.Vector, mathutils.Euler, mathutils.Vector]]:
    return {
        bone.name: (bone.location.copy(), bone.rotation_euler.copy(), bone.scale.copy())
        for bone in arm.pose.bones
    }


def restore_pose_state(
    arm: bpy.types.Object,
    state: dict[str, tuple[mathutils.Vector, mathutils.Euler, mathutils.Vector]],
) -> None:
    for bone in arm.pose.bones:
        loc, rot, scale = state[bone.name]
        bone.location = loc
        bone.rotation_euler = rot
        bone.scale = scale


def set_translation(arm: bpy.types.Object, name: str, point: np.ndarray) -> None:
    bone = arm.pose.bones.get(name)
    if bone is None:
        return
    mat = bone.matrix.copy()
    mat.translation = mathutils.Vector((float(point[0]), float(point[1]), float(point[2])))
    bone.matrix = mat


def set_ikfk_defaults(arm: bpy.types.Object) -> None:
    prop = arm.pose.bones.get("Properties_IKFK")
    if prop is None:
        return
    for key in ("ik_arm_left", "ik_arm_right", "ik_leg_left", "ik_leg_right", "ik_spine"):
        if key in prop:
            prop[key] = 1.0
    for key in ("ik_stretch_arms", "ik_stretch_spine"):
        if key in prop:
            prop[key] = 0.0


def apply_rain_pose(arm: bpy.types.Object, frame: np.ndarray) -> None:
    # HumanML3D joint indices.
    root = frame[0]
    l_hip, r_hip = frame[1], frame[2]
    spine3, neck, head = frame[9], frame[12], frame[15]
    l_knee, r_knee = frame[4], frame[5]
    l_foot, r_foot = frame[10], frame[11]
    l_elbow, r_elbow = frame[18], frame[19]
    l_wrist, r_wrist = frame[20], frame[21]

    hips = (l_hip + r_hip) * 0.5
    chest = (spine3 + neck) * 0.5
    set_translation(arm, "ROOT", root * np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
    set_translation(arm, "MSTR-Pelvis", hips)
    set_translation(arm, "MSTR-Hips", hips)
    set_translation(arm, "MSTR-Chest", chest)
    set_translation(arm, "FK-Head", head)

    # Rain uses .L for positive X in its default file; HumanML3D left/right match after x/z/y conversion.
    set_translation(arm, f"IK-Hand{LEFT}", l_wrist)
    set_translation(arm, f"IK-Hand{RIGHT}", r_wrist)
    set_translation(arm, f"IK-Foot{LEFT}", l_foot)
    set_translation(arm, f"IK-Foot{RIGHT}", r_foot)
    set_translation(arm, f"IK-Pole-Forearm{LEFT}", l_elbow)
    set_translation(arm, f"IK-Pole-Forearm{RIGHT}", r_elbow)
    set_translation(arm, f"IK-Pole-Shin{LEFT}", l_knee)
    set_translation(arm, f"IK-Pole-Shin{RIGHT}", r_knee)


def bake_current_rain_mesh(frame_id: int, collection: bpy.types.Collection) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.name.startswith("GEO-rain"):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        baked = bpy.data.objects.new(f"{obj.name}_pose_{frame_id:02d}", mesh)
        baked.matrix_world = obj.matrix_world.copy()
        collection.objects.link(baked)
        for slot in obj.material_slots:
            if slot.material is not None:
                baked.data.materials.append(slot.material)


def hide_rain_rig() -> None:
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" or obj.name.startswith("WGT-"):
            obj.hide_render = True
            obj.hide_viewport = True
        elif obj.name.startswith("GEO-rain") and "_pose_" not in obj.name:
            obj.hide_render = True
            obj.hide_viewport = True


def configure_render(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("rain_panel_world")
    scene.world.color = (0.982, 0.982, 0.962)


def main() -> None:
    args = parse_args()
    joints = np.load(args.joints)
    coords, _, _ = normalize_scene(joints)
    if args.align_route_x:
        coords = rotate_xy_to_route_x(coords, route_forward_direction(coords[:, 0, :2]))
    frame_ids = np.linspace(0, len(coords) - 1, args.frame_count).astype(int).tolist()
    coords = coords[frame_ids]
    source_height = max(float(coords[..., 2].max() - coords[..., 2].min()), 1.0)
    scale = rain_height() / source_height * float(args.scale)
    coords *= scale

    arm = main_armature()
    state = save_pose_state(arm)
    set_ikfk_defaults(arm)
    baked_collection = bpy.data.collections.new("rain_baked_poses")
    bpy.context.scene.collection.children.link(baked_collection)

    for display_i, frame in enumerate(coords):
        restore_pose_state(arm, state)
        set_ikfk_defaults(arm)
        apply_rain_pose(arm, frame)
        bpy.context.view_layer.update()
        bake_current_rain_mesh(display_i, baked_collection)

    hide_rain_rig()
    configure_render(args.width, args.height)
    route_mat = make_mat("generated_root", (1.0, 0.34, 0.10, 1.0))
    ground_mat = make_mat("ground", (0.88, 0.88, 0.84, 1.0))
    add_curve(coords[:, 0], 0.014, route_mat, "generated_root")

    bounds_min = coords.reshape(-1, 3).min(axis=0)
    bounds_max = coords.reshape(-1, 3).max(axis=0)
    center = mathutils.Vector(
        (
            (bounds_min[0] + bounds_max[0]) / 2,
            (bounds_min[1] + bounds_max[1]) / 2,
            (bounds_min[2] + bounds_max[2]) / 2,
        )
    )
    span = float(max(bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], (bounds_max[2] - bounds_min[2]) * 1.3, 2.6))
    bpy.ops.mesh.primitive_plane_add(size=span * 1.55, location=(center.x, center.y, -0.015))
    ground = bpy.context.object
    ground.name = "ground"
    ground.data.materials.append(ground_mat)

    camera_pos = (
        center.x + span * args.camera_side,
        center.y - span * args.camera_distance,
        center.z + span * args.camera_height,
    )
    bpy.ops.object.camera_add(location=camera_pos)
    camera = bpy.context.object
    camera.data.lens = args.lens
    look_at(camera, center)
    bpy.context.scene.camera = camera

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
