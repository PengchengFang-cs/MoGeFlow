#!/usr/bin/env python3
"""Render a static Mixamo-character panel from HumanML3D joints.

This is a fast automatic preview path. The paper-ready MoMask-style path should
still use BVH plus a verified Blender/KeeMap retarget when available.

Run inside Blender:
  blender -b --python tools/render_mixamo_retarget_panel.py -- ...
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
    parser.add_argument("--character-fbx", required=True)
    parser.add_argument("--joints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--align-route-x", action="store_true")
    parser.add_argument("--body-scale", type=float, default=1.0)
    parser.add_argument("--path-scale", type=float, default=1.0)
    parser.add_argument("--character-scale", type=float, default=0.68)
    parser.add_argument("--lens", type=float, default=58.0)
    parser.add_argument("--camera-side", type=float, default=0.62)
    parser.add_argument("--camera-distance", type=float, default=2.05)
    parser.add_argument("--camera-height", type=float, default=0.92)
    parser.add_argument("--bone-prefix", default="")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def normalize_scene(joints: np.ndarray) -> np.ndarray:
    coords = np.stack([joints[..., 0], joints[..., 2], joints[..., 1]], axis=-1).astype(np.float32)
    root = coords[:, 0]
    center_xy = root[:, :2].mean(axis=0)
    min_z = float(coords[..., 2].min())
    coords[..., 0] -= center_xy[0]
    coords[..., 1] -= center_xy[1]
    coords[..., 2] -= min_z
    return coords


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


def import_character(fbx_path: str) -> bpy.types.Object:
    clear_scene()
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature in {fbx_path}, found {len(armatures)}")
    return armatures[0]


def infer_bone_prefix(arm: bpy.types.Object, explicit_prefix: str) -> str:
    if explicit_prefix:
        return explicit_prefix
    bone_names = {bone.name for bone in arm.pose.bones}
    if "mixamorig:Hips" in bone_names:
        return "mixamorig:"
    if "mixamorig_Hips" in bone_names:
        return "mixamorig_"
    return ""


def character_height() -> float:
    z_min = math.inf
    z_max = -math.inf
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "_pose_" in obj.name:
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


def orient_bone(arm: bpy.types.Object, name: str, head: np.ndarray, tail: np.ndarray) -> None:
    pose_bone = arm.pose.bones.get(name)
    rest_bone = arm.data.bones.get(name)
    if pose_bone is None or rest_bone is None:
        return
    desired = mathutils.Vector(
        (
            float(tail[0] - head[0]),
            float(tail[1] - head[1]),
            float(tail[2] - head[2]),
        )
    )
    rest = rest_bone.tail_local - rest_bone.head_local
    if desired.length < 1e-5 or rest.length < 1e-5:
        return
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = rest.normalized().rotation_difference(desired.normalized())


def hand_tail(wrist: np.ndarray, elbow: np.ndarray) -> np.ndarray:
    direction = wrist - elbow
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        direction = np.asarray([0.12, 0.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm * 0.16
    return wrist + direction


def head_tail(head: np.ndarray, neck: np.ndarray) -> np.ndarray:
    direction = head - neck
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        direction = np.asarray([0.0, 0.0, 0.2], dtype=np.float32)
    else:
        direction = direction / norm * 0.24
    return head + direction


def apply_pose(arm: bpy.types.Object, frame: np.ndarray, prefix: str) -> None:
    root = frame[0]
    l_hip, r_hip = frame[1], frame[2]
    spine1, spine2, spine3 = frame[3], frame[6], frame[9]
    l_knee, r_knee = frame[4], frame[5]
    l_ankle, r_ankle = frame[7], frame[8]
    l_foot, r_foot = frame[10], frame[11]
    neck, head = frame[12], frame[15]
    l_shoulder, r_shoulder = frame[16], frame[17]
    l_elbow, r_elbow = frame[18], frame[19]
    l_wrist, r_wrist = frame[20], frame[21]

    hips_mid = (l_hip + r_hip) * 0.5
    orient_bone(arm, f"{prefix}Hips", hips_mid, spine1)
    orient_bone(arm, f"{prefix}Spine", spine1, spine2)
    orient_bone(arm, f"{prefix}Spine1", spine2, spine3)
    orient_bone(arm, f"{prefix}Spine2", spine3, neck)
    orient_bone(arm, f"{prefix}Neck", neck, head)
    orient_bone(arm, f"{prefix}Head", head, head_tail(head, neck))

    orient_bone(arm, f"{prefix}LeftShoulder", neck, l_shoulder)
    orient_bone(arm, f"{prefix}LeftArm", l_shoulder, l_elbow)
    orient_bone(arm, f"{prefix}LeftForeArm", l_elbow, l_wrist)
    orient_bone(arm, f"{prefix}LeftHand", l_wrist, hand_tail(l_wrist, l_elbow))

    orient_bone(arm, f"{prefix}RightShoulder", neck, r_shoulder)
    orient_bone(arm, f"{prefix}RightArm", r_shoulder, r_elbow)
    orient_bone(arm, f"{prefix}RightForeArm", r_elbow, r_wrist)
    orient_bone(arm, f"{prefix}RightHand", r_wrist, hand_tail(r_wrist, r_elbow))

    orient_bone(arm, f"{prefix}LeftUpLeg", l_hip, l_knee)
    orient_bone(arm, f"{prefix}LeftLeg", l_knee, l_ankle)
    orient_bone(arm, f"{prefix}LeftFoot", l_ankle, l_foot)
    orient_bone(arm, f"{prefix}RightUpLeg", r_hip, r_knee)
    orient_bone(arm, f"{prefix}RightLeg", r_knee, r_ankle)
    orient_bone(arm, f"{prefix}RightFoot", r_ankle, r_foot)


def bake_current_mesh(frame_id: int, collection: bpy.types.Collection, offset: np.ndarray, character_scale: float) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "_pose_" in obj.name:
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        baked = bpy.data.objects.new(f"{obj.name}_pose_{frame_id:02d}", mesh)
        baked.matrix_world = (
            mathutils.Matrix.Translation(
                (float(offset[0]), float(offset[1]), float(offset[2]))
            )
            @ mathutils.Matrix.Scale(float(character_scale), 4)
            @ obj.matrix_world
        )
        collection.objects.link(baked)
        for slot in obj.material_slots:
            if slot.material is not None:
                baked.data.materials.append(slot.material)


def hide_source_objects() -> None:
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" or (obj.type == "MESH" and "_pose_" not in obj.name):
            obj.hide_render = True
            obj.hide_viewport = True


def configure_render(width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("mixamo_panel_world")
    scene.world.color = (0.982, 0.982, 0.962)


def scaled_frames(
    coords: np.ndarray,
    frame_ids: list[int],
    body_scale: float,
    path_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = coords[frame_ids]
    local = np.zeros_like(selected)
    offsets = np.zeros((len(selected), 3), dtype=np.float32)
    root_z_mean = float(selected[:, 0, 2].mean())
    for i, frame in enumerate(selected):
        root = frame[0]
        local[i] = (frame - root) * body_scale
        offsets[i] = np.asarray(
            [root[0] * path_scale, root[1] * path_scale, (root[2] - root_z_mean) * body_scale],
            dtype=np.float32,
        )
    return local, offsets


def main() -> None:
    args = parse_args()
    arm = import_character(args.character_fbx)
    prefix = infer_bone_prefix(arm, args.bone_prefix)
    coords = normalize_scene(np.load(args.joints))
    if args.align_route_x:
        coords = rotate_xy_to_route_x(coords, route_forward_direction(coords[:, 0, :2]))
    frame_ids = np.linspace(0, len(coords) - 1, args.frame_count).astype(int).tolist()
    source_height = max(float(coords[..., 2].max() - coords[..., 2].min()), 1.0)
    auto_body_scale = character_height() / source_height
    frames, offsets = scaled_frames(
        coords,
        frame_ids,
        auto_body_scale * float(args.body_scale),
        float(args.path_scale),
    )

    state = save_pose_state(arm)
    baked_collection = bpy.data.collections.new("mixamo_baked_poses")
    bpy.context.scene.collection.children.link(baked_collection)

    for display_i, (frame, offset) in enumerate(zip(frames, offsets)):
        restore_pose_state(arm, state)
        apply_pose(arm, frame, prefix)
        bpy.context.view_layer.update()
        bake_current_mesh(display_i, baked_collection, offset, float(args.character_scale))

    hide_source_objects()
    configure_render(args.width, args.height)
    route_mat = make_mat("generated_root", (1.0, 0.34, 0.10, 1.0))
    ground_mat = make_mat("ground", (0.88, 0.88, 0.84, 1.0))
    route_points = offsets.copy()
    route_points[:, 2] += character_height() * float(args.character_scale) * 0.42
    add_curve(route_points, 0.014, route_mat, "generated_root")

    char_h = character_height() * float(args.character_scale)
    bounds_min = offsets.min(axis=0) + np.asarray([-0.9, -0.9, 0.0], dtype=np.float32)
    bounds_max = offsets.max(axis=0) + np.asarray([0.9, 0.9, char_h], dtype=np.float32)
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
