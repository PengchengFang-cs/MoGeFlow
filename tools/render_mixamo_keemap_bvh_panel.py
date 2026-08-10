#!/usr/bin/env python3
"""Render a static Mixamo-character panel by retargeting BVH with KeeMap.

Run inside Blender:
  blender -b --python tools/render_mixamo_keemap_bvh_panel.py -- ...
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.render_mixamo_retarget_panel import add_curve, clear_scene, look_at, make_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-fbx", required=True)
    parser.add_argument("--bvh", required=True)
    parser.add_argument("--sample-joints")
    parser.add_argument("--sampling", choices=("uniform", "adaptive"), default="uniform")
    parser.add_argument("--mapping", default=str(REPO / "assets" / "mapping.json"))
    parser.add_argument("--keemap-source", default="/tmp/Keemap-Blender-Rig-ReTargeting-Addon/Source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame-count", type=int, default=7)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=196)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--lens", type=float, default=58.0)
    parser.add_argument("--camera-side", type=float, default=0.70)
    parser.add_argument("--camera-distance", type=float, default=2.10)
    parser.add_argument("--camera-height", type=float, default=0.95)
    parser.add_argument("--display-scale", type=float, default=3.35)
    parser.add_argument("--fallback-line-scale", type=float, default=2.85)
    parser.add_argument("--display-align", choices=("raw", "pca"), default="raw")
    parser.add_argument("--display-perp-scale", type=float, default=0.35)
    parser.add_argument("--material-lighten", type=float, default=0.0)
    parser.add_argument("--material-desaturate", type=float, default=0.0)
    parser.add_argument("--ground-gray", type=float, default=0.88)
    parser.add_argument("--world-gray", type=float, default=0.982)
    parser.add_argument("--show-cavity", action="store_true")
    parser.add_argument("--route-z-ratio", type=float, default=0.06)
    parser.add_argument("--route-width", type=float, default=0.010)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def register_keemap(source_dir: str) -> None:
    source_path = Path(source_dir)
    if not (source_path / "SourceFiles" / "__init__.py").exists():
        raise FileNotFoundError(source_path / "SourceFiles" / "__init__.py")
    sys.path.insert(0, str(source_path))
    addon = importlib.import_module("SourceFiles")
    addon.register()


def import_bvh(path: str, name: str) -> bpy.types.Object:
    try:
        bpy.ops.preferences.addon_enable(module="io_anim_bvh")
    except Exception:
        pass
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(filepath=path)
    new_armatures = [obj for obj in bpy.data.objects if obj not in before and obj.type == "ARMATURE"]
    if len(new_armatures) != 1:
        raise RuntimeError(f"Expected one BVH armature, found {len(new_armatures)}")
    arm = new_armatures[0]
    arm.name = name
    arm.data.name = f"{name}Data"
    return arm


def import_character(path: str, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new_armatures = [obj for obj in bpy.data.objects if obj not in before and obj.type == "ARMATURE"]
    if len(new_armatures) != 1:
        raise RuntimeError(f"Expected one FBX armature, found {len(new_armatures)}")
    arm = new_armatures[0]
    arm.name = name
    arm.data.name = f"{name}Data"
    return arm


def adjust_materials(lighten: float, desaturate: float) -> None:
    if lighten <= 0.0 and desaturate <= 0.0:
        return
    lighten = float(np.clip(lighten, 0.0, 1.0))
    desaturate = float(np.clip(desaturate, 0.0, 1.0))

    def adjust_rgba(rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        rgb = np.asarray(rgba[:3], dtype=np.float32)
        gray = float(np.dot(rgb, np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)))
        rgb = rgb * (1.0 - desaturate) + gray * desaturate
        rgb = rgb * (1.0 - lighten) + lighten
        return (float(np.clip(rgb[0], 0.0, 1.0)), float(np.clip(rgb[1], 0.0, 1.0)), float(np.clip(rgb[2], 0.0, 1.0)), float(rgba[3]))

    for mat in bpy.data.materials:
        mat.diffuse_color = adjust_rgba(tuple(mat.diffuse_color))
        if mat.use_nodes and mat.node_tree is not None:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None and "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = adjust_rgba(tuple(bsdf.inputs["Base Color"].default_value))


def character_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.type == "MESH" and "_pose_" not in obj.name]


def run_keemap_transfer(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    mapping_path: str,
    start_frame: int,
    num_frames: int,
) -> None:
    settings = bpy.context.scene.keemap_settings
    settings.bone_mapping_file = str(Path(mapping_path).resolve())
    bpy.ops.wm.keemap_read_file()
    settings.source_rig_name = source_arm.name
    settings.destination_rig_name = target_arm.name
    settings.start_frame_to_apply = int(start_frame)
    settings.number_of_frames_to_apply = int(num_frames)
    settings.keyframe_every_n_frames = 1
    bpy.ops.wm.perform_animation_transfer()


def bake_character_meshes(
    target_arm: bpy.types.Object,
    frames: list[int],
    collection: bpy.types.Collection,
    display_positions: np.ndarray | None = None,
) -> list[bpy.types.Object]:
    baked_objects: list[bpy.types.Object] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    root_bone = target_root_bone(target_arm)
    for display_i, frame_id in enumerate(frames):
        bpy.context.scene.frame_set(frame_id)
        bpy.context.view_layer.update()
        translation = mathutils.Matrix.Identity(4)
        if display_positions is not None and root_bone is not None:
            root_world = target_arm.matrix_world @ root_bone.matrix.translation
            desired = display_positions[display_i]
            translation = mathutils.Matrix.Translation(
                (
                    float(desired[0] - root_world.x),
                    float(desired[1] - root_world.y),
                    0.0,
                )
            )
        for obj in character_meshes():
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
            baked = bpy.data.objects.new(f"{obj.name}_pose_{display_i:02d}", mesh)
            baked.matrix_world = translation @ obj.matrix_world.copy()
            collection.objects.link(baked)
            for slot in obj.material_slots:
                if slot.material is not None:
                    baked.data.materials.append(slot.material)
            baked_objects.append(baked)
    target_arm.hide_render = True
    target_arm.hide_viewport = True
    for obj in character_meshes():
        obj.hide_render = True
        obj.hide_viewport = True
    return baked_objects


def target_root_bone(target_arm: bpy.types.Object) -> bpy.types.PoseBone | None:
    for name in ("mixamorig:Hips", "mixamorig_Hips", "Hips"):
        bone = target_arm.pose.bones.get(name)
        if bone is not None:
            return bone
    return None


def load_joints(path: str) -> np.ndarray:
    joints = np.load(path)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints [T,J,3], got {joints.shape} from {path}")
    return joints.astype(np.float32)


def uniform_indices(length: int, count: int) -> np.ndarray:
    count = max(2, min(int(count), int(length)))
    return np.linspace(0, length - 1, count).round().astype(int)


def adaptive_indices(joints: np.ndarray, count: int) -> np.ndarray:
    length = len(joints)
    count = max(2, min(int(count), int(length)))
    if count == length:
        return np.arange(length, dtype=int)

    data = joints.copy()
    data[:, :, 1] -= data[:, :, 1].min()
    root = data[:, 0]
    local = data - root[:, None, :]
    body_height = max(float(np.percentile(data[:, :, 1].max(axis=1), 90)), 1e-4)
    root_xy = root[:, [0, 2]]
    root_extent = max(float(np.linalg.norm(np.ptp(root_xy, axis=0))), body_height * 0.25, 1e-4)
    pose_feat = local.reshape(length, -1) / body_height
    root_feat = (root_xy - root_xy[:1]) / root_extent
    feat = np.concatenate([pose_feat * 0.62, root_feat * 1.15], axis=1).astype(np.float32)

    chosen = [0, length - 1]
    min_dist = np.full(length, np.inf, dtype=np.float32)
    for idx in chosen:
        dist = np.linalg.norm(feat - feat[idx], axis=1)
        min_dist = np.minimum(min_dist, dist)

    min_gap = max(1, length // (count * 3))
    while len(chosen) < count:
        scores = min_dist.copy()
        for idx in chosen:
            lo = max(0, idx - min_gap)
            hi = min(length, idx + min_gap + 1)
            scores[lo:hi] = -1.0
        next_idx = int(np.argmax(scores))
        if scores[next_idx] < 0:
            next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
        dist = np.linalg.norm(feat - feat[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return np.asarray(sorted(chosen), dtype=int)


def display_positions_from_joints(
    joints: np.ndarray,
    frame_indices: np.ndarray,
    display_scale: float,
    fallback_line_scale: float,
    display_align: str,
    display_perp_scale: float,
) -> np.ndarray:
    data = joints.copy()
    data[:, :, 1] -= data[:, :, 1].min()
    root_xy = data[:, 0, [0, 2]]
    path = root_xy[frame_indices] - root_xy[frame_indices[0]]
    path_extent = np.ptp(path, axis=0)
    path_scale = max(float(np.linalg.norm(path_extent)), 1e-6)
    if path_scale < 0.45:
        return np.stack(
            [
                np.linspace(-fallback_line_scale / 2.0, fallback_line_scale / 2.0, len(frame_indices)),
                np.zeros(len(frame_indices), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if display_align == "pca":
        points = root_xy[frame_indices]
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            vt = None
        if vt is not None and len(vt) >= 1:
            direction = vt[0].astype(np.float32)
            if float(np.dot(direction, points[-1] - points[0])) < 0.0:
                direction *= -1.0
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            along = centered @ direction
            across = (centered @ normal) * float(display_perp_scale)
            coords = np.stack([along, across], axis=1)
            coord_scale = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])) * 1.35, 1e-6)
            if coord_scale >= 0.20:
                return (coords / coord_scale * float(display_scale)).astype(np.float32)
    return (path / path_scale * float(display_scale)).astype(np.float32)


def select_frames(
    start_frame: int,
    num_frames: int,
    frame_count: int,
    sample_joints: str | None,
    sampling: str,
    display_scale: float,
    fallback_line_scale: float,
    display_align: str,
    display_perp_scale: float,
) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
    if sample_joints is None:
        frame_ids = np.linspace(start_frame, start_frame + num_frames - 1, frame_count).round().astype(int)
        return frame_ids.tolist(), None, None
    joints = load_joints(sample_joints)
    usable = min(len(joints), int(num_frames))
    joints = joints[:usable]
    if sampling == "adaptive":
        frame_indices = adaptive_indices(joints, frame_count)
    else:
        frame_indices = uniform_indices(usable, frame_count)
    display_positions = display_positions_from_joints(
        joints,
        frame_indices,
        display_scale,
        fallback_line_scale,
        display_align,
        display_perp_scale,
    )
    frame_ids = (frame_indices + int(start_frame)).astype(int)
    print("Selected source frame indices:", ",".join(str(int(x)) for x in frame_indices))
    return frame_ids.tolist(), display_positions, frame_indices


def object_bounds(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.asarray([np.inf, np.inf, np.inf], dtype=np.float32)
    maxs = np.asarray([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            values = np.asarray([point.x, point.y, point.z], dtype=np.float32)
            mins = np.minimum(mins, values)
            maxs = np.maximum(maxs, values)
    return mins, maxs


def source_root_points(source_arm: bpy.types.Object, frames: list[int]) -> np.ndarray:
    points = []
    bone = source_arm.pose.bones.get("Hips")
    if bone is None:
        return np.zeros((0, 3), dtype=np.float32)
    for frame_id in frames:
        bpy.context.scene.frame_set(frame_id)
        bpy.context.view_layer.update()
        loc = source_arm.matrix_world @ bone.matrix.translation
        points.append([loc.x, loc.y, loc.z])
    return np.asarray(points, dtype=np.float32)


def configure_render(width: int, height: int, show_cavity: bool, world_gray: float) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = bool(show_cavity)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass
    if scene.world is None:
        scene.world = bpy.data.worlds.new("keemap_panel_world")
    gray = float(np.clip(world_gray, 0.0, 1.0))
    scene.world.color = (gray, gray, gray)


def main() -> None:
    args = parse_args()
    register_keemap(args.keemap_source)
    clear_scene()
    source_arm = import_bvh(args.bvh, "SourceBVH")
    target_arm = import_character(args.character_fbx, "TargetMixamo")
    adjust_materials(float(args.material_lighten), float(args.material_desaturate))

    run_keemap_transfer(source_arm, target_arm, args.mapping, args.start_frame, args.num_frames)
    frame_ids, display_positions, _ = select_frames(
        int(args.start_frame),
        int(args.num_frames),
        int(args.frame_count),
        args.sample_joints,
        args.sampling,
        float(args.display_scale),
        float(args.fallback_line_scale),
        args.display_align,
        float(args.display_perp_scale),
    )
    baked_collection = bpy.data.collections.new("keemap_baked_poses")
    bpy.context.scene.collection.children.link(baked_collection)
    baked_objects = bake_character_meshes(target_arm, frame_ids, baked_collection, display_positions)

    source_arm.hide_render = True
    source_arm.hide_viewport = True
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            obj.hide_render = True
            obj.hide_viewport = True

    configure_render(args.width, args.height, bool(args.show_cavity), float(args.world_gray))
    route_mat = make_mat("generated_root", (1.0, 0.34, 0.10, 1.0))
    ground_gray = float(np.clip(args.ground_gray, 0.0, 1.0))
    ground_mat = make_mat("ground", (ground_gray, ground_gray, ground_gray, 1.0))

    bounds_min, bounds_max = object_bounds(baked_objects)
    center_np = (bounds_min + bounds_max) * 0.5
    span = float(max(bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], (bounds_max[2] - bounds_min[2]) * 1.3, 2.6))
    center = mathutils.Vector((float(center_np[0]), float(center_np[1]), float(center_np[2])))

    if display_positions is not None:
        route_points = np.zeros((len(display_positions), 3), dtype=np.float32)
        route_points[:, :2] = display_positions
    else:
        route_points = source_root_points(source_arm, frame_ids)
    if len(route_points) >= 2:
        route_points[:, 2] = float(bounds_min[2] + (bounds_max[2] - bounds_min[2]) * float(args.route_z_ratio))
        add_curve(route_points, float(args.route_width), route_mat, "generated_root")

    bpy.ops.mesh.primitive_plane_add(size=span * 1.55, location=(center.x, center.y, float(bounds_min[2]) - 0.015))
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
