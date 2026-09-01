#!/usr/bin/env python3
"""Blender cell renderer for the SMPL qualitative grid.

Reproduces the composition of tools/render_motion_qualitative_grid.py (keyframes
spread along the normalised root path, ground patch, method-coloured trajectory
arrow, elev=18 / azim=-64 view) but renders the fitted SMPL meshes properly:
smooth shading, real lighting, contact shadows.

Run as:  blender -b --python tools/blender_qual_cell.py -- --ply-dir ... --joints ... --out ...
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

ELEV, AZIM = 18.0, -64.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply-dir", required=True)
    p.add_argument("--joints", required=True, help="full source joints npy [T,J,3]")
    p.add_argument("--out", required=True)
    p.add_argument("--num-keyframes", type=int, default=7)
    p.add_argument("--arrow-color", default="#525252")
    p.add_argument("--body-color", default="#a8a8a8")
    p.add_argument("--width", type=int, default=1800)
    p.add_argument("--height", type=int, default=1150)
    p.add_argument("--engine", choices=("cycles", "workbench"), default="cycles")
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--floor-mode", choices=("per_frame_up", "min", "median"), default="per_frame_up")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return p.parse_args(argv)


def hex_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


# ------------------------------------------------------------------ layout


def keyframe_indices(length: int, count: int) -> np.ndarray:
    count = max(2, min(count, length))
    return np.linspace(0, length - 1, count).round().astype(int)


def prepare_layout(joints: np.ndarray, num_keyframes: int):
    """Identical rule to the stick figure."""
    frames = keyframe_indices(len(joints), num_keyframes)
    root_xy = joints[:, 0, [0, 2]]
    path = root_xy[frames] - root_xy[frames[0]]
    path_scale = max(float(np.linalg.norm(np.ptp(path, axis=0))), 1e-6)
    if path_scale < 0.45:
        positions = np.stack([np.linspace(-1.8, 1.8, len(frames)), np.zeros(len(frames))], axis=1)
    else:
        positions = path / path_scale * 3.6
    return frames, positions.astype(np.float64)


# ------------------------------------------------------------------ scene


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            block.remove(item)


def diffuse_mat(name: str, rgb, roughness: float = 0.65):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    lin = srgb_to_linear(rgb)
    bsdf.inputs["Base Color"].default_value = (*lin, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.25
    return mat


def emission_mat(name: str, rgb, strength: float = 1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*srgb_to_linear(rgb), 1.0)
    emit.inputs["Strength"].default_value = strength
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def import_ply(path: Path):
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    new = list(set(bpy.data.objects) - before)
    if not new:
        raise RuntimeError(f"PLY import produced no object: {path}")
    return new[0]


def add_tube(points: np.ndarray, radius: float, mat, name: str) -> None:
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 4
    curve.bevel_depth = radius
    curve.bevel_resolution = 6
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for pnt, xyz in zip(spline.points, points):
        pnt.co = (float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)


def add_arrow_head(tip: np.ndarray, direction: np.ndarray, length: float, radius: float, mat, name: str) -> None:
    d = mathutils.Vector(direction.tolist())
    if d.length < 1e-6:
        return
    d.normalize()
    base = mathutils.Vector(tip.tolist()) - d * length
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius, depth=length, location=(base + d * length * 0.5))
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)


def main() -> None:
    args = parse_args()
    joints = np.load(args.joints).astype(np.float64)
    frames, positions = prepare_layout(joints, args.num_keyframes)
    ply_dir = Path(args.ply_dir)

    clear_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.world = bpy.data.worlds.new("qual_world")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.92, 0.92, 0.92, 1.0)
    bg.inputs["Strength"].default_value = 0.42

    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 4
        scene.cycles.transparent_max_bounces = 4
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_cavity = True
        scene.display.shading.show_shadows = True
    scene.view_settings.view_transform = "Standard"

    body_mat = diffuse_mat("smpl_body", hex_rgb(args.body_color), roughness=0.62)
    ground_mat = diffuse_mat("ground", (0.80, 0.80, 0.80), roughness=0.92)
    arrow_mat = emission_mat("arrow", hex_rgb(args.arrow_color), strength=1.0)

    # ---- bodies: same placement rule as the stick figure, then grounded
    all_pts = []
    for k, (frame, pos) in enumerate(zip(frames, positions)):
        obj = import_ply(ply_dir / f"{k:04d}.ply")
        obj.name = f"smpl_{k:02d}"
        root = joints[frame, 0, [0, 2]]
        dx = float(pos[0] - root[0])
        dz = float(pos[1] - root[1])
        v = np.empty((len(obj.data.vertices), 3), dtype=np.float64)
        obj.data.vertices.foreach_get("co", v.ravel())
        # source is y-up; blender scene is z-up with X=x, Y=depth(z), Z=height
        scene_v = np.stack([v[:, 0] + dx, v[:, 2] + dz, v[:, 1]], axis=1)
        all_pts.append((obj, scene_v))

    per_frame_min = np.array([sv[:, 2].min() for _, sv in all_pts])
    if args.floor_mode == "min":
        shifts = np.full(len(all_pts), -per_frame_min.min())
    elif args.floor_mode == "median":
        shifts = np.full(len(all_pts), -float(np.median(per_frame_min)))
    else:  # per_frame_up: only lift bodies that would clip through the floor
        shifts = np.maximum(0.0, -per_frame_min)

    verts_all = []
    for (obj, scene_v), shift in zip(all_pts, shifts):
        scene_v = scene_v.copy()
        scene_v[:, 2] += shift
        obj.data.vertices.foreach_set("co", scene_v.ravel())
        obj.data.update()
        obj.matrix_world = mathutils.Matrix.Identity(4)
        obj.data.materials.append(body_mat)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        obj.select_set(False)
        verts_all.append(scene_v)

    pts = np.concatenate(verts_all, axis=0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    # ---- ground patch, same proportions as the stick figure's floor
    pad_x = max(0.28, 0.05 * (hi[0] - lo[0]))
    pad_y = max(0.28, 0.05 * (hi[1] - lo[1]))
    gx0, gx1 = lo[0] - pad_x, hi[0] + pad_x
    gy0, gy1 = lo[1] - pad_y, hi[1] + pad_y
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=((gx0 + gx1) / 2, (gy0 + gy1) / 2, 0.0))
    ground = bpy.context.object
    ground.name = "ground"
    ground.scale = (gx1 - gx0, gy1 - gy0, 1.0)
    ground.data.materials.append(ground_mat)

    # ---- trajectory + arrow head on the ground
    traj = np.stack([positions[:, 0], positions[:, 1], np.full(len(positions), 0.012)], axis=1)
    seg = traj[-1] - traj[-2]
    seg_len = float(np.linalg.norm(seg[:2]))
    if seg_len > 1e-4:
        head_len = min(0.38, max(0.20, seg_len * 0.42))
        shaft = traj.copy()
        shaft[-1] = traj[-1] - seg / max(seg_len, 1e-6) * head_len * 0.9
        add_tube(shaft, 0.021, arrow_mat, "traj")
        add_arrow_head(traj[-1], seg, head_len, 0.068, arrow_mat, "traj_head")
    else:
        add_tube(traj, 0.021, arrow_mat, "traj")

    # ---- lights
    bpy.ops.object.light_add(type="SUN", location=(gx1 + 2.0, gy0 - 3.0, 6.0))
    sun = bpy.context.object
    sun.data.energy = 1.7
    sun.data.angle = math.radians(20.0)
    sun.rotation_euler = mathutils.Vector((-0.55, 0.72, -3.0)).to_track_quat("-Z", "Y").to_euler()

    bpy.ops.object.light_add(type="AREA", location=(gx0 - 3.0, gy0 - 4.0, 3.4))
    fill = bpy.context.object
    fill.data.energy = 170.0
    fill.data.size = 6.0
    fill.rotation_euler = (
        mathutils.Vector(((gx0 + gx1) / 2, (gy0 + gy1) / 2, 0.8)) - fill.location
    ).to_track_quat("-Z", "Y").to_euler()

    # ---- camera: same spherical direction as matplotlib's elev/azim, orthographic
    e, a = math.radians(ELEV), math.radians(AZIM)
    view = mathutils.Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))).normalized()
    center = mathutils.Vector(
        ((gx0 + gx1) / 2, (gy0 + gy1) / 2, (0.0 + hi[2]) / 2)
    )
    bpy.ops.object.camera_add(location=center + view * 40.0)
    cam = bpy.context.object
    cam.data.type = "ORTHO"
    cam.rotation_euler = view.to_track_quat("Z", "Y").to_euler()
    scene.camera = cam

    basis = view.to_track_quat("Z", "Y").to_matrix()
    right, up = basis @ mathutils.Vector((1, 0, 0)), basis @ mathutils.Vector((0, 1, 0))
    corners = np.array([[gx0, gy0, 0.0], [gx0, gy1, 0.0], [gx1, gy0, 0.0], [gx1, gy1, 0.0]])
    fit = np.concatenate([pts, corners], axis=0) - np.array(center)
    u = fit @ np.array(right)
    v = fit @ np.array(up)
    need_w = (u.max() - u.min()) * 1.02
    need_h = (v.max() - v.min()) * 1.02
    aspect = args.width / args.height
    cam.data.ortho_scale = max(need_w, need_h * aspect)
    cam.location = cam.location + right * float((u.max() + u.min()) / 2) + up * float((v.max() + v.min()) / 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
