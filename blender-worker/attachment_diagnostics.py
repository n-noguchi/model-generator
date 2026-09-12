"""Non-destructive deformation candidates, NOT a semantic adhesion diagnosis."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector

VERSION = "attachment-candidates-v1"
THRESHOLDS = {"stretch_ratio": 4.0, "min_rest_edge_height": .001,
              "min_extension_height": .02, "branch_weight": .15}
PALETTE = ((.25, .32, .40, 1), (1, .45, .025, 1), (1, .035, .06, 1))


class DiagnosticUnavailable(Exception):
    """An unsupported evaluation is not a failed model-generation job."""


def stretched_edges(rest, current, edges, height):
    """Scale-independent; short seams must not dominate the ratio."""
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    posed = np.linalg.norm(current[edges[:, 0]] - current[edges[:, 1]], axis=1)
    ratios = posed / np.maximum(lengths, 1e-12)
    selected = ((lengths >= height * THRESHOLDS["min_rest_edge_height"])
                & (ratios >= THRESHOLDS["stretch_ratio"])
                & (posed - lengths >= height * THRESHOLDS["min_extension_height"]))
    return selected, ratios, (posed - lengths) / height


@contextmanager
def isolated_pose(arm):
    """Keep diagnostic poses and NLA evaluation out of the real output."""
    frame = bpy.context.scene.frame_current
    subframe = bpy.context.scene.frame_subframe
    position = arm.data.pose_position
    transforms = [(b, b.rotation_mode, b.matrix_basis.copy()) for b in arm.pose.bones]
    existed = arm.animation_data is not None
    animation = arm.animation_data_create()
    action, use_nla = animation.action, animation.use_nla
    try:
        animation.action = None
        animation.use_nla = False
        arm.data.pose_position = "POSE"
        for bone, _, _ in transforms:
            bone.rotation_mode = "QUATERNION"
            bone.matrix_basis.identity()
        bpy.context.view_layer.update()
        yield
    finally:
        animation.action, animation.use_nla = action, use_nla
        arm.data.pose_position = position
        bpy.context.scene.frame_set(frame, subframe=subframe)
        for bone, mode, basis in transforms:
            bone.rotation_mode = mode
            bone.matrix_basis = basis
        if not existed:
            arm.animation_data_clear()
        bpy.context.view_layer.update()


def coordinates(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if (len(mesh.vertices) != len(obj.data.vertices) or len(mesh.polygons) != len(obj.data.polygons)
                or any(tuple(a.vertices) != tuple(b.vertices) for a, b in zip(mesh.polygons, obj.data.polygons))):
            raise DiagnosticUnavailable("診断中にメッシュの面対応が変更されました")
        return np.array([tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices], dtype=float)
    finally:
        evaluated.to_mesh_clear()


def branch(name):
    if name.startswith(("UpperArm.", "LowerArm.", "Hand.")):
        return "arm_" + name[-1]
    if name.startswith(("UpperLeg.", "LowerLeg.", "Foot.")):
        return "leg_" + name[-1]
    return "trunk"


def regions_for(obj, record, height):
    """Face IDs refer to this exact imported source, not any other LOD."""
    edge_faces = {}
    face_edges = {}
    groups = {g.index: branch(g.name) for g in obj.vertex_groups}
    weights = []
    for vertex in obj.data.vertices:
        values = {}
        for item in vertex.groups:
            if item.group in groups:
                name = groups[item.group]
                values[name] = values.get(name, 0) + item.weight
        weights.append(values)
    lookup = {tuple(sorted(edge)): i for i, edge in enumerate(record["edges"].tolist())}
    faces = {}
    for face in obj.data.polygons:
        keys = [tuple(sorted(key)) for key in face.edge_keys]
        ids = [lookup[key] for key in keys]
        face_edges[face.index] = keys
        for key in keys:
            edge_faces.setdefault(key, []).append(face.index)
        hits = [i for i in ids if record["ratio"][i] > 0]
        if not hits:
            continue
        totals = {}
        for index in face.vertices:
            for name, value in weights[index].items():
                totals[name] = totals.get(name, 0) + value / len(face.vertices)
        names = sorted(name for name, value in totals.items() if value >= THRESHOLDS["branch_weight"])
        # Shoulder/hip blends are normal. This flag means only that multiple
        # branches influence a stretched face, never that it should be cut.
        cross = (any(n.startswith("arm_") for n in names)
                 and any(n.startswith("leg_") for n in names)) or ("arm_L" in names and "arm_R" in names) or ("leg_L" in names and "leg_R" in names)
        distal = (any(n.startswith("arm_") for n in names) and "trunk" in names
                  and np.mean(record["rest"][list(face.vertices), 2]) < record["low_z"] + height * .65)
        worst = max(hits, key=lambda i: record["ratio"][i])
        faces[face.index] = {"severity": 2 if cross or distal else 1, "branches": names,
                             "edge": worst}
    pending = set(faces)
    regions = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        stack, ids = [seed], []
        while stack:
            current = stack.pop()
            ids.append(current)
            for key in face_edges[current]:
                for neighbor in edge_faces[key]:
                    if neighbor in pending:
                        pending.remove(neighbor)
                        stack.append(neighbor)
        vertices = sorted({v for i in ids for v in obj.data.polygons[i].vertices})
        points = record["rest"][vertices]
        worst = max((faces[i]["edge"] for i in ids), key=lambda i: record["ratio"][i])
        cross_count = sum(faces[i]["severity"] == 2 for i in ids)
        regions.append({"face_ids": sorted(ids), "face_count": len(ids),
                        "cross_branch_faces": cross_count,
                        "branches": sorted({n for i in ids for n in faces[i]["branches"]}),
                        "max_stretch": float(record["ratio"][worst]),
                        "extension_height": float(record["extension"][worst]),
                        "pose": record["poses"][int(record["pose_index"][worst])],
                        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                        "reasons": ["extreme_stretch", "shared_mesh_edges"] + (["cross_branch_weights"] if cross_count else [])})
    return sorted(regions, key=lambda r: -r["max_stretch"]), faces


def analyze(meshes, arm, include_actions=False):
    try:
        return _analyze(meshes, arm, include_actions)
    except DiagnosticUnavailable as exc:
        return {"version": VERSION, "status": "not_evaluated", "thresholds": THRESHOLDS,
                "reason": str(exc), "regions": [], "meshes": [], "candidate_faces": 0,
                "tested_poses": 0, "partial_results_discarded": True,
                "image_comparison": "not_performed"}, {}


def _analyze(meshes, arm, include_actions=False):
    report = {"version": VERSION, "status": "not_evaluated", "thresholds": THRESHOLDS,
              "image_comparison": "not_performed", "regions": [], "meshes": [],
              "candidate_faces": 0, "tested_poses": 0,
              "note": "癒着またはウェイト異常の候補です。癒着の確定・分離は行いません。4方向画像との自動照合は未実施です。"}
    required = {f"{part}.{side}" for part in ("UpperArm", "UpperLeg") for side in ("L", "R")}
    if not arm or not meshes or any(not obj.data.vertices for obj in meshes) or not required.issubset({b.name for b in arm.pose.bones}):
        report["reason"] = "対応する人体骨格がないため検査できません。"
        return report, {}
    # Arbitrary constraints/drivers or geometry-changing modifiers invalidate
    # face correspondence and isolated poses. Report unavailable, not zero risk.
    if (any(b.constraints for b in arm.pose.bones) or arm.constraints
            or (arm.animation_data and arm.animation_data.drivers)
            or any(obj.data.shape_keys or obj.constraints or (obj.animation_data and obj.animation_data.drivers)
                   or any(m.type != "ARMATURE" for m in obj.modifiers) for obj in meshes)):
        report["reason"] = "制約・形状変更を含むため、この検査では面の対応を保証できません。"
        return report, {}
    records, coloring = [], {}
    with isolated_pose(arm):
        rest = [coordinates(obj) for obj in meshes]
        if any(len(points) != len(obj.data.vertices) for obj, points in zip(meshes, rest)):
            raise DiagnosticUnavailable("診断時の頂点対応が一致しません")
        points = np.concatenate(rest)
        height = float(points[:, 2].max() - points[:, 2].min())
        if not np.isfinite(points).all() or height <= 1e-8:
            report["reason"] = "有効な全身形状を計測できません。"
            return report, {}
        for obj, points in zip(meshes, rest):
            edges = np.array([tuple(e.vertices) for e in obj.data.edges], dtype=int).reshape((-1, 2))
            records.append({"rest": points, "edges": edges, "ratio": np.zeros(len(edges)),
                            "extension": np.zeros(len(edges)), "pose_index": np.zeros(len(edges), dtype=int),
                            "poses": [], "low_z": min(p[:, 2].min() for p in rest)})

        def sample(label):
            bpy.context.view_layer.update()
            report["tested_poses"] += 1
            for obj, record in zip(meshes, records):
                posed = coordinates(obj)
                if posed.shape != record["rest"].shape or not np.isfinite(posed).all():
                    raise DiagnosticUnavailable("診断ポーズの形状が無効です")
                selected, ratios, extension = stretched_edges(record["rest"], posed, record["edges"], height)
                better = selected & (ratios > record["ratio"])
                record["ratio"][better] = ratios[better]
                record["extension"][better] = extension[better]
                record["pose_index"][better] = len(record["poses"])
                record["poses"].append(label)

        # Evaluate each branch independently in body axes, independent of bone
        # roll. Two directions also catch a bridge hidden by one particular pose.
        for part in ("UpperArm", "UpperLeg"):
            for side in ("L", "R"):
                bone = arm.pose.bones[f"{part}.{side}"]
                for axis, axis_name in ((Vector((1, 0, 0)), "前後"), (Vector((0, 1, 0)), "左右")):
                    local = bone.bone.matrix_local.to_3x3().inverted() @ arm.matrix_world.to_3x3().inverted() @ axis
                    for angle in (-.6, .6):
                        bone.rotation_quaternion = Quaternion(local.normalized(), angle)
                        sample(f'{"左" if side == "L" else "右"}{"腕" if part == "UpperArm" else "脚"}・{axis_name} {round(angle * 180 / np.pi)}°')
                        bone.matrix_basis.identity()
        if include_actions:
            for action in list(bpy.data.actions):
                if not any(curve.data_path.startswith('pose.bones[') for curve in action.fcurves):
                    continue
                start, end = (round(v) for v in action.frame_range)
                if end - start > 300:
                    raise DiagnosticUnavailable("300フレームを超えるクリップは検査対象外です。部分的な結果は破棄しました。")
                for bone in arm.pose.bones:
                    bone.matrix_basis.identity()
                arm.animation_data.action = action
                for frame in range(start, end + 1):
                    bpy.context.scene.frame_set(frame)
                    sample(f"{action.name} / frame {frame}")
                arm.animation_data.action = None
                for bone in arm.pose.bones:
                    bone.matrix_basis.identity()
        for index, (obj, record) in enumerate(zip(meshes, records)):
            regions, faces = regions_for(obj, record, height)
            report["meshes"].append({"name": obj.name, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons)})
            for region in regions:
                region["mesh"] = obj.name
                region["mesh_index"] = index
                report["regions"].append(region)
            report["candidate_faces"] += len(faces)
            coloring[obj.name] = {i: value["severity"] for i, value in faces.items()}
    report["regions"].sort(key=lambda r: -r["max_stretch"])
    report["status"] = "candidates_found" if report["candidate_faces"] else "no_candidates"
    return report, coloring


def export_preview(meshes, coloring, path, source_hash):
    """Static, per-face colors in a separate scene. Never recolor real assets."""
    original_scene = bpy.context.window.scene
    scene = bpy.data.scenes.new("Attachment diagnostic preview")
    scene["diagnostic_version"] = VERSION
    scene["source_sha256"] = source_hash
    copies, materials = [], []
    try:
        bpy.context.window.scene = scene
        for index, color in enumerate(PALETTE):
            material = bpy.data.materials.new(f"Diagnostic {index}")
            material.diffuse_color = color
            materials.append(material)
        for original in meshes:
            mesh = original.data.copy()
            obj = bpy.data.objects.new(original.name + " diagnostic", mesh)
            scene.collection.objects.link(obj)
            obj.matrix_world = original.matrix_world.copy()
            copies.append(obj)
            mesh.materials.clear()
            for material in materials:
                mesh.materials.append(material)
            for polygon in mesh.polygons:
                polygon.material_index = coloring.get(original.name, {}).get(polygon.index, 0)
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_active_scene=True, export_animations=False, export_extras=True)
    finally:
        bpy.context.window.scene = original_scene
        for obj in copies:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(scene)
        for material in materials:
            bpy.data.materials.remove(material)


def write_diagnostics(meshes, arm, source, root, label, include_actions=False):
    source, root = Path(source), Path(root)
    report, coloring = analyze(meshes, arm, include_actions)
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["source_file"] = source.name
    report_path = root / f"{label}-attachment-report.json"
    paths = {f"{label}_attachment_report": report_path}
    if report["status"] != "not_evaluated":
        preview = root / f"{label}-attachment-preview.glb"
        export_preview(meshes, coloring, preview, report["source_sha256"])
        report["preview_sha256"] = hashlib.sha256(preview.read_bytes()).hexdigest()
        paths[f"{label}_attachment_preview"] = preview
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
