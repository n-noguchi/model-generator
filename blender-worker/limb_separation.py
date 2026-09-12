"""Conservative contact-loop separation for the generated upright humanoid rig."""
import heapq
import math
from collections import Counter
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Quaternion
from attachment_diagnostics import analyze, branch, isolated_pose, coordinates

VERSION = "contact-loop-separation-v1"


class CannotSeparate(Exception):
    pass


def edge_components(edges):
    pending = set(edges)
    result = []
    while pending:
        seed = pending.pop()
        component, stack = {seed}, [seed]
        while stack:
            edge = stack.pop()
            for vert in edge.verts:
                for other in vert.link_edges:
                    if other in pending:
                        pending.remove(other); component.add(other); stack.append(other)
        result.append(component)
    return result


def ordered_loop(edges):
    vertices = {v for e in edges for v in e.verts}
    if not vertices or any(sum(e in edges for e in v.link_edges) != 2 for v in vertices):
        raise CannotSeparate("切り口が単純な閉曲線ではありません")
    start = min(vertices, key=lambda v: v.index)
    ordered, used, current = [start], set(), start
    while len(used) < len(edges):
        edge = next(e for e in current.link_edges if e in edges and e not in used)
        used.add(edge); current = edge.other_vert(current)
        if current != start:
            ordered.append(current)
    if current != start or len(ordered) != len(vertices):
        raise CannotSeparate("切り口が閉じていません")
    return ordered


def distances_from(seeds, blocked, matrix, limit):
    distances, queue = {}, []
    for vertex in seeds:
        distances[vertex] = 0.
        heapq.heappush(queue, (0., vertex.index, vertex))
    while queue:
        distance, _, vertex = heapq.heappop(queue)
        if distance > distances[vertex]: continue
        for edge in vertex.link_edges:
            if edge in blocked: continue
            neighbor = edge.other_vert(vertex)
            value = distance + ((matrix @ vertex.co) - (matrix @ neighbor.co)).length
            if value <= limit and value < distances.get(neighbor, math.inf):
                distances[neighbor] = value
                heapq.heappush(queue, (value, neighbor.index, neighbor))
    return distances


def weights_for(vertex, deform, groups):
    result = {}
    for index, weight in vertex[deform].items():
        name = groups.get(index)
        if name:
            result[name] = result.get(name, 0) + weight
    return result


def distance_to_branch(point, segments):
    values = []
    for start, end in segments:
        delta = end - start
        t = min(1., max(0., (point-start).dot(delta) / max(delta.length_squared, 1e-12)))
        values.append((point-start-delta*t).length)
    return min(values, default=math.inf)


def simple_cap_geometry(vertices, matrix, height):
    """Reject folded/non-star-shaped loops and return a safe local cap center."""
    points = np.array([tuple(matrix @ v.co) for v in vertices])
    centered = points - points.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    if max(abs(centered @ axes[2])) > height * .03:
        raise CannotSeparate("切り口の曲がりが大きく表面を補完できません")
    def cross(a, b): return a[0]*b[1] - a[1]*b[0]
    for i in range(len(vertices)):
        a, b = projected[i], projected[(i+1) % len(vertices)]
        for j in range(i+2, len(vertices)):
            if i == 0 and j == len(vertices)-1: continue
            c, d = projected[j], projected[(j+1) % len(vertices)]
            if cross(b-a,c-a)*cross(b-a,d-a) < -1e-16 and cross(d-c,a-c)*cross(d-c,b-c) < -1e-16:
                raise CannotSeparate("切り口が自己交差しています")
    center = projected.mean(0)
    # The explicit triangle fan below must stay inside the contour.  This
    # intentionally rejects concave loops whose mean is not a safe kernel.
    signs = [cross(projected[(i+1) % len(vertices)]-projected[i], center-projected[i])
             for i in range(len(vertices))]
    if any(value > 1e-10 for value in signs) and any(value < -1e-10 for value in signs):
        raise CannotSeparate("切り口の中心を安全に補完できません")
    return sum((vertex.co for vertex in vertices), Vector()) / len(vertices)


def connected_to_root(seeds, branch_name, face_label, arm, matrix, height):
    root = arm.data.bones.get(("UpperArm." if branch_name.startswith("arm_") else "UpperLeg.") + branch_name[-1])
    if not root: return False
    target = arm.matrix_world @ root.head_local
    pending = list(seeds); visited = set(seeds)
    while pending:
        vertex = pending.pop()
        if (matrix @ vertex.co - target).length < height * .12: return True
        for edge in vertex.link_edges:
            if not any(face_label(face) == branch_name for face in edge.link_faces): continue
            other = edge.other_vert(vertex)
            if other not in visited: visited.add(other); pending.append(other)
    return False


def repair_mesh(obj, arm, height, candidate_faces):
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
        deform = bm.verts.layers.deform.active
        if deform is None: raise CannotSeparate("部位のウェイトがありません")
        groups = {g.index: branch(g.name) for g in obj.vertex_groups if arm.data.bones.get(g.name)}
        label_layer = bm.faces.layers.int.new("separation_part")
        source_layer = bm.faces.layers.int.new("separation_source_face")
        part_names = ["trunk", "arm_L", "arm_R", "leg_L", "leg_R"]
        def face_label(face): return part_names[face[label_layer]]
        def face_signature(face):
            return tuple(sorted((tuple(loop.vert.co), tuple(tuple(loop[layer].uv) for layer in bm.loops.layers.uv.values())) for loop in face.loops)), face.material_index
        original_signatures = {}
        for face in bm.faces:
            total = {}
            for vertex in face.verts:
                for name, weight in weights_for(vertex, deform, groups).items():
                    total[name] = total.get(name, 0) + weight / len(face.verts)
            if not total: raise CannotSeparate("部位を判断できない面があります")
            face[label_layer] = part_names.index(max(total, key=total.get))
            face[source_layer] = face.index + 1
            original_signatures[face.index + 1] = face_signature(face)
        original_boundary = sum(e.is_boundary for e in bm.edges)
        def invalid_topology():
            # Distinguish loose/one-sided edges from edges shared by 3+ faces.
            # `is_manifold` alone is too coarse for a repair decision.
            return dict(sorted(Counter(len(edge.link_faces) for edge in bm.edges
                                       if not edge.is_manifold and not edge.is_boundary).items()))
        original_invalid = invalid_topology()
        original_nonmanifold = sum(original_invalid.values())
        topology_stages = [("開始時", original_invalid)]
        original_components = len(edge_components(set(bm.edges)))
        segments = {}
        for bone in arm.data.bones:
            segments.setdefault(branch(bone.name), []).append((arm.matrix_world @ bone.head_local, arm.matrix_world @ bone.tail_local))
        pairs = {}
        for edge in bm.edges:
            # A contact boundary must be a clean two-sided manifold loop.
            # Splitting an already non-manifold edge can create a new
            # non-manifold seam even when the cap itself is valid.
            if not edge.is_manifold or len(edge.link_faces) != 2: continue
            names = tuple(sorted({face_label(f) for f in edge.link_faces}))
            if len(names) != 2: continue
            if not any(n.startswith("arm_") for n in names) and set(names) != {"leg_L", "leg_R"}: continue
            pairs.setdefault(names, set()).add(edge)
        plans, skipped = [], []
        for names, edges in sorted(pairs.items()):
            # A hand beside a hip is not a normal anatomical joint between
            # those branches. Protect shoulder seams for arms, hip seams for
            # two legs; also verify each side's route to its actual root below.
            protected_parts = ("UpperArm.",) if any(n.startswith("arm_") for n in names) else ("UpperLeg.",)
            joint_centers = [arm.matrix_world @ bone.head_local for bone in arm.data.bones if bone.name.startswith(protected_parts)]
            for loop in edge_components(edges):
                verts = {v for e in loop for v in e.verts}
                adjacent = {f for v in verts for f in v.link_faces}
                if not any(f.index in candidate_faces for f in adjacent): continue
                points = np.array([tuple(obj.matrix_world @ v.co) for v in verts])
                if np.max(points.max(0)-points.min(0)) > height * .12: continue
                try:
                    ordered = ordered_loop(loop)
                    if len(loop) < 6 or sum(((obj.matrix_world @ e.verts[0].co)-(obj.matrix_world @ e.verts[1].co)).length for e in loop) > height * .5:
                        raise CannotSeparate("接触境界が大きすぎるか粗すぎます")
                    if any((obj.matrix_world @ v.co-center).length < height * .08 for v in verts for center in joint_centers):
                        raise CannotSeparate("肩または股関節の保護範囲です")
                    if any({face_label(f) for f in v.link_faces} - set(names) for v in verts):
                        raise CannotSeparate("3部位以上が接する境界です")
                    cap_center = simple_cap_geometry(ordered, obj.matrix_world, height)
                    # Both sides must have independently corroborated seeds.
                    # Use the immutable pre-split topology for all proposals.
                    for name in names:
                        side_faces = {f for e in loop for f in e.link_faces if face_label(f) == name}
                        seeds = {v for f in side_faces for v in f.verts} - verts
                        near = distances_from(seeds, loop, obj.matrix_world, height * .08)
                        supported = []
                        other = next(n for n in names if n != name)
                        for vertex in near:
                            weights = weights_for(vertex, deform, groups)
                            point = obj.matrix_world @ vertex.co
                            own = distance_to_branch(point, segments[name])
                            foreign = distance_to_branch(point, segments[other])
                            if weights.get(name, 0) >= .8 and own < foreign * .85:
                                supported.append(vertex)
                        if len(supported) < 3:
                            raise CannotSeparate("両側の部位を形状とウェイトで確認できません")
                    if any(verts & other["vertices"] for other in plans):
                        raise CannotSeparate("分離境界が重なっています")
                    plans.append({"edges": loop, "vertices": verts, "parts": names,
                                  "cap_center": cap_center})
                except CannotSeparate as exc:
                    skipped.append(str(exc))
        if not plans:
            return {"loops": 0, "skipped": sorted(set(skipped)) or ["局所的な閉じた接触境界が見つかりません"]}
        cap_layer = bm.faces.layers.int.new("auto_separation_cap")
        cap_faces, changed_vertices = set(), set()
        for plan in plans:
            existing_boundary = {e for e in bm.edges if e.is_boundary}
            expected = len(plan["edges"])
            bmesh.ops.split_edges(bm, edges=list(plan["edges"]))
            topology_stages.append(("境界分離後", invalid_topology()))
            bm.verts.index_update()
            new_boundary = {e for e in bm.edges if e.is_boundary} - existing_boundary
            cuts = edge_components(new_boundary)
            if len(cuts) != 2 or any(len(c) != expected for c in cuts):
                raise CannotSeparate("分離した2つの切り口の対応が一致しません")
            sides = []
            for cut in cuts:
                ordered = ordered_loop(cut)
                names = {face_label(f) for e in cut for f in e.link_faces}
                if len(names) != 1: raise CannotSeparate("切り口に異なる部位が混ざっています")
                name = names.pop(); sides.append(name)
                if name != "trunk" and not connected_to_root(ordered, name, face_label, arm, obj.matrix_world, height):
                    raise CannotSeparate("分離により手足の正常な接続が失われます")
                other = next(n for n in plan["parts"] if n != name)
                near = distances_from(ordered, set(), obj.matrix_world, height * .08)
                for vertex, distance in near.items():
                    # A smooth transition prevents merely moving the bad edge
                    # to the next ring. At the cut all foreign weights vanish.
                    blend = max(0., 1.-distance/(height*.08))
                    values = dict(vertex[deform])
                    if weights_for(vertex, deform, groups).get(name, 0) <= 1e-6: continue
                    for index in values:
                        if groups.get(index) == other: values[index] *= (1.-blend)**2
                    total = sum(values.values())
                    if total <= 1e-8: raise CannotSeparate("分離部にウェイトを割り当てられません")
                    if any(abs(values[i]/total-vertex[deform][i]) > 1e-6 for i in values):
                        changed_vertices.add(vertex)
                    for index, value in values.items():
                        if value/total > 1e-8: vertex[deform][index] = value/total
                        elif index in vertex[deform]: del vertex[deform][index]
                # Copy UVs per side and per corner before creating each cap.
                uv_layers = list(bm.loops.layers.uv.values())
                uv = {v: {layer: next(loop[layer].uv.copy() for loop in v.link_loops if loop.face[source_layer] > 0) for layer in uv_layers} for v in ordered}
                material = next(iter(cut)).link_faces[0].material_index
                # Do not triangulate an n-gon here: Blender may select a
                # pre-existing surface diagonal and turn it into a four-face
                # edge.  A fresh center vertex guarantees fresh diagonals.
                center = bm.verts.new(plan["cap_center"])
                boundary_weights = [dict(vertex[deform]) for vertex in ordered]
                for index in set().union(*[set(values) for values in boundary_weights]):
                    value = sum(values.get(index, 0.) for values in boundary_weights) / len(boundary_weights)
                    if value > 1e-8: center[deform][index] = value
                filled = []
                for index, vertex in enumerate(ordered):
                    following = ordered[(index+1) % len(ordered)]
                    try:
                        face = bm.faces.new((center, vertex, following))
                    except ValueError as exc:
                        raise CannotSeparate("切り口の表面を補完できません") from exc
                    filled.append(face)
                topology_stages.append(("扇状補完後", invalid_topology()))
                for face in filled:
                    face.material_index = material
                    face[cap_layer] = 1
                    face[source_layer] = 0
                    face[label_layer] = part_names.index(name)
                    for loop in face.loops:
                        for layer in uv_layers:
                            loop[layer].uv = (uv[loop.vert][layer] if loop.vert != center
                                               else sum((values[layer] for values in uv.values()), Vector((0., 0.))) / len(uv))
                cap_faces.update(filled)
            if set(sides) != set(plan["parts"]): raise CannotSeparate("切り口の部位が一致しません")
        actual_signatures = {face[source_layer]:face_signature(face) for face in bm.faces if face[source_layer] > 0}
        if actual_signatures != original_signatures:
            raise CannotSeparate("元の表面またはUVが変更されています")
        if sum(e.is_boundary for e in bm.edges) != original_boundary:
            raise CannotSeparate("分離後に穴が残っています")
        nonmanifold_after = sum(not e.is_manifold and not e.is_boundary for e in bm.edges)
        if nonmanifold_after > original_nonmanifold:
            raise CannotSeparate(
                f"分離後の面接続が不正です ({original_invalid}→{invalid_topology()}; {topology_stages})"
            )
        if len(edge_components(set(bm.edges))) != original_components:
            raise CannotSeparate("手足が独立した破片になっています")
        cap_area = sum(face.calc_area() for face in cap_faces) * obj.matrix_world.to_scale().length_squared/3
        if cap_area > height**2 * .03:
            raise CannotSeparate("補完する面積が大きすぎます")
        if any(face.calc_area() <= height**2*1e-12 for face in cap_faces):
            raise CannotSeparate("補完部に退化面があります")
        bmesh.ops.recalc_face_normals(bm, faces=list(cap_faces))
        bm.to_mesh(obj.data); obj.data.update()
        return {"loops": len(plans), "parts": [list(p["parts"]) for p in plans],
                "cap_faces": len(cap_faces), "cap_area_height2": cap_area/height**2,
                "reweighted_vertices": len(changed_vertices), "original_faces_retained": len(original_signatures),
                "boundary_edges_before": original_boundary, "boundary_edges_after": original_boundary,
                "skipped": sorted(set(skipped))}
    finally:
        bm.free()


def validate_caps(meshes, arm):
    """Includes all cap edges (even tiny ones) and triangle area changes."""
    with isolated_pose(arm):
        records = []
        for obj in meshes:
            attribute = obj.data.attributes.get("auto_separation_cap")
            indices = [p.index for p in obj.data.polygons if attribute and attribute.data[p.index].value]
            triangles = np.array([list(obj.data.polygons[i].vertices) for i in indices], dtype=int).reshape((-1, 3))
            if not len(triangles): continue
            rest = coordinates(obj)
            areas = np.linalg.norm(np.cross(rest[triangles[:,1]]-rest[triangles[:,0]],rest[triangles[:,2]]-rest[triangles[:,0]]),axis=1)
            edges = np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
            lengths = np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
            records.append((obj,triangles,areas,edges,lengths))
        max_edge, max_area = 1., 1.
        for part in ("UpperArm", "UpperLeg"):
            for side in ("L", "R"):
                bone = arm.pose.bones[f"{part}.{side}"]
                for axis in (Vector((1,0,0)), Vector((0,1,0))):
                    local = bone.bone.matrix_local.to_3x3().inverted() @ arm.matrix_world.to_3x3().inverted() @ axis
                    for angle in (-.6,.6):
                        bone.rotation_quaternion = Quaternion(local.normalized(),angle)
                        bpy.context.view_layer.update()
                        for obj,triangles,areas,edges,lengths in records:
                            points=coordinates(obj)
                            new_area=np.linalg.norm(np.cross(points[triangles[:,1]]-points[triangles[:,0]],points[triangles[:,2]]-points[triangles[:,0]]),axis=1)
                            max_area=max(max_area,float(np.max(new_area/np.maximum(areas,1e-12))))
                            max_edge=max(max_edge,float(np.max(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)/np.maximum(lengths,1e-12))))
                        bone.matrix_basis.identity()
        if max_edge > 4 or max_area > 4: raise CannotSeparate("補完した面が動作時に大きく変形します")
        return {"max_edge_stretch":max_edge,"max_area_ratio":max_area}


def separate_contacts(meshes, arm):
    before, _ = analyze(meshes, arm)
    report = {"version":VERSION,"status":"unchanged","before_faces":before["candidate_faces"],
              "after_faces":before["candidate_faces"],"meshes":[],"reason":"検出候補がありません"}
    if before["status"] == "not_evaluated":
        report.update(status="not_evaluated",reason=before.get("reason")); return report
    if before["status"] == "no_candidates": return report
    points=[obj.matrix_world@v.co for obj in meshes for v in obj.data.vertices]
    height=max(p.z for p in points)-min(p.z for p in points)
    originals=[obj.data for obj in meshes]
    accepted=False
    try:
        for obj in meshes: obj.data=obj.data.copy()
        for index,obj in enumerate(meshes):
            ids={i for r in before["regions"] if r["mesh_index"]==index for i in r["face_ids"]}
            report["meshes"].append(repair_mesh(obj,arm,height,ids))
        if not any(row["loops"] for row in report["meshes"]):
            raise CannotSeparate("局所境界を確定できません: " + " / ".join(sorted({s for row in report["meshes"] for s in row["skipped"]})))
        report["caps_validation"]=validate_caps(meshes,arm)
        after,_=analyze(meshes,arm)
        report["attempted_after_faces"]=after["candidate_faces"]
        report["before_max_stretch"]=max((r["max_stretch"] for r in before["regions"]),default=1.)
        report["after_max_stretch"]=max((r["max_stretch"] for r in after["regions"]),default=1.)
        if after["status"]=="not_evaluated" or after["candidate_faces"] > before["candidate_faces"]*.75 or report["after_max_stretch"] > report["before_max_stretch"]*.95:
            raise CannotSeparate("分離後の変形改善が十分ではないため元に戻しました")
        accepted=True
        report.update(status="separated",after_faces=after["candidate_faces"],reason="局所接触を分離し、表面と動作を検証しました")
        for obj in meshes: obj["automatic_separation_version"]=VERSION
    except CannotSeparate as exc:
        report.update(status="not_separated",reason=str(exc))
    finally:
        for obj,original in zip(meshes,originals):
            if accepted:
                if original.users==0: bpy.data.meshes.remove(original)
            else:
                temporary=obj.data; obj.data=original
                if temporary.users==0: bpy.data.meshes.remove(temporary)
    return report
