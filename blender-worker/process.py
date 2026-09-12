"""Blender post-processing jobs. Rig output is always a separate artifact."""
import bpy, json, sys, shutil, statistics
from mathutils import Vector, Quaternion
from pathlib import Path
# Blender --python does not add the script's directory to Python's import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attachment_diagnostics import write_diagnostics
from limb_separation import separate_contacts

args=sys.argv[sys.argv.index("--")+1:]
data,candidate_id,kind,payload,result=Path(args[0]),args[1],args[2],json.loads(args[3]),Path(args[4])

def rel(path): return str(path.relative_to(data)).replace("\\","/")

def load_model():
    if kind in {"rig", "game_prepare", "motion_prepare"}:
        source=data/payload["source_model_path"]
        if not source.is_file(): raise RuntimeError("fixed source model is missing")
        actual_hash=__import__("hashlib").sha256(source.read_bytes()).hexdigest()
        if actual_hash != payload["source_sha256"]: raise RuntimeError("入力モデルがジョブ作成後に変更されています。再実行してください")
    else:
        models=list((data/"projects").glob(f"*/runs/{candidate_id}/mesh.glb"))+list((data/"projects").glob(f"*/runs/{candidate_id}/model.obj"))
        if not models: raise RuntimeError("candidate model not found")
        source=models[0]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source.suffix.lower()==".glb": bpy.ops.import_scene.gltf(filepath=str(source))
    else: bpy.ops.wm.obj_import(filepath=str(source))
    meshes=[o for o in bpy.context.scene.objects if o.type=="MESH"]
    if not meshes: raise RuntimeError("no mesh")
    return meshes, source

def cleanup(meshes, target):
    total=sum(sum(max(0, len(face.vertices)-2) for face in o.data.polygons) for o in meshes); ratio=min(1.0, target/max(total,1))
    for obj in meshes:
        bpy.context.view_layer.objects.active=obj; obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        if ratio < .999:
            mod=obj.modifiers.new("Game budget decimation", "DECIMATE"); mod.ratio=ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)

def triangle_count(meshes):
    return sum(sum(max(0, len(face.vertices)-2) for face in obj.data.polygons) for obj in meshes)

def skinned_meshes(arm):
    return [o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" and m.object==arm for m in o.modifiers)]

def decimate_to(meshes, target):
    total=triangle_count(meshes); ratio=min(1.0, target/max(total,1))
    for obj in meshes:
        mod=obj.modifiers.new("LOD decimation", "DECIMATE"); mod.ratio=ratio
        bpy.context.view_layer.objects.active=obj; bpy.ops.object.modifier_apply(modifier=mod.name)

def normalize_weights(meshes, arm):
    """Make every skin vertex engine-safe: 1--4 finite, normalized influences."""
    stats={"vertices":0,"pruned_influences":0,"normalized_vertices":0}
    valid=set(b.name for b in arm.data.bones)
    for obj in meshes:
        groups={g.index:g for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            items=[(groups[x.group], x.weight) for x in vertex.groups if x.group in groups and groups[x.group].name in valid and x.weight > 1e-8]
            if not items: raise RuntimeError("ウェイト無し頂点が検出されました。自動補完せず停止しました")
            items.sort(key=lambda item:item[1], reverse=True)
            kept=items[:4]; stats["pruned_influences"]+=len(items)-len(kept)
            total=sum(weight for _,weight in kept)
            if not total or not all(float(weight)==float(weight) for _,weight in kept): raise RuntimeError("無効なスキニングウェイトが検出されました")
            for group,_ in items[4:]: group.remove([vertex.index])
            for group,weight in kept: group.add([vertex.index], weight/total, "REPLACE")
            stats["vertices"]+=1; stats["normalized_vertices"]+=1
    return stats

def limit_limb_weight_bleed(meshes, arm):
    """Remove only clearly remote limb weights before normalizing.

    Automatic heat weights can occasionally reach through a connected torso or
    pelvis surface.  A shoulder/hip needs blended weights, so this uses the
    distance to each limb bone segment rather than a hard body-side cut.  A
    remote influence is removed only when the vertex retains another valid
    influence; this never creates an unweighted vertex.
    """
    limb_names=("UpperArm.L","LowerArm.L","Hand.L","UpperArm.R","LowerArm.R","Hand.R",
                "UpperLeg.L","LowerLeg.L","Foot.L","UpperLeg.R","LowerLeg.R","Foot.R")
    points=[obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices]
    if not points: return {"removed":0,"retained_only_influence":0}
    height=max(point.z for point in points)-min(point.z for point in points)
    segments={}
    for name in limb_names:
        bone=arm.data.bones.get(name)
        if not bone: continue
        start=arm.matrix_world @ bone.head_local; end=arm.matrix_world @ bone.tail_local
        length=max((end-start).length, height*.025)
        # Keep a shoulder/hip blending zone, while excluding the torso/pelvis
        # centre from a limb several bone-widths away.
        segments[name]=(start,end,max(length*.60,height*.07))
    def distance_to_segment(point, start, end):
        vector=end-start; length_sq=vector.length_squared
        if length_sq <= 1e-12: return (point-start).length
        factor=max(0.0,min(1.0,(point-start).dot(vector)/length_sq))
        return (point-(start+vector*factor)).length
    stats={"removed":0,"retained_only_influence":0}
    valid=set(bone.name for bone in arm.data.bones)
    for obj in meshes:
        groups={group.index:group for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            items=[(groups[item.group],item.weight) for item in vertex.groups if item.group in groups and groups[item.group].name in valid and item.weight > 1e-8]
            remote=[]
            point=obj.matrix_world @ vertex.co
            for group,weight in items:
                segment=segments.get(group.name)
                if segment and distance_to_segment(point,*segment[:2]) > segment[2]: remote.append((group,weight))
            if not remote: continue
            remaining=sum(weight for group,weight in items if (group,weight) not in remote)
            if remaining <= 1e-8:
                stats["retained_only_influence"]+=len(remote)
                continue
            for group,_ in remote:
                group.remove([vertex.index]); stats["removed"]+=1
    return stats

def remove_cross_limb_weights(meshes, arm):
    """Prevent an arm/leg from pulling the other branch through a close pose.

    Downward hands can sit next to thighs, so proximity alone cannot identify
    their ownership.  When one branch already has at least 70% of a vertex's
    valid skinning weight, discard only the other branch's minority weights.
    The conservative threshold keeps genuine hip/shoulder trunk blending.
    """
    arms={"UpperArm.L","LowerArm.L","Hand.L","UpperArm.R","LowerArm.R","Hand.R"}
    legs={"UpperLeg.L","LowerLeg.L","Foot.L","UpperLeg.R","LowerLeg.R","Foot.R"}
    valid=set(bone.name for bone in arm.data.bones); removed=0
    for obj in meshes:
        groups={group.index:group for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            items=[(groups[item.group],item.weight) for item in vertex.groups if item.group in groups and groups[item.group].name in valid and item.weight > 1e-8]
            arm_weight=sum(weight for group,weight in items if group.name in arms)
            leg_weight=sum(weight for group,weight in items if group.name in legs)
            if arm_weight >= .70 and leg_weight > 0:
                for group,_ in items:
                    if group.name in legs: group.remove([vertex.index]); removed+=1
            elif leg_weight >= .70 and arm_weight > 0:
                for group,_ in items:
                    if group.name in arms: group.remove([vertex.index]); removed+=1
    return {"removed":removed}

def make_humanoid_armature(meshes):
    points=[obj.matrix_world @ v.co for obj in meshes for v in obj.data.vertices]
    low=[min(p[i] for p in points) for i in range(3)]; high=[max(p[i] for p in points) for i in range(3)]
    height=max(high[2]-low[2], .01); center=((low[0]+high[0])*.5,(low[1]+high[1])*.5)
    def pos(x,y,z): return (center[0]+x,center[1]+y,low[2]+z*height)
    width=max(high[0]-low[0], height*.12)

    def outer_limb_point(side, z, fallback_x):
        """Use the mesh silhouette at a height band for a down-arm rest pose.

        The earlier fixed A-pose layout placed elbows beyond a narrow, arms-down
        character.  Automatic weights then had no vertices to assign to the
        forearm.  Sampling the actual outer silhouette keeps each rest bone in
        the source mesh's coordinate system.  A robust median avoids a single
        texture/mesh outlier pulling a joint outside the body.
        """
        half_band=height*.045
        target=low[2]+z*height
        band=[p for p in points if abs(p[2]-target) <= half_band and (p[0]-center[0])*side > 0]
        if len(band) < 12:
            return pos(fallback_x, 0, z)
        band.sort(key=lambda p:p[0], reverse=side > 0)
        edge=band[:max(4, len(band)//8)]
        return (statistics.median(p[0] for p in edge), statistics.median(p[1] for p in edge), target)

    def leg_axis_point(side, z, fallback_x):
        """Estimate a leg centerline independently of the arm-span width.

        A T-pose makes the global X extent mostly arms.  Using that extent for
        the hips placed leg bones outside the thighs.  At calf/ankle heights a
        side-specific median is robust to the front/back surface vertices and
        remains local to the actual leg.
        """
        half_band=height*.035; target=low[2]+z*height
        band=[p for p in points if abs(p[2]-target) <= half_band and (p[0]-center[0])*side > height*.018]
        if len(band) < 8: return pos(fallback_x, 0, z)
        return (statistics.median(p[0] for p in band), statistics.median(p[1] for p in band), target)

    def tpose_arm(side):
        """Build a horizontal arm chain from upper-body cross sections."""
        chest=[p for p in points if low[2]+height*.60 <= p[2] <= low[2]+height*.68]
        torso=sorted(abs(p[0]-center[0]) for p in chest)
        torso_half=torso[min(len(torso)-1, int(len(torso)*.85))] if len(torso) >= 16 else height*.13
        arm_points=[p for p in points if low[2]+height*.62 <= p[2] <= low[2]+height*.88 and (p[0]-center[0])*side > torso_half*.8]
        if len(arm_points) < 16: raise RuntimeError("Tポーズの腕領域を検出できませんでした。腕を肩の高さで左右へ伸ばした4方向画像で再生成してください")
        xs=sorted((p[0]-center[0])*side for p in arm_points)
        outer=xs[max(0, int(len(xs)*.92)-1)]
        shoulder=max(torso_half*.92, height*.10)
        if outer-shoulder < height*.16: raise RuntimeError("Tポーズの腕の長さを確認できませんでした。腕を胴体から離してください")
        wrist=outer-height*.075; elbow=shoulder+(wrist-shoulder)*.52
        def point_at(offset):
            near=[p for p in arm_points if abs((p[0]-center[0])*side-offset) <= height*.025]
            if not near: return (center[0]+side*offset, center[1], low[2]+height*.76)
            return (center[0]+side*offset, statistics.median(p[1] for p in near), statistics.median(p[2] for p in near))
        shoulder_point=point_at(shoulder)
        # The inner shoulder slice also contains the chest/underarm wall.  Use
        # only the adjacent outer upper-arm band for its Y/Z centerline.
        upper=[p for p in arm_points if shoulder+height*.04 <= (p[0]-center[0])*side <= elbow-height*.03]
        if len(upper) >= 6:
            shoulder_point=(shoulder_point[0], statistics.median(p[1] for p in upper), statistics.median(p[2] for p in upper))
        return shoulder_point,point_at(elbow),point_at(wrist),point_at(outer)

    spec=[("Hips",pos(0,0,.48),pos(0,0,.58),None), ("Spine",pos(0,0,.58),pos(0,0,.70),"Hips"),
          ("Chest",pos(0,0,.70),pos(0,0,.80),"Spine"), ("Neck",pos(0,0,.80),pos(0,0,.86),"Chest"),
          ("Head",pos(0,0,.86),pos(0,0,.98),"Neck")]
    for side in (-1,1):
        tag="L" if side>0 else "R"
        shoulder,elbow,wrist,hand=tpose_arm(side)
        ankle=leg_axis_point(side,.06,side*height*.07)
        knee=leg_axis_point(side,.27,side*height*.07)
        # Waist/garment hems widen the .49 cross-section; use the upper-thigh
        # centerline and extend it to the anatomical hip height instead.
        thigh_axis=leg_axis_point(side,.39,side*height*.07)
        hip=(thigh_axis[0], thigh_axis[1], low[2]+height*.49)
        spec += [(f"UpperArm.{tag}",shoulder,elbow,"Chest"),
                 (f"LowerArm.{tag}",elbow,wrist,f"UpperArm.{tag}"),
                 (f"Hand.{tag}",wrist,hand,f"LowerArm.{tag}"),
                 (f"UpperLeg.{tag}",hip,knee,"Hips"),
                 (f"LowerLeg.{tag}",knee,ankle,f"UpperLeg.{tag}"),
                 (f"Foot.{tag}",ankle,(ankle[0],ankle[1]-height*.08,low[2]+height*.02),f"LowerLeg.{tag}")]
    bpy.ops.object.armature_add(enter_editmode=True, location=(0,0,0)); arm=bpy.context.object; arm.name="AutoRig"
    bpy.ops.armature.select_all(action="SELECT"); bpy.ops.armature.delete()
    bones={}
    for name,head,tail,parent in spec:
        bone=arm.data.edit_bones.new(name); bone.head=head; bone.tail=tail; bone.parent=bones.get(parent); bones[name]=bone
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm

def validate_skinning(meshes, arm):
    # A merely non-empty vertex-group list is insufficient: an armature can
    # export successfully while an elbow or knee has zero influence.
    required=("UpperArm.L","LowerArm.L","UpperArm.R","LowerArm.R",
              "UpperLeg.L","LowerLeg.L","UpperLeg.R","LowerLeg.R")
    totals={name:{"vertices":0,"weight":0.0} for name in required}
    for obj in meshes:
        mod=next((m for m in obj.modifiers if m.type=="ARMATURE" and m.object==arm),None)
        if not mod or not obj.vertex_groups: raise RuntimeError("自動ウェイト検証に失敗しました")
        groups={g.index:g.name for g in obj.vertex_groups}
        valid=set(b.name for b in arm.data.bones)
        for vertex in obj.data.vertices:
            weights=[g.weight for g in vertex.groups if groups.get(g.group) in valid]
            if not weights or len(weights)>4 or any(w < 0 for w in weights) or not all(float(w)==float(w) for w in weights) or abs(sum(weights)-1.0)>1e-3:
                raise RuntimeError("無効なスキニングウェイトが検出されました")
            for assignment in vertex.groups:
                name=groups.get(assignment.group)
                if name in totals and assignment.weight > 1e-6:
                    totals[name]["vertices"]+=1; totals[name]["weight"]+=assignment.weight
    missing=[name for name, value in totals.items() if value["vertices"] == 0]
    if missing:
        raise RuntimeError("自動ウェイトが必須関節に割り当てられませんでした: " + ", ".join(missing))
    return totals

def rig(meshes, target_triangles=None):
    cleanup(meshes, target_triangles if target_triangles is not None else payload["target_triangles"])
    arm=make_humanoid_armature(meshes)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes: obj.select_set(True)
    arm.select_set(True); bpy.context.view_layer.objects.active=arm
    try: bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc: raise RuntimeError("自動ウェイトの計算に失敗しました。腕を肩の高さで伸ばし、脚を離したTポーズの4方向画像で再生成してください") from exc
    cross_branch=remove_cross_limb_weights(meshes, arm)
    bleed=limit_limb_weight_bleed(meshes, arm)
    correction=normalize_weights(meshes, arm)
    separation=separate_contacts(meshes, arm)
    # New cap-center vertices receive interpolated weights; normalize only
    # after a successful topology change. Failed/uncertain repairs are already
    # restored by separate_contacts.
    if separation["status"] == "separated": normalize_weights(meshes, arm)
    return arm, {"correction":correction,"cross_limb_weight":cross_branch,
                 "limb_weight_bleed":bleed,"separation":separation,
                 "coverage":validate_skinning(meshes, arm)}

UNITY_HUMANOID_MAP={"Hips":"Hips","Spine":"Spine","Chest":"Chest","Neck":"Neck","Head":"Head","UpperArm.L":"LeftUpperArm","LowerArm.L":"LeftLowerArm","Hand.L":"LeftHand","UpperArm.R":"RightUpperArm","LowerArm.R":"RightLowerArm","Hand.R":"RightHand","UpperLeg.L":"LeftUpperLeg","LowerLeg.L":"LeftLowerLeg","Foot.L":"LeftFoot","UpperLeg.R":"RightUpperLeg","LowerLeg.R":"RightLowerLeg","Foot.R":"RightFoot"}

def export_and_validate(root, label):
    glb=root/f"{label}.glb"; fbx=root/f"{label}.fbx"
    bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB")
    # Unity can import this FBX without depending on the worker's /data path.
    # COPY + embed_textures keeps the source GLB material images inside the FBX.
    bpy.ops.export_scene.fbx(filepath=str(fbx),use_armature_deform_only=True,path_mode="COPY",embed_textures=True)
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(glb))
    arm=next((o for o in bpy.context.scene.objects if o.type=="ARMATURE"),None)
    meshes=skinned_meshes(arm) if arm else []
    if not arm or not meshes: raise RuntimeError(f"{label}: GLBのリグ検証に失敗しました")
    validation={"triangles":triangle_count(meshes),"skinning":validate_skinning(meshes,arm)}
    diagnostics={key:rel(path) for key,path in write_diagnostics(meshes,arm,glb,root,label.replace("-","_")).items()}
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=str(fbx))
    fbx_arm=next((o for o in bpy.context.scene.objects if o.type=="ARMATURE"),None)
    fbx_meshes=skinned_meshes(fbx_arm) if fbx_arm else []
    if not fbx_arm or not fbx_meshes: raise RuntimeError(f"{label}: FBXのリグ検証に失敗しました")
    validate_skinning(fbx_meshes,fbx_arm)
    if not any(image.size[0] > 0 and image.size[1] > 0 for image in bpy.data.images): raise RuntimeError(f"{label}: FBXのテクスチャ検証に失敗しました")
    return {"glb":rel(glb),"fbx":rel(fbx),**validation,"fbx_validated":True,"diagnostics":diagnostics}

def game_prepare(meshes):
    cleanup(meshes, payload["lod0_triangles"])
    arm, skinning=rig(meshes, payload["lod0_triangles"]); root=result.parent
    points=[obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices]
    low=[min(point[i] for point in points) for i in range(3)]; high=[max(point[i] for point in points) for i in range(3)]
    height=max(high[2]-low[2], .01); radius=max((high[0]-low[0])*.5, (high[1]-low[1])*.5, height*.06)
    radius=min(radius, height*.49)
    collision={"shape":"capsule","axis":"Y","radius":radius,"height":max(height, radius*2),"center":[(low[0]+high[0])*.5,(low[1]+high[1])*.5,(low[2]+high[2])*.5],"unity":"モデルに CapsuleCollider または CharacterController を一方だけ追加し、この値を入力してください。"}
    collision_path=root/"unity-collision.json"; collision_path.write_text(json.dumps(collision,ensure_ascii=False,indent=2),encoding="utf-8")
    separation_path=root/"unity-separation-report.json"
    separation_path.write_text(json.dumps(skinning["separation"],ensure_ascii=False,indent=2),encoding="utf-8")
    blend=root/"unity-editable.blend"; bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    package={"lod0":export_and_validate(root,"unity-lod0")}
    # Re-import each preceding LOD so lower LODs retain exactly the same rig.
    for label, ratio in (("lod1", payload["lod_ratios"][1]),("lod2", payload["lod_ratios"][2])):
        arm=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE"); meshes=skinned_meshes(arm)
        decimate_to(meshes, max(1, round(payload["lod0_triangles"]*ratio)))
        remove_cross_limb_weights(meshes, arm)
        limit_limb_weight_bleed(meshes, arm)
        normalize_weights(meshes, arm)
        validate_skinning(meshes, arm)
        package[label]=export_and_validate(root, f"unity-{label}")
    manifest={"pipeline":payload["pipeline"],"engine_profile":payload["engine_profile"],"source_model_path":payload["source_model_path"],"source_sha256":payload["source_sha256"],"unity_humanoid_map":UNITY_HUMANOID_MAP,"lods":package,"lod0_skinning":skinning,"warning":"UnityではFBX Import Settings の Rig を Humanoid にし、Humanoid Configure でこの対応表を確認してください。"}
    manifest_path=root/"unity-manifest.json"; manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    artifacts={"unity_lod0_glb":package["lod0"]["glb"],"unity_lod0_fbx":package["lod0"]["fbx"],"unity_lod1_glb":package["lod1"]["glb"],"unity_lod1_fbx":package["lod1"]["fbx"],"unity_lod2_glb":package["lod2"]["glb"],"unity_lod2_fbx":package["lod2"]["fbx"],"unity_editable_blend":rel(blend),"unity_manifest":rel(manifest_path),"unity_collision":rel(collision_path),"unity_separation_report":rel(separation_path)}
    for lod in package.values(): artifacts.update(lod["diagnostics"])
    json.dump({"artifacts":artifacts,"metadata":{"pipeline":payload["pipeline"],"validated":True,"engine_profile":payload["engine_profile"]}},result.open("w"))

def motion_prepare():
    arm=next((obj for obj in bpy.context.scene.objects if obj.type=="ARMATURE"),None)
    meshes=skinned_meshes(arm) if arm else []
    if not arm or not meshes: raise RuntimeError("モーション入力にリグ付きメッシュがありません")
    # The source rig is authored in T-pose.  Procedural locomotion must first
    # establish a down-arm locomotion pose instead of treating T-pose as idle.
    down_arm={}
    for side in ("L", "R"):
        bone=arm.pose.bones.get(f"UpperArm.{side}")
        if bone:
            rest=bone.bone.matrix_local.to_3x3()
            # Keep a small lateral clearance from the torso throughout gait.
            # This is a posture target, not a model-specific coordinate hack.
            side_sign=1 if side == "L" else -1
            down_outward=Vector((side_sign*.13,0,-1)).normalized()
            target=arm.matrix_world.to_3x3().inverted() @ down_outward
            down_arm[bone.name]=Vector((0,1,0)).rotation_difference((rest.inverted() @ target).normalized())
    for template in payload["templates"]:
        action=bpy.data.actions.new(template); action.use_fake_user=True
        frames=48 if template in {"Walk","Run"} else 32
        targets=[("UpperArm.L",1),("UpperArm.R",-1),("UpperLeg.L",-1),("UpperLeg.R",1)]
        if template=="Wave": targets=[("UpperArm.R",1),("LowerArm.R",1),("Hand.R",1)]
        if template=="Idle": targets=[("Chest",1)]
        if template=="Jump": targets=[("UpperLeg.L",-1),("UpperLeg.R",-1),("LowerLeg.L",1),("LowerLeg.R",1)]
        for name, sign in targets:
            bone=arm.pose.bones.get(name)
            if not bone: continue
            # Rotation mode belongs to the pose bone, not to an Action.  Keep
            # every generated clip in Quaternion form so switching a Wave
            # cannot invalidate the Euler tracks of Walk/Run.
            bone.rotation_mode="QUATERNION"
            baseline=down_arm.get(name, Quaternion()) if template in {"Walk","Run"} else Quaternion()
            if template=="Wave":
                rest=bone.bone.matrix_local.to_3x3()
                # The generated rig is Z-up in world space.  Convert its
                # forward axis through the armature and rest transforms.
                body_axis=arm.matrix_world.to_3x3().inverted() @ Vector((0,1,0))
                local_axis=(rest.inverted() @ body_axis).normalized()
            elif name.startswith("UpperArm.") and template in {"Walk","Run"}:
                # Swing around the body's left/right axis.  Convert through
                # both the rest basis and the down-arm baseline: the result is
                # a forward/back swing after lowering the T-pose arm.
                rest=bone.bone.matrix_local.to_3x3()
                body_lateral=arm.matrix_world.to_3x3().inverted() @ Vector((1,0,0))
                local_axis=(baseline.inverted() @ (rest.inverted() @ body_lateral)).normalized()
            else:
                # Preserve the existing gait/jump bending convention while
                # representing it as a Quaternion rather than Euler curves.
                local_axis=Vector((1,0,0))
            amplitude=0.12 if template=="Idle" else .38 if template in {"Walk","Run"} else 0.8
            if template=="Wave":
                # Small body-forward Wave rotations avoid pulling a down-arm
                # through the torso regardless of the generated bone roll.
                amplitude={"UpperArm.R":.16,"LowerArm.R":.32,"Hand.R":.18}[name]
            path=f'pose.bones["{name}"].rotation_quaternion'
            curves=[]
            for index,value in enumerate(baseline):
                curve=action.fcurves.new(path,index=index); curves.append(curve)
                curve.keyframe_points.insert(1,value); curve.keyframe_points.insert(frames,value)
            rotation=baseline @ Quaternion(local_axis,sign*amplitude)
            reverse=baseline @ Quaternion(local_axis,-sign*amplitude)
            middle=baseline if template in {"Walk","Run"} else rotation
            for curve,value in zip(curves,middle): curve.keyframe_points.insert(frames//2,value)
            for curve,value in zip(curves,rotation): curve.keyframe_points.insert(frames//4,value)
            for curve,value in zip(curves,reverse): curve.keyframe_points.insert(frames*3//4,value)
        track=arm.animation_data_create().nla_tracks.new(); track.name=template; track.strips.new(template,1,action)
    root=result.parent; glb=root/"template-motions.glb"; fbx=root/"template-motions.fbx"
    bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB",export_animations=True,export_nla_strips=True)
    bpy.ops.export_scene.fbx(filepath=str(fbx),bake_anim=True,path_mode="COPY",embed_textures=True)
    diagnostics={}
    def validate_motion_file(path, importer, label):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        importer(filepath=str(path))
        imported_arm=next((obj for obj in bpy.context.scene.objects if obj.type=="ARMATURE"),None)
        imported_meshes=skinned_meshes(imported_arm) if imported_arm else []
        if not imported_arm or not imported_meshes:
            raise RuntimeError(f"{label}: モーションのリグ検証に失敗しました")
        actions=list(bpy.data.actions)
        # Importers qualify animation names differently: FBX stores the
        # armature's take as "AutoRig|Walk", while Blender's GLB importer
        # appends the target object as "Walk_AutoRig".  Do not match generic
        # suffixes: FBX also imports mesh/world takes with the same template
        # name, which do not animate the armature.
        def action_for(template):
            names={template, f"{imported_arm.name}|{template}", f"{template}_{imported_arm.name}"}
            matches=[action for action in actions if action.name in names]
            if len(matches) > 1:
                raise RuntimeError(f"{label}: {template} に対応するモーションクリップが複数あります")
            return matches[0] if matches else None
        clips={template:action_for(template) for template in payload["templates"]}
        missing=[template for template,action in clips.items() if action is None]
        if missing:
            raise RuntimeError(f"{label}: モーションクリップがありません: " + ", ".join(missing))
        validate_skinning(imported_meshes, imported_arm)
        if not any(image.size[0] > 0 and image.size[1] > 0 for image in bpy.data.images):
            raise RuntimeError(f"{label}: テクスチャ検証に失敗しました")
        # Evaluate each imported action independently.  Checking actions alone
        # is not enough: a clip with unmapped bone tracks exports but never moves
        # the skinned geometry in a viewer or engine.
        imported_arm.animation_data_create()
        for track in list(imported_arm.animation_data.nla_tracks):
            imported_arm.animation_data.nla_tracks.remove(track)
        depsgraph=bpy.context.evaluated_depsgraph_get()
        for template in payload["templates"]:
            action=clips[template]
            start,end=action.frame_range
            imported_arm.animation_data.action=action
            bpy.context.scene.frame_set(round(start))
            before=[obj.matrix_world @ vertex.co for obj in imported_meshes for vertex in obj.evaluated_get(depsgraph).data.vertices]
            # Every template now has a periodic neutral midpoint; check a
            # quarter phase where the actual template motion peaks.
            sample=start+(end-start)*.25
            bpy.context.scene.frame_set(round(sample))
            after=[obj.matrix_world @ vertex.co for obj in imported_meshes for vertex in obj.evaluated_get(depsgraph).data.vertices]
            if len(before) != len(after) or not any((a-b).length > 1e-5 for a,b in zip(before,after)):
                raise RuntimeError(f"{label}: {template} でメッシュ変形を確認できません")
        if path == glb:
            diagnostics.update({key:rel(value) for key,value in write_diagnostics(imported_meshes,imported_arm,glb,root,"motion",include_actions=True).items()})
        return {"clips":payload["templates"],"mesh_deformation_validated":True}
    validation={"glb":validate_motion_file(glb,bpy.ops.import_scene.gltf,"template-motions.glb"),
                "fbx":validate_motion_file(fbx,bpy.ops.import_scene.fbx,"template-motions.fbx")}
    manifest=root/"motion-manifest.json"; manifest.write_text(json.dumps({"templates":payload["templates"],"warning":"手続き的テンプレートです。足接地IK・自由文モーション生成は含みません。"},ensure_ascii=False,indent=2),encoding="utf-8")
    json.dump({"artifacts":{"template_motion_glb":rel(glb),"template_motion_fbx":rel(fbx),"motion_manifest":rel(manifest),**diagnostics},"metadata":{"pipeline":payload["pipeline"],"validated":True,"motion_validation":validation}},result.open("w"))

meshes, source=load_model()
if kind=="optimize":
    cleanup(meshes, payload["triangles"]); out=result.parent/"optimized.glb"; bpy.ops.export_scene.gltf(filepath=str(out),export_format="GLB")
    json.dump({"model_path":rel(out)},result.open("w"))
elif kind=="rig":
    arm, skinning=rig(meshes); root=result.parent; blend=root/"editable-rig.blend"; glb=root/"rigged.glb"; fbx=root/"rigged.fbx"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend)); bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB"); bpy.ops.export_scene.fbx(filepath=str(fbx),use_armature_deform_only=True)
    # Validate the distributed GLB, not only Blender's pre-export scene.  GLB
    # export limits vertex influences, so validate after that reduction too.
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(glb))
    exported_arm=next((o for o in bpy.context.scene.objects if o.type=="ARMATURE"),None)
    # Source GLBs may also contain static helper meshes.  They are not skin
    # data, so validate the mesh(es) actually bound to the exported armature.
    exported_meshes=[o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" and m.object==exported_arm for m in o.modifiers)]
    if not exported_arm or not exported_meshes: raise RuntimeError("rigged GLB validation failed")
    exported_skinning=validate_skinning(exported_meshes, exported_arm)
    diagnostics={key:rel(value) for key,value in write_diagnostics(exported_meshes,exported_arm,glb,root,"rig").items()}
    report=root/"report.json"; report.write_text(json.dumps({"source_model_path":payload["source_model_path"],"source_sha256":payload["source_sha256"],"target_triangles":payload["target_triangles"],"skinning":exported_skinning,"warning":"自動骨配置・自動ウェイトです。ゲーム投入前に関節変形を必ず確認してください。"},ensure_ascii=False,indent=2),encoding="utf-8")
    json.dump({"artifacts":{"rigged_glb":rel(glb),"rigged_fbx":rel(fbx),"editable_blend":rel(blend),"rig_report":rel(report),**diagnostics},"metadata":{"pipeline":payload["pipeline"],"source_sha256":payload["source_sha256"],"validated":True}},result.open("w"))
elif kind=="game_prepare":
    game_prepare(meshes)
elif kind=="motion_prepare":
    motion_prepare()
else:
    artifacts={}; formats=payload["formats"]
    if "glb" in formats:
        p=result.parent/"model.glb";bpy.ops.export_scene.gltf(filepath=str(p),export_format="GLB");artifacts["glb"]=rel(p)
    if "fbx" in formats:
        p=result.parent/"model.fbx";bpy.ops.export_scene.fbx(filepath=str(p));artifacts["fbx"]=rel(p)
    for p in artifacts.values():
        bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(data/p)) if p.endswith('.glb') else bpy.ops.import_scene.fbx(filepath=str(data/p))
        if not any(o.type=="MESH" for o in bpy.context.scene.objects): raise RuntimeError("export validation: mesh missing")
    zip_path=shutil.make_archive(str(result.parent/"export"),"zip",root_dir=result.parent);artifacts["zip"]=rel(Path(zip_path))
    json.dump({"artifacts":artifacts,"metadata":{"validated":True,"preset":payload["preset"]}},result.open("w"))
