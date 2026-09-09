"""Blender post-processing jobs. Rig output is always a separate artifact."""
import bpy, json, sys, shutil, statistics
from pathlib import Path

args=sys.argv[sys.argv.index("--")+1:]
data,candidate_id,kind,payload,result=Path(args[0]),args[1],args[2],json.loads(args[3]),Path(args[4])

def rel(path): return str(path.relative_to(data)).replace("\\","/")

def load_model():
    if kind == "rig":
        source=data/payload["source_model_path"]
        if not source.is_file(): raise RuntimeError("fixed source model is missing")
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
    total=sum(len(o.data.polygons) for o in meshes); ratio=min(1.0, target/max(total,1))
    for obj in meshes:
        bpy.context.view_layer.objects.active=obj; obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001); bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        if ratio < .999:
            mod=obj.modifiers.new("Game budget decimation", "DECIMATE"); mod.ratio=ratio
            bpy.ops.object.modifier_apply(modifier=mod.name)
        obj.select_set(False)

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

    spec=[("Hips",pos(0,0,.48),pos(0,0,.58),None), ("Spine",pos(0,0,.58),pos(0,0,.70),"Hips"),
          ("Chest",pos(0,0,.70),pos(0,0,.80),"Spine"), ("Neck",pos(0,0,.80),pos(0,0,.86),"Chest"),
          ("Head",pos(0,0,.86),pos(0,0,.98),"Neck")]
    for side in (-1,1):
        tag="L" if side>0 else "R"
        shoulder=outer_limb_point(side,.77,side*width*.28)
        elbow=outer_limb_point(side,.59,side*width*.42)
        wrist=outer_limb_point(side,.43,side*width*.45)
        hand=outer_limb_point(side,.34,side*width*.45)
        spec += [(f"UpperArm.{tag}",shoulder,elbow,"Chest"),
                 (f"LowerArm.{tag}",elbow,wrist,f"UpperArm.{tag}"),
                 (f"Hand.{tag}",wrist,hand,f"LowerArm.{tag}"),
                 (f"UpperLeg.{tag}",pos(side*width*.22,0,.49),pos(side*width*.27,0,.27),"Hips"),
                 (f"LowerLeg.{tag}",pos(side*width*.27,0,.27),pos(side*width*.27,0,.06),f"UpperLeg.{tag}"),
                 (f"Foot.{tag}",pos(side*width*.27,0,.06),pos(side*width*.27,-height*.08,.02),f"LowerLeg.{tag}")]
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
        for vertex in obj.data.vertices:
            weights=[g.weight for g in vertex.groups]
            if not weights or any(w < 0 for w in weights) or not all(float(w)==float(w) for w in weights):
                raise RuntimeError("無効なスキニングウェイトが検出されました")
            for assignment in vertex.groups:
                name=groups.get(assignment.group)
                if name in totals and assignment.weight > 1e-6:
                    totals[name]["vertices"]+=1; totals[name]["weight"]+=assignment.weight
    missing=[name for name, value in totals.items() if value["vertices"] == 0]
    if missing:
        raise RuntimeError("自動ウェイトが必須関節に割り当てられませんでした: " + ", ".join(missing))
    return totals

def rig(meshes):
    cleanup(meshes, payload["target_triangles"])
    arm=make_humanoid_armature(meshes)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes: obj.select_set(True)
    arm.select_set(True); bpy.context.view_layer.objects.active=arm
    try: bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError as exc: raise RuntimeError("自動ウェイトの計算に失敗しました。腕を自然に下げた直立全身画像で再生成してください") from exc
    return arm, validate_skinning(meshes, arm)

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
    report=root/"report.json"; report.write_text(json.dumps({"source_model_path":payload["source_model_path"],"source_sha256":payload["source_sha256"],"target_triangles":payload["target_triangles"],"skinning":exported_skinning,"warning":"自動骨配置・自動ウェイトです。ゲーム投入前に関節変形を必ず確認してください。"},ensure_ascii=False,indent=2),encoding="utf-8")
    json.dump({"artifacts":{"rigged_glb":rel(glb),"rigged_fbx":rel(fbx),"editable_blend":rel(blend),"rig_report":rel(report)},"metadata":{"pipeline":payload["pipeline"],"source_sha256":payload["source_sha256"],"validated":True}},result.open("w"))
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
