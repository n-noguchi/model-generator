"""HumanML3D joint positions -> an existing AutoRig, preserving mesh and rest bones."""
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

LIMBS = {"UpperArm.L":(16,18),"LowerArm.L":(18,20),"Hand.L":(18,20),
         "UpperArm.R":(17,19),"LowerArm.R":(19,21),"Hand.R":(19,21),
         "UpperLeg.L":(1,4),"LowerLeg.L":(4,7),"Foot.L":(7,10),
         "UpperLeg.R":(2,5),"LowerLeg.R":(5,8),"Foot.R":(8,11)}
TORSO = {"Hips":(0,3,1,2),"Spine":(3,6,1,2),"Chest":(6,9,16,17),"Neck":(12,15,16,17),"Head":(12,15,16,17)}
REQUIRED = set(LIMBS) | set(TORSO)

def body_frame(up, across):
    up = up.normalized()
    lateral = across - up * across.dot(up)
    if lateral.length < 1e-5: raise ValueError("生成された胴体の方向を判定できません")
    lateral.normalize(); forward = up.cross(lateral).normalized()
    return Matrix((lateral, forward, up)).transposed().to_quaternion()

def signature(meshes):
    return [(o.name,len(o.data.vertices),len(o.data.polygons),
             hashlib.sha256(np.array([tuple(v.co) for v in o.data.vertices],dtype=np.float32).tobytes()).hexdigest()) for o in meshes]


def root_translation(root, first_root, scale, target_floor, rest_hip_height):
    # HumanML3D's height is absolute relative to its floor at zero. Only XY
    # starts at the target's origin; subtracting initial Z loses crouched poses.
    translation=Vector((root-first_root)*scale)
    translation.z=target_floor+float(root[2])*scale-rest_hip_height
    return translation

def retarget(source, joints_path, out, payload):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=payload["source_sha256"]: raise ValueError("入力リグがジョブ作成後に変更されています")
    with np.load(joints_path,allow_pickle=False) as data:
        points=data["joints"].copy(); fps=int(data["fps"])
    if points.ndim!=3 or points.shape[1:]!=(22,3) or not 40<=len(points)<=160 or fps!=20 or not np.isfinite(points).all(): raise ValueError("生成関節データが不正です")
    # HumanML3D is Y-up, +Z forward. Blender AutoRig is Z-up, -Y forward.
    points=points[:,:,[0,2,1]]; points[:,:,1]*=-1
    initial_across=points[0,1]-points[0,2]
    yaw=-math.atan2(initial_across[1],initial_across[0])
    alignment=np.array(((math.cos(yaw),-math.sin(yaw),0),(math.sin(yaw),math.cos(yaw),0),(0,0,1)))
    points=points @ alignment.T
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(source))
    arm=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
    if not REQUIRED <= set(arm.pose.bones.keys()): raise ValueError("このリグはText-to-Motionの対象骨格ではありません")
    meshes=[o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" and m.object==arm for m in o.modifiers)]
    if not meshes: raise ValueError("スキニングされたメッシュがありません")
    before=signature(meshes)
    # Preserve an existing rig's lengths; only root translation is scaled.
    source_leg=np.median(np.linalg.norm(points[:,1]-points[:,4],axis=1)+np.linalg.norm(points[:,4]-points[:,7],axis=1))
    target_leg=sum((arm.matrix_world.to_3x3() @ (arm.data.bones[n].tail_local-arm.data.bones[n].head_local)).length for n in ("UpperLeg.L","LowerLeg.L"))
    scale=target_leg/max(float(source_leg),1e-5)
    target_floor=min((o.matrix_world @ v.co).z for o in meshes for v in o.data.vertices)
    rest_hip_height=(arm.matrix_world @ arm.data.bones["Hips"].head_local).z
    world_inv=arm.matrix_world.to_3x3().inverted(); world_rotation=arm.matrix_world.to_quaternion()
    arm.animation_data_clear()
    for action in list(bpy.data.actions): bpy.data.actions.remove(action)
    action=bpy.data.actions.new("CustomMotion"); arm.animation_data_create().action=action
    order=sorted(arm.pose.bones,key=lambda b:len(b.parent_recursive))
    rest={bone.name:bone.bone.matrix_local.to_quaternion() for bone in order}
    prev={}; scene=bpy.context.scene; scene.render.fps=fps; scene.frame_start=1; scene.frame_end=len(points)
    for index,frame in enumerate(points):
        desired={}; positions=[Vector(p) for p in frame]
        for bone in order:
            bone.rotation_mode="QUATERNION"; bone.location=(0,0,0); bone.scale=(1,1,1)
            # Rest basis under the already posed parent: local delta is
            # computed relative to this, never copied from the source rig.
            parent=bone.parent
            base=desired[parent.name] @ rest[parent.name].inverted() @ rest[bone.name] if parent else rest[bone.name]
            if bone.name in TORSO:
                a,b,l,r=TORSO[bone.name]
                orientation=world_rotation.inverted() @ body_frame(positions[b]-positions[a],positions[l]-positions[r]) @ world_rotation @ rest[bone.name]
            elif bone.name in LIMBS:
                a,b=LIMBS[bone.name]; direction=world_inv @ (positions[b]-positions[a])
                if direction.length<1e-5: raise ValueError("生成された手足の骨長がゼロです")
                orientation=(base @ Vector((0,1,0))).rotation_difference(direction.normalized()) @ base
            else: orientation=base
            delta=base.inverted() @ orientation; delta.normalize()
            if bone.name in prev and prev[bone.name].dot(delta)<0: delta.negate()
            bone.rotation_quaternion=delta; prev[bone.name]=delta.copy(); desired[bone.name]=orientation
            if bone.name=="Hips":
                translation=root_translation(frame[0],points[0,0],scale,target_floor,rest_hip_height)
                bone.location=rest[bone.name].inverted() @ (world_inv @ translation)
                bone.keyframe_insert("location",frame=index+1,group=bone.name)
            bone.keyframe_insert("rotation_quaternion",frame=index+1,group=bone.name)
    for curve in action.fcurves:
        for key in curve.keyframe_points: key.interpolation="LINEAR"
    if signature(meshes)!=before: raise RuntimeError("リターゲット処理で元のメッシュが変わっています")
    scene.frame_set(1)
    glb=out/"custom-motion.glb"; fbx=out/"custom-motion.fbx"
    bpy.ops.export_scene.gltf(filepath=str(glb),export_format="GLB",export_animations=True,export_frame_range=True)
    bpy.ops.export_scene.fbx(filepath=str(fbx),bake_anim=True,bake_anim_use_nla_strips=False,bake_anim_use_all_actions=False,path_mode="COPY",embed_textures=True,add_leaf_bones=False)
    verification={}
    for path,importer in [(glb,bpy.ops.import_scene.gltf),(fbx,bpy.ops.import_scene.fbx)]:
        bpy.ops.wm.read_factory_settings(use_empty=True); bpy.context.scene.render.fps=fps; importer(filepath=str(path))
        rig=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
        if not REQUIRED <= set(rig.pose.bones.keys()): raise RuntimeError("出力リグの骨が不足しています")
        bound=[o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" and m.object==rig for m in o.modifiers)]
        actions=[a for a in bpy.data.actions if any(c.data_path.startswith('pose.bones[') for c in a.fcurves)]
        if len(actions)!=1 or not bound: raise RuntimeError("出力モーションを確認できません")
        rig.animation_data_create()
        for track in list(rig.animation_data.nla_tracks): rig.animation_data.nla_tracks.remove(track)
        rig.animation_data.action=actions[0]
        start,end=actions[0].frame_range
        if abs((end-start)-(len(points)-1))>.2: raise RuntimeError("出力したモーションの長さが不正です")
        for obj in bound:
            if not obj.vertex_groups or any(not v.groups for v in obj.data.vertices): raise RuntimeError("出力メッシュのウェイトがありません")
        if not any(image.size[0]>0 for image in bpy.data.images): raise RuntimeError("出力テクスチャがありません")
        samples=[]
        for t in np.linspace(start,end,9):
            whole=int(t); bpy.context.scene.frame_set(whole,subframe=float(t-whole))
            graph=bpy.context.evaluated_depsgraph_get()
            coords=np.array([tuple(o.matrix_world @ v.co) for o in bound for v in o.evaluated_get(graph).data.vertices])
            if not np.isfinite(coords).all(): raise RuntimeError("出力した動作の頂点が不正です")
            samples.append(coords)
        # Stationary prompts may legitimately have little motion; report it.
        displacement=max(float(np.max(np.linalg.norm(p-samples[0],axis=1))) for p in samples)
        verification[path.suffix[1:]]={"frames":len(points),"fps":fps,"sampled_poses":len(samples),"max_vertex_displacement":displacement,"skinned_meshes":len(bound)}
    return {"version":"humanml22-autorig-v1","source_scale":scale,"target_floor":target_floor,"root_height":"absolute-floor-relative","preserved_mesh":True,"validation":verification}

if __name__=="__main__":
    job_path=Path(sys.argv[sys.argv.index("--")+1]); job=json.loads(job_path.read_text()); out=job_path.parent
    result=retarget(Path("/data")/job["source_model_path"],out/"joints.npz",out,job)
    (out/"retarget.json").write_text(json.dumps(result,indent=2))
