import bpy, json, sys, shutil
from pathlib import Path
args=sys.argv[sys.argv.index("--")+1:];data,candidate_id,kind,payload,result=Path(args[0]),args[1],args[2],json.loads(args[3]),Path(args[4])
# Locate input model by candidate directory; production SF3D GLB is the canonical input.
models=list((data/"projects").glob(f"*/runs/{candidate_id}/mesh.glb"))+list((data/"projects").glob(f"*/runs/{candidate_id}/model.obj"))
if not models: raise RuntimeError("candidate model not found")
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(models[0])) if models[0].suffix==".glb" else bpy.ops.wm.obj_import(filepath=str(models[0]))
meshes=[o for o in bpy.context.scene.objects if o.type=="MESH"]
if not meshes: raise RuntimeError("no mesh")
if kind=="optimize":
 target=payload["triangles"]; total=sum(len(o.data.polygons) for o in meshes); ratio=min(1,target/max(total,1))
 for o in meshes:
  bpy.context.view_layer.objects.active=o;o.select_set(True);bpy.ops.object.mode_set(mode="EDIT");bpy.ops.mesh.remove_doubles();bpy.ops.mesh.normals_make_consistent(inside=False);bpy.ops.object.mode_set(mode="OBJECT");m=o.modifiers.new("Decimate","DECIMATE");m.ratio=ratio;bpy.ops.object.modifier_apply(modifier=m.name)
 out=result.parent/"optimized.glb";bpy.ops.export_scene.gltf(filepath=str(out),export_format="GLB");json.dump({"model_path":str(out.relative_to(data)).replace('\\','/')},result.open("w"))
else:
 artifacts={}; formats=payload["formats"]
 if "glb" in formats:
  p=result.parent/"model.glb";bpy.ops.export_scene.gltf(filepath=str(p),export_format="GLB");artifacts["glb"]=str(p.relative_to(data)).replace('\\','/')
 if "fbx" in formats:
  p=result.parent/"model.fbx";bpy.ops.export_scene.fbx(filepath=str(p));artifacts["fbx"]=str(p.relative_to(data)).replace('\\','/')
 # re-import validation in clean scene
 for p in artifacts.values():
  bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=str(data/p)) if p.endswith('.glb') else bpy.ops.import_scene.fbx(filepath=str(data/p));
  if not any(o.type=="MESH" for o in bpy.context.scene.objects): raise RuntimeError("export validation: mesh missing")
 zip_path=shutil.make_archive(str(result.parent/"export"),"zip",root_dir=result.parent);artifacts["zip"]=str(Path(zip_path).relative_to(data)).replace('\\','/')
 json.dump({"artifacts":artifacts,"metadata":{"validated":True,"preset":payload["preset"]}},result.open("w"))
