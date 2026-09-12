"""End-to-end trial on a saved model, writing only to a new output folder."""
import sys,json,hashlib,time
from pathlib import Path
import bpy,bmesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limb_separation import separate_contacts

source,output=(Path(p) for p in sys.argv[sys.argv.index("--")+1:])
output.mkdir(parents=True,exist_ok=False)
digest=hashlib.sha256(source.read_bytes()).hexdigest()
if source.suffix==".blend": bpy.ops.wm.open_mainfile(filepath=str(source))
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
arm=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
meshes=[o for o in bpy.context.scene.objects if o.type=="MESH" and any(m.type=="ARMATURE" and m.object==arm for m in o.modifiers)]
if source.suffix!=".blend":
    for obj in meshes:
        bm=bmesh.new();bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
        bm.to_mesh(obj.data);bm.free()
started=time.monotonic()
report=separate_contacts(meshes,arm)
report["seconds"]=round(time.monotonic()-started,2)
(output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
bpy.ops.export_scene.gltf(filepath=str(output/"result.glb"),export_format="GLB",export_extras=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output/"result.blend"))
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
