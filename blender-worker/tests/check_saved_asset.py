"""Integration check of an existing GLB; writes only a new diagnostic folder."""
import hashlib
import json
import sys
import time
from pathlib import Path
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attachment_diagnostics import write_diagnostics

source, output = (Path(p) for p in sys.argv[sys.argv.index("--") + 1:])
output.mkdir(parents=True, exist_ok=False)
before = hashlib.sha256(source.read_bytes()).hexdigest()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(source))
arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
started = time.monotonic()
paths = write_diagnostics(meshes, arm, source, output, "check", include_actions=True)
elapsed = time.monotonic() - started
report = json.loads(paths["check_attachment_report"].read_text())
assert before == hashlib.sha256(source.read_bytes()).hexdigest()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(paths["check_attachment_preview"]))
assert any(o.type == "MESH" for o in bpy.context.scene.objects)
assert not any(o.type == "ARMATURE" for o in bpy.context.scene.objects)
print(json.dumps({"status": report["status"], "faces": report["candidate_faces"],
                  "regions": len(report["regions"]), "poses": report["tested_poses"],
                  "seconds": round(elapsed, 2), "source_unchanged": True,
                  "worst": [{k:v for k,v in r.items() if k != "face_ids"} for r in report["regions"][:2]]}))
