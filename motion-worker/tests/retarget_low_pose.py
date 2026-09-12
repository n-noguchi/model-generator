"""Blender integration check using an existing test rig and synthetic crouch.

Usage: blender -b --python-exit-code 1 --python retarget_low_pose.py -- job.json
Writes only to a new temporary test directory, never to registered artifacts.
"""
import json
import math
import sys
import tempfile
from pathlib import Path
import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]))
from retarget import retarget

job_path=Path(sys.argv[sys.argv.index("--")+1])
job=json.loads(job_path.read_text())
source=Path("/data")/job["source_model_path"]
with np.load(job_path.parent/"joints.npz",allow_pickle=False) as data:
    standing=data["joints"][0].copy()

# Lower the entire upper body; keep feet planted and solve each knee on its
# two-bone sphere intersection so the original leg lengths remain unchanged.
crouch=standing.copy(); crouch[:,1]-=.35
for hip,knee,ankle,toe in [(1,4,7,10),(2,5,8,11)]:
    crouch[ankle]=standing[ankle]; crouch[toe]=standing[toe]
    upper=np.linalg.norm(standing[knee]-standing[hip])
    lower=np.linalg.norm(standing[ankle]-standing[knee])
    direction=crouch[ankle]-crouch[hip]; distance=np.linalg.norm(direction)
    assert abs(upper-lower)<distance<upper+lower
    direction/=distance
    along=(upper*upper-lower*lower+distance*distance)/(2*distance)
    bend=np.array([0.,0.,1.]); bend-=direction*np.dot(bend,direction); bend/=np.linalg.norm(bend)
    crouch[knee]=crouch[hip]+direction*along+bend*math.sqrt(max(0,upper*upper-along*along))

out=Path(tempfile.mkdtemp(prefix="text-motion-low-pose-",dir="/data"))
np.savez_compressed(out/"joints.npz",joints=np.repeat(crouch[None,:,:],40,axis=0),fps=np.array(20))
result=retarget(source,out/"joints.npz",out,job)
# retarget() has reimported the FBX and sampled its animation by this point.
rig=next(o for o in bpy.context.scene.objects if o.type=="ARMATURE")
actual=(rig.matrix_world @ rig.pose.bones["Hips"].head).z
expected=result["target_floor"]+float(crouch[0,1])*result["source_scale"]
assert abs(actual-expected)<1e-4,(actual,expected)
assert (out/"custom-motion.glb").is_file() and (out/"custom-motion.fbx").is_file()
result.update({"test":"synthetic-crouch-on-real-rig","initial_source_hip_height":float(crouch[0,1]),"expected_hip_z":expected,"actual_fbx_hip_z":actual})
(out/"test-result.json").write_text(json.dumps(result,indent=2))
print("PASS low initial pose GLB/FBX:",json.dumps(result),"output:",out)
