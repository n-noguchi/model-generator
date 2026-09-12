"""Run with Blender --python: synthetic low initial poses retain root height."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np
from retarget import root_translation, body_frame
from mathutils import Vector

# Synthetic Y-up HumanML frames transformed to the retargeter's Z-up space.
for hip_height in (.35,.6,1.0):
    joints=np.zeros((40,22,3)); joints[:,0]=(3,-2,hip_height)
    first=joints[0,0]; target_floor=-.1; rest_hip=1.1; scale=1.2
    offset=root_translation(joints[0,0],first,scale,target_floor,rest_hip)
    assert abs(rest_hip+offset.z-(target_floor+hip_height*scale))<1e-6
    assert offset.x==0 and offset.y==0
    moved=joints[-1,0]+(1,2,.2)
    delta=root_translation(moved,first,scale,target_floor,rest_hip)
    assert np.allclose(tuple(delta-offset),(1.2,2.4,.24))
assert body_frame(Vector((0,0,1)),Vector((1,0,0))).angle<1e-6
print("PASS: low/standing initial root height, XY origin, translation scale, body axes")
