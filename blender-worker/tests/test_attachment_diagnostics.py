"""Run with Blender: blender -b --python tests/test_attachment_diagnostics.py."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attachment_diagnostics import analyze, isolated_pose, stretched_edges, write_diagnostics


def fixture(bridge=False, scale=1, translation=(0, 0, 0)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.armature_add(enter_editmode=True)
    arm = bpy.context.object
    arm.name = "TestRig"
    bpy.ops.armature.select_all(action="SELECT")
    bpy.ops.armature.delete()
    for name in ("Hips", "UpperArm.L", "UpperArm.R", "UpperLeg.L", "UpperLeg.R"):
        bone = arm.data.edit_bones.new(name)
        bone.head = (.1, 0, .8)
        bone.tail = (.1, 0, .3)
    bpy.ops.object.mode_set(mode="OBJECT")
    def box(x):
        return [(x, 0, 0), (x+.05, 0, 0), (x+.05, .1, 0), (x, .1, 0),
                (x, 0, 1), (x+.05, 0, 1), (x+.05, .1, 1), (x, .1, 1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    faces = quads + [tuple(i+8 for i in face) for face in quads]
    if bridge:
        faces.append((1, 8, 12, 5))
    mesh = bpy.data.meshes.new("Nearby separated surfaces")
    mesh.from_pydata(box(0)+box(.07), [], faces)
    obj = bpy.data.objects.new("Body and arm", mesh)
    bpy.context.collection.objects.link(obj)
    obj.vertex_groups.new(name="Hips").add(list(range(8)), 1, "REPLACE")
    obj.vertex_groups.new(name="UpperArm.R").add(list(range(8,16)), 1, "REPLACE")
    obj.modifiers.new("Skin", "ARMATURE").object = arm
    transform = Matrix.Translation(translation) @ Matrix.Scale(scale, 4)
    obj.matrix_world = transform
    arm.matrix_world = transform
    bpy.context.view_layer.update()
    return obj, arm


class DiagnosticsTests(unittest.TestCase):
    def test_nearby_separate_surfaces_do_not_trigger(self):
        obj, arm = fixture()
        report, _ = analyze([obj], arm)
        self.assertEqual(report["status"], "no_candidates")
        self.assertEqual(report["candidate_faces"], 0)
        self.assertEqual(report["tested_poses"], 16)

    def test_bridge_and_scale_translation(self):
        results = []
        for scale, shift in ((1, (0,0,0)), (3, (4,2,-7)), (.01, (0,0,0))):
            obj, arm = fixture(True, scale, shift)
            report, colors = analyze([obj], arm)
            self.assertEqual(report["status"], "candidates_found")
            self.assertEqual(report["candidate_faces"], 1)
            self.assertEqual(set(colors[obj.name]), {12})
            self.assertIn("cross_branch_weights", report["regions"][0]["reasons"])
            results.append(report["regions"][0]["max_stretch"])
        self.assertAlmostEqual(results[0], results[1], places=3)
        self.assertAlmostEqual(results[0], results[2], places=3)

    def test_weight_error_is_candidate_not_confirmed_fusion(self):
        obj, arm = fixture()
        obj.vertex_groups["Hips"].remove([1])
        obj.vertex_groups["UpperArm.R"].add([1], 1, "REPLACE")
        report, _ = analyze([obj], arm)
        self.assertEqual(report["status"], "candidates_found")
        self.assertNotIn("confirmed", report)
        self.assertEqual(report["image_comparison"], "not_performed")

    def test_unavailable_is_not_zero_risk(self):
        obj, arm = fixture()
        self.assertEqual(analyze([obj], None)[0]["status"], "not_evaluated")
        obj.modifiers.new("Topology changes", "SUBSURF")
        self.assertEqual(analyze([obj], arm)[0]["status"], "not_evaluated")

    def test_long_clip_discards_partial_results_and_restores_action(self):
        obj, arm = fixture(True)
        action = bpy.data.actions.new("Too long")
        curve = action.fcurves.new('pose.bones["UpperArm.R"].rotation_quaternion', index=0)
        curve.keyframe_points.insert(1, 1)
        curve.keyframe_points.insert(302, 1)
        arm.animation_data_create().action = action
        report, colors = analyze([obj], arm, include_actions=True)
        self.assertEqual(report["status"], "not_evaluated")
        self.assertTrue(report["partial_results_discarded"])
        self.assertEqual(colors, {})
        self.assertEqual(arm.animation_data.action, action)

    def test_empty_mesh_and_nonfinite_geometry(self):
        obj, arm = fixture()
        obj.data.vertices[0].co.x = float("nan")
        self.assertEqual(analyze([obj], arm)[0]["status"], "not_evaluated")
        obj, arm = fixture()
        obj.data.clear_geometry()
        self.assertEqual(analyze([obj], arm)[0]["status"], "not_evaluated")

    def test_tiny_edges_are_ignored(self):
        rest = np.array([[0,0,0], [1e-7,0,0]])
        current = np.array([[0,0,0], [.1,0,0]])
        selected, _, _ = stretched_edges(rest,current,np.array([[0,1]]),1)
        self.assertFalse(selected[0])

    def test_state_restored_even_on_error(self):
        obj, arm = fixture(True)
        action = bpy.data.actions.new("Existing")
        arm.animation_data_create().action = action
        arm.animation_data.use_nla = True
        arm.data.pose_position = "REST"
        bone = arm.pose.bones["UpperArm.R"]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler.x = .4
        bpy.context.scene.frame_set(9, subframe=.25)
        basis = bone.matrix_basis.copy()
        with self.assertRaises(ValueError):
            with isolated_pose(arm):
                raise ValueError("test")
        self.assertEqual(arm.animation_data.action, action)
        self.assertTrue(arm.animation_data.use_nla)
        self.assertEqual(arm.data.pose_position, "REST")
        self.assertEqual(bone.rotation_mode, "XYZ")
        self.assertEqual(bone.matrix_basis, basis)
        self.assertEqual(bpy.context.scene.frame_current, 9)
        self.assertEqual(bpy.context.scene.frame_subframe, .25)

    def test_export_is_separate_and_reimportable(self):
        obj, arm = fixture(True)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source.glb"
            bpy.ops.export_scene.gltf(filepath=str(source),export_format="GLB")
            before = source.read_bytes()
            original_mesh = obj.data
            original_scene = bpy.context.scene
            paths = write_diagnostics([obj],arm,source,folder,"test")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(obj.data, original_mesh)
            self.assertEqual(bpy.context.scene, original_scene)
            self.assertEqual(len(obj.data.materials), 0)
            report=json.loads(paths["test_attachment_report"].read_text())
            self.assertEqual(report["source_sha256"],hashlib.sha256(before).hexdigest())
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.gltf(filepath=str(paths["test_attachment_preview"]))
            meshes=[o for o in bpy.context.scene.objects if o.type=="MESH"]
            self.assertTrue(meshes)
            self.assertTrue(any(len(o.data.materials)>1 for o in meshes))
            self.assertFalse(any(o.type=="ARMATURE" for o in bpy.context.scene.objects))


suite=unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticsTests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    sys.exit(1)
