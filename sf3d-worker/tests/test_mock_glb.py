import sys
from pathlib import Path
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import worker

def test_mock_glb_is_loadable_by_trimesh(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "DATA", tmp_path)
    # This is a GLB contract test, not a model-dependent cutout integration test.
    image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    image.paste((100, 120, 140, 255), (32, 16, 96, 112))
    image.save(tmp_path / "input.png")
    model, _, _ = worker.MockImageTo3DBackend().generate({"input_path": "input.png"}, tmp_path / "out")
    scene = trimesh.load(model, force="scene")
    assert scene.geometry and sum(len(mesh.vertices) for mesh in scene.geometry.values()) == 3
