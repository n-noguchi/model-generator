import sys
import types
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import hunyuan_backend
from hunyuan_backend import HunyuanMultiViewBackend, VALID_OCTREE_RESOLUTIONS, view_paths


def test_multiview_requires_all_named_views():
    with pytest.raises(RuntimeError, match="4画像"):
        view_paths({"input_path": "uploads/front.png", "back_path": "uploads/back.png"})


def test_multiview_preserves_named_view_mapping():
    payload = {
        "input_path": "front.png", "back_path": "back.png",
        "left_path": "left.png", "right_path": "right.png",
    }
    assert view_paths(payload) == {
        "front": "front.png", "back": "back.png", "left": "left.png", "right": "right.png"
    }


def test_quality_resolution_choices_are_bounded_for_8gb_worker():
    assert VALID_OCTREE_RESOLUTIONS == (256, 320, 384)


def test_explicit_cpu_offload_uses_hunyuan_module_order_and_cleans_once(monkeypatch):
    calls, cleanup = [], []

    class Hook:
        def __init__(self, name): self.name = name
        def offload(self): cleanup.append(("offload", self.name))
        def remove(self): cleanup.append(("remove", self.name))

    def cpu_offload_with_hook(module, device, prev_module_hook=None):
        calls.append((module.name, device, getattr(prev_module_hook, "name", None)))
        return module, Hook(module.name)

    monkeypatch.setitem(sys.modules, "accelerate", types.SimpleNamespace(cpu_offload_with_hook=cpu_offload_with_hook))
    pipeline = types.SimpleNamespace(
        conditioner=types.SimpleNamespace(name="conditioner"),
        model=types.SimpleNamespace(name="model"),
        vae=types.SimpleNamespace(name="vae"),
    )
    hooks = hunyuan_backend._enable_pipeline_cpu_offload(pipeline)
    hunyuan_backend._release_pipeline_cpu_offload(hooks)

    assert calls == [
        ("conditioner", "cuda", None),
        ("model", "cuda", "conditioner"),
        ("vae", "cuda", "model"),
    ]
    assert cleanup == [
        ("offload", "conditioner"), ("remove", "conditioner"),
        ("offload", "model"), ("remove", "model"),
        ("offload", "vae"), ("remove", "vae"),
    ]


class _FakeProcess:
    def __init__(self, lines, exit_code=0):
        self.stdout = iter(lines)
        self.exit_code = exit_code
        self.terminated = False

    def poll(self):
        return None if not self.terminated else -15

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return -15 if self.terminated else self.exit_code


def _payload():
    return {"input_path": "front.png", "back_path": "back.png", "left_path": "left.png", "right_path": "right.png",
            "octree_resolution": 320, "bake_resolution": 2048}


def _make_backend(tmp_path, monkeypatch, process):
    for name in ("front", "back", "left", "right"):
        Image.new("RGBA", (4, 4), (20, 30, 40, 255)).save(tmp_path / f"{name}.png")
    runtime = tmp_path / "python"; runtime.touch()
    monkeypatch.setenv("HUNYUAN_PYTHON", str(runtime))
    monkeypatch.setattr(hunyuan_backend, "_assert_checkpoint", lambda: None)
    monkeypatch.setattr(hunyuan_backend.subprocess, "Popen", lambda *args, **kwargs: process)
    return HunyuanMultiViewBackend(tmp_path, lambda image, payload: (image.convert("RGBA"), "ready"))


def test_subprocess_logs_progress_and_returns_mesh(tmp_path, monkeypatch):
    process = _FakeProcess(["ordinary detail\n", "HUNYUAN_PROGRESS: shape complete\n"])
    backend = _make_backend(tmp_path, monkeypatch, process)
    out = tmp_path / "out"; out.mkdir()
    (out / "mesh.glb").write_bytes(b"glTF")
    updates = []
    model, preview, score = backend.generate(_payload(), out, updates.append)
    assert (model, preview, score) == (out / "mesh.glb", out / "preview.png", 0)
    assert updates[-1] == "shape complete"
    assert (out / "hunyuan.log").read_text(encoding="utf-8") == "ordinary detail\nHUNYUAN_PROGRESS: shape complete\n"
    assert not process.terminated


def test_report_failure_terminates_running_subprocess(tmp_path, monkeypatch):
    process = _FakeProcess(["HUNYUAN_PROGRESS: loading\n"])
    backend = _make_backend(tmp_path, monkeypatch, process)
    def fail_only_on_child(message):
        if message == "loading":
            raise RuntimeError("status endpoint unavailable")
    with pytest.raises(RuntimeError, match="status endpoint unavailable"):
        backend.generate(_payload(), tmp_path / "out", fail_only_on_child)
    assert process.terminated
