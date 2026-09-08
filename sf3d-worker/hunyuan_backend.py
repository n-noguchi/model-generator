"""Offline Hunyuan3D-2mv generation, isolated from SF3D Python packages."""
import argparse
from collections import deque
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

VIEW_PATH_FIELDS = {"front": "input_path", "back": "back_path", "left": "left_path", "right": "right_path"}
VALID_OCTREE_RESOLUTIONS = (256, 320, 384)


def view_paths(payload):
    missing = [name for name, field in VIEW_PATH_FIELDS.items() if not payload.get(field)]
    if missing:
        raise RuntimeError("高品質マルチビュー生成には前・後・本人の左・本人の右の4画像が必要です: " + "、".join(missing))
    return {name: payload[field] for name, field in VIEW_PATH_FIELDS.items()}


def _model_directory(): return Path(os.getenv("HUNYUAN_MODEL_DIR", "/models/hunyuan3d-2mv"))


def _assert_checkpoint():
    directory = _model_directory() / "hunyuan3d-dit-v2-mv"
    if not all(path.is_file() for path in (directory / "config.yaml", directory / "model.fp16.safetensors")):
        raise RuntimeError("Hunyuan3D-2mv のローカル重みがありません。scripts/download-hunyuan.ps1 を実行してください。")


def _validate_quality(payload):
    resolution = int(payload.get("octree_resolution", 320))
    if resolution not in VALID_OCTREE_RESOLUTIONS:
        raise RuntimeError("octree_resolution は " + "/".join(map(str, VALID_OCTREE_RESOLUTIONS)) + " のいずれかです")
    texture = int(payload.get("bake_resolution", 2048))
    if texture not in (1024, 2048, 4096): raise RuntimeError("高品質テクスチャ解像度は 1024/2048/4096 のいずれかです")
    return resolution, texture


def _enable_pipeline_cpu_offload(pipeline, device="cuda"):
    """Install the three Hunyuan module hooks without using its broken components property.

    The upstream pipeline copies Diffusers' helper but does not define ``components``.
    Keeping this list local also lets cleanup remove hooks without attempting to install
    them a second time through the upstream helper.
    """
    try:
        from accelerate import cpu_offload_with_hook
    except ImportError as error:
        raise RuntimeError("Hunyuan3D-2mv には accelerate が必要です") from error

    previous_hook = None
    hooks = []
    for module in (pipeline.conditioner, pipeline.model, pipeline.vae):
        _, hook = cpu_offload_with_hook(module, device, prev_module_hook=previous_hook)
        hooks.append(hook)
        previous_hook = hook
    return hooks


def _release_pipeline_cpu_offload(hooks):
    """Return hooked modules to CPU and detach hooks exactly once."""
    for hook in hooks:
        hook.offload()
        hook.remove()


class HunyuanMultiViewBackend:
    """Cut out images in the SF3D process, then invoke the isolated runtime."""
    def __init__(self, data_dir, foreground_preparer):
        self.data_dir, self.prepare_foreground = Path(data_dir), foreground_preparer

    @staticmethod
    def _stop_process(process):
        """Do not leave an expensive CUDA child alive when reporting fails."""
        try:
            running = process.poll() is None
        except AttributeError:
            running = True
        if not running:
            return
        try:
            process.terminate()
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def generate(self, payload, out, report=lambda _: None):
        out = Path(out); out.mkdir(parents=True, exist_ok=True); _validate_quality(payload)
        prepared = {}
        for name, relative in view_paths(payload).items():
            source = self.data_dir / relative
            if not source.is_file(): raise RuntimeError(f"入力画像が見つかりません: {relative}")
            image, message = self.prepare_foreground(Image.open(source), payload)
            path = out / f"input-{name}-alpha.png"; image.save(path); prepared[name] = str(path)
            report(f"{name}: {message}")
        Image.open(prepared["front"]).save(out / "preview.png")
        _assert_checkpoint()
        runtime = Path(os.getenv("HUNYUAN_PYTHON", "/opt/hunyuan-venv/bin/python"))
        if not runtime.is_file(): raise RuntimeError("Hunyuan3D-2 実行環境がありません。sf3d-workerイメージを再ビルドしてください。")
        request = out / "hunyuan-request.json"
        request.write_text(json.dumps({"payload": payload, "views": prepared}), encoding="utf-8")
        process = subprocess.Popen([str(runtime), str(Path(__file__).resolve()), "--request", str(request), "--out", str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        recent = deque(maxlen=20)
        log_path = out / "hunyuan.log"
        try:
            assert process.stdout is not None
            with log_path.open("w", encoding="utf-8") as log:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    log.write(raw_line if raw_line.endswith(("\n", "\r")) else raw_line + "\n")
                    log.flush()
                    recent.append(line)
                    if line.startswith("HUNYUAN_PROGRESS:"):
                        report(line.removeprefix("HUNYUAN_PROGRESS:").strip())
            exit_code = process.wait()
        except BaseException:
            self._stop_process(process)
            raise
        if exit_code != 0:
            raise RuntimeError("Hunyuan3D-2mv subprocess failed (full log: hunyuan.log):\n" + "\n".join(recent))
        model_path = out / "mesh.glb"
        if not model_path.is_file(): raise RuntimeError("Hunyuan3D-2mv subprocess did not create mesh.glb")
        return model_path, out / "preview.png", 0


def generate_prepared_views(view_paths_map, payload, out, report=lambda _: None):
    """Run only in /opt/hunyuan-venv; it may safely upgrade Transformers."""
    out = Path(out); octree_resolution, texture_resolution = _validate_quality(payload); _assert_checkpoint()
    source = Path(os.getenv("HUNYUAN3D_DIR", "/opt/hunyuan3d-2"))
    if not (source / "hy3dgen" / "shapegen" / "pipelines.py").is_file(): raise RuntimeError(f"Hunyuan3D-2 code not found at {source}")
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    os.environ.setdefault("HY3DGEN_MODELS", str(_model_directory().parent))
    model_identifier = os.getenv("HUNYUAN_MODEL_ID", _model_directory().name)
    torch = None
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    if not torch.cuda.is_available(): raise RuntimeError("高品質マルチビュー生成にはCUDA GPUが必要です")
    views = {name: Image.open(path).convert("RGBA") for name, path in view_paths_map.items()}
    pipeline = mesh = None
    offload_hooks = []
    try:
        report("Hunyuan3D-2mv: 形状モデルを読み込み中")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_identifier, subfolder="hunyuan3d-dit-v2-mv", variant="fp16", use_safetensors=True, device="cpu")
        offload_hooks = _enable_pipeline_cpu_offload(pipeline, device="cuda")
        # Hooks execute modules on CUDA, so sampling tensors must use CUDA too.
        pipeline.device = torch.device("cuda")
        report(f"Hunyuan3D-2mv: 50ステップ、形状解像度 {octree_resolution}")
        def on_step(step, _timestep, _outputs):
            if step and step % 10 == 0: report(f"Hunyuan3D-2mv: 形状生成 {min(step * 2, 98)}%")
        mesh = pipeline(image=views, num_inference_steps=50, octree_resolution=octree_resolution, num_chunks=20000, generator=torch.manual_seed(int(payload.get("seed", 12345))), output_type="trimesh", enable_pbar=False, callback=on_step, callback_steps=5)[0]
        mesh.export(out / "mesh-shape.glb", include_normals=True)
        report(f"Hunyuan3D-2mv: 形状生成完了（{len(mesh.vertices):,} 頂点 / {len(mesh.faces):,} 面）")
        _release_pipeline_cpu_offload(offload_hooks); offload_hooks = []
        del pipeline; pipeline = None; gc.collect(); torch.cuda.empty_cache()
        report(f"元画像4方向を投影して {texture_resolution}px テクスチャを作成")
        from projection import project_texture
        model_path = out / "mesh.glb"
        _, metrics = project_texture(mesh, views, model_path, resolution=texture_resolution, front_axis="+z")
        if not model_path.is_file(): raise RuntimeError("元画像テクスチャ投影がGLBを出力しませんでした")
        coverage = float(metrics["coverage"])
        iou = metrics.get("silhouette_iou", {})
        iou_summary = ", ".join(f"{name} {float(value):.0%}" for name, value in iou.items())
        report(f"元画像投影完了: テクスチャ被覆 {coverage:.1%}; シルエット一致 {iou_summary}")
        if coverage < 0.95:
            report(f"警告: テクスチャ被覆が {coverage:.1%} です（95%未満）。未投影領域は近傍色で補完されています。")
        return model_path
    finally:
        if offload_hooks:
            try: _release_pipeline_cpu_offload(offload_hooks)
            except Exception: pass
        del mesh, pipeline
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()


def _cli():
    parser = argparse.ArgumentParser(); parser.add_argument("--request", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    generate_prepared_views(request["views"], request["payload"], args.out, lambda message: print(f"HUNYUAN_PROGRESS:{message}", flush=True))


if __name__ == "__main__": _cli()
