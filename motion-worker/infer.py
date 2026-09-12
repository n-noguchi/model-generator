"""One isolated GPU process; English text -> HumanML3D joints, no network."""
import json
import os
import sys
import time
import fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import clip
from model.mdm import MDM
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from prepare_models import digest, EXPECTED, REVISION

class LocalMDM(MDM):
    def load_and_freeze_clip(self, clip_version):
        model, _ = clip.load(str(Path(os.environ["MOTION_MODEL_DIR"]) / "clip/ViT-B-32.pt"), device="cpu", jit=False)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad = False
        return model

def generate(payload, output):
    root = Path(os.environ["MOTION_MODEL_DIR"])
    ready = json.loads((root / "ready.json").read_text())
    if ready["files"] != EXPECTED or ready["revision"] != REVISION: raise RuntimeError("モデルの準備情報が一致しません。再準備してください。")
    for name, expected in EXPECTED.items():
        if digest(root / name) != expected: raise RuntimeError(f"モデルファイルの検証に失敗: {name}")
    # MDM was trained on 20 CLIP tokens. Never silently discard user text.
    try: clip.tokenize([payload["prompt"]], context_length=22, truncate=False)
    except RuntimeError as exc: raise ValueError("指示が長すぎます。動作をより短い英語1文で指定してください。") from exc
    if not torch.cuda.is_available(): raise RuntimeError("Text-to-Motion用のCUDA GPUが見つかりません")
    torch.set_num_threads(4)
    torch.manual_seed(payload["seed"]); np.random.seed(payload["seed"])
    settings = json.loads((root / "args.json").read_text())
    if settings.get("diffusion_steps") != 50 or settings.get("arch") != "trans_enc":
        raise RuntimeError("MDM 50-step encoderのチェックポイントではありません")
    started = time.monotonic()
    model = LocalMDM(modeltype="", njoints=263, nfeats=1, num_actions=1, translation=True,
                     pose_rep="rot6d", glob=True, glob_rot=True, latent_dim=settings["latent_dim"],
                     num_layers=settings["layers"], data_rep="hml_vec", dataset="humanml", cond_mode="text",
                     cond_mask_prob=.1, arch="trans_enc", clip_version="ViT-B/32")
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    load_model_wo_clip(model, state)
    model.to("cuda"); model.eval()
    diffusion = create_gaussian_diffusion(SimpleNamespace(**{**dict(lambda_vel=0, lambda_rcxyz=0, lambda_fc=0, sigma_small=True), **settings}))
    frames = round(payload["seconds"] * 20)
    with torch.inference_mode():
        condition = {"text": [payload["prompt"]], "lengths": torch.tensor([frames], device="cuda"),
                     "mask": torch.ones((1, 1, 1, frames), device="cuda", dtype=torch.bool),
                     "scale": torch.tensor([2.5], device="cuda")}
        condition["text_embed"] = model.encode_text(condition["text"])
        sample = diffusion.p_sample_loop(ClassifierFreeSampleModel(model), (1, 263, 1, frames),
                                         clip_denoised=False, model_kwargs={"y": condition}, progress=True)
        data = sample.cpu().permute(0, 2, 3, 1).float()
        mean = torch.from_numpy(np.load(root / "Mean.npy", allow_pickle=False)).float()
        std = torch.from_numpy(np.load(root / "Std.npy", allow_pickle=False)).float()
        joints = recover_from_ric(data * std + mean, 22).reshape(frames, 22, 3).numpy()
    if not np.isfinite(joints).all(): raise RuntimeError("モデルが不正な関節座標を出力しました")
    np.savez_compressed(output, joints=joints, fps=np.array(20))
    report = {"model": ready["model"], "revision": ready["revision"], "checkpoint_sha256": ready["files"]["model.pt"],
              "device": torch.cuda.get_device_name(), "peak_vram_mb": round(torch.cuda.max_memory_allocated()/1024**2),
              "inference_seconds": round(time.monotonic()-started, 2), "fps": 20, "frames": frames,
              "prompt": payload["prompt"], "seed": payload["seed"], "requested_seconds": payload["seconds"],
              "coordinate_system": "HumanML3D Y-up, 22 joints"}
    Path(output).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    with (Path(os.environ["MOTION_MODEL_DIR"]) / "prepare.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        generate(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
