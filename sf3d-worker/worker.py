"""HTTP-polling generation worker. BACKEND_MODE=mock is intentionally separate from SF3D."""
import hashlib, json, os, time, struct, sys
from pathlib import Path
import requests
from PIL import Image
API=os.getenv("API_BASE_URL","http://localhost:8000"); DATA=Path("/data")
_AUTO_REMBG_SESSION = None
U2NET_HUMAN_SEG_SHA256 = "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c"
def alpha_mask_background(image, background: str, tolerance: int):
    """Replace pixels within Euclidean RGB tolerance with transparent pixels."""
    if not (len(background) == 7 and background.startswith("#")): raise ValueError("background must be #RRGGBB")
    target=tuple(int(background[i:i+2],16) for i in (1,3,5)); out=[]
    for r,g,b,a in image.convert("RGBA").getdata():
        out.append((r,g,b,0 if ((r-target[0])**2+(g-target[1])**2+(b-target[2])**2)**.5<=tolerance else a))
    result=image.convert("RGBA");result.putdata(out);return result

def _foreground_ratio(image):
    alpha=image.convert("RGBA").getchannel("A")
    return sum(value >= 128 for value in alpha.getdata()) / (image.width * image.height)

def _has_usable_alpha(image):
    """Only trust an existing alpha channel when it represents a real cutout."""
    alpha=image.convert("RGBA").getchannel("A")
    low, high=alpha.getextrema()
    return low < 128 and high >= 128 and .01 <= _foreground_ratio(image) <= .98

def _rembg_session(session_factory=None):
    global _AUTO_REMBG_SESSION
    if session_factory is None and _AUTO_REMBG_SESSION is not None:
        return _AUTO_REMBG_SESSION
    model=Path(os.getenv("U2NET_HOME","/models/rembg")) / "u2net_human_seg.onnx"
    if not model.is_file():
        raise RuntimeError("人物切り抜きモデルがありません: /models/rembg/u2net_human_seg.onnx。scripts/download-model.ps1 を実行してください（実行時ダウンロードは行いません）。")
    checksum=hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): checksum.update(chunk)
    digest=checksum.hexdigest()
    if digest != U2NET_HUMAN_SEG_SHA256:
        raise RuntimeError("人物切り抜きモデルのSHA-256が一致しません。破損した可能性があるため scripts/download-model.ps1 を再実行してください。")
    if session_factory is not None: return session_factory("u2net_human_seg")
    from rembg import new_session
    # The cutout model intentionally runs on CPU so it does not compete with
    # SF3D for VRAM or probe unavailable CUDA/TensorRT provider libraries.
    _AUTO_REMBG_SESSION=new_session("u2net_human_seg",providers=["CPUExecutionProvider"])
    return _AUTO_REMBG_SESSION

def prepare_foreground(image, payload, session=None, session_factory=None, remove_fn=None):
    """Return an RGBA person image and an operator-visible preprocessing message."""
    mode=payload.get("cutout_mode", "auto")
    image=image.convert("RGBA")
    if mode == "color":
        result=alpha_mask_background(image,payload.get("background","#FF00FF"),int(payload.get("tolerance",48)))
        message="背景除去: 指定単色を透過"
    elif mode == "auto":
        if _has_usable_alpha(image):
            result=image
            message="背景除去: 入力画像の透明度を使用"
        else:
            if session is None: session=_rembg_session(session_factory)
            if remove_fn is None:
                from rembg import remove
                remove_fn=remove
            result=remove_fn(image, session=session).convert("RGBA")
            message="背景除去: AI人物切り抜き (u2net_human_seg)"
    else:
        raise RuntimeError(f"不明な切り抜き方式: {mode}")
    ratio=_foreground_ratio(result)
    if ratio < .01: raise RuntimeError("背景除去で人物がほぼ消去されました。別の人物画像または単色背景方式を試してください。")
    if ratio > .98: message+="（警告: 人物領域が画像のほぼ全体です。切り抜き結果を確認してください）"
    return result,message
class ImageTo3DBackend:
    def generate(self, payload, out): raise NotImplementedError
class MockImageTo3DBackend(ImageTo3DBackend):
    def generate(self,payload,out,report=lambda _:None):
        out.mkdir(parents=True,exist_ok=True); model=out/"mesh.glb"
        # Small but spec-valid GLB: one triangle, so the browser/Blender contract matches SF3D.
        binary=struct.pack("<9f3H",-.5,0,0,.5,0,0,0,1,0,0,1,2); pad=(-len(binary))%4; binary+=b"\0"*pad
        doc={"asset":{"version":"2.0","generator":"modelgenerator mock"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":len(binary)}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962},{"buffer":0,"byteOffset":36,"byteLength":6,"target":34963}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-.5,0,0],"max":[.5,1,0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}]}
        raw=json.dumps(doc,separators=(",", ":")).encode();raw+=b" "*((-len(raw))%4)
        model.write_bytes(struct.pack("<4sII",b"glTF",2,12+8+len(raw)+8+len(binary))+struct.pack("<I4s",len(raw),b"JSON")+raw+struct.pack("<I4s",len(binary),b"BIN\0")+binary)
        src=DATA/payload["input_path"]; preview=out/"preview.png"; im,message=prepare_foreground(Image.open(src),payload);report(message);im.thumbnail((512,512));im.save(out/"input-alpha.png");im.save(preview)
        return model,preview,50
class StableFast3DBackend(ImageTo3DBackend):
    def generate(self,payload,out,report=lambda _:None):
        """Use SF3D's Python API after retaining or producing an RGBA person mask."""
        out.mkdir(parents=True,exist_ok=True); src=DATA/payload["input_path"]
        workdir=Path(os.getenv("SF3D_DIR","/opt/stable-fast-3d"))
        if not (workdir/"sf3d"/"system.py").is_file(): raise RuntimeError(f"SF3D code not found in image at {workdir}")
        cache=Path(os.getenv("HUGGINGFACE_HUB_CACHE", "/models/huggingface/hub"))
        if not cache.is_dir(): raise RuntimeError("Hugging Face model cache is missing; run scripts/download-model.ps1 on the host after hf auth login")
        if not sys.path or sys.path[0] != str(workdir): sys.path.insert(0,str(workdir))
        import torch
        from sf3d.system import SF3D
        from sf3d.utils import resize_foreground
        if not torch.cuda.is_available(): raise RuntimeError("SF3D mode requires CUDA; use BACKEND_MODE=mock without a GPU")
        model=mesh=None
        try:
            bake_resolution=payload.get("bake_resolution",512)
            if bake_resolution not in (512,1024,2048):
                raise RuntimeError(f"Unsupported bake resolution: {bake_resolution}")
            image,message=prepare_foreground(Image.open(src),payload)
            report(message)
            report(f"テクスチャ生成解像度: {bake_resolution}px")
            image=resize_foreground(image,.85)
            image.save(out/"input-alpha.png")
            model=SF3D.from_pretrained("stabilityai/stable-fast-3d",config_name="config.yaml",weight_name="model.safetensors")
            model.to("cuda");model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    mesh,_=model.run_image([image],bake_resolution=bake_resolution,remesh="none",vertex_count=-1)
            model_path=out/"mesh.glb";mesh.export(model_path,include_normals=True)
            preview=out/"preview.png";image.save(preview)
            return model_path,preview,0
        finally:
            del mesh, model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

class ProductionImageTo3DBackend(ImageTo3DBackend):
    """Keep legacy SF3D jobs working while opting explicit jobs into 4-view shape generation."""
    def __init__(self):
        self.sf3d=StableFast3DBackend()
        self._hunyuan=None

    def generate(self,payload,out,report=lambda _:None):
        if payload.get("generation_mode", "sf3d") != "multiview":
            return self.sf3d.generate(payload,out,report)
        if self._hunyuan is None:
            from hunyuan_backend import HunyuanMultiViewBackend
            self._hunyuan=HunyuanMultiViewBackend(DATA,prepare_foreground)
        return self._hunyuan.generate(payload,out,report)
def post(path, **kwargs):
    """Treat worker callback HTTP errors as failures, never as a completed job."""
    response = requests.post(API + path, timeout=30, **kwargs)
    response.raise_for_status()
    return response
def report_connection_error(error): print(f"API unavailable; retrying in 2 seconds: {error}",flush=True)
def main():
    backend=MockImageTo3DBackend() if os.getenv("BACKEND_MODE","mock")=="mock" else ProductionImageTo3DBackend()
    while True:
        try: job=post("/worker/jobs/claim?kind=generate").json()
        except (requests.RequestException, ValueError) as e:
            report_connection_error(e);time.sleep(2);continue
        if not job: time.sleep(2);continue
        try:
            post(f"/worker/jobs/{job['id']}/start"); post(f"/worker/jobs/{job['id']}/update",json={"progress":10,"log":"generation started"})
            out=DATA/"projects"/job["run_id"]/"runs"/job["candidate_id"]
            def report(message): post(f"/worker/jobs/{job['id']}/update",json={"progress":20,"log":message})
            model,preview,score=backend.generate(json.loads(job["payload"]),out,report)
            post(f"/worker/jobs/{job['id']}/complete",json={"model_path":str(model.relative_to(DATA)).replace('\\','/'),"preview_path":str(preview.relative_to(DATA)).replace('\\','/'),"score":score})
        except Exception as e:
            try: post(f"/worker/jobs/{job['id']}/update",json={"error":str(e)})
            except requests.RequestException as report_error: report_connection_error(report_error)
if __name__=="__main__": main()
