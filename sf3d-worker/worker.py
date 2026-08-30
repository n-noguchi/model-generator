"""HTTP-polling generation worker. BACKEND_MODE=mock is intentionally separate from SF3D."""
import json, os, time, struct, sys
from pathlib import Path
import requests
from PIL import Image
API=os.getenv("API_BASE_URL","http://localhost:8000"); DATA=Path("/data")
def alpha_mask_background(image, background: str, tolerance: int):
    """Replace pixels within Euclidean RGB tolerance with transparent pixels."""
    if not (len(background) == 7 and background.startswith("#")): raise ValueError("background must be #RRGGBB")
    target=tuple(int(background[i:i+2],16) for i in (1,3,5)); out=[]
    for r,g,b,a in image.convert("RGBA").getdata():
        out.append((r,g,b,0 if ((r-target[0])**2+(g-target[1])**2+(b-target[2])**2)**.5<=tolerance else a))
    result=image.convert("RGBA");result.putdata(out);return result
class ImageTo3DBackend:
    def generate(self, payload, out): raise NotImplementedError
class MockImageTo3DBackend(ImageTo3DBackend):
    def generate(self,payload,out):
        out.mkdir(parents=True,exist_ok=True); model=out/"mesh.glb"
        # Small but spec-valid GLB: one triangle, so the browser/Blender contract matches SF3D.
        binary=struct.pack("<9f3H",-.5,0,0,.5,0,0,0,1,0,0,1,2); pad=(-len(binary))%4; binary+=b"\0"*pad
        doc={"asset":{"version":"2.0","generator":"modelgenerator mock"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":len(binary)}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962},{"buffer":0,"byteOffset":36,"byteLength":6,"target":34963}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-.5,0,0],"max":[.5,1,0]},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}]}
        raw=json.dumps(doc,separators=(",", ":")).encode();raw+=b" "*((-len(raw))%4)
        model.write_bytes(struct.pack("<4sII",b"glTF",2,12+8+len(raw)+8+len(binary))+struct.pack("<I4s",len(raw),b"JSON")+raw+struct.pack("<I4s",len(binary),b"BIN\0")+binary)
        src=DATA/payload["input_path"]; preview=out/"preview.png"; im=Image.open(src).convert("RGB");im.thumbnail((512,512));im.save(preview)
        return model,preview,50
class StableFast3DBackend(ImageTo3DBackend):
    def generate(self,payload,out):
        """Use SF3D's Python API so the user-provided alpha mask replaces rembg."""
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
            image=alpha_mask_background(Image.open(src),payload["background"],int(payload["tolerance"]))
            if image.getextrema()[3][1] == 0: raise RuntimeError("background mask removed the entire image")
            image=resize_foreground(image,.85)
            image.save(out/"input-alpha.png")
            model=SF3D.from_pretrained("stabilityai/stable-fast-3d",config_name="config.yaml",weight_name="model.safetensors")
            model.to("cuda");model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    mesh,_=model.run_image([image],bake_resolution=512,remesh="none",vertex_count=-1)
            model_path=out/"mesh.glb";mesh.export(model_path,include_normals=True)
            preview=out/"preview.png";image.convert("RGB").save(preview)
            return model_path,preview,0
        finally:
            del mesh, model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
def post(path,**kwargs): return requests.post(API+path,timeout=30,**kwargs)
def report_connection_error(error): print(f"API unavailable; retrying in 2 seconds: {error}",flush=True)
def main():
    backend=MockImageTo3DBackend() if os.getenv("BACKEND_MODE","mock")=="mock" else StableFast3DBackend()
    while True:
        try: job=post("/worker/jobs/claim?kind=generate").json()
        except (requests.RequestException, ValueError) as e:
            report_connection_error(e);time.sleep(2);continue
        if not job: time.sleep(2);continue
        try:
            post(f"/worker/jobs/{job['id']}/start"); post(f"/worker/jobs/{job['id']}/update",json={"progress":10,"log":"generation started"})
            out=DATA/"projects"/job["run_id"]/"runs"/job["candidate_id"]; model,preview,score=backend.generate(json.loads(job["payload"]),out)
            post(f"/worker/jobs/{job['id']}/complete",json={"model_path":str(model.relative_to(DATA)).replace('\\','/'),"preview_path":str(preview.relative_to(DATA)).replace('\\','/'),"score":score})
        except Exception as e:
            try: post(f"/worker/jobs/{job['id']}/update",json={"error":str(e)})
            except requests.RequestException as report_error: report_connection_error(report_error)
if __name__=="__main__": main()
