"""Local text-to-motion queue worker; a fresh process releases GPU per job."""
import hashlib
import json
import os
import sys
from pathlib import Path
from gpu_runtime import api, run_process, serve

API=os.getenv("API_BASE_URL","http://api:8000"); DATA=Path("/data")
MODELS=Path(os.getenv("MOTION_MODEL_DIR","/models/mdm"))

def heartbeat():
    ready=(MODELS/"ready.json").is_file()
    value={"ready":ready,"model":"MDM 50-step (local)","message":"ローカルモデルの準備完了" if ready else "初回のモデルダウンロードが必要です。"}
    temporary=DATA/"text-motion-status.tmp"; temporary.write_text(json.dumps(value,ensure_ascii=False)); temporary.replace(DATA/"text-motion-status.json")
    return ready

def execute(job):
    out=DATA/"projects"/job["run_id"]/"text-motions"/job["candidate_id"]/job["id"]
    out.mkdir(parents=True,exist_ok=True); payload=json.loads(job["payload"])
    source=(DATA/payload["source_model_path"]).resolve()
    if not source.is_relative_to(DATA) or hashlib.sha256(source.read_bytes()).hexdigest()!=payload["source_sha256"]: raise RuntimeError("入力リグのSHA256が一致しません")
    job_path=out/"job.json"; job_path.write_text(json.dumps(payload,ensure_ascii=False))
    api(f"/worker/jobs/{job['id']}/start",{})
    api(f"/worker/jobs/{job['id']}/update",{"progress":10,"log":"MDMで文章から全身モーションを生成中（ローカルGPU）"})
    run_process([sys.executable,"/worker/infer.py",str(job_path),str(out/"joints.npz")],out,job,"inference.log",heartbeat)
    api(f"/worker/jobs/{job['id']}/update",{"progress":65,"log":"既存キャラクターへ動作を適用し、GLB・FBXを検証中"})
    run_process(["blender","-b","--python-exit-code","1","--python","/worker/retarget.py","--",str(job_path)],out,job,"retarget.log",heartbeat)
    if not (out/"retarget.json").exists(): raise RuntimeError("Blenderの出力検証結果がありません")
    manifest={**payload,**json.loads((out/"joints.json").read_text()),"retarget":json.loads((out/"retarget.json").read_text()),
              "limitations":["学習データにない動作や複雑な指示は再現できない場合があります", "手指・物体との接触・足接地IKは未対応", "生成物の商用利用条件はモデルと学習データのライセンス確認が必要です"]}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    artifacts={key:str((out/name).relative_to(DATA)) for key,name in {"text_motion_glb":"custom-motion.glb","text_motion_fbx":"custom-motion.fbx","text_motion_manifest":"manifest.json","text_motion_joints":"joints.npz"}.items()}
    result={"artifacts":artifacts,"metadata":{"pipeline":payload["pipeline"],"validated":True,"prompt":payload["prompt"],"seed":payload["seed"]}}
    return result

def main():
    serve("text_motion",execute,heartbeat)

if __name__=="__main__": main()
