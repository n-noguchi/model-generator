from __future__ import annotations
import json, os, shutil, uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Optional
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from PIL import Image, ImageChops

DATA = Path(os.getenv("DATA_DIR", "data")).resolve(); DATA.mkdir(parents=True, exist_ok=True)
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA/'modelgenerator.db'}")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
Sessions = sessionmaker(engine, expire_on_commit=False)
def now(): return datetime.now(timezone.utc)
def rel(path: Path) -> str:
    try: return path.resolve().relative_to(DATA).as_posix()
    except ValueError: raise HTTPException(400, "data directory outside of storage is not allowed")
def abs_path(value: str) -> Path:
    p = (DATA / value).resolve()
    if DATA not in p.parents and p != DATA: raise HTTPException(400, "invalid relative path")
    return p

class Base(DeclarativeBase): pass
class Project(Base):
    __tablename__="projects"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); name: Mapped[str]=mapped_column(String); created_at: Mapped[datetime]=mapped_column(DateTime,default=now)
class Run(Base):
    __tablename__="runs"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); project_id: Mapped[str]=mapped_column(ForeignKey("projects.id")); input_path: Mapped[str]=mapped_column(String); original_path: Mapped[Optional[str]]=mapped_column(String,nullable=True); background: Mapped[str]=mapped_column(String,default="#FF00FF"); tolerance: Mapped[int]=mapped_column(Integer,default=48); status: Mapped[str]=mapped_column(String,default="queued"); selected_candidate_id: Mapped[Optional[str]]=mapped_column(String,nullable=True); warning: Mapped[Optional[str]]=mapped_column(Text,nullable=True); created_at: Mapped[datetime]=mapped_column(DateTime,default=now)
class Candidate(Base):
    __tablename__="candidates"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); run_id: Mapped[str]=mapped_column(ForeignKey("runs.id")); number: Mapped[int]=mapped_column(Integer); seed: Mapped[int]=mapped_column(Integer); model_path: Mapped[Optional[str]]=mapped_column(String,nullable=True); preview_path: Mapped[Optional[str]]=mapped_column(String,nullable=True); status: Mapped[str]=mapped_column(String,default="waiting"); score: Mapped[Optional[int]]=mapped_column(Integer,nullable=True)
class Job(Base):
    __tablename__="jobs"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); kind: Mapped[str]=mapped_column(String); run_id: Mapped[Optional[str]]=mapped_column(String,nullable=True); candidate_id: Mapped[Optional[str]]=mapped_column(String,nullable=True); payload: Mapped[str]=mapped_column(Text,default="{}"); status: Mapped[str]=mapped_column(String,default="queued"); progress: Mapped[int]=mapped_column(Integer,default=0); log: Mapped[str]=mapped_column(Text,default=""); created_at: Mapped[datetime]=mapped_column(DateTime,default=now); claimed_at: Mapped[Optional[datetime]]=mapped_column(DateTime,nullable=True)
class Evaluation(Base):
    __tablename__="evaluations"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); candidate_id: Mapped[str]=mapped_column(ForeignKey("candidates.id")); geometry: Mapped[int]=mapped_column(Integer); texture: Mapped[int]=mapped_column(Integer); overall: Mapped[int]=mapped_column(Integer); verdict: Mapped[str]=mapped_column(String); comment: Mapped[str]=mapped_column(Text,default="")
class Artifact(Base):
    __tablename__="artifacts"; id: Mapped[str]=mapped_column(String,primary_key=True,default=lambda:str(uuid.uuid4())); candidate_id: Mapped[str]=mapped_column(ForeignKey("candidates.id")); kind: Mapped[str]=mapped_column(String); path: Mapped[str]=mapped_column(String); metadata_json: Mapped[str]=mapped_column("metadata",Text,default="{}")
class Setting(Base):
    __tablename__="settings"; key: Mapped[str]=mapped_column(String,primary_key=True); value: Mapped[str]=mapped_column(Text)
Base.metadata.create_all(engine)

class CreateProject(BaseModel): name:str=Field(min_length=1,max_length=80)
class CreateRun(BaseModel): project_id:str; input_path:str; original_path:Optional[str]=None; background:str="#FF00FF"; tolerance:int=Field(48,ge=0,le=441); confirm_warning:bool=False
class EvaluationIn(BaseModel): geometry:int=Field(ge=1,le=5); texture:int=Field(ge=1,le=5); overall:int=Field(ge=1,le=5); verdict:str; comment:str=""
class OptimizeIn(BaseModel): triangles:int=Field(20000,ge=500,le=200000); texture_size:Literal[1024,2048,4096]=2048; lod:bool=False; collision:bool=False; height_m:Optional[float]=Field(None,gt=0)
class WorkerUpdate(BaseModel): progress:Optional[int]=Field(None,ge=0,le=100); log:Optional[str]=None; error:Optional[str]=None
class JobOut(BaseModel): model_config=ConfigDict(from_attributes=True); id:str; kind:str; run_id:Optional[str]; candidate_id:Optional[str]; payload:str; status:str; progress:int; log:str

app=FastAPI(title="Person Image-to-3D MVP")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost","http://127.0.0.1"], allow_methods=["*"], allow_headers=["*"])
def db():
    with Sessions() as s: yield s
def serialize(o):
    return {a.key:getattr(o,a.key) for a in o.__mapper__.column_attrs}
def get_or_404(s, cls, id):
    o=s.get(cls,id)
    if not o: raise HTTPException(404,"not found")
    return o
@app.get("/health")
def health(): return {"mode":os.getenv("BACKEND_MODE","mock"),"data_dir":str(DATA),"gpu":"checked by sf3d worker"}
@app.post("/uploads")
async def upload(image:UploadFile=File(...), original:bool=False):
    if image.content_type not in {"image/png","image/jpeg"}: raise HTTPException(415,"PNG/JPEG only")
    suffix=".png" if image.content_type=="image/png" else ".jpg"; dest=DATA/"uploads"/f"{uuid.uuid4()}{suffix}"; dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("wb") as f: shutil.copyfileobj(image.file,f)
    try:
        with Image.open(dest) as im:
            if im.width<128 or im.height<128: raise HTTPException(422,"image must be at least 128px")
    except HTTPException: dest.unlink(missing_ok=True); raise
    return {"path":rel(dest)}
@app.get("/projects")
def projects(s:Session=Depends(db)): return [serialize(x) for x in s.scalars(select(Project).order_by(Project.created_at.desc()))]
@app.post("/projects")
def create_project(body:CreateProject,s:Session=Depends(db)):
    p=Project(name=body.name);s.add(p);s.commit();(DATA/"projects"/p.id).mkdir(parents=True,exist_ok=True);return serialize(p)
def image_warning(input_path:str, original_path:Optional[str], bg:str, tolerance:int):
    im=Image.open(abs_path(input_path)).convert("RGBA"); pix=list(im.getdata()); target=tuple(int(bg[i:i+2],16) for i in (1,3,5)); bg_count=sum(sum((p[i]-target[i])**2 for i in range(3))**.5<=tolerance for p in pix)
    warnings=[]
    if bg_count/len(pix)<.03: warnings.append("背景色が十分に検出できません")
    if original_path:
        a=Image.open(abs_path(input_path)).convert("RGB").resize((64,64)); b=Image.open(abs_path(original_path)).convert("RGB").resize((64,64)); diff=sum(sum(v) for v in ImageChops.difference(a,b).getdata())/(64*64*3)
        if diff>70: warnings.append("元画像との簡易差分が大きいため、人物の変化を確認してください")
    return " / ".join(warnings) or None
@app.post("/runs")
def create_run(body:CreateRun,s:Session=Depends(db)):
    get_or_404(s,Project,body.project_id); abs_path(body.input_path)
    warning=image_warning(body.input_path,body.original_path,body.background,body.tolerance)
    if warning and not body.confirm_warning: return {"warning":warning,"requires_confirmation":True}
    r=Run(**body.model_dump(exclude={"confirm_warning"}),warning=warning);s.add(r);s.flush()
    for n in range(1,4):
        c=Candidate(run_id=r.id,number=n,seed=1000+n);s.add(c);s.flush();s.add(Job(kind="generate",run_id=r.id,candidate_id=c.id,payload=json.dumps({"input_path":r.input_path,"background":r.background,"tolerance":r.tolerance,"seed":c.seed})))
    s.commit();return serialize(r)
@app.get("/runs/{run_id}")
def run(run_id:str,s:Session=Depends(db)):
    r=get_or_404(s,Run,run_id); candidates=list(s.scalars(select(Candidate).where(Candidate.run_id==run_id).order_by(Candidate.number))); ids=[c.id for c in candidates]
    return {"run":serialize(r),"candidates":[serialize(c) for c in candidates],"jobs":[serialize(j) for j in s.scalars(select(Job).where(Job.run_id==run_id).order_by(Job.created_at))],"artifacts":[serialize(a) for a in s.scalars(select(Artifact).where(Artifact.candidate_id.in_(ids))) ] if ids else []}
@app.get("/artifacts/{artifact_id}")
def artifact(artifact_id:str,s:Session=Depends(db)):
    return serialize(get_or_404(s,Artifact,artifact_id))
@app.post("/runs/{run_id}/candidates")
def add_candidate(run_id:str,s:Session=Depends(db)):
    r=get_or_404(s,Run,run_id); n=len(list(s.scalars(select(Candidate).where(Candidate.run_id==run_id))))+1;c=Candidate(run_id=run_id,number=n,seed=1000+n);s.add(c);s.flush();s.add(Job(kind="generate",run_id=run_id,candidate_id=c.id,payload=json.dumps({"input_path":r.input_path,"background":r.background,"tolerance":r.tolerance,"seed":c.seed})));s.commit();return serialize(c)
@app.post("/candidates/{candidate_id}/select")
def choose(candidate_id:str,s:Session=Depends(db)):
    c=get_or_404(s,Candidate,candidate_id); r=get_or_404(s,Run,c.run_id);r.selected_candidate_id=c.id;s.commit();return serialize(r)
@app.post("/candidates/{candidate_id}/evaluation")
def evaluate(candidate_id:str,body:EvaluationIn,s:Session=Depends(db)):
    get_or_404(s,Candidate,candidate_id); e=Evaluation(candidate_id=candidate_id,**body.model_dump());s.add(e);s.commit();return serialize(e)
@app.post("/candidates/{candidate_id}/optimize")
def optimize(candidate_id:str,body:OptimizeIn,s:Session=Depends(db)):
    c=get_or_404(s,Candidate,candidate_id);j=Job(kind="optimize",run_id=c.run_id,candidate_id=c.id,payload=body.model_dump_json());s.add(j);s.commit();return serialize(j)
@app.post("/candidates/{candidate_id}/export")
def export(candidate_id:str,formats:list[str],preset:str="Generic",s:Session=Depends(db)):
    c=get_or_404(s,Candidate,candidate_id)
    if not set(formats)<= {"glb","fbx"} or not formats: raise HTTPException(422,"formats must contain glb and/or fbx")
    j=Job(kind="export",run_id=c.run_id,candidate_id=c.id,payload=json.dumps({"formats":formats,"preset":preset}));s.add(j);s.commit();return serialize(j)
@app.post("/worker/jobs/claim",response_model=Optional[JobOut])
def claim(kind:str,s:Session=Depends(db)):
    j=s.scalar(select(Job).where(Job.kind==kind,Job.status=="queued").order_by(Job.created_at))
    if not j:return None
    j.status="claimed";j.claimed_at=now();s.commit();return j
@app.post("/worker/jobs/{job_id}/start")
def start(job_id:str,s:Session=Depends(db)): j=get_or_404(s,Job,job_id);j.status="running";j.progress=0;s.commit();return serialize(j)
@app.post("/worker/jobs/{job_id}/update")
def update(job_id:str,body:WorkerUpdate,s:Session=Depends(db)):
    j=get_or_404(s,Job,job_id)
    if body.progress is not None:j.progress=body.progress
    if body.log:j.log+=(body.log+"\n")
    if body.error:j.status="failed";j.log+=body.error
    s.commit();return serialize(j)
@app.post("/worker/jobs/{job_id}/complete")
def complete(job_id:str,result:dict,s:Session=Depends(db)):
    j=get_or_404(s,Job,job_id); j.status="completed";j.progress=100;j.log+="completed\n"
    c=get_or_404(s,Candidate,j.candidate_id) if j.candidate_id else None
    if j.kind=="generate" and c: c.status="completed";c.model_path=result.get("model_path");c.preview_path=result.get("preview_path");c.score=result.get("score")
    if j.kind=="export" and c:
        for kind,path in result.get("artifacts",{}).items(): s.add(Artifact(candidate_id=c.id,kind=kind,path=path,metadata_json=json.dumps(result.get("metadata",{}))))
    if j.kind=="optimize" and c and result.get("model_path"): c.model_path=result["model_path"]
    s.commit();return serialize(j)
@app.post("/worker/jobs/{job_id}/cancel")
def cancel(job_id:str,s:Session=Depends(db)): j=get_or_404(s,Job,job_id);j.status="cancelled";s.commit();return serialize(j)
@app.get("/files/{path:path}")
def file_path(path:str):
    p=abs_path(path)
    if not p.is_file(): raise HTTPException(404,"file not found")
    from fastapi.responses import FileResponse
    return FileResponse(p)
