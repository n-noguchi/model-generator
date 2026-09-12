"""Contracts for local motion jobs and shared GPU queue ownership."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from test_api import client, project_and_four_uploads, create_high_quality_run
from app.main import DATA, Artifact, Candidate, Job, Sessions

@pytest.fixture
def source():
    project, paths=project_and_four_uploads()
    detail=create_high_quality_run(project,paths).json(); candidate=detail["candidates"][0]
    with Sessions() as session:
        # Isolate queued work from tests that deliberately leave jobs pending.
        for job in session.query(Job).all(): job.status="completed"
        row=session.get(Candidate,candidate["id"]); row.status="completed"; row.model_path=paths[0]
        artifact=Artifact(candidate_id=row.id,kind="rigged_glb",path=paths[0]); session.add(artifact); session.commit()
        artifact_id=artifact.id
    (DATA/"text-motion-status.json").write_text(json.dumps({"ready":True}))
    return candidate["id"],artifact_id,paths[0]

def submit(source, **options):
    cid,aid,_=source
    return client.post(f"/candidates/{cid}/text-motions",json={"artifact_id":aid,"prompt":"A person walks forward.",**options})

def test_inputs_are_fixed_and_invalid_requests_do_not_enqueue(source):
    response=submit(source,seconds=4,seed=77); assert response.status_code==200
    job=response.json(); payload=json.loads(job["payload"])
    assert payload["source_artifact_id"]==source[1]
    assert payload["source_sha256"]==hashlib.sha256((DATA/source[2]).read_bytes()).hexdigest()
    assert payload["prompt"]=="A person walks forward." and payload["seed"]==77
    for bad in ({"prompt":"歩いてください"},{"prompt":"   "},{"seconds":9},{"seed":-1}):
        assert submit(source,**bad).status_code==422
    (DATA/"text-motion-status.json").unlink()
    assert submit(source).status_code==409

def test_wrong_candidate_artifact_is_rejected(source):
    with Sessions() as session:
        artifact=session.get(Artifact,source[1]); artifact.kind="template_motion_glb"; session.commit()
    assert submit(source).status_code==422

def test_shared_gpu_claim_is_atomic_and_cancel_keeps_lease(source):
    job=submit(source).json()
    with Sessions() as session:
        other=Job(kind="generate",run_id=job["run_id"],candidate_id=source[0]); session.add(other); session.commit(); other_id=other.id
    def claim(kind): return client.post(f"/worker/jobs/claim?kind={kind}").json()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(claim,["text_motion","generate"]))
    assert sum(result is not None for result in results)==1
    winner=next(result for result in results if result)
    # Release the selected job, then test the text-motion cancellation lease.
    with Sessions() as session:
        session.get(Job,winner["id"]).status="completed"
        session.get(Job,job["id"]).status="queued"
        session.get(Job,other_id).status="queued"; session.commit()
    assert claim("text_motion")["id"]==job["id"]
    client.post(f"/worker/jobs/{job['id']}/start")
    assert client.post(f"/worker/jobs/{job['id']}/cancel").json()["status"]=="cancelling"
    assert claim("generate") is None
    assert client.post(f"/worker/jobs/{job['id']}/complete",json={}).status_code==409
    client.post(f"/worker/jobs/{job['id']}/cancelled")
    assert claim("generate")["id"]==other_id
    next_motion=submit(source).json()
    assert "id" in next_motion
    client.post(f"/worker/jobs/{other_id}/start")
    assert client.post(f"/worker/jobs/{other_id}/cancel").json()["status"]=="cancelling"
    assert claim("text_motion") is None
    assert client.post(f"/worker/jobs/{other_id}/start").status_code==409
    assert client.post(f"/worker/jobs/{other_id}/update",json={"error":"late"}).json()["status"]=="cancelling"
    assert client.post(f"/worker/jobs/{other_id}/complete",json={}).status_code==409
    client.post(f"/worker/jobs/{other_id}/cancelled")
    assert claim("text_motion")["id"]==next_motion["id"]

def test_complete_is_idempotent_and_artifacts_cannot_escape(source):
    job=submit(source).json(); jid=job["id"]
    root=DATA/"projects"/job["run_id"]/"text-motions"/source[0]/jid; root.mkdir(parents=True)
    artifacts={}
    for kind in ("glb","fbx","manifest","joints"):
        file=root/kind; file.write_bytes(b"test"); artifacts[f"text_motion_{kind}"]=str(file.relative_to(DATA))
    escaped={**artifacts,"text_motion_glb":source[2]}
    assert client.post(f"/worker/jobs/{jid}/complete",json={"artifacts":escaped}).status_code==422
    result={"artifacts":artifacts,"metadata":{"pipeline":"test"}}
    assert client.post(f"/worker/jobs/{jid}/complete",json=result).status_code==200
    assert client.post(f"/worker/jobs/{jid}/complete",json=result).status_code==200
    with Sessions() as session:
        assert len([a for a in session.query(Artifact).all() if jid in a.path])==4
    assert client.post(f"/worker/jobs/{jid}/cancel").json()["status"]=="completed"
