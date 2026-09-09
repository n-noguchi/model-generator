import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{Path(os.environ['DATA_DIR']) / 'test.db'}"
sys.path.insert(0, str(Path(__file__).parents[1]))
from app.main import app

client = TestClient(app)


def project_and_four_uploads():
    project = client.post("/projects", json={"name": "high quality"}).json()
    image = Image.new("RGB", (128, 128), (20, 40, 60))
    source = Path(os.environ["DATA_DIR"]) / "input.png"
    image.save(source)
    paths = [client.post("/uploads", files={"image": (f"{name}.png", source.read_bytes(), "image/png")}).json()["path"]
             for name in ("front", "back", "left", "right")]
    return project, paths


def create_high_quality_run(project, paths, **options):
    body = {"project_id": project["id"], "input_path": paths[0], "back_path": paths[1],
            "left_path": paths[2], "right_path": paths[3], **options}
    return client.post("/runs", json=body)


def test_high_quality_run_requires_all_views_and_queues_one_candidate():
    project, paths = project_and_four_uploads()
    response = create_high_quality_run(project, paths, bake_resolution=4096)
    assert response.status_code == 200
    detail = response.json()
    assert detail["run"]["generation_mode"] == "multiview"
    assert len(detail["candidates"]) == len(detail["jobs"]) == 1
    payload = json.loads(detail["jobs"][0]["payload"])
    assert {key: payload[key] for key in ("front_path", "back_path", "left_path", "right_path")} == dict(
        zip(("front_path", "back_path", "left_path", "right_path"), paths))
    assert payload["bake_resolution"] == 4096


def test_api_rejects_single_image_and_fast_mode_requests():
    project, paths = project_and_four_uploads()
    response = client.post("/runs", json={"project_id": project["id"], "input_path": paths[0]})
    assert response.status_code == 422
    response = create_high_quality_run(project, paths, generation_mode="sf3d")
    assert response.status_code == 422
    response = create_high_quality_run(project, paths, bake_resolution=1024)
    assert response.status_code == 422


def test_added_candidate_preserves_all_four_high_quality_inputs():
    project, paths = project_and_four_uploads()
    detail = create_high_quality_run(project, paths).json()
    candidate = client.post(f"/runs/{detail['run']['id']}/candidates").json()
    jobs = client.get(f"/jobs?run_id={detail['run']['id']}").json()
    payload = json.loads(next(job for job in jobs if job["candidate_id"] == candidate["id"])["payload"])
    assert payload["generation_mode"] == "multiview"
    assert payload["right_path"] == paths[3]


def test_old_single_image_run_cannot_create_an_invalid_new_candidate():
    project, paths = project_and_four_uploads()
    from app.main import Candidate, Job, Run, Sessions
    with Sessions() as session:
        old = Run(project_id=project["id"], input_path=paths[0], generation_mode="sf3d")
        session.add(old)
        session.flush()
        session.add(Candidate(run_id=old.id, number=1, seed=1001))
        session.add(Job(kind="generate", run_id=old.id, status="completed"))
        session.commit()
        old_id = old.id
    assert client.post(f"/runs/{old_id}/candidates").status_code == 409


def test_high_quality_run_status_and_health():
    project, paths = project_and_four_uploads()
    detail = create_high_quality_run(project, paths).json()
    job = detail["jobs"][0]
    client.post(f"/worker/jobs/{job['id']}/start")
    assert client.get(f"/runs/{detail['run']['id']}").json()["run"]["status"] == "running"
    client.post(f"/worker/jobs/{job['id']}/update", json={"error": "broken"})
    assert client.get(f"/runs/{detail['run']['id']}").json()["candidates"][0]["status"] == "failed"
    assert client.get("/health").json()["mode"] == "multiview"


def test_rig_job_preserves_candidate_model_and_fixes_source_input():
    project, paths = project_and_four_uploads()
    detail = create_high_quality_run(project, paths).json()
    candidate = detail["candidates"][0]
    from app.main import Candidate, Sessions
    with Sessions() as session:
        row = session.get(Candidate, candidate["id"])
        row.status, row.model_path = "completed", paths[0]
        session.commit()
    job = client.post(f"/candidates/{candidate['id']}/rig", json={"target_triangles": 20000}).json()
    payload = json.loads(job["payload"])
    assert job["kind"] == "rig" and payload["source_model_path"] == paths[0]
    with Sessions() as session:
        assert session.get(Candidate, candidate["id"]).model_path == paths[0]
