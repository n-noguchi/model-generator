"""Single-owner GPU jobs: stop children before releasing the database lease."""
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path
import requests

API = os.getenv("API_BASE_URL", "http://api:8000")
DATA = Path("/data")
LOCK_FD = None


def api(path, data=None):
    response = requests.get(API + path, timeout=15) if data is None else requests.post(API + path, json=data, timeout=30)
    response.raise_for_status()
    return response.json()


def status(job):
    return next(row["status"] for row in api(f"/jobs?run_id={job['run_id']}") if row["id"] == job["id"])


class Cancelled(Exception):
    pass


def group_is_running(group_id):
    # Linux /proc: wait for all descendants, not only the direct Popen child.
    # Zombies have released CUDA resources and need their own parent's reap.
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit(): continue
        try:
            fields=(directory / "stat").read_text().rsplit(")",1)[1].split()
            if int(fields[2]) == group_id and fields[0] != "Z": return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def run_process(args, out, job, log_name, heartbeat=lambda: True):
    with (out / log_name).open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=() if LOCK_FD is None else (LOCK_FD,),
                                   env={**os.environ, "GPU_JOB_LOCK_FD": "" if LOCK_FD is None else str(LOCK_FD)})
        try:
            while process.poll() is None:
                heartbeat()
                if status(job) in {"cancelling", "cancelled"}:
                    raise Cancelled()
                time.sleep(2)
            if process.returncode:
                raise RuntimeError((out / log_name).read_text(errors="replace")[-1800:])
        finally:
            # Include nested model processes (e.g. Hunyuan's isolated venv).
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            while group_is_running(process.pid):
                heartbeat()
                time.sleep(.05)


def settle(job, result=None, error=None, heartbeat=lambda: True):
    """Keep ownership across API outages and uncertain completion responses."""
    while True:
        try:
            heartbeat()
            state = status(job)
            if state in {"completed", "failed", "cancelled"}:
                return
            if state == "cancelling":
                api(f"/worker/jobs/{job['id']}/cancelled", {})
            elif result is not None:
                try:
                    api(f"/worker/jobs/{job['id']}/complete", result)
                except requests.HTTPError as exc:
                    if exc.response.status_code in {400, 422}:
                        error, result = "成果物の登録検証に失敗しました。ワーカーログを確認してください。", None
                    else:
                        raise
            else:
                api(f"/worker/jobs/{job['id']}/update", {"error": error or "生成が中断しました。再生成してください。"})
        except requests.RequestException:
            # Do not drop the current job or claim new work while unacknowledged.
            time.sleep(3)


def recover(kind, heartbeat):
    for job in api("/jobs"):
        if job["kind"] == kind and job["status"] in {"claimed", "running", "cancelling"}:
            settle(job, error="ワーカーの再起動で中断しました。再生成してください。", heartbeat=heartbeat)


def serve(kind, execute, heartbeat=lambda: True):
    global LOCK_FD
    with (DATA / f"{kind}-worker.lock").open("a") as lock:
        # Child inherits this descriptor, so a parent-only crash cannot recover
        # a job whose inference process is still alive. No timeout lease theft.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        LOCK_FD = lock.fileno()
        needs_recovery = True
        while True:
            try:
                if needs_recovery:
                    recover(kind, heartbeat)
                    needs_recovery = False
                if not heartbeat():
                    time.sleep(5)
                    continue
                job = api(f"/worker/jobs/claim?kind={kind}", {})
                if not job:
                    time.sleep(2)
                    continue
                try:
                    result = execute(job)
                except Exception as exc:
                    settle(job, error=str(exc)[-2000:], heartbeat=heartbeat)
                else:
                    settle(job, result=result, heartbeat=heartbeat)
            except requests.RequestException:
                # Includes a successful claim whose HTTP response was lost.
                needs_recovery = True
                print("API unavailable; retaining/recovering GPU ownership", flush=True)
                time.sleep(3)
