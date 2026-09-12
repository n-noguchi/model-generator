import importlib.util
import fcntl
import os
import sys
from pathlib import Path
import pytest
import requests

spec=importlib.util.spec_from_file_location("gpu_runtime",Path(__file__).parents[1]/"gpu_runtime.py")
runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
JOB={"id":"test","run_id":"run","kind":"generate","status":"running"}

@pytest.mark.parametrize("result",[None,{"model_path":"output.glb"}])
def test_settle_retains_job_across_api_outage_and_uncertain_ack(monkeypatch,result):
    state="running"; reads=0; writes=[]
    def fake(path,data=None):
        nonlocal state,reads
        if data is None:
            reads+=1
            if reads<=3: raise requests.ConnectionError("temporary outage")
            return [{**JOB,"status":state}]
        writes.append(path)
        state="completed" if path.endswith("complete") else "failed"
        raise requests.ConnectionError("response lost after committed write")
    monkeypatch.setattr(runtime,"api",fake)
    monkeypatch.setattr(runtime.time,"sleep",lambda _:None)
    runtime.settle(JOB,result=result,error="test failure")
    assert reads==5 and len(writes)==1

def test_recovery_only_releases_owned_kind_and_acknowledges_cancel(monkeypatch):
    jobs=[{**JOB,"id":"stale"},{**JOB,"id":"cancel","status":"cancelling"},{**JOB,"id":"other","kind":"text_motion"}]
    writes=[]
    def fake(path,data=None):
        if data is None: return jobs
        row=next(row for row in jobs if row["id"]==path.split("/")[-2])
        writes.append(path)
        row["status"]="cancelled" if path.endswith("cancelled") else "failed"
    monkeypatch.setattr(runtime,"api",fake)
    runtime.recover("generate",lambda:True)
    assert [row["status"] for row in jobs]==["failed","cancelled","running"]
    assert len(writes)==2

@pytest.mark.parametrize("failure",[runtime.Cancelled,requests.ConnectionError])
def test_child_is_stopped_before_cancellation_or_api_error_escapes(monkeypatch,tmp_path,failure):
    children=[]; original=runtime.subprocess.Popen
    def capture(*args,**kwargs):
        child=original(*args,**kwargs); children.append(child); return child
    def fail(_): raise failure()
    monkeypatch.setattr(runtime.subprocess,"Popen",capture)
    monkeypatch.setattr(runtime,"status",fail)
    with pytest.raises(failure):
        runtime.run_process([sys.executable,"-c","import time; time.sleep(30)"],tmp_path,JOB,"test.log")
    assert len(children)==1 and children[0].poll() is not None

@pytest.mark.parametrize("middle_crash",[False,True])
def test_nested_processes_are_stopped_and_inherit_lock(monkeypatch,tmp_path,middle_crash):
    pid_file=tmp_path/"grandchild.pid"
    nested="import os,time,signal,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(os.environ['TEST_PID_FILE']).write_text(str(os.getpid())); time.sleep(30)"
    script="import os,sys,subprocess,time; subprocess.Popen([sys.executable,'-c',"+repr(nested)+"],pass_fds=(int(os.environ['GPU_JOB_LOCK_FD']),)); "+("time.sleep(.2); sys.exit(1)" if middle_crash else "time.sleep(30)")
    def state(_):
        return "cancelling" if pid_file.exists() and not middle_crash else "running"
    monkeypatch.setattr(runtime,"status",state)
    monkeypatch.setenv("TEST_PID_FILE",str(pid_file))
    with (tmp_path/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        monkeypatch.setattr(runtime,"LOCK_FD",lock.fileno())
        with pytest.raises((runtime.Cancelled,RuntimeError)):
            runtime.run_process([sys.executable,"-c",script],tmp_path,JOB,"nested.log")
        grandchild=int(pid_file.read_text())
        proc=Path(f"/proc/{grandchild}/stat")
        assert not proc.exists() or proc.read_text().rsplit(")",1)[1].split()[0]=="Z"
    with (tmp_path/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
