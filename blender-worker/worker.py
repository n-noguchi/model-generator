import json, os, subprocess, time
from pathlib import Path
import requests
API=os.getenv("API_BASE_URL","http://localhost:8000");DATA=Path("/data")
def main():
  while True:
    worked=False
    for kind in ("optimize","export"):
      try: j=requests.post(f"{API}/worker/jobs/claim?kind={kind}",timeout=30).json()
      except (requests.RequestException, ValueError) as e:
        print(f"API unavailable; retrying in 2 seconds: {e}",flush=True);time.sleep(2);continue
      if not j: continue
      worked=True
      try:
        requests.post(f"{API}/worker/jobs/{j['id']}/start",timeout=30)
        out=DATA/"projects"/j["run_id"]/("exports" if kind=="export" else "optimized")/j["candidate_id"];out.mkdir(parents=True,exist_ok=True)
        result=out/"result.json";subprocess.run(["blender","--background","--python","/worker/process.py","--",str(DATA),j["candidate_id"],j["kind"],j["payload"],str(result)],check=True)
        requests.post(f"{API}/worker/jobs/{j['id']}/complete",json=json.loads(result.read_text()),timeout=30)
      except Exception as e:
        try: requests.post(f"{API}/worker/jobs/{j['id']}/update",json={"error":str(e)},timeout=30)
        except requests.RequestException as report_error: print(f"Could not report job failure: {report_error}",flush=True)
    if not worked: time.sleep(2)
if __name__=="__main__":main()
