import os,tempfile,sys
from pathlib import Path
os.environ["DATA_DIR"]=tempfile.mkdtemp();os.environ["DATABASE_URL"]=f"sqlite:///{Path(os.environ['DATA_DIR'])/'test.db'}"
sys.path.insert(0,str(Path(__file__).parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
client=TestClient(app)
def test_project_upload_and_confirmed_run():
 p=client.post('/projects',json={'name':'a'}).json(); image=Image.new('RGB',(128,128),(255,0,255)); f=Path(os.environ['DATA_DIR'])/'in.png';image.save(f)
 up=client.post('/uploads',files={'image':('in.png',f.read_bytes(),'image/png')}).json()['path']
 r=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json();assert 'run' in r
 run=client.get('/runs/'+r['run']['id']).json();assert len(run['candidates'])==3 and len(run['jobs'])==3
 assert client.get('/projects/'+p['id']+'/runs').json()[0]['id']==r['run']['id']
 assert client.get('/jobs?run_id='+r['run']['id']).status_code==200
 assert r['run']['cutout_mode']=='auto'
 assert all(__import__('json').loads(job['payload'])['cutout_mode']=='auto' for job in r['jobs'])

def test_color_mode_and_added_candidate_keep_cutout_payload():
 p=client.post('/projects',json={'name':'cutout'}).json(); image=Image.new('RGB',(128,128),(1,2,3)); f=Path(os.environ['DATA_DIR'])/'cutout.png';image.save(f)
 up=client.post('/uploads',files={'image':('cutout.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up,'cutout_mode':'color','confirm_warning':True}).json()
 assert detail['run']['cutout_mode']=='color'
 added=client.post('/runs/'+detail['run']['id']+'/candidates').json()
 jobs=client.get('/jobs?run_id='+detail['run']['id']).json()
 assert __import__('json').loads(next(j for j in jobs if j['candidate_id']==added['id'])['payload'])['cutout_mode']=='color'
def test_transparent_background_and_run_status():
 p=client.post('/projects',json={'name':'transparent'}).json(); image=Image.new('RGBA',(128,128),(20,40,60,0)); f=Path(os.environ['DATA_DIR'])/'alpha.png';image.save(f)
 up=client.post('/uploads',files={'image':('alpha.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json(); assert detail['run']['warning'] is None
 job=detail['jobs'][0]; client.post('/worker/jobs/'+job['id']+'/start')
 assert client.get('/runs/'+detail['run']['id']).json()['run']['status']=='running'
 client.post('/worker/jobs/'+job['id']+'/update',json={'error':'broken'})
 assert client.get('/runs/'+detail['run']['id']).json()['candidates'][0]['status']=='failed'
def test_completed_and_partial_failure_run_states():
 p=client.post('/projects',json={'name':'states'}).json(); image=Image.new('RGB',(128,128),(255,0,255)); f=Path(os.environ['DATA_DIR'])/'states.png';image.save(f)
 up=client.post('/uploads',files={'image':('states.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json(); rid=detail['run']['id']
 for job in detail['jobs']: client.post('/worker/jobs/'+job['id']+'/complete',json={})
 assert client.get('/runs/'+rid).json()['run']['status']=='completed'
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json(); rid=detail['run']['id']
 client.post('/worker/jobs/'+detail['jobs'][0]['id']+'/complete',json={})
 client.post('/worker/jobs/'+detail['jobs'][1]['id']+'/update',json={'error':'broken'})
 client.post('/worker/jobs/'+detail['jobs'][2]['id']+'/complete',json={})
 assert client.get('/runs/'+rid).json()['run']['status']=='completed_with_errors'
def test_rejects_traversal():
 assert client.get('/files/%2E%2E/%2E%2E/Windows/win.ini').status_code in (400,404)
