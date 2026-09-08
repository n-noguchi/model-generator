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
 assert r['run']['bake_resolution']==1024
 assert all(__import__('json').loads(job['payload'])['cutout_mode']=='auto' for job in r['jobs'])
 assert all(__import__('json').loads(job['payload'])['bake_resolution']==1024 for job in r['jobs'])

def test_generation_texture_resolution_is_persisted_and_added_candidates_keep_it():
 p=client.post('/projects',json={'name':'texture resolution'}).json(); image=Image.new('RGB',(128,128),(255,0,255)); f=Path(os.environ['DATA_DIR'])/'resolution.png';image.save(f)
 up=client.post('/uploads',files={'image':('resolution.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up,'bake_resolution':2048}).json()
 assert detail['run']['bake_resolution']==2048
 assert all(__import__('json').loads(job['payload'])['bake_resolution']==2048 for job in detail['jobs'])
 added=client.post('/runs/'+detail['run']['id']+'/candidates').json()
 jobs=client.get('/jobs?run_id='+detail['run']['id']).json()
 assert __import__('json').loads(next(j for j in jobs if j['candidate_id']==added['id'])['payload'])['bake_resolution']==2048

def test_color_mode_and_added_candidate_keep_cutout_payload():
 p=client.post('/projects',json={'name':'cutout'}).json(); image=Image.new('RGB',(128,128),(1,2,3)); f=Path(os.environ['DATA_DIR'])/'cutout.png';image.save(f)
 up=client.post('/uploads',files={'image':('cutout.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up,'cutout_mode':'color','confirm_warning':True}).json()
 assert detail['run']['cutout_mode']=='color'
 added=client.post('/runs/'+detail['run']['id']+'/candidates').json()
 jobs=client.get('/jobs?run_id='+detail['run']['id']).json()
 assert __import__('json').loads(next(j for j in jobs if j['candidate_id']==added['id'])['payload'])['cutout_mode']=='color'
def test_optional_reference_views_are_saved_but_not_generation_inputs():
 p=client.post('/projects',json={'name':'views'}).json(); image=Image.new('RGB',(128,128),(3,4,5)); f=Path(os.environ['DATA_DIR'])/'view.png';image.save(f)
 def upload(name): return client.post('/uploads',files={'image':(name,f.read_bytes(),'image/png')}).json()['path']
 front,back,left,right=(upload(n) for n in ('front.png','back.png','left.png','right.png'))
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':front,'back_path':back,'left_path':left,'right_path':right}).json()
 assert {k:detail['run'][k] for k in ('input_path','back_path','left_path','right_path')}=={'input_path':front,'back_path':back,'left_path':left,'right_path':right}
 assert all(__import__('json').loads(job['payload'])['input_path']==front for job in detail['jobs'])
 assert all(__import__('json').loads(job['payload'])['generation_view']=='front' for job in detail['jobs'])
 assert detail['run']['generation_mode']=='sf3d'

def test_multiview_queues_one_candidate_with_all_four_inputs_and_quality_options():
 p=client.post('/projects',json={'name':'multiview'}).json(); image=Image.new('RGB',(128,128),(3,4,5)); f=Path(os.environ['DATA_DIR'])/'multiview.png';image.save(f)
 def upload(name): return client.post('/uploads',files={'image':(name,f.read_bytes(),'image/png')}).json()['path']
 front,back,left,right=(upload(n) for n in ('front.png','back.png','left.png','right.png'))
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':front,'back_path':back,'left_path':left,'right_path':right,'generation_mode':'multiview','bake_resolution':4096,'inference_steps':50,'octree_resolution':384}).json()
 assert detail['run']['generation_mode']=='multiview' and len(detail['candidates'])==1 and len(detail['jobs'])==1
 payload=__import__('json').loads(detail['jobs'][0]['payload'])
 assert {k:payload[k] for k in ('front_path','back_path','left_path','right_path')}=={'front_path':front,'back_path':back,'left_path':left,'right_path':right}
 assert payload['generation_mode']=='multiview' and payload['inference_steps']==50 and payload['octree_resolution']==384 and payload['bake_resolution']==4096
 added=client.post('/runs/'+detail['run']['id']+'/candidates').json()
 payload=__import__('json').loads(next(j for j in client.get('/jobs?run_id='+detail['run']['id']).json() if j['candidate_id']==added['id'])['payload'])
 assert payload['generation_mode']=='multiview' and payload['right_path']==right

def test_multiview_requires_all_directions_and_high_resolution_texture():
 p=client.post('/projects',json={'name':'invalid multiview'}).json(); image=Image.new('RGB',(128,128),(3,4,5)); f=Path(os.environ['DATA_DIR'])/'invalid-mv.png';image.save(f)
 up=client.post('/uploads',files={'image':('invalid-mv.png',f.read_bytes(),'image/png')}).json()['path']
 response=client.post('/runs',json={'project_id':p['id'],'input_path':up,'generation_mode':'multiview','bake_resolution':2048})
 assert response.status_code==422
 response=client.post('/runs',json={'project_id':p['id'],'input_path':up,'back_path':up,'left_path':up,'right_path':up,'generation_mode':'multiview','bake_resolution':1024})
 assert response.status_code==422
def test_old_run_defaults_reference_views_to_null():
 p=client.post('/projects',json={'name':'legacy'}).json(); image=Image.new('RGB',(128,128),(7,8,9)); f=Path(os.environ['DATA_DIR'])/'legacy.png';image.save(f)
 up=client.post('/uploads',files={'image':('legacy.png',f.read_bytes(),'image/png')}).json()['path']
 detail=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json()
 assert detail['run']['back_path'] is None and detail['run']['left_path'] is None and detail['run']['right_path'] is None
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
