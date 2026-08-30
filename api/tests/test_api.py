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
 r=client.post('/runs',json={'project_id':p['id'],'input_path':up}).json();assert 'id' in r
 run=client.get('/runs/'+r['id']).json();assert len(run['candidates'])==3 and len(run['jobs'])==3
def test_rejects_traversal():
 assert client.get('/files/%2E%2E/%2E%2E/Windows/win.ini').status_code in (400,404)
