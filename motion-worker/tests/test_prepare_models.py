import hashlib
import importlib.util
import io
import sys
import types
import zipfile
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("prepare_models",Path(__file__).parents[1]/"prepare_models.py")
prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)

@pytest.fixture
def assets(monkeypatch,tmp_path):
    values={"model.pt":b"checkpoint","args.json":b"args","Mean.npy":b"mean","Std.npy":b"std","clip/ViT-B-32.pt":b"clip"}
    monkeypatch.setattr(prepare,"ROOT",tmp_path)
    monkeypatch.setattr(prepare,"EXPECTED",{name:hashlib.sha256(value).hexdigest() for name,value in values.items()})
    for name,value in values.items():
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(value)
    monkeypatch.setitem(sys.modules,"clip",types.SimpleNamespace(load=lambda *a,**kw:pytest.fail("unexpected download")))
    return values

@pytest.mark.parametrize("broken_zip",[False,True])
def test_retry_repairs_partial_checkpoint_and_missing_args(monkeypatch,tmp_path,assets,broken_zip):
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,"w") as archive:
        archive.writestr("model000750000.pt",assets["model.pt"])
        archive.writestr("args.json",assets["args.json"])
    (tmp_path/"humanml_enc_512_50steps.zip").write_bytes(b"partial" if broken_zip else bundle.getvalue())
    (tmp_path/"model.pt").write_bytes(b"partial")
    (tmp_path/"args.json").unlink()
    def download(**kwargs): Path(kwargs["output"]).write_bytes(bundle.getvalue())
    monkeypatch.setitem(sys.modules,"gdown",types.SimpleNamespace(download=download))
    prepare.main()
    assert all(prepare.valid(name) for name in assets)
    assert (tmp_path/"ready.json").is_file()

def test_bad_digest_is_not_published(tmp_path,assets):
    with pytest.raises(RuntimeError,match="SHA256"):
        prepare.atomic_file("model.pt",io.BytesIO(b"bad download"))
    assert (tmp_path/"model.pt").read_bytes()==assets["model.pt"]
