import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import os
import pytest
import worker
from worker import alpha_mask_background, prepare_foreground

def test_background_tolerance_becomes_transparent():
    image=Image.new("RGBA",(2,1));image.putdata([(255,0,255,255),(255,0,230,255)])
    masked=alpha_mask_background(image,"#FF00FF",20)
    assert list(masked.getdata())[0][3] == 0
    assert list(masked.getdata())[1][3] == 255

def test_auto_preserves_usable_input_alpha():
    image=Image.new("RGBA",(10,10),(10,20,30,255))
    for x in range(2):
        for y in range(10): image.putpixel((x,y),(10,20,30,0))
    result,message=prepare_foreground(image,{"cutout_mode":"auto"})
    assert result.getpixel((0,0))[3] == 0
    assert "入力画像の透明度" in message

def test_auto_uses_injected_rembg_session_and_remove(tmp_path, monkeypatch):
    (tmp_path/"u2net_human_seg.onnx").write_bytes(b"mock")
    monkeypatch.setenv("U2NET_HOME",str(tmp_path))
    monkeypatch.setattr(worker,"U2NET_HUMAN_SEG_SHA256",__import__("hashlib").sha256(b"mock").hexdigest())
    calls=[]
    def factory(name): calls.append(("session",name)); return "session"
    def remove(image,session):
        calls.append(("remove",session)); output=image.copy(); output.putpixel((0,0),(1,2,3,0)); return output
    result,message=prepare_foreground(Image.new("RGB",(20,20),(1,2,3)),{"cutout_mode":"auto"},session_factory=factory,remove_fn=remove)
    assert result.getpixel((0,0))[3] == 0 and calls == [("session","u2net_human_seg"),("remove","session")]
    assert "AI人物切り抜き" in message

def test_color_mode_is_compatible_with_existing_chroma_key():
    image=Image.new("RGB",(10,10),(255,0,255)); image.putpixel((5,5),(10,20,30))
    result,message=prepare_foreground(image,{"cutout_mode":"color","background":"#FF00FF","tolerance":0})
    assert result.getpixel((0,0))[3] == 0 and result.getpixel((5,5))[3] == 255
    assert "指定単色" in message

def test_auto_fails_without_preloaded_model(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME",str(tmp_path))
    with pytest.raises(RuntimeError,match="人物切り抜きモデルがありません"):
        prepare_foreground(Image.new("RGB",(20,20),(1,2,3)),{"cutout_mode":"auto"})

def test_auto_rejects_corrupt_preloaded_model(tmp_path, monkeypatch):
    (tmp_path/"u2net_human_seg.onnx").write_bytes(b"corrupt")
    monkeypatch.setenv("U2NET_HOME",str(tmp_path))
    with pytest.raises(RuntimeError,match="SHA-256"):
        prepare_foreground(Image.new("RGB",(20,20),(1,2,3)),{"cutout_mode":"auto"})
