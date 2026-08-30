import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from worker import alpha_mask_background

def test_background_tolerance_becomes_transparent():
    image=Image.new("RGBA",(2,1));image.putdata([(255,0,255,255),(255,0,230,255)])
    masked=alpha_mask_background(image,"#FF00FF",20)
    assert list(masked.getdata())[0][3] == 0
    assert list(masked.getdata())[1][3] == 255
