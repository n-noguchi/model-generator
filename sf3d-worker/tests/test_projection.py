import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
import projection
from projection import _fill_unseen, _project, _sample_rgba, _view_axes, _zbuffer


def test_default_view_directions_are_distinct_and_y_up():
    axes = _view_axes("+z")
    assert np.allclose(axes["front"][0], (0, 0, 1))
    assert np.allclose(axes["back"][0], (0, 0, -1))
    assert np.allclose(axes["left"][0], (1, 0, 0))
    assert np.allclose(axes["right"][0], (-1, 0, 0))
    # Screen right for the front is world +X.  Top of the image is world +Y.
    bounds = np.array(((-1, -1, -1), (1, 1, 1)), dtype=float)
    xy, _ = _project(np.array(((0.8, 0.8, 1), (-0.8, -0.8, 1)), dtype=float), *axes["front"], bounds,
                     (0, 99, 0, 99), (100, 100))
    assert xy[0, 0] > xy[1, 0]
    assert xy[0, 1] < xy[1, 1]
    # Deliberately asymmetric source data catches a silent left/right mirror.
    source = np.zeros((100, 100, 4), dtype=np.uint8)
    source[:, :50] = (255, 0, 0, 255)
    source[:, 50:] = (0, 0, 255, 255)
    rgb, alpha = _sample_rgba(source, xy)
    assert alpha.min() > 0.99
    assert rgb[0, 2] > rgb[0, 0]  # world +X must sample source-image right (blue)
    assert rgb[1, 0] > rgb[1, 2]


def test_front_zbuffer_keeps_near_cube_face_not_back_face():
    cube = trimesh.creation.box(extents=(2, 2, 2))
    direction, right = _view_axes("+z")["front"]
    z, scale, _ = _zbuffer(np.asarray(cube.vertices), np.asarray(cube.faces), direction, right,
                            np.asarray(cube.bounds), (0, 99, 0, 99), (100, 100), max_size=100)
    # Camera looks from +Z, therefore the centre pixel must hold the +Z face.
    assert scale == 1.0
    assert z[50, 50] > 0.9


def test_front_axis_can_be_calibrated_to_negative_z():
    axes = _view_axes("-z")
    assert np.allclose(axes["front"][0], (0, 0, -1))
    assert np.allclose(axes["front"][1], (-1, 0, 0))


def test_fill_unseen_does_not_cross_between_nearby_uv_charts():
    """A nearby packed chart must not donate its colour through UV space."""
    rgb = np.zeros((5, 12, 3), dtype=np.float32)
    confidence = np.zeros((5, 12), dtype=np.float32)
    atlas = np.zeros((5, 12), dtype=bool)
    # Separate islands: red lies physically closer in 2-D to the left edge of
    # the blue chart, so the old global EDT would have copied it there.
    atlas[1:4, 1:3] = True
    atlas[1:4, 4:11] = True
    rgb[2, 1] = (1, 0, 0)
    confidence[2, 1] = 1
    rgb[2, 10] = (0, 0, 1)
    confidence[2, 10] = 1
    positions = np.zeros((5, 12, 3), dtype=np.float16)
    result = _fill_unseen(rgb, confidence, atlas, positions)
    assert result[2, 4, 2] > 0.9
    assert result[2, 4, 0] < 0.1


def test_project_texture_exports_textured_glb_with_dominant_front_view(tmp_path, monkeypatch):
    # A UV-ready single +Z triangle lets this be an integration test without
    # requiring the optional xatlas binary package on every test runner.
    triangle = trimesh.Trimesh(vertices=[(-1, -1, 1), (1, -1, 1), (0, 1, 1)], faces=[(0, 1, 2)], process=False)
    triangle.visual = trimesh.visual.texture.TextureVisuals(uv=np.array(((0, 0), (1, 0), (0.5, 1)), dtype=np.float32))
    monkeypatch.setattr(projection, "_unwrap", lambda _: triangle.copy())
    colors = {"front": (240, 20, 10, 255), "back": (10, 30, 240, 255),
              "left": (10, 240, 30, 255), "right": (240, 220, 10, 255)}
    images = {key: Image.new("RGBA", (64, 64), value) for key, value in colors.items()}
    out = tmp_path / "textured.glb"
    _, metrics = projection.project_texture(triangle, images, out, resolution=512)
    assert out.exists()
    assert (tmp_path / "texture.png").exists()
    assert metrics["coverage"] > 0.95
    texture = np.asarray(Image.open(tmp_path / "texture.png"))
    painted = texture[texture[..., 0] > 150]
    assert painted.size > 0
    assert painted[:, 0].mean() > painted[:, 2].mean() * 3
    exported = trimesh.load(out, force="mesh")
    assert len(exported.faces) == 1


def test_project_texture_uses_subject_side_views_with_real_xatlas_unwrap(tmp_path):
    """Exercise the production unwrap and ensure side photographs are not mirrored.

    The deliberately non-uniform left/right images make a swapped view direction
    detectable in both the ownership diagnostic and the baked texture.  This is
    skipped only on lightweight development environments; the Hunyuan worker
    image installs xatlas as a required dependency.
    """
    pytest.importorskip("xatlas")
    cube = trimesh.creation.box(extents=(2, 3, 1.5))
    width = height = 96
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    left = np.zeros((height, width, 4), dtype=np.uint8)
    left[..., 1] = ramp[None, :]
    left[..., 2] = 255 - ramp[None, :]
    left[..., 3] = 255
    right = np.zeros((height, width, 4), dtype=np.uint8)
    right[..., 0] = ramp[None, :]
    right[..., 1] = 255 - ramp[None, :]
    right[..., 3] = 255
    images = {
        "front": Image.new("RGBA", (width, height), (230, 20, 10, 255)),
        "back": Image.new("RGBA", (width, height), (10, 30, 230, 255)),
        "left": Image.fromarray(left, "RGBA"),
        "right": Image.fromarray(right, "RGBA"),
    }
    out = tmp_path / "cube.glb"
    _, metrics = projection.project_texture(cube, images, out, resolution=512)

    assert out.exists()
    assert metrics["source_samples"]["left"] > 0
    assert metrics["source_samples"]["right"] > 0
    source_map = np.asarray(Image.open(tmp_path / "projection-diagnostics" / "source-map.png"))
    # Owner colours: left=green, right=yellow.  Both actual subject sides must
    # be present after the real xatlas seam split.
    assert np.count_nonzero(np.all(source_map == (80, 205, 125), axis=-1)) > 20
    assert np.count_nonzero(np.all(source_map == (240, 185, 60), axis=-1)) > 20
    texture = np.asarray(Image.open(tmp_path / "texture.png"))
    # The gradients survive baking; side input must not collapse to a flat fill.
    green_blue = texture[(texture[..., 1] > 40) & (texture[..., 2] > 40)]
    red_green = texture[(texture[..., 0] > 40) & (texture[..., 1] > 40) & (texture[..., 2] < 25)]
    assert np.ptp(green_blue[:, 1]) > 80
    assert np.ptp(red_green[:, 0]) > 80
