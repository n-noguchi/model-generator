"""CPU multiview texture projection for generated character meshes.

The shape pipeline is deliberately not coupled to this module.  It receives the
four already cut-out, RGBA source views and writes a conventional textured GLB.
Coordinates are Y-up and the default convention is that the front is visible
from +Z.  ``front_axis`` can be changed while calibrating a different generator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import trimesh


VIEW_ORDER = ("front", "back", "left", "right")
_AXES = {
    "+z": np.array((0.0, 0.0, 1.0)),
    "-z": np.array((0.0, 0.0, -1.0)),
    "+x": np.array((1.0, 0.0, 0.0)),
    "-x": np.array((-1.0, 0.0, 0.0)),
}


def _view_axes(front_axis: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return (view direction, screen-right) for the configurable convention."""
    front = _AXES.get(front_axis)
    if front is None:
        raise ValueError("front_axis must be one of +z, -z, +x, -x")
    # This is a rotation about the Y axis.  The image's top always maps to +Y.
    right = np.cross(np.array((0.0, 1.0, 0.0)), front)
    return {
        "front": (front, right),
        "back": (-front, -right),
        # ``left`` and ``right`` are the subject's own sides, rather than the
        # viewer's.  Starting at front, the Hunyuan multiview convention turns
        # clockwise to the subject's left (+90 degrees), then to the back.
        "left": (right, -front),
        "right": (-right, front),
    }


def _alpha_bounds(alpha: np.ndarray) -> tuple[float, float, float, float]:
    yy, xx = np.where(alpha >= 250)
    h, w = alpha.shape
    if len(xx) < 16:
        return (0.0, float(w - 1), 0.0, float(h - 1))
    return (float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max()))


def _source_rgba(value: Image.Image | str | Path) -> np.ndarray:
    if not isinstance(value, Image.Image):
        with Image.open(value) as opened:
            value = opened.copy()
    return np.asarray(value.convert("RGBA"), dtype=np.uint8)


def _project(points: np.ndarray, direction: np.ndarray, screen_right: np.ndarray,
             mesh_bounds: np.ndarray, source_bounds: tuple[float, float, float, float],
             shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Orthographically project points, fitting each cut-out to the mesh extent."""
    h, w = shape
    sx0, sx1, sy0, sy1 = source_bounds
    center = (mesh_bounds[0] + mesh_bounds[1]) * 0.5
    extent = np.maximum(mesh_bounds[1] - mesh_bounds[0], 1e-6)
    # horizontal mesh span follows screen-right; vertical always follows Y.
    horizontal = abs(screen_right[0]) * extent[0] + abs(screen_right[2]) * extent[2]
    vertical = extent[1]
    u = np.dot(points - center, screen_right) / max(horizontal, 1e-6) + 0.5
    v = 0.5 - (points[:, 1] - center[1]) / max(vertical, 1e-6)
    x = sx0 + u * max(sx1 - sx0, 1.0)
    y = sy0 + v * max(sy1 - sy0, 1.0)
    depth = np.dot(points - center, direction)
    return np.stack((x, y), axis=1), depth


def _sample_rgba(image: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear source sampling. Returns RGB in 0..1 and alpha in 0..1."""
    h, w = image.shape[:2]
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    x0 = np.clip(np.floor(x).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    a = image[y0, x0].astype(np.float32) * (1 - fx) + image[y0, x1] * fx
    b = image[y1, x0].astype(np.float32) * (1 - fx) + image[y1, x1] * fx
    rgba = (a * (1 - fy) + b * fy) / 255.0
    rgba[~inside] = 0.0
    return rgba[:, :3], rgba[:, 3]


def _sample_scalar(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinearly sample a single-channel trust mask at image coordinates."""
    h, w = image.shape
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    x0 = np.clip(np.floor(x).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    value = (image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy) + (
        image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy
    value[~inside] = 0.0
    return value


def _zbuffer(vertices: np.ndarray, faces: np.ndarray, direction: np.ndarray,
             screen_right: np.ndarray, bounds: np.ndarray,
             source_bounds: tuple[float, float, float, float], shape: tuple[int, int],
             max_size: int = 768) -> tuple[np.ndarray, float, float]:
    """Small orthographic depth buffer.  It is intentionally CPU-only."""
    src_h, src_w = shape
    scale = min(1.0, max_size / max(src_h, src_w))
    h, w = max(2, round(src_h * scale)), max(2, round(src_w * scale))
    xy, depth = _project(vertices, direction, screen_right, bounds, source_bounds, shape)
    xy *= scale
    z = np.full((h, w), -np.inf, dtype=np.float32)
    for tri in faces:
        p = xy[tri]
        xmin, ymin = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        xmax, ymax = np.minimum(np.ceil(p.max(axis=0)).astype(int), (w - 1, h - 1))
        if xmax < xmin or ymax < ymin:
            continue
        area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(area) < 1e-10:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        px, py = gx + 0.5, gy + 0.5
        b1 = ((px - p[0, 0]) * (p[2, 1] - p[0, 1]) - (py - p[0, 1]) * (p[2, 0] - p[0, 0])) / area
        b2 = ((p[1, 0] - p[0, 0]) * (py - p[0, 1]) - (p[1, 1] - p[0, 1]) * (px - p[0, 0])) / area
        b0 = 1.0 - b1 - b2
        mask = (b0 >= -1e-5) & (b1 >= -1e-5) & (b2 >= -1e-5)
        if np.any(mask):
            values = (b0 * depth[tri[0]] + b1 * depth[tri[1]] + b2 * depth[tri[2]]).astype(np.float32)
            current = z[ymin:ymax + 1, xmin:xmax + 1]
            current[mask] = np.maximum(current[mask], values[mask])
    return z, scale, float(bounds[1].max() - bounds[0].min())


def _unwrap(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        import xatlas
    except ImportError as exc:  # Make missing image dependency actionable in the worker log.
        raise RuntimeError("xatlas is required for multiview texture projection") from exc
    # xatlas duplicates vertices at chart boundaries.  Asking trimesh for
    # normals after that split makes each chart edge look like a hard edge,
    # which in turn changes the preferred camera and produces patchy clothing.
    # Preserve the smooth normals from the generated mesh through ``vmapping``.
    source_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    atlas = xatlas.Atlas()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = 2048
    pack_options.padding = 8
    atlas.add_mesh(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32))
    atlas.generate(pack_options=pack_options)
    vmapping, indices, uvs = atlas[0]
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[vmapping], faces=np.asarray(indices), process=False)
    result.visual = trimesh.visual.texture.TextureVisuals(uv=np.asarray(uvs, dtype=np.float32))
    preserved_normals = source_normals[np.asarray(vmapping, dtype=np.int64)]
    preserved_normals /= np.maximum(np.linalg.norm(preserved_normals, axis=1, keepdims=True), 1e-8)
    result.vertex_attributes["projection_normals"] = preserved_normals
    return result


def _fill_unseen(rgb: np.ndarray, confidence: np.ndarray, atlas_mask: np.ndarray,
                 positions: np.ndarray | None = None) -> np.ndarray:
    """Fill only within a UV chart, never across unrelated packed charts.

    A global 2-D nearest-pixel fill can copy a shoe or face into a nearby UV
    island.  Components that have observations use a local nearest fill.  A
    wholly unobserved chart falls back to the closest *surface* sample in 3-D,
    so it still inherits plausible material colour rather than neutral grey.
    """
    seen = (confidence > 0) & atlas_mask
    if not np.any(atlas_mask):
        return rgb
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree

        labels, count = ndimage.label(atlas_mask)
        objects = ndimage.find_objects(labels)
        known_points = positions[seen].astype(np.float32) if positions is not None and np.any(seen) else None
        known_colours = rgb[seen].copy()
        tree = cKDTree(known_points) if known_points is not None and len(known_points) else None
        for label, box in enumerate(objects, start=1):
            if box is None:
                continue
            # Work only in this chart's compact UV rectangle.  Running an EDT
            # over the entire 2K/4K atlas for every chart is prohibitively slow.
            chart = labels[box] == label
            seen_crop = seen[box]
            missing = chart & ~seen_crop
            if not np.any(missing):
                continue
            chart_seen = chart & seen_crop
            if np.any(chart_seen):
                # distance_transform_edt returns source coordinates for every
                # destination, including a compact chart with many UV gaps.
                _, nearest = ndimage.distance_transform_edt(~chart_seen, return_indices=True)
                cropped_rgb = rgb[box]
                cropped_rgb[missing] = cropped_rgb[tuple(nearest[:, missing])]
            elif tree is not None:
                # Query in chunks: a 4096 map can contain large hidden charts.
                yy, xx = np.where(missing)
                yy += box[0].start
                xx += box[1].start
                indices = yy * rgb.shape[1] + xx
                for start in range(0, len(indices), 262144):
                    chunk = indices[start:start + 262144]
                    points = positions.reshape(-1, 3)[chunk].astype(np.float32)
                    _, nearest = tree.query(points, k=1)
                    rgb.reshape(-1, 3)[chunk] = known_colours[nearest]
            elif np.any(seen):
                # Only reached without position data (mainly lightweight tests).
                rgb_crop = rgb[box]
                rgb_crop[missing] = np.median(rgb[seen], axis=0)
    except ImportError:
        # Deterministic fallback for development images without scipy.  It is
        # deliberately chart-local where possible, never a whole-atlas bleed.
        for y, x in zip(*np.where(atlas_mask & ~seen)):
            rgb[y, x] = np.median(rgb[seen], axis=0) if np.any(seen) else (0.45, 0.45, 0.45)
    # Extend chart edge colours into the atlas padding for bilinear filtering
    # and mipmaps. This never changes occupied texels or fills another chart.
    if np.any(atlas_mask):
        from scipy import ndimage
        _, nearest = ndimage.distance_transform_edt(~atlas_mask, return_indices=True)
        rgb[~atlas_mask] = rgb[tuple(nearest[:, ~atlas_mask])]
    return rgb


def _silhouette_iou(zbuffer: np.ndarray, rgba: np.ndarray) -> tuple[float, np.ndarray]:
    """Compare the fitted mesh silhouette to the supplied cut-out silhouette."""
    predicted = np.isfinite(zbuffer)
    alpha = Image.fromarray(rgba[..., 3], "L").resize((zbuffer.shape[1], zbuffer.shape[0]), Image.Resampling.NEAREST)
    observed = np.asarray(alpha) >= 250
    intersection = predicted & observed
    union = predicted | observed
    iou = float(intersection.sum() / max(int(union.sum()), 1))
    # Green is agreement, red is generated-only, blue is source-only.
    overlay = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    overlay[predicted & ~observed] = (230, 70, 55)
    overlay[observed & ~predicted] = (70, 120, 230)
    overlay[intersection] = (70, 205, 105)
    return iou, overlay


def project_texture(mesh: trimesh.Trimesh | str | Path,
                    images: Mapping[str, Image.Image | str | Path], out: str | Path,
                    resolution: int = 2048, front_axis: str = "+z") -> tuple[trimesh.Trimesh, dict]:
    """Bake cut-out original views to an xatlas UV map and export ``out`` GLB.

    Fully opaque pixels only are trusted.  This excludes anti-aliased alpha
    boundaries (where coloured backgrounds otherwise leak into the model).
    """
    if resolution not in (512, 1024, 2048, 4096):
        raise ValueError("resolution must be 512, 1024, 2048, or 4096")
    missing = [name for name in VIEW_ORDER if name not in images]
    if missing:
        raise ValueError(f"missing source views: {', '.join(missing)}")
    if not isinstance(mesh, trimesh.Trimesh):
        loaded = trimesh.load(mesh, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("input must contain one mesh")
        mesh = loaded
    if mesh.faces.size == 0:
        raise ValueError("input mesh has no faces")
    mesh = _unwrap(mesh)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    # See _unwrap: projection normals deliberately survive UV chart splits.
    normals = np.asarray(mesh.vertex_attributes.get("projection_normals", mesh.vertex_normals), dtype=np.float32)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    source = {name: _source_rgba(images[name]) for name in VIEW_ORDER}
    source_bounds = {name: _alpha_bounds(source[name][..., 3]) for name in VIEW_ORDER}
    # Fully opaque alone is insufficient for generated images: pixels just
    # inside a hard cut-out edge often contain one-pixel pink/green background
    # contamination.  Keep a three-pixel interior safety margin without ever
    # keying a colour (a character may legitimately wear pink).
    try:
        from scipy import ndimage
        interior_distance = {
            name: ndimage.distance_transform_edt(source[name][..., 3] >= 250).astype(np.float32)
            for name in VIEW_ORDER
        }
    except ImportError:
        interior_distance = {name: (source[name][..., 3] >= 250).astype(np.float32) * 4.0 for name in VIEW_ORDER}
    axes = _view_axes(front_axis)
    depth = {}
    for name in VIEW_ORDER:
        direction, screen_right = axes[name]
        depth[name] = _zbuffer(vertices, faces, direction, screen_right, bounds,
                               source_bounds[name], source[name].shape[:2])

    size = resolution
    colors = np.zeros((size, size, 3), dtype=np.float32)
    weights = np.zeros((size, size), dtype=np.float32)
    owners = np.full((size, size), 255, dtype=np.uint8)
    atlas_mask = np.zeros((size, size), dtype=bool)
    # Kept as float16 to make a 4096 bake practical (~96 MiB).  It is used only
    # for wholly hidden UV islands, where millimetre-level precision is not
    # meaningful and avoids a cross-chart 2-D colour leak.
    positions = np.zeros((size, size, 3), dtype=np.float16)
    uv = np.asarray(mesh.visual.uv)
    # xatlas UV is bottom-left origin; image arrays are top-left origin.
    uv_pixels = np.column_stack((uv[:, 0] * (size - 1), (1.0 - uv[:, 1]) * (size - 1)))
    head_start = bounds[0, 1] + (bounds[1, 1] - bounds[0, 1]) * 0.82
    visible_samples = dict.fromkeys(VIEW_ORDER, 0)
    for face in faces:
        p = uv_pixels[face]
        xmin, ymin = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        xmax, ymax = np.minimum(np.ceil(p.max(axis=0)).astype(int), size - 1)
        if xmax < xmin or ymax < ymin:
            continue
        area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(area) < 1e-10:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        px, py = gx + 0.5, gy + 0.5
        b1 = ((px - p[0, 0]) * (p[2, 1] - p[0, 1]) - (py - p[0, 1]) * (p[2, 0] - p[0, 0])) / area
        b2 = ((p[1, 0] - p[0, 0]) * (py - p[0, 1]) - (p[1, 1] - p[0, 1]) * (px - p[0, 0])) / area
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-5) & (b1 >= -1e-5) & (b2 >= -1e-5)
        if not np.any(inside):
            continue
        bary = np.column_stack((b0[inside], b1[inside], b2[inside]))
        points = bary @ vertices[face]
        point_normals = bary @ normals[face]
        point_normals /= np.maximum(np.linalg.norm(point_normals, axis=1, keepdims=True), 1e-8)
        # Do not average opposing photos: it causes immediately visible double
        # eyes and logos.  The best geometrically-supported source owns a texel.
        local_rgb = np.zeros((len(points), 3), dtype=np.float32)
        local_weight = np.zeros(len(points), dtype=np.float32)
        second_rgb = np.zeros((len(points), 3), dtype=np.float32)
        second_weight = np.zeros(len(points), dtype=np.float32)
        local_owner = np.full(len(points), 255, dtype=np.uint8)
        for view_index, name in enumerate(VIEW_ORDER):
            direction, screen_right = axes[name]
            xy, point_depth = _project(points, direction, screen_right, bounds, source_bounds[name], source[name].shape[:2])
            sampled_rgb, alpha = _sample_rgba(source[name], xy)
            interior = _sample_scalar(interior_distance[name], xy)
            z, scale, span = depth[name]
            zzx = np.clip((xy[:, 0] * scale).astype(int), 0, z.shape[1] - 1)
            zzy = np.clip((xy[:, 1] * scale).astype(int), 0, z.shape[0] - 1)
            # A small tolerance allows the low-resolution z-buffer to represent slanted faces.
            visible = point_depth >= z[zzy, zzx] - max(span / max(z.shape), 1e-5) * 2.5
            facing = np.clip(point_normals @ direction, 0.0, 1.0)
            valid = visible & (alpha >= 0.98) & (interior >= 3.0) & (facing > 0.03)
            edge_confidence = np.clip((interior - 3.0) / 6.0, 0.0, 1.0)
            w = np.where(valid, facing ** 4 * alpha * edge_confidence, 0.0).astype(np.float32)
            # Portrait detail is usually clearest in the true front image.  The
            # boost is only used on confidently front-facing upper-body pixels.
            if name == "front":
                frontal_head = valid & (points[:, 1] >= head_start) & (facing >= 0.65)
                w[frontal_head] *= 4.0
            # Track two best views.  Clothing near a view boundary benefits
            # from a narrow blend; facial pixels retain a single strongest view
            # to prevent a profile photo printing a second face on the cheek.
            replace = w > local_weight
            second_rgb[replace] = local_rgb[replace]
            second_weight[replace] = local_weight[replace]
            local_owner[replace] = view_index
            local_weight[replace] = w[replace]
            local_rgb[replace] = sampled_rgb[replace]
            runner_up = ~replace & (w > second_weight)
            second_weight[runner_up] = w[runner_up]
            second_rgb[runner_up] = sampled_rgb[runner_up]
            visible_samples[name] += int(valid.sum())
        yy, xx = gy[inside], gx[inside]
        # Only blend a genuinely close runner-up, and never above the lower
        # face.  This removes sharp clothing patches while keeping image detail
        # (eyes, mouths, emblems) owned by the best-aligned camera.
        clothing = points[:, 1] < head_start
        narrow = clothing & (second_weight > 0) & (local_weight > 0)
        ratio = second_weight[narrow] / local_weight[narrow]
        transition = np.clip((ratio - 0.25) / 0.75, 0.0, 1.0)
        # A smooth ramp reaches an equal blend at the ownership boundary.
        # The previous hard threshold jumped immediately from 0 to ~42%.
        blend = (0.5 * transition * transition * (3.0 - 2.0 * transition))[:, None]
        local_rgb[narrow] = local_rgb[narrow] * (1.0 - blend) + second_rgb[narrow] * blend
        atlas_mask[yy, xx] = True
        colors[yy, xx] = local_rgb
        weights[yy, xx] = local_weight
        owners[yy, xx] = local_owner
        positions[yy, xx] = points.astype(np.float16)

    atlas_pixels = max(int(np.count_nonzero(atlas_mask)), 1)
    coverage = float(np.count_nonzero(weights) / atlas_pixels)
    if coverage <= 0.0:
        raise RuntimeError("no opaque, visible source pixels could be projected onto the mesh")
    texture = np.clip(_fill_unseen(colors, weights, atlas_mask, positions) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    texture_path = Path(out).with_name("texture.png")
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture, "RGB").save(texture_path, optimize=True)
    # PBR settings avoid a plastic-looking default material in game engines.
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture, "RGB"),
                                                   metallicFactor=0.0, roughnessFactor=1.0)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.fromarray(texture, "RGB"), material=material)
    out_path = Path(out)
    mesh.export(out_path)
    diagnostics = out_path.with_name("projection-diagnostics")
    diagnostics.mkdir(exist_ok=True)
    Image.fromarray((np.clip(weights / max(float(weights.max()), 1e-8), 0, 1) * 255).astype(np.uint8), "L").save(diagnostics / "coverage.png")
    palette = np.array(((235, 80, 70), (70, 155, 235), (80, 205, 125), (240, 185, 60), (45, 45, 45)), dtype=np.uint8)
    Image.fromarray(palette[np.minimum(owners, 4)], "RGB").save(diagnostics / "source-map.png")
    # These depth overlays make camera-sign or left/right calibration errors visible
    # without having to load the GLB in another application.
    silhouette_iou = {}
    for name in VIEW_ORDER:
        z, _, _ = depth[name]
        finite = np.isfinite(z)
        preview = np.zeros_like(z, dtype=np.uint8)
        if np.any(finite):
            lo, hi = z[finite].min(), z[finite].max()
            preview[finite] = np.clip((z[finite] - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(preview, "L").save(diagnostics / f"visibility-{name}.png")
        iou, overlay = _silhouette_iou(z, source[name])
        silhouette_iou[name] = iou
        Image.fromarray(overlay, "RGB").save(diagnostics / f"silhouette-{name}.png")
    metrics = {"texture_resolution": resolution, "front_axis": front_axis, "coverage": coverage,
               "atlas_pixels": atlas_pixels, "source_samples": visible_samples,
               "silhouette_iou": silhouette_iou, "views": list(VIEW_ORDER), "texture": str(texture_path.name)}
    (diagnostics / "coverage.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return mesh, metrics
