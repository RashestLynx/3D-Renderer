"""CPU rasteriser: z-buffer, perspective-correct UVs, bilinear textures, alpha cut-outs, three-point lighting."""
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Camera:
    yaw: float = 0.0          # degrees around the up axis; 0 looks from +Z toward the model
    pitch: float = 5.0        # degrees above the horizon
    zoom: float = 1.0         # >1 moves closer
    fov: float = 30.0         # vertical field of view, degrees
    target: object = None     # look-at point (default: centre of the model)
    span: float = None        # height of the framed area (default: the model's height)
    pan: tuple = (0.0, 0.0)   # screen-space offset, fraction of the framed height


@dataclass
class Style:
    background: tuple = ((0.20, 0.20, 0.22), (0.07, 0.07, 0.08))   # top, bottom
    ambient: float = 0.28
    lights: tuple = (((-0.5, 0.6, 0.8), 0.85), ((0.7, 0.2, 0.4), 0.35), ((0.0, 0.5, -1.0), 0.45))
    gamma: float = 2.2
    supersample: int = 2


def sample(tex, uv):
    """Bilinear, wrapped. uv (N,2) -> (N,4). glTF UV origin is the image's top-left."""
    h, w = tex.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy) +
            tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def view_matrix(model, cam):
    lo, hi = model.bounds()
    target = np.asarray(cam.target if cam.target is not None else (lo + hi) / 2, np.float64)
    span = cam.span if cam.span else max(hi[1] - lo[1], (hi[0] - lo[0]) * 0.7, (hi[2] - lo[2]) * 0.7) * 1.1
    dist = span / 2 / np.tan(np.radians(cam.fov) / 2) / cam.zoom
    a, b = np.radians(cam.yaw), np.radians(cam.pitch)
    eye = target + dist * np.array([np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)])
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0, 1, 0))
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f])
    eye = eye - (r * cam.pan[0] + u * cam.pan[1]) * span
    return R, eye


def render(model, cam=None, size=(640, 900), style=None):
    """Render a Model to a PIL RGB image."""
    cam = cam or Camera()
    style = style or Style()
    W, H = size
    ss = max(1, style.supersample)
    Ws, Hs = W * ss, H * ss
    R, eye = view_matrix(model, cam)
    fpx = (Hs / 2) / np.tan(np.radians(cam.fov) / 2)
    top, bot = np.array(style.background[0], np.float32), np.array(style.background[1], np.float32)
    t = np.linspace(0, 1, Hs, dtype=np.float32)[:, None, None]
    col = np.broadcast_to(top * (1 - t) + bot * t, (Hs, Ws, 3)).copy()
    zb = np.full((Hs, Ws), np.inf, np.float32)
    lights = [(np.asarray(d, np.float64) / np.linalg.norm(d), k) for d, k in style.lights]
    for part in model.parts:
        cam_p = (part.positions - eye) @ R.T
        z = -cam_p[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            sx = cam_p[:, 0] / z * fpx + Ws / 2
            sy = -cam_p[:, 1] / z * fpx + Hs / 2
        nv = part.normals @ R.T
        mat = part.material
        tex = mat.texture
        bc = np.asarray(mat.base_color, np.float32)
        cut = mat.alpha_cutoff if mat.alpha_mode == 'MASK' else (0.45 if tex is not None and mat.alpha_mode != 'OPAQUE' else None)
        if cut is None and tex is not None and tex[..., 3].min() < 0.5:
            cut = 0.45                                   # textures with a cut-out alpha (hair, leaves) but no flag
        for i0, i1, i2 in part.triangles:
            if z[i0] <= 1e-3 or z[i1] <= 1e-3 or z[i2] <= 1e-3:
                continue
            xs = (sx[i0], sx[i1], sx[i2])
            ys = (sy[i0], sy[i1], sy[i2])
            x0, x1 = max(int(min(xs)), 0), min(int(max(xs)) + 1, Ws)
            y0, y1 = max(int(min(ys)), 0), min(int(max(ys)) + 1, Hs)
            if x0 >= x1 or y0 >= y1:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            px = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
            py = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
            w0 = ((ys[1] - ys[2]) * (px - xs[2]) + (xs[2] - xs[1]) * (py - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (px - xs[2]) + (xs[0] - xs[2]) * (py - ys[2])) / den
            w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            iz0, iz1, iz2 = 1 / z[i0], 1 / z[i1], 1 / z[i2]
            a0, a1, a2 = w0[m] * iz0, w1[m] * iz1, w2[m] * iz2
            d = 1 / (a0 + a1 + a2)
            yy, xx = np.nonzero(m)
            yy += y0
            xx += x0
            closer = d < zb[yy, xx]
            if not closer.any():
                continue
            yy, xx, d = yy[closer], xx[closer], d[closer]
            b = np.stack([a0[closer], a1[closer], a2[closer]], 1) * d[:, None]
            if tex is not None:
                uv = b[:, :1] * part.uvs[i0] + b[:, 1:2] * part.uvs[i1] + b[:, 2:3] * part.uvs[i2]
                c = sample(tex, uv) * bc
            else:
                c = np.broadcast_to(bc, (len(yy), 4))
            if cut is not None:
                keep = c[:, 3] > cut
                if not keep.any():
                    continue
                yy, xx, d, b, c = yy[keep], xx[keep], d[keep], b[keep], c[keep]
            n = b[:, :1] * nv[i0] + b[:, 1:2] * nv[i1] + b[:, 2:3] * nv[i2]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
            n[n[:, 2] < 0] *= -1                         # two-sided
            shade = np.full((len(n), 1), style.ambient)
            for l, k in lights:
                shade = shade + k * np.clip(n @ l, 0, None)[:, None]
            zb[yy, xx] = d
            col[yy, xx] = np.clip((c[:, :3] ** style.gamma) * shade, 0, 1) ** (1 / style.gamma)
    img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8))
    return img.resize((W, H), Image.LANCZOS) if ss > 1 else img
