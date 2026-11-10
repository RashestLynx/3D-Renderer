"""Writes examples/demo.glb: a torus knot with a generated brick colour texture and its normal map, all original
content (no game assets), so the viewer has something to open out of the box.

    python examples/make_demo.py
"""
import io
import json
import os
import struct

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def knot(p=2, q=3, seg=600, ring=48, r_tube=0.32):
    t = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    r = 1 + 0.45 * np.cos(q * t)
    c = np.stack([r * np.cos(p * t), 0.45 * np.sin(q * t) + 1.6, r * np.sin(p * t)], 1)
    d = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    tng = d / np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.cross(tng, [0, 1, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    bin_ = np.cross(tng, nrm)
    a = np.linspace(0, 2 * np.pi, ring + 1)
    pos, nor, uv = [], [], []
    for i in range(seg + 1):
        k = i % seg
        for j, ang in enumerate(a):
            n = np.cos(ang) * nrm[k] + np.sin(ang) * bin_[k]
            pos.append(c[k] + r_tube * n)
            nor.append(n)
            uv.append((i / seg * 12, j / ring))
    idx = []
    w = ring + 1
    for i in range(seg):
        for j in range(ring):
            a0, a1, b0, b1 = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
            idx += [a0, b0, a1, a1, b0, b1]
    return np.array(pos, 'f4'), np.array(nor, 'f4'), np.array(uv, 'f4'), np.array(idx, 'u4')


def bricks(n=512):
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:n, 0:n] / n
    rows = 8
    row = np.floor(y * rows)
    xs = x * 4 + (row % 2) * 0.5
    bx, by = xs % 1, (y * rows) % 1
    mortar = (np.minimum(bx, 1 - bx) < 0.03) | (np.minimum(by, 1 - by) < 0.06)
    tint = rng.uniform(0.75, 1.1, (rows, 5))[row.astype(int), np.floor(xs).astype(int) % 5]
    noise = rng.normal(0, 0.05, (n, n))
    col = np.stack([0.62 * tint, 0.30 * tint, 0.22 * tint], 2) + noise[..., None]
    col[mortar] = [0.72, 0.70, 0.66]
    height = np.where(mortar, 0.0, 1.0) + noise * 0.6
    gy, gx = np.gradient(height)
    nm = np.stack([-gx * 6, -gy * 6, np.ones_like(gx)], 2)
    nm /= np.linalg.norm(nm, axis=2, keepdims=True)
    return (np.clip(col, 0, 1) * 255).astype(np.uint8), ((nm * 0.5 + 0.5) * 255).astype(np.uint8)


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, 'PNG')
    return b.getvalue()


def write_glb(path):
    pos, nor, uv, idx = knot()
    col, nm = bricks()
    blobs = [pos.tobytes(), nor.tobytes(), uv.tobytes(), idx.tobytes(), png(col), png(nm)]
    views, data = [], b''
    for b in blobs:
        data += b'\0' * (-len(data) % 4)
        views.append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(b)})
        data += b
    data += b'\0' * (-len(data) % 4)
    j = {'asset': {'version': '2.0', 'generator': 'modelview examples/make_demo.py'},
         'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0, 'name': 'knot'}],
         'buffers': [{'byteLength': len(data)}], 'bufferViews': views,
         'accessors': [{'bufferView': 0, 'componentType': 5126, 'type': 'VEC3', 'count': len(pos),
                        'min': pos.min(0).tolist(), 'max': pos.max(0).tolist()},
                       {'bufferView': 1, 'componentType': 5126, 'type': 'VEC3', 'count': len(nor)},
                       {'bufferView': 2, 'componentType': 5126, 'type': 'VEC2', 'count': len(uv)},
                       {'bufferView': 3, 'componentType': 5125, 'type': 'SCALAR', 'count': len(idx)}],
         'images': [{'bufferView': 4, 'mimeType': 'image/png'}, {'bufferView': 5, 'mimeType': 'image/png'}],
         'textures': [{'source': 0}, {'source': 1}],
         'materials': [{'name': 'bricks', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}},
                        'normalTexture': {'index': 1}}],
         'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3,
                                     'material': 0}]}]}
    js = json.dumps(j).encode()
    js += b' ' * (-len(js) % 4)
    out = b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data))
    out += struct.pack('<I', len(js)) + b'JSON' + js + struct.pack('<I', len(data)) + b'BIN\0' + data
    open(path, 'wb').write(out)
    return path


if __name__ == '__main__':
    print(write_glb(os.path.join(HERE, 'demo.glb')))
