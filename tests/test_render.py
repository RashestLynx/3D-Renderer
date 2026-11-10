"""Builds small glTF files in memory and checks what the renderer draws.  python -m pytest tests"""
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelview import Camera, Style, load, render  # noqa: E402
from modelview.cli import main  # noqa: E402


def make_glb(path, colour=(255, 0, 0, 255), alpha_hole=False, skinned_offset=None):
    """A unit quad facing +Z with a 4x4 texture; optionally a transparent texel column and a joint moving it."""
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
    nrm = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint16)
    tex = np.zeros((4, 4, 4), np.uint8)
    tex[:] = colour
    if alpha_hole:
        tex[:, :2, 3] = 0                        # left half transparent
    png = io.BytesIO()
    Image.fromarray(tex, 'RGBA').save(png, 'PNG')
    blobs = [pos.tobytes(), nrm.tobytes(), uv.tobytes(), idx.tobytes(), png.getvalue()]
    j = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
         'nodes': [{'mesh': 0}], 'bufferViews': [], 'accessors': [],
         'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3,
                                     'material': 0}]}],
         'materials': [{'name': 'quad', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}},
                        'alphaMode': 'MASK' if alpha_hole else 'OPAQUE'}],
         'textures': [{'source': 0}], 'images': [{'bufferView': 4, 'mimeType': 'image/png'}]}
    if skinned_offset is not None:
        joints = np.zeros((4, 4), np.uint8)
        weights = np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1))
        inv = np.eye(4, dtype=np.float32).T.ravel()
        blobs += [joints.tobytes(), weights.tobytes(), inv.tobytes()]
        j['nodes'] = [{'mesh': 0, 'skin': 0}, {'translation': list(skinned_offset)}]
        j['skins'] = [{'joints': [1], 'inverseBindMatrices': 7}]
        j['meshes'][0]['primitives'][0]['attributes'].update({'JOINTS_0': 5, 'WEIGHTS_0': 6})
    data = b''
    specs = [(5126, 'VEC3', 4), (5126, 'VEC3', 4), (5126, 'VEC2', 4), (5123, 'SCALAR', 6), None,
             (5121, 'VEC4', 4), (5126, 'VEC4', 4), (5126, 'MAT4', 1)]
    for k, b in enumerate(blobs):
        while len(data) % 4:
            data += b'\0'
        j['bufferViews'].append({'buffer': 0, 'byteOffset': len(data), 'byteLength': len(b)})
        if specs[k]:
            ct, typ, n = specs[k]
            a = {'bufferView': k, 'componentType': ct, 'type': typ, 'count': n}
            if k == 0:
                a.update(min=[-1, -1, 0], max=[1, 1, 0])
            j['accessors'].append(a)
        else:
            j['accessors'].append({'bufferView': 0, 'componentType': 5126, 'type': 'VEC3', 'count': 1})  # placeholder
        data += b
    # accessor indices must match the spec positions above: rebuild without the image placeholder
    acc = [a for k, a in enumerate(j['accessors']) if specs[k]]
    remap = {}
    n = 0
    for k in range(len(specs)):
        if specs[k]:
            remap[k] = n
            n += 1
    prim = j['meshes'][0]['primitives'][0]
    prim['attributes'] = {key: remap[v] for key, v in prim['attributes'].items()}
    prim['indices'] = remap[3]
    if 'skins' in j:
        j['skins'][0]['inverseBindMatrices'] = remap[7]
    j['accessors'] = acc
    j['buffers'] = [{'byteLength': len(data)}]
    js = json.dumps(j).encode()
    js += b' ' * (-len(js) % 4)
    data += b'\0' * (-len(data) % 4)
    out = b'glTF' + struct.pack('<II', 2, 12 + 8 + len(js) + 8 + len(data))
    out += struct.pack('<I', len(js)) + b'JSON' + js + struct.pack('<I', len(data)) + b'BIN\0' + data
    open(path, 'wb').write(out)
    return path


def centre(img):
    a = np.asarray(img)
    return a[a.shape[0] // 2, a.shape[1] // 2]


def test_textured_quad_is_drawn_red(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb')))
    assert m.triangle_count() == 2
    img = render(m, Camera(yaw=0, pitch=0), (200, 200), Style(supersample=1, ambient=1.0, lights=()))
    r, g, b = centre(img)
    assert r > 200 and g < 30 and b < 30


def test_background_outside_model(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb')))
    img = render(m, Camera(yaw=0, pitch=0, zoom=0.3), (200, 200), Style(supersample=1))
    r, g, b = np.asarray(img)[2, 2]
    assert r < 80 and abs(int(r) - int(b)) < 20          # grey background, not red


def test_alpha_cutout_shows_background(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb'), alpha_hole=True))
    a = np.asarray(render(m, Camera(yaw=0, pitch=0), (200, 200), Style(supersample=1, ambient=1.0, lights=())))
    left, right = a[100, 70], a[100, 130]
    assert right[0] > 200 and left[0] < 80                # left half cut out


def test_texture_override(tmp_path):
    green = tmp_path / 'g.png'
    Image.new('RGBA', (2, 2), (0, 255, 0, 255)).save(green)
    m = load(make_glb(str(tmp_path / 'q.glb')), {'quad': str(green)})
    r, g, b = centre(render(m, Camera(yaw=0, pitch=0), (100, 100), Style(supersample=1, ambient=1.0, lights=())))
    assert g > 200 and r < 30


def test_empty_texture_path_hides_material(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb')), {'quad': ''})
    assert m.parts == []


def test_skinned_mesh_uses_bind_pose(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb'), skinned_offset=(5, 0, 0)))
    lo, hi = m.bounds()
    assert np.allclose(lo[0], 4) and np.allclose(hi[0], 6)


def test_back_view_still_draws_two_sided(tmp_path):
    m = load(make_glb(str(tmp_path / 'q.glb')))
    r, g, b = centre(render(m, Camera(yaw=180, pitch=0), (100, 100), Style(supersample=1, ambient=1.0, lights=())))
    assert r > 200


def test_cli_render_and_compare(tmp_path):
    q = make_glb(str(tmp_path / 'q.glb'))
    out = tmp_path / 'r.png'
    main(['render', q, '-o', str(out), '--views', '0,90', '--size', '120x160'])
    assert Image.open(out).size[0] == 120 * 2 + 10
    green = tmp_path / 'g.png'
    Image.new('RGBA', (2, 2), (0, 255, 0, 255)).save(green)
    (tmp_path / 'a.json').write_text('{}')
    (tmp_path / 'b.json').write_text(json.dumps({'quad': str(green)}))
    out2 = tmp_path / 'c.png'
    main(['compare', q, '-o', str(out2), '--a', str(tmp_path / 'a.json'), '--b', str(tmp_path / 'b.json'),
          '--views', '0', '--size', '120x160'])
    assert Image.open(out2).size == (120, 50 + 2 * (160 + 42))
