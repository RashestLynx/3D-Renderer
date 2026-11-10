"""glTF 2.0 / GLB loading: meshes in their bind pose, materials and their colour textures."""
import io
import json
import os
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


@dataclass
class Material:
    name: str
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    texture: object = None          # (H, W, 4) float32 0..1, or None
    alpha_mode: str = 'OPAQUE'      # OPAQUE | MASK | BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = True
    normal_texture: object = None   # (H, W, 4) float32 tangent-space normal map, or None
    metallic: float = 0.0
    roughness: float = 0.7
    mr_texture: object = None       # glTF metallicRoughness: G = roughness, B = metallic
    emissive: tuple = (0.0, 0.0, 0.0)
    emissive_texture: object = None
    occlusion_texture: object = None   # R = ambient occlusion
    clearcoat: float = 0.0          # KHR_materials_clearcoat (car paint)
    clearcoat_roughness: float = 0.05
    transmission: float = 0.0       # KHR_materials_transmission (glass): drawn see-through
    uv_sets: dict = field(default_factory=dict)   # texture slot -> TEXCOORD index (0 or 1)
    uv_transforms: dict = field(default_factory=dict)   # texture slot -> 3x3 KHR_texture_transform matrix


@dataclass
class Part:
    positions: np.ndarray           # (N, 3)
    normals: np.ndarray             # (N, 3)
    uvs: np.ndarray                 # (N, 2)
    triangles: np.ndarray           # (M, 3) int
    material: Material
    uvs2: np.ndarray = None         # TEXCOORD_1 (often the ambient-occlusion set), or None


@dataclass
class Model:
    parts: list = field(default_factory=list)

    def bounds(self):
        if not self.parts:
            return np.full(3, -1.0), np.full(3, 1.0)
        p = np.concatenate([q.positions for q in self.parts])
        return p.min(0), p.max(0)

    def triangle_count(self):
        return sum(len(p.triangles) for p in self.parts)


class Gltf:
    """Reads .glb (binary) and .gltf (JSON + buffers) files."""

    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(os.path.abspath(path))
        data = open(path, 'rb').read()
        if data[:4] == b'glTF':
            n = struct.unpack_from('<I', data, 12)[0]
            self.j = json.loads(data[20:20 + n])
            at = 20 + n
            self.buffers = [data[at + 8:at + 8 + struct.unpack_from('<I', data, at)[0]]] if at < len(data) else [b'']
        else:
            self.j = json.loads(data)
            self.buffers = []
            for b in self.j.get('buffers', []):
                uri = b.get('uri', '')
                if uri.startswith('data:'):
                    import base64
                    self.buffers.append(base64.b64decode(uri.split(',', 1)[1]))
                else:
                    self.buffers.append(open(os.path.join(self.dir, uri), 'rb').read())

    def view_bytes(self, i):
        bv = self.j['bufferViews'][i]
        buf = self.buffers[bv.get('buffer', 0)]
        off = bv.get('byteOffset', 0)
        return buf[off:off + bv['byteLength']], bv

    def accessor(self, i):
        a = self.j['accessors'][i]
        dt = np.dtype(_COMP[a['componentType']])
        k = _NCOMP[a['type']]
        if 'bufferView' not in a:
            return np.zeros((a['count'], k), dt)
        raw, bv = self.view_bytes(a['bufferView'])
        off = a.get('byteOffset', 0)
        stride = bv.get('byteStride', 0) or dt.itemsize * k
        arr = np.frombuffer(raw, np.uint8, stride * (a['count'] - 1) + dt.itemsize * k, off)
        rows = np.lib.stride_tricks.as_strided(arr, (a['count'], dt.itemsize * k), (stride, 1))
        out = rows.copy().view(dt).reshape(a['count'], k)
        if a.get('normalized'):
            out = out.astype(np.float32) / np.iinfo(dt).max
        return out

    def node_matrix(self, n):
        node = self.j['nodes'][n]
        if 'matrix' in node:
            return np.array(node['matrix'], np.float64).reshape(4, 4).T
        t = np.array(node.get('translation', [0, 0, 0]), np.float64)
        x, y, z, w = node.get('rotation', [0, 0, 0, 1])
        s = np.array(node.get('scale', [1, 1, 1]), np.float64)
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = r * s
        m[:3, 3] = t
        return m

    def world_matrices(self):
        nodes = self.j.get('nodes', [])
        parent = {c: i for i, nd in enumerate(nodes) for c in nd.get('children', [])}
        cache = {}

        def w(i):
            if i not in cache:
                cache[i] = (w(parent[i]) if i in parent else np.eye(4)) @ self.node_matrix(i)
            return cache[i]
        return [w(i) for i in range(len(nodes))]

    def image(self, index):
        """(H, W, 4) float32 of glTF image `index`, or None when the file is missing."""
        im = self.j['images'][index]
        try:
            if 'bufferView' in im:
                raw, _ = self.view_bytes(im['bufferView'])
                pil = Image.open(io.BytesIO(raw))
            else:
                uri = im.get('uri', '')
                if uri.startswith('data:'):
                    import base64
                    pil = Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1])))
                else:
                    pil = Image.open(os.path.join(self.dir, uri))
            return np.asarray(pil.convert('RGBA'), np.float32) / 255
        except Exception:
            return None


def load_image(path):
    return np.asarray(Image.open(path).convert('RGBA'), np.float32) / 255


def load(path, texture_overrides=None):
    """Load a glTF/GLB as a Model in its bind pose.

    texture_overrides: {material name: image path | {"color": path, "normal": path}} replaces (or supplies) a
    material's textures - for files whose images live elsewhere, or to compare two texture sets on the same mesh.
    An empty value hides the material (clear shells such as eye corneas or glass that would otherwise draw solid)."""
    g = Gltf(path)
    overrides = texture_overrides or {}
    mats, tex_cache = [], {}

    def file_tex(p):
        if p not in tex_cache:
            tex_cache[p] = load_image(p) if p and os.path.exists(p) else None
        return tex_cache[p]

    def gltf_tex(info):
        if not info:
            return None
        src = g.j['textures'][info['index']].get('source')
        if src is None:
            return None
        if ('img', src) not in tex_cache:
            tex_cache[('img', src)] = g.image(src)
        return tex_cache[('img', src)]

    for m in g.j.get('materials', []) or [{}]:
        name = m.get('name', '')
        pbr = m.get('pbrMetallicRoughness', {})
        tex, ntex = gltf_tex(pbr.get('baseColorTexture')), gltf_tex(m.get('normalTexture'))
        ov = overrides.get(name)
        if isinstance(ov, str) and ov:
            tex = file_tex(ov)
        elif isinstance(ov, dict):
            tex = file_tex(ov['color']) if ov.get('color') else tex
            ntex = file_tex(ov['normal']) if ov.get('normal') else ntex
        ext = m.get('extensions', {})
        cc = ext.get('KHR_materials_clearcoat', {})
        tr = ext.get('KHR_materials_transmission', {})
        has_mr = 'metallicFactor' in pbr or 'metallicRoughnessTexture' in pbr
        mat = Material(name, tuple(pbr.get('baseColorFactor', (1, 1, 1, 1))), tex, m.get('alphaMode', 'OPAQUE'),
                       m.get('alphaCutoff', 0.5), m.get('doubleSided', True), ntex,
                       # glTF's default metallic is 1, but exporters that write no PBR data at all (game rips, old
                       # tools) mean a plain non-metal surface
                       metallic=float(pbr.get('metallicFactor', 1.0 if has_mr else 0.0)),
                       roughness=float(pbr.get('roughnessFactor', 1.0 if has_mr else 0.7)),
                       mr_texture=gltf_tex(pbr.get('metallicRoughnessTexture')),
                       emissive=tuple(m.get('emissiveFactor', (0, 0, 0))),
                       emissive_texture=gltf_tex(m.get('emissiveTexture')),
                       occlusion_texture=gltf_tex(m.get('occlusionTexture')),
                       clearcoat=float(cc.get('clearcoatFactor', 0.0)),
                       clearcoat_roughness=float(cc.get('clearcoatRoughnessFactor', 0.05)),
                       transmission=float(tr.get('transmissionFactor', 0.0)))
        strength = ext.get('KHR_materials_emissive_strength', {}).get('emissiveStrength', 1.0)
        mat.emissive = tuple(float(e) * strength for e in mat.emissive)
        for slot, info in (('color', pbr.get('baseColorTexture')), ('normal', m.get('normalTexture')),
                           ('mr', pbr.get('metallicRoughnessTexture')), ('emissive', m.get('emissiveTexture')),
                           ('occlusion', m.get('occlusionTexture'))):
            if info:
                mat.uv_sets[slot] = info.get('texCoord', 0)
                tt = info.get('extensions', {}).get('KHR_texture_transform')
                if tt:
                    mat.uv_transforms[slot] = uv_transform(tt)
                    mat.uv_sets[slot] = tt.get('texCoord', mat.uv_sets[slot])
        mats.append(mat)
    world = g.world_matrices()
    skin_mats = None
    if g.j.get('skins'):
        skin = g.j['skins'][0]
        inv = (g.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
               if 'inverseBindMatrices' in skin else np.tile(np.eye(4), (len(skin['joints']), 1, 1)))
        skin_mats = np.array([world[j] @ inv[k] for k, j in enumerate(skin['joints'])])
    model = Model()
    mesh_nodes = [(i, nd) for i, nd in enumerate(g.j.get('nodes', [])) if 'mesh' in nd]
    if not mesh_nodes:
        mesh_nodes = [(None, {'mesh': k}) for k in range(len(g.j.get('meshes', [])))]
    for ni, nd in mesh_nodes:
        mesh = g.j['meshes'][nd['mesh']]
        for prim in mesh['primitives']:
            if prim.get('mode', 4) != 4:
                continue
            at = prim['attributes']
            pos = g.accessor(at['POSITION']).astype(np.float64)
            nrm = g.accessor(at['NORMAL']).astype(np.float64) if 'NORMAL' in at else None
            uv = g.accessor(at['TEXCOORD_0']).astype(np.float64) if 'TEXCOORD_0' in at else np.zeros((len(pos), 2))
            uv2 = g.accessor(at['TEXCOORD_1']).astype(np.float64) if 'TEXCOORD_1' in at else None
            idx = g.accessor(prim['indices']).ravel().astype(np.int64) if 'indices' in prim else np.arange(len(pos))
            tris = idx[:len(idx) // 3 * 3].reshape(-1, 3)
            if skin_mats is not None and 'JOINTS_0' in at and 'skin' in nd:
                pos, nrm = _skin(pos, nrm, g.accessor(at['JOINTS_0']).astype(int),
                                 g.accessor(at['WEIGHTS_0']).astype(np.float64), skin_mats)
            elif ni is not None:
                m4 = world[ni]
                pos = pos @ m4[:3, :3].T + m4[:3, 3]
                if nrm is not None:
                    nrm = nrm @ np.linalg.inv(m4[:3, :3])
            if nrm is None:
                nrm = vertex_normals(pos, tris)
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            mat = mats[prim['material']] if prim.get('material') is not None else Material('default')
            if mat.name in overrides and not overrides[mat.name]:
                continue                                 # "" / null in the texture map hides the material
            model.parts.append(Part(pos, nrm, uv, tris, mat, uv2))
    return model


def uv_transform(tt):
    """KHR_texture_transform -> 3x3 matrix applied to (u, v, 1): translation * rotation * scale."""
    ox, oy = tt.get('offset', (0, 0))
    r = tt.get('rotation', 0.0)
    sx, sy = tt.get('scale', (1, 1))
    c, s = np.cos(r), np.sin(r)
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], np.float64)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)
    return T @ R @ S


def _skin(pos, nrm, joints, weights, mats):
    w = weights / np.maximum(weights.sum(1, keepdims=True), 1e-9)
    p = np.c_[pos, np.ones(len(pos))]
    out_p = np.zeros((len(pos), 3))
    out_n = np.zeros((len(pos), 3))
    for k in range(joints.shape[1]):
        m = mats[joints[:, k]]
        out_p += w[:, k:k + 1] * np.einsum('nij,nj->ni', m, p)[:, :3]
        if nrm is not None:
            out_n += w[:, k:k + 1] * np.einsum('nij,nj->ni', m[:, :3, :3], nrm)
    return out_p, (out_n if nrm is not None else None)


def vertex_normals(pos, tris):
    n = np.zeros_like(pos)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n
