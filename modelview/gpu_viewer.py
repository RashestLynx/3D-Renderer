"""Real-time viewer (OpenGL 3.3 via moderngl + pyglet) with physically based shading.

    Drop a .glb/.gltf on the window (or press O) to open it; drop a texture map .json to re-texture it.
    Mouse: left-drag orbit, right/middle-drag pan, wheel zoom, double-click re-frames.
    Keys:  F frame   1 2 3 4 front/right/back/left   SPACE turntable   E lighting environment   B background
           +/- exposure   P shadows   R floor reflection   W wireframe   T textures   N normal maps   G flip normal-map green   L lighting
           H help   S screenshot   O open   ESC quit
"""
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moderngl
import pyglet

from . import shaders as S
from .gltf import load

HELP = [
    'Open: drop a .glb/.gltf here, or press O',
    'Left-drag orbit  |  right-drag pan  |  wheel zoom  |  double-click frame',
    'F frame   1-4 front/right/back/left   SPACE turntable',
    'E environment   B background   +/- exposure   P shadows   R floor reflection',
    'W wireframe   T textures   N normal maps   G flip green   L lighting',
    'S screenshot   H hide help   ESC quit',
]
ENVS = ['studio', 'overcast', 'night', 'white studio']
BACKGROUNDS = [('environment', None), ('dark', ((0.20, 0.21, 0.24), (0.05, 0.05, 0.06))),
               ('grey', ((0.55, 0.57, 0.60), (0.32, 0.33, 0.35))), ('black', ((0, 0, 0), (0, 0, 0))),
               ('white', ((0.95, 0.95, 0.96), (0.80, 0.80, 0.82)))]
SHADOW = 2048
CONTACT = 512


def perspective(fov, aspect, near, far):
    f = 1 / math.tan(math.radians(fov) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float64)


def ortho(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)], [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]], np.float64)


def look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, (0, 0, 1))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def tangents(pos, nrm, uv, tris):
    t = np.zeros_like(pos)
    b = np.zeros_like(pos)
    p0, p1, p2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    u0, u1, u2 = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = u1 - u0, u2 - u0
    den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(den) > 1e-12, 1 / np.where(den == 0, 1, den), 0)[:, None]
    ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    fb = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    for k in range(3):
        np.add.at(t, tris[:, k], ft)
        np.add.at(b, tris[:, k], fb)
    t = t - nrm * (t * nrm).sum(1, keepdims=True)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(ln > 1e-9, t / np.maximum(ln, 1e-9), np.array([1.0, 0, 0]))
    w = np.where((np.cross(nrm, t) * b).sum(1) < 0, -1.0, 1.0)
    return np.c_[t, w]


def _font(n):
    for f in ('segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


class Viewer(pyglet.window.Window):
    def __init__(self, path=None, textures=None, size=(1100, 900), visible=True):
        cfg = None
        for c in (pyglet.gl.Config(major_version=3, minor_version=3, sample_buffers=1, samples=4, depth_size=24,
                                   double_buffer=True),
                  pyglet.gl.Config(major_version=3, minor_version=3, depth_size=24, double_buffer=True)):
            try:
                disp = getattr(pyglet, 'display', None) or getattr(pyglet, 'canvas')    # pyglet 2.1 renamed canvas
                cfg = disp.get_display().get_default_screen().get_best_config(c)
                break
            except pyglet.window.NoSuchConfigException:
                continue
        super().__init__(size[0], size[1], caption='modelview', resizable=True, config=cfg, file_drops=True,
                         visible=visible)
        ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
        if os.path.exists(ico):
            self.set_icon(pyglet.image.load(ico))
        ctx = self.ctx = moderngl.create_context()
        self.prog = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS)
        self.depth = ctx.program(vertex_shader=S.DEPTH_VS, fragment_shader=S.DEPTH_FS)
        self.bg = ctx.program(vertex_shader=S.BG_VS, fragment_shader=S.BG_FS)
        self.bg_vao = ctx.vertex_array(self.bg, [])
        self.ground = ctx.program(vertex_shader=S.GROUND_VS, fragment_shader=S.GROUND_FS)
        self.line = ctx.program(vertex_shader=S.LINE_VS, fragment_shader=S.LINE_FS)
        self.ui = ctx.program(vertex_shader=S.UI_VS, fragment_shader=S.UI_FS)
        self.white = ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
        self.flat = ctx.texture((1, 1), 4, bytes([128, 128, 255, 255]))
        self.shadow_tex = ctx.depth_texture((SHADOW, SHADOW))
        self.shadow_tex.compare_func = ''            # sample raw depth (the shader does the PCF compare)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.contact_tex = ctx.depth_texture((CONTACT, CONTACT))
        self.contact_tex.compare_func = ''
        self.contact_tex.repeat_x = self.contact_tex.repeat_y = False
        self.contact_fbo = ctx.framebuffer(depth_attachment=self.contact_tex)
        self.parts, self.grid, self.ground_vao = [], None, None
        self.textures = textures
        self.path = None
        self.opts = dict(tex=True, nmap=True, lit=True, flip_g=False, wire=False, grid=True, help=True, spin=False,
                         bg=1, env=0, shadows=True, floor=True)
        self.exposure = 1.0
        self.yaw, self.pitch, self.dist, self.pan = 35.0, 12.0, 3.0, np.zeros(3)
        self.target, self.radius, self.extent = np.zeros(3), 1.0, np.ones(3)
        self.floor = 0.0
        self.hide_hud = False
        self.toast, self.toast_until = '', 0
        self.hud_key, self.hud = None, None
        self.info = 'no model'
        # yaw/pitch/dist/pan above are where the camera is heading; `cur` is where it is - it glides there each
        # frame, and a flicked orbit keeps turning for a moment (spin velocity in degrees/second)
        self.cur = dict(yaw=self.yaw, pitch=self.pitch, dist=self.dist, pan=self.pan.copy())
        self.spin_vel, self.dragging, self._drag_t = 0.0, False, time.time()
        self.set_vsync(True)
        pyglet.clock.schedule(self._tick)            # every frame, at the display's refresh rate
        if path:
            self.open(path)

    # ---------- loading
    def open(self, path):
        self._toast('loading %s ...' % os.path.basename(path))
        self.on_draw()
        self.flip()
        t0 = time.time()
        try:
            model = load(path, self.textures)
        except Exception as e:
            self._toast('could not open %s: %s' % (os.path.basename(path), e), 8)
            return
        for p in self.parts:
            p['vao'].release()
            p['dvao'].release()
        self.parts = []
        cache = {}

        def gltex(arr):
            if arr is None:
                return None
            k = id(arr)
            if k not in cache:
                a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                tx = self.ctx.texture((a.shape[1], a.shape[0]), 4, np.ascontiguousarray(a).tobytes())
                tx.build_mipmaps()
                tx.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tx.anisotropy = 8.0
                cache[k] = tx
            return cache[k]
        for part in model.parts:
            m = part.material
            uv2 = part.uvs2 if part.uvs2 is not None else part.uvs
            tan = tangents(part.positions, part.normals, part.uvs, part.triangles)
            data = np.hstack([part.positions, part.normals, part.uvs, uv2, tan]).astype('f4')
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(part.triangles.astype('i4').tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f 2f 2f 4f', 'in_pos', 'in_nrm', 'in_uv', 'in_uv2', 'in_tan')], ibo)
            dvao = self.ctx.vertex_array(self.depth, [(vbo, '3f 12x 2f 24x', 'in_pos', 'in_uv')], ibo)
            cut = m.alpha_cutoff if m.alpha_mode == 'MASK' else -1.0
            blend = m.alpha_mode == 'BLEND' or m.transmission > 0
            if m.texture is not None and cut < 0 and not blend and m.texture[..., 3].min() < 0.5:
                cut = 0.45                               # cut-out alpha in a texture with no alpha flag (hair, leaves)
            self.parts.append(dict(vao=vao, dvao=dvao, m=m, blend=blend, cut=cut,
                                   centre=part.positions.mean(0),
                                   c=gltex(m.texture), n=gltex(m.normal_texture), mr=gltex(m.mr_texture),
                                   e=gltex(m.emissive_texture), o=gltex(m.occlusion_texture),
                                   o_uv2=m.uv_sets.get('occlusion', 0) == 1 and part.uvs2 is not None,
                                   has_uv2=part.uvs2 is not None))
        lo, hi = model.bounds()
        self.target = (lo + hi) / 2
        self.radius = float(np.linalg.norm(hi - lo) / 2) or 1.0
        self.extent = np.maximum(hi - lo, 1e-3)
        self.floor = float(lo[1])
        self._make_ground(lo, hi)
        self.path = path
        self.frame(keep_angle=False)
        self.snap()
        ntex = sum(1 for p in self.parts if p['c'] is not None)
        self.info = '%s   %d triangles   %d parts   %d textured   loaded in %.1fs' % (
            os.path.basename(path), model.triangle_count(), len(self.parts), ntex, time.time() - t0)
        self.set_caption('modelview - ' + os.path.basename(path))
        self._toast('opened ' + os.path.basename(path))

    def _make_ground(self, lo, hi):
        size = max(hi[0] - lo[0], hi[2] - lo[2], (hi[1] - lo[1]) * 0.8) * 1.6
        step = 10 ** math.floor(math.log10(size / 10))
        if size / step > 40:
            step *= 5
        elif size / step > 20:
            step *= 2
        n = int(size / step / 2) + 1
        cx, cz = round(self.target[0] / step) * step, round(self.target[2] / step) * step
        lines = []
        for i in range(-n, n + 1):
            f = 1 - abs(i) / (n + 1)
            for a, b in (((cx + i * step, cz - n * step), (cx + i * step, cz + n * step)),
                         ((cx - n * step, cz + i * step), (cx + n * step, cz + i * step))):
                lines += [a[0], f * 0.6, a[1], b[0], f * 0.6, b[1]]
        if self.grid:
            self.grid[0].release()
        buf = self.ctx.buffer(np.array(lines, 'f4').tobytes())
        self.grid = (buf, self.ctx.vertex_array(self.line, [(buf, '3f', 'in_pos')]))
        r = self.radius * 3
        q = np.array([self.target[0] - r, self.target[2] - r, self.target[0] + r, self.target[2] - r,
                      self.target[0] - r, self.target[2] + r, self.target[0] + r, self.target[2] + r], 'f4')
        gb = self.ctx.buffer(q.tobytes())
        self.ground_vao = self.ctx.vertex_array(self.ground, [(gb, '2f', 'in_xz')])

    def frame(self, keep_angle=True):
        """Fit the model's bounding sphere to the window from any angle, filling most of it."""
        w, h = self.get_framebuffer_size()
        aspect = w / max(h, 1)
        half = math.tan(math.radians(30) / 2)
        self.dist = self.radius * 0.82 / (half * min(1.0, aspect))
        self.pan = np.zeros(3)
        if not keep_angle:
            self.yaw, self.pitch = 35.0, 12.0

    # ---------- drawing
    def snap(self):
        """Jump the camera to where it is heading (no glide): new models, screenshots."""
        self.cur = dict(yaw=self.yaw, pitch=self.pitch, dist=self.dist, pan=self.pan.copy())

    def _matrices(self):
        c = self.cur
        a, b = math.radians(c['yaw']), math.radians(c['pitch'])
        tgt = self.target + c['pan']
        eye = tgt + c['dist'] * np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        view = look_at(eye, tgt)
        w, h = self.get_framebuffer_size()
        proj = perspective(30, w / max(h, 1), c['dist'] * 0.01, c['dist'] * 100)
        return proj, view, eye

    def _light(self):
        key = np.array([0.45, 0.85, 0.35])
        key /= np.linalg.norm(key)
        r = self.radius * 1.05
        eye = self.target + key * r * 3
        lv = look_at(eye, self.target)
        return ortho(-r, r, -r, r, r * 0.5, r * 5.5) @ lv, key

    def _set_mat(self, name, m):
        self.prog[name].write(m.T.astype('f4').tobytes())

    def _contact(self):
        """Straight-down orthographic view of the model for the soft contact shadow under it."""
        r = self.radius * 1.3
        top = self.target + np.array([0, self.extent[1] / 2 + r, 0])
        lv = look_at(top, top - np.array([0, 1, 0]), up=(0, 0, -1))
        return ortho(-r, r, -r, r, 0.0, self.extent[1] + r * 1.01) @ lv

    def _depth_pass(self, fbo, size, mat):
        ctx = self.ctx
        fbo.use()
        ctx.viewport = (0, 0, size, size)
        ctx.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        self.depth['light_mvp'].write(mat.T.astype('f4').tobytes())
        self.depth['tex_c'].value = 0
        for p in self.parts:
            if p['blend']:
                continue
            (p['c'] or self.white).use(0)
            self.depth['has_c'].value = p['c'] is not None
            self.depth['cutoff'].value = p['cut']
            p['dvao'].render()

    def _setup_prog(self, mvp, light_mvp, eye, key, shadows):
        p = self.prog
        self._set_mat('mvp', mvp)
        self._set_mat('light_mvp', light_mvp)
        p['cam'].value = tuple(eye)
        p['key_dir'].value = tuple(key)
        p['exposure'].value = self.exposure
        p['env_mode'].value = float(self.opts['env'])
        p['shadows'].value = shadows
        p['shadow_texel'].value = (1 / SHADOW, 1 / SHADOW)
        p['floor_y'].value = self.floor
        for k, u in (('use_tex', 'tex'), ('use_nmap', 'nmap'), ('lit', 'lit'), ('flip_g', 'flip_g')):
            p[k].value = self.opts[u]
        for i, nm in enumerate(('tex_c', 'tex_n', 'tex_mr', 'tex_e', 'tex_o', 'shadow_map')):
            p[nm].value = i
        self.shadow_tex.use(5)

    def _draw_model(self, eye):
        ctx = self.ctx
        opaque = [q for q in self.parts if not q['blend']]
        clear = sorted((q for q in self.parts if q['blend']), key=lambda q: -np.linalg.norm(q['centre'] - eye))
        ctx.disable(moderngl.BLEND)
        for q in opaque:
            self._draw_part(q)
        if clear:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.depth_mask = False
            for q in clear:
                self._draw_part(q)
            ctx.depth_mask = True

    def on_draw(self):
        ctx = self.ctx
        w, h = self.get_framebuffer_size()
        proj, view, eye = self._matrices()
        mvp = proj @ view
        light_mvp, key = self._light()
        contact_mvp = self._contact()
        lit_shadows = self.opts['shadows'] and self.opts['lit'] and bool(self.parts)
        # 1. shadow maps: the key light's, and a straight-down one for the contact shadow
        if lit_shadows:
            self._depth_pass(self.shadow_fbo, SHADOW, light_mvp)
            self._depth_pass(self.contact_fbo, CONTACT, contact_mvp)
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0, 0, 0, 1, depth=1.0)
        # 2. background
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        name, grad = BACKGROUNDS[self.opts['bg']]
        self.bg['show_env'].value = grad is None
        self.bg['top'].value, self.bg['bottom'].value = grad if grad else ((0, 0, 0), (0, 0, 0))
        self.bg['inv_vp'].write(np.linalg.inv(mvp).T.astype('f4').tobytes())
        self.bg['env_mode'].value = float(self.opts['env'])
        self.bg['exposure'].value = self.exposure
        self.bg_vao.render(moderngl.TRIANGLES, vertices=3)
        if not self.parts:
            self._draw_hud(w, h)
            return
        light_bg = name in ('white',) or (grad is None and self.opts['env'] == 3)
        above = eye[1] > self.floor
        ctx.wireframe = self.opts['wire']
        # 3. glossy floor: the model mirrored below it, then the floor drawn over that, see-through near the model
        if self.opts['floor'] and above:
            ctx.enable(moderngl.DEPTH_TEST)
            self._setup_prog(mvp, light_mvp, eye, key, False)
            self.prog['mirror'].value = True
            self._draw_model(eye)
            self.prog['mirror'].value = False
            # no depth clear: from above the floor the mirror image always lies beyond it, so the real model
            # still wins the depth test (and a clear here would wipe the background too)
        if above:
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.shadow_tex.use(5)
            self.contact_tex.use(6)
            g = self.ground
            g['mvp'].write(mvp.T.astype('f4').tobytes())
            g['light_mvp'].write(light_mvp.T.astype('f4').tobytes())
            g['contact_mvp'].write(contact_mvp.T.astype('f4').tobytes())
            g['floor_y'].value = self.floor
            g['shadow_map'].value, g['contact_map'].value = 5, 6
            g['shadow_texel'].value = (1 / SHADOW, 1 / SHADOW)
            g['contact_texel'].value = (1 / CONTACT, 1 / CONTACT)
            g['shadows'].value = lit_shadows
            g['centre'].value = (float(self.target[0]), float(self.target[2]))
            g['radius'].value = self.radius * 3
            bottom = grad[1] if grad else (0.02, 0.02, 0.024)
            g['tint'].value = tuple(float(x) for x in bottom)
            g['reflect_amount'].value = (0.0 if not self.opts['floor'] else (0.25 if light_bg else 0.42))
            self.ground_vao.render(moderngl.TRIANGLE_STRIP)
            if self.opts['grid'] and self.grid:
                m = mvp @ np.array([[1, 0, 0, 0], [0, 1, 0, self.floor], [0, 0, 1, 0], [0, 0, 0, 1]], float)
                self.line['mvp'].write(m.T.astype('f4').tobytes())
                self.line['color'].value = (0, 0, 0, 0.3) if light_bg else (1, 1, 1, 0.08)
                self.grid[1].render(moderngl.LINES)
        # 4. the model: opaque first, then see-through parts back to front
        ctx.enable(moderngl.DEPTH_TEST)
        self._setup_prog(mvp, light_mvp, eye, key, lit_shadows)
        self.prog['mirror'].value = False
        self._draw_model(eye)
        ctx.wireframe = False
        if not self.hide_hud:
            self._draw_hud(w, h)

    def _draw_part(self, q):
        p, m = self.prog, q['m']
        for i, k in enumerate(('c', 'n', 'mr', 'e', 'o')):
            (q[k] or (self.flat if k == 'n' else self.white)).use(i)
            p['has_' + k].value = q[k] is not None
        p['o_uv2'].value = q['o_uv2']
        for k, slot in (('c', 'color'), ('n', 'normal'), ('mr', 'mr'), ('e', 'emissive'), ('o', 'occlusion')):
            tf = m.uv_transforms.get(slot)
            p['tf_' + k].write((tf if tf is not None else np.eye(3)).T.astype('f4').tobytes())
            if k != 'o':
                p['uv2_' + k].value = m.uv_sets.get(slot, 0) == 1 and q['has_uv2']
        p['base'].value = tuple(m.base_color)
        p['metallic'].value = m.metallic
        p['roughness'].value = m.roughness
        p['emissive'].value = tuple(m.emissive)
        p['clearcoat'].value = m.clearcoat
        p['cc_rough'].value = m.clearcoat_roughness
        p['transmission'].value = m.transmission
        p['cutoff'].value = q['cut']
        p['blend'].value = q['blend']
        q['vao'].render()

    def _draw_hud(self, w, h):
        lines = []
        if self.opts['help']:
            lines += HELP + ['']
        lines.append(self.info)
        flags = ['env: ' + ENVS[self.opts['env']], 'bg: ' + BACKGROUNDS[self.opts['bg']][0],
                 'exposure %.2f' % self.exposure] + \
                ([] if self.opts['tex'] else ['NO textures']) + ([] if self.opts['nmap'] else ['no normal maps']) + \
                ([] if self.opts['lit'] else ['unlit']) + ([] if self.opts['shadows'] else ['no shadows']) + \
                (['wireframe'] if self.opts['wire'] else []) + (['turntable'] if self.opts['spin'] else []) + \
                (['green flipped'] if self.opts['flip_g'] else [])
        lines.append('   '.join(flags))
        toast = self.toast if time.time() < self.toast_until else ''
        key = (tuple(lines), toast, w, h)
        if key != self.hud_key:
            self.hud_key = key
            f = _font(15)
            pad, lh = 10, 21
            tw = max(int(f.getlength(s)) for s in lines + [toast or ' ']) + pad * 2
            th = lh * (len(lines) + (2 if toast else 0)) + pad * 2
            im = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((0, 0, tw - 1, th - 1), 8, fill=(10, 10, 12, 150))
            y = pad
            for s in lines:
                d.text((pad, y), s, font=f, fill=(225, 225, 230, 255))
                y += lh
            if toast:
                d.text((pad, y + lh // 2), toast, font=f, fill=(130, 230, 150, 255))
            if self.hud:
                for o in self.hud:
                    o.release()
            tex = self.ctx.texture(im.size, 4, im.transpose(Image.FLIP_TOP_BOTTOM).tobytes())
            x0, y0 = 12, h - th - 12                     # top-left (GL y goes up)
            q = np.array([x0, y0, 0, 0, x0 + tw, y0, 1, 0, x0, y0 + th, 0, 1, x0 + tw, y0 + th, 1, 1], 'f4')
            buf = self.ctx.buffer(q.tobytes())
            self.hud = (tex, buf, self.ctx.vertex_array(self.ui, [(buf, '2f 2f', 'in_pos', 'in_uv')]))
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ui['screen'].value = (w, h)
        self.hud[0].use(0)
        self.ui['tex'].value = 0
        self.hud[2].render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _tick(self, dt):
        dt = min(dt, 0.1)
        if self.opts['spin']:
            self.yaw += dt * 25
        if not self.dragging and abs(self.spin_vel) > 0.5:     # inertia after a flick
            self.yaw += self.spin_vel * dt
            self.spin_vel *= math.exp(-dt * 3.5)
        k = 1 - math.exp(-dt * 14)                              # exponential ease toward the goal
        c = self.cur
        c['yaw'] += (self.yaw - c['yaw']) * k
        c['pitch'] += (self.pitch - c['pitch']) * k
        c['dist'] *= (self.dist / c['dist']) ** k
        c['pan'] = c['pan'] + (self.pan - c['pan']) * k

    def _toast(self, s, secs=3):
        self.toast, self.toast_until = s, time.time() + secs

    # ---------- input
    def on_resize(self, w, h):
        self.ctx.viewport = (0, 0, *self.get_framebuffer_size())
        return pyglet.event.EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        if buttons & pyglet.window.mouse.LEFT and not mods & pyglet.window.key.MOD_SHIFT:
            now = time.time()
            self.dragging = True
            self.yaw -= dx * 0.4
            self.pitch = max(-89, min(89, self.pitch - dy * 0.4))
            step = max(now - self._drag_t, 1 / 240)
            self.spin_vel = 0.7 * self.spin_vel + 0.3 * (-dx * 0.4 / step)
            self._drag_t = now
        else:
            a, b = math.radians(self.yaw), math.radians(self.pitch)
            right = np.array([math.cos(a), 0, -math.sin(a)])
            up = np.array([-math.sin(a) * math.sin(b), math.cos(b), -math.cos(a) * math.sin(b)])
            s = self.dist * 0.0012
            self.pan -= (right * dx + up * dy) * s

    def on_mouse_scroll(self, x, y, sx, sy):
        self.dist *= 0.88 ** sy

    def on_mouse_release(self, x, y, button, mods):
        self.dragging = False
        if time.time() - self._drag_t > 0.08:          # held still before letting go: no flick
            self.spin_vel = 0.0

    def on_mouse_press(self, x, y, button, mods):
        self.spin_vel = 0.0
        now = time.time()
        if button == pyglet.window.mouse.LEFT and now - getattr(self, '_click', 0) < 0.3:
            self.frame()
        self._click = now

    def on_key_press(self, sym, mods):
        k = pyglet.window.key
        o = self.opts
        toggles = {k.T: ('tex', 'textures'), k.N: ('nmap', 'normal maps'), k.L: ('lit', 'lighting'),
                   k.G: ('flip_g', 'normal-map green flipped'), k.W: ('wire', 'wireframe'), k.SPACE: ('spin', 'turntable'),
                   k.H: ('help', 'help'), k.F1: ('help', 'help'), k.P: ('shadows', 'shadows'),
                   k.R: ('floor', 'reflective floor')}
        if sym in toggles:
            key, name = toggles[sym]
            o[key] = not o[key]
            self._toast('%s %s' % (name, 'on' if o[key] else 'off'))
        elif sym == k.F:
            self.frame()
        elif sym in (k._1, k._2, k._3, k._4):
            self.yaw, self.pitch = {k._1: 0, k._2: 90, k._3: 180, k._4: 270}[sym], 8.0
            self.pan = np.zeros(3)
        elif sym == k.B:
            o['bg'] = (o['bg'] + 1) % len(BACKGROUNDS)
        elif sym == k.E:
            o['env'] = (o['env'] + 1) % len(ENVS)
            self._toast('environment: ' + ENVS[o['env']])
        elif sym in (k.PLUS, k.EQUAL, k.NUM_ADD):
            self.exposure *= 1.15
        elif sym in (k.MINUS, k.NUM_SUBTRACT):
            self.exposure /= 1.15
        elif sym == k.S:
            self.screenshot()
        elif sym == k.O:
            self.ask_open()
        elif sym == k.ESCAPE:
            self.close()
        return pyglet.event.EVENT_HANDLED

    def on_file_drop(self, x, y, paths):
        for p in paths:
            if p.lower().endswith('.json'):
                try:
                    self.textures = json.load(open(p))
                except Exception as e:
                    self._toast('bad texture map: %s' % e, 6)
                    continue
                if self.path:
                    self.open(self.path)
                self._toast('texture map: ' + os.path.basename(p))
            elif p.lower().endswith(('.glb', '.gltf')):
                self.open(p)
            else:
                self._toast('not a model: ' + os.path.basename(p), 5)

    def ask_open(self):
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        p = filedialog.askopenfilename(title='Open model', filetypes=[('glTF models', '*.glb *.gltf'), ('All', '*.*')])
        r.destroy()
        if p:
            self.open(p)

    def grab(self):
        """The current frame (without the on-screen panel) as a PIL image."""
        w, h = self.get_framebuffer_size()
        self.snap()
        self.hide_hud = True
        self.on_draw()
        data = self.ctx.screen.read(components=3)
        self.hide_hud = False
        return Image.frombytes('RGB', (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)

    def screenshot(self):
        img = self.grab()
        folder = os.path.join(os.path.expanduser('~'), 'Pictures', 'modelview')
        os.makedirs(folder, exist_ok=True)
        name = os.path.splitext(os.path.basename(self.path or 'view'))[0]
        out = os.path.join(folder, '%s_%s.png' % (name, time.strftime('%Y%m%d_%H%M%S')))
        img.save(out)
        self._toast('screenshot saved: ' + out, 5)


def run(path=None, textures=None):
    Viewer(path, textures)
    pyglet.app.run()
