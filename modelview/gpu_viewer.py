"""Real-time viewer (OpenGL 3.3 via moderngl + pyglet).

    Drop a .glb/.gltf on the window (or press O) to open it; drop a texture map .json to re-texture it.
    Mouse: left-drag orbit, right/middle-drag pan, wheel zoom, double-click re-frames.
    Keys:  F frame   1 2 3 4 front/right/back/left   SPACE turntable   W wireframe   T textures   N normal maps
           G flip normal-map green   L lighting   B background   F1/H help   S screenshot   O open   ESC quit
"""
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moderngl
import pyglet

from .gltf import load

HELP = [
    'Open: drop a .glb/.gltf here, or press O',
    'Left-drag orbit  |  right-drag pan  |  wheel zoom  |  double-click frame',
    'F frame   1-4 front/right/back/left   SPACE turntable',
    'W wireframe   T textures   N normal maps   G flip green   L lighting',
    'B background   S screenshot   H hide help   ESC quit',
]

VS = """
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec4 in_tan;
uniform mat4 mvp; uniform mat4 view;
out vec3 v_n; out vec3 v_t; out float v_w; out vec3 v_vpos; out vec2 v_uv;
void main() {
    v_n = mat3(view) * in_nrm;
    v_t = mat3(view) * in_tan.xyz;
    v_w = in_tan.w;
    v_vpos = (view * vec4(in_pos, 1.0)).xyz;
    v_uv = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FS = """
#version 330
in vec3 v_n; in vec3 v_t; in float v_w; in vec3 v_vpos; in vec2 v_uv;
uniform sampler2D tex_c; uniform sampler2D tex_n;
uniform bool has_c; uniform bool has_n; uniform bool use_tex; uniform bool use_nmap; uniform bool lit; uniform bool flip_g;
uniform vec4 base; uniform float cutoff;
out vec4 f_color;
void main() {
    vec4 c = (has_c && use_tex) ? texture(tex_c, v_uv) * base : (use_tex ? base : vec4(0.72, 0.72, 0.74, 1.0));
    if (c.a < cutoff) discard;
    vec3 n = normalize(v_n);
    if (!gl_FrontFacing) n = -n;
    if (has_n && use_nmap) {
        vec3 t = normalize(v_t - n * dot(n, v_t));
        vec3 b = cross(n, t) * (v_w < 0.0 ? -1.0 : 1.0);
        vec3 tn = texture(tex_n, v_uv).xyz * 2.0 - 1.0;
        if (flip_g) tn.y = -tn.y;
        n = normalize(mat3(t, b, n) * tn);
    }
    vec3 albedo = pow(c.rgb, vec3(2.2));
    if (!lit) { f_color = vec4(c.rgb, 1.0); return; }
    vec3 v = normalize(-v_vpos);
    vec3 col = albedo * mix(vec3(0.10, 0.09, 0.08), vec3(0.30, 0.32, 0.36), n.y * 0.5 + 0.5);
    vec3 L[3] = vec3[](normalize(vec3(-0.5, 0.6, 0.8)), normalize(vec3(0.7, 0.2, 0.4)), normalize(vec3(0.0, 0.5, -1.0)));
    float K[3] = float[](1.0, 0.38, 0.55);
    for (int i = 0; i < 3; i++) {
        float d = max(dot(n, L[i]), 0.0);
        vec3 h = normalize(L[i] + v);
        col += K[i] * (albedo * d + vec3(0.06) * pow(max(dot(n, h), 0.0), 40.0) * d);
    }
    col = col / (1.0 + col * 0.12);
    f_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""

BG_VS = """
#version 330
out vec2 uv;
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.999, 1.0); }
"""
BG_FS = """
#version 330
in vec2 uv; uniform vec3 top; uniform vec3 bottom; out vec4 f_color;
void main() { vec2 d = uv - vec2(0.5, 0.55); float vig = 1.0 - dot(d, d) * 0.6;
              f_color = vec4(mix(bottom, top, uv.y) * vig, 1.0); }
"""
LINE_VS = """
#version 330
in vec3 in_pos; uniform mat4 mvp; out float fade;
void main() { fade = in_pos.y; gl_Position = mvp * vec4(in_pos.x, 0.0, in_pos.z, 1.0); }
"""
LINE_FS = """
#version 330
in float fade; uniform vec4 color; out vec4 f_color;
void main() { f_color = vec4(color.rgb, color.a * fade); }
"""
UI_VS = """
#version 330
in vec2 in_pos; in vec2 in_uv; uniform vec2 screen; out vec2 uv;
void main() { uv = in_uv; gl_Position = vec4(in_pos / screen * 2.0 - 1.0, 0.0, 1.0); }
"""
UI_FS = """
#version 330
in vec2 uv; uniform sampler2D tex; out vec4 f_color;
void main() { f_color = texture(tex, uv); }
"""

BACKGROUNDS = [((0.20, 0.21, 0.24), (0.05, 0.05, 0.06)), ((0.55, 0.57, 0.60), (0.32, 0.33, 0.35)),
               ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.92, 0.92, 0.93), (0.78, 0.78, 0.80))]


def perspective(fov, aspect, near, far):
    f = 1 / math.tan(math.radians(fov) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float64)


def look_at(eye, target, up=(0, 1, 0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-9 else np.array([1.0, 0, 0])
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
    def __init__(self, path=None, textures=None):
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
        super().__init__(1100, 900, caption='modelview', resizable=True, config=cfg, file_drops=True)
        ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
        if os.path.exists(ico):
            self.set_icon(pyglet.image.load(ico))
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.bg = self.ctx.program(vertex_shader=BG_VS, fragment_shader=BG_FS)
        self.bg_vao = self.ctx.vertex_array(self.bg, [])
        self.line = self.ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.ui = self.ctx.program(vertex_shader=UI_VS, fragment_shader=UI_FS)
        self.white = self.ctx.texture((1, 1), 4, bytes([255, 255, 255, 255]))
        self.flat = self.ctx.texture((1, 1), 4, bytes([128, 128, 255, 255]))
        self.parts, self.grid = [], None
        self.textures = textures
        self.path = None
        self.opts = dict(tex=True, nmap=True, lit=True, flip_g=False, wire=False, grid=True, help=True, spin=False, bg=0)
        self.yaw, self.pitch, self.dist, self.pan = 30.0, 8.0, 3.0, np.zeros(3)
        self.target, self.radius, self.extent = np.zeros(3), 1.0, np.ones(3)
        self.floor = 0.0
        self.hide_hud = False
        self.toast, self.toast_until = '', 0
        self.hud_key, self.hud = None, None
        self.last = time.time()
        self.info = 'no model'
        pyglet.clock.schedule_interval(self._tick, 1 / 60)
        if path:
            self.open(path)

    # ---------- loading
    def open(self, path):
        self._toast('loading %s ...' % os.path.basename(path))
        self.on_draw()
        self.flip()
        try:
            t = time.time()
            model = load(path, self.textures)
        except Exception as e:
            self._toast('could not open %s: %s' % (os.path.basename(path), e), 8)
            return
        for p in self.parts:
            p['vao'].release()
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
            tan = tangents(part.positions, part.normals, part.uvs, part.triangles)
            data = np.hstack([part.positions, part.normals, part.uvs, tan]).astype('f4')
            vbo = self.ctx.buffer(data.tobytes())
            ibo = self.ctx.buffer(part.triangles.astype('i4').tobytes())
            vao = self.ctx.vertex_array(self.prog, [(vbo, '3f 3f 2f 4f', 'in_pos', 'in_nrm', 'in_uv', 'in_tan')], ibo)
            cut = m.alpha_cutoff if m.alpha_mode == 'MASK' else -1.0
            if m.texture is not None and (m.alpha_mode == 'BLEND' or (cut < 0 and m.texture[..., 3].min() < 0.5)):
                cut = 0.45
            self.parts.append(dict(vao=vao, c=gltex(m.texture), n=gltex(m.normal_texture), base=m.base_color, cut=cut))
        lo, hi = model.bounds()
        self.target = (lo + hi) / 2
        self.radius = float(np.linalg.norm(hi - lo) / 2) or 1.0
        self.extent = np.maximum(hi - lo, 1e-3)
        self.floor = float(lo[1])
        self._make_grid(lo, hi)
        self.path = path
        self.frame(keep_angle=False)
        ntex = sum(1 for p in self.parts if p['c'] is not None)
        self.info = '%s   %d triangles   %d parts   %d textured   loaded in %.1fs' % (
            os.path.basename(path), model.triangle_count(), len(self.parts), ntex, time.time() - t)
        self.set_caption('modelview - ' + os.path.basename(path))
        self._toast('opened ' + os.path.basename(path))

    def _make_grid(self, lo, hi):
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

    def frame(self, keep_angle=True):
        """Distance at which the whole model fits the window with a small margin."""
        w, h = self.get_framebuffer_size()
        aspect = w / max(h, 1)
        half = math.tan(math.radians(30) / 2)
        sx, sy, sz = self.extent
        horiz = max(sx, sz) / 2
        self.dist = max(sy / 2 / half, horiz / half / aspect) * 1.18 + max(sx, sz) / 2
        self.pan = np.zeros(3)
        if not keep_angle:
            self.yaw, self.pitch = 30.0, 8.0

    # ---------- drawing
    def _matrices(self):
        a, b = math.radians(self.yaw), math.radians(self.pitch)
        tgt = self.target + self.pan
        eye = tgt + self.dist * np.array([math.sin(a) * math.cos(b), math.sin(b), math.cos(a) * math.cos(b)])
        view = look_at(eye, tgt)
        w, h = self.get_framebuffer_size()
        proj = perspective(30, w / max(h, 1), self.dist * 0.01, self.dist * 100)
        return proj, view

    def on_draw(self):
        ctx = self.ctx
        w, h = self.get_framebuffer_size()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0, 0, 0, 1, depth=1.0)
        top, bottom = BACKGROUNDS[self.opts['bg']]
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.bg['top'].value, self.bg['bottom'].value = top, bottom
        self.bg_vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
        proj, view = self._matrices()
        mvp = proj @ view
        if self.parts:
            p = self.prog
            p['mvp'].write(mvp.T.astype('f4').tobytes())
            p['view'].write(view.T.astype('f4').tobytes())
            for k, u in (('use_tex', 'tex'), ('use_nmap', 'nmap'), ('lit', 'lit'), ('flip_g', 'flip_g')):
                p[k].value = self.opts[u]
            p['tex_c'].value, p['tex_n'].value = 0, 1
            ctx.wireframe = self.opts['wire']
            for part in self.parts:
                (part['c'] or self.white).use(0)
                (part['n'] or self.flat).use(1)
                p['has_c'].value = part['c'] is not None
                p['has_n'].value = part['n'] is not None
                p['base'].value = tuple(part['base'])
                p['cutoff'].value = part['cut']
                part['vao'].render()
            ctx.wireframe = False
            if self.opts['grid'] and self.grid:
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                m = mvp @ np.array([[1, 0, 0, 0], [0, 1, 0, self.floor], [0, 0, 1, 0], [0, 0, 0, 1]], float)
                self.line['mvp'].write(m.T.astype('f4').tobytes())
                light = self.opts['bg'] == 3
                self.line['color'].value = (0, 0, 0, 0.35) if light else (1, 1, 1, 0.16)
                self.grid[1].render(moderngl.LINES)
        if not self.hide_hud:
            self._draw_hud(w, h)

    def _draw_hud(self, w, h):
        lines = []
        if self.opts['help']:
            lines += HELP + ['']
        lines.append(self.info)
        flags = ['textures' if self.opts['tex'] else 'NO textures', 'normal maps' if self.opts['nmap'] else 'no normal maps',
                 'lit' if self.opts['lit'] else 'unlit'] + (['wireframe'] if self.opts['wire'] else []) + \
                (['turntable'] if self.opts['spin'] else []) + (['green flipped'] if self.opts['flip_g'] else [])
        lines.append('   '.join(flags))
        toast = self.toast if time.time() < self.toast_until else ''
        key = (tuple(lines), toast, w, h)
        if key != self.hud_key:
            self.hud_key = key
            f = _font(15)
            pad = 10
            lh = 21
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
                self.hud[0].release()
                self.hud[1].release()
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
        if self.opts['spin']:
            self.yaw += dt * 25

    def _toast(self, s, secs=3):
        self.toast, self.toast_until = s, time.time() + secs

    # ---------- input
    def on_resize(self, w, h):
        self.ctx.viewport = (0, 0, *self.get_framebuffer_size())
        return pyglet.event.EVENT_HANDLED

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        if buttons & pyglet.window.mouse.LEFT and not mods & pyglet.window.key.MOD_SHIFT:
            self.yaw -= dx * 0.4
            self.pitch = max(-89, min(89, self.pitch - dy * 0.4))
        else:
            a, b = math.radians(self.yaw), math.radians(self.pitch)
            right = np.array([math.cos(a), 0, -math.sin(a)])
            up = np.array([-math.sin(a) * math.sin(b), math.cos(b), -math.cos(a) * math.sin(b)])
            s = self.dist * 0.0012
            self.pan -= (right * dx + up * dy) * s

    def on_mouse_scroll(self, x, y, sx, sy):
        self.dist *= 0.88 ** sy

    def on_mouse_press(self, x, y, button, mods):
        now = time.time()
        if button == pyglet.window.mouse.LEFT and now - getattr(self, '_click', 0) < 0.3:
            self.frame()
        self._click = now

    def on_key_press(self, sym, mods):
        k = pyglet.window.key
        o = self.opts
        toggles = {k.T: ('tex', 'textures'), k.N: ('nmap', 'normal maps'), k.L: ('lit', 'lighting'),
                   k.G: ('flip_g', 'normal-map green flipped'), k.W: ('wire', 'wireframe'), k.SPACE: ('spin', 'turntable'),
                   k.H: ('help', 'help'), k.F1: ('help', 'help')}
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

    def screenshot(self):
        w, h = self.get_framebuffer_size()
        self.hide_hud = True
        self.on_draw()
        data = self.ctx.screen.read(components=3)
        self.hide_hud = False
        img = Image.frombytes('RGB', (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
        folder = os.path.join(os.path.expanduser('~'), 'Pictures', 'modelview')
        os.makedirs(folder, exist_ok=True)
        name = os.path.splitext(os.path.basename(self.path or 'view'))[0]
        out = os.path.join(folder, '%s_%s.png' % (name, time.strftime('%Y%m%d_%H%M%S')))
        img.save(out)
        self._toast('screenshot saved: ' + out, 5)


def run(path=None, textures=None):
    Viewer(path, textures)
    pyglet.app.run()
