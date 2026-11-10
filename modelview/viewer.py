"""Desktop app (Tk): open a model, orbit around it, swap texture sets, save screenshots.

Mouse: left-drag orbits, right-drag pans, wheel zooms. Keys: R reset, 1/2/3/4 front/right/back/left, S save, O open.
A quick half-resolution frame is drawn while the mouse moves; the full-quality frame follows when it stops.
"""
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from .gltf import load
from .raster import Camera, Style, render

APP = 'modelview'


class App:
    def __init__(self, root, path=None, textures=None):
        self.root = root
        self.model = None
        self.path = None
        self.textures = textures
        self.cam = Camera()
        self.drag = None
        self.job = None
        self.img = None
        self.busy = False
        self.pending = None
        self.quality = tk.BooleanVar(value=True)
        root.title(APP)
        root.configure(bg='#111')
        self._menus()
        self.canvas = tk.Label(root, bg='#141417', fg='#888', font=('Segoe UI', 13),
                               text='File > Open model...  (or drag a .glb onto modelview.exe)')
        self.canvas.pack(fill='both', expand=True)
        self.status = tk.Label(root, anchor='w', fg='#bbb', bg='#1b1b1f', font=('Segoe UI', 9),
                               text='left-drag: orbit   right-drag: pan   wheel: zoom   R: reset   1-4: views   S: save')
        self.status.pack(fill='x')
        c = self.canvas
        c.bind('<ButtonPress-1>', lambda e: self._press(e, 'orbit'))
        c.bind('<ButtonPress-3>', lambda e: self._press(e, 'pan'))
        c.bind('<B1-Motion>', self._motion)
        c.bind('<B3-Motion>', self._motion)
        c.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', None))
        c.bind('<ButtonRelease-3>', lambda e: setattr(self, 'drag', None))
        c.bind('<MouseWheel>', self._wheel)
        c.bind('<Configure>', lambda e: self._later(150))
        root.bind('<Key>', self._key)
        self.done = queue.Queue()
        self._poll()
        if path:
            root.after(100, lambda: self.open(path))

    # ---- menus
    def _menus(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label='Open model...', accelerator='O', command=self.ask_open)
        f.add_command(label='Load texture map (JSON)...', command=self.ask_textures)
        f.add_command(label='Clear texture map', command=lambda: self._set_textures(None))
        f.add_separator()
        f.add_command(label='Save screenshot...', accelerator='S', command=self.save)
        f.add_separator()
        f.add_command(label='Exit', command=self.root.destroy)
        m.add_cascade(label='File', menu=f)
        v = tk.Menu(m, tearoff=0)
        for label, yaw, key in (('Front', 0, '1'), ('Right', 90, '2'), ('Back', 180, '3'), ('Left', 270, '4')):
            v.add_command(label=label, accelerator=key, command=lambda y=yaw: self.set_view(y))
        v.add_command(label='Reset camera', accelerator='R', command=self.reset)
        v.add_separator()
        v.add_checkbutton(label='High quality (anti-aliased)', variable=self.quality, command=lambda: self._later(0))
        m.add_cascade(label='View', menu=v)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label='Controls', command=lambda: messagebox.showinfo(APP, __doc__))
        m.add_cascade(label='Help', menu=h)
        self.root.config(menu=m)

    # ---- actions
    def ask_open(self):
        p = filedialog.askopenfilename(title='Open model', filetypes=[('glTF models', '*.glb *.gltf'), ('All files', '*.*')])
        if p:
            self.open(p)

    def ask_textures(self):
        p = filedialog.askopenfilename(title='Texture map', filetypes=[('JSON', '*.json')])
        if p:
            try:
                self._set_textures(json.load(open(p)))
            except Exception as e:
                messagebox.showerror(APP, 'Could not read %s:\n%s' % (p, e))

    def _set_textures(self, t):
        self.textures = t
        if self.path:
            self.open(self.path, keep_camera=True)

    def open(self, path, keep_camera=False):
        self.status.configure(text='loading %s ...' % os.path.basename(path))
        self.root.update_idletasks()
        try:
            self.model = load(path, self.textures)
        except Exception as e:
            messagebox.showerror(APP, 'Could not open %s:\n%s' % (path, e))
            return
        self.path = path
        if not keep_camera:
            self.cam = Camera()
        self.root.title('%s - %s  (%d triangles)' % (APP, os.path.basename(path), self.model.triangle_count()))
        self._later(0)

    def save(self):
        if self.img is None:
            return
        base = os.path.splitext(os.path.basename(self.path or 'render'))[0]
        p = filedialog.asksaveasfilename(defaultextension='.png', initialfile=base + '.png',
                                         filetypes=[('PNG image', '*.png'), ('JPEG image', '*.jpg')])
        if p:
            w, h = self.img.size
            render(self.model, self.cam, (w * 2, h * 2), Style(supersample=2)).save(p)   # a sharper file than the screen
            self.status.configure(text='saved ' + p)

    def set_view(self, yaw):
        self.cam.yaw, self.cam.pitch, self.cam.pan = yaw, 5.0, (0.0, 0.0)
        self._later(0)

    def reset(self):
        self.cam = Camera()
        self._later(0)

    # ---- drawing (renders run on a worker thread so the window stays responsive)
    def _draw(self, fast):
        if self.model is None:
            return
        if self.busy:
            self.pending = fast
            return
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 200)
        cam = Camera(**vars(self.cam))
        self.busy = True

        def work():
            t = time.time()
            if fast:
                img = render(self.model, cam, (w // 2, h // 2), Style(supersample=1)).resize((w, h), Image.BILINEAR)
            else:
                img = render(self.model, cam, (w, h), Style(supersample=2 if self.quality.get() else 1))
            self.done.put((img, fast, time.time() - t))     # Tk is not thread-safe: the main loop picks it up
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                self._show(*self.done.get_nowait())
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def _show(self, img, fast, secs):
        self.busy = False
        self.img = img
        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.tkimg, text='')
        c = self.cam
        self.status.configure(text='yaw %.0f   pitch %.0f   zoom %.2f   %s %.1fs   |   left-drag orbit  right-drag pan  '
                              'wheel zoom  R reset  1-4 views  S save' % (c.yaw % 360, c.pitch, c.zoom,
                                                                         'preview' if fast else 'frame', secs))
        if self.pending is not None:
            f, self.pending = self.pending, None
            self._draw(f)

    def _later(self, ms=350):
        if self.job:
            self.root.after_cancel(self.job)
        self.job = self.root.after(ms, lambda: self._draw(False))

    # ---- input
    def _press(self, e, kind):
        self.drag = (kind, e.x, e.y)

    def _motion(self, e):
        if not self.drag or self.model is None:
            return
        kind, x, y = self.drag
        c = self.cam
        if kind == 'orbit':
            c.yaw -= (e.x - x) * 0.5
            c.pitch = max(-85, min(85, c.pitch + (e.y - y) * 0.4))
        else:
            h = max(self.canvas.winfo_height(), 1)
            c.pan = (c.pan[0] - (e.x - x) / h / c.zoom, c.pan[1] + (e.y - y) / h / c.zoom)
        self.drag = (kind, e.x, e.y)
        self._draw(True)
        self._later()

    def _wheel(self, e):
        if self.model is None:
            return
        self.cam.zoom *= 1.15 if e.delta > 0 else 1 / 1.15
        self._draw(True)
        self._later()

    def _key(self, e):
        k = e.char.lower()
        if k == 'r':
            self.reset()
        elif k == 's':
            self.save()
        elif k == 'o':
            self.ask_open()
        elif k in '1234' and k:
            self.set_view({'1': 0, '2': 90, '3': 180, '4': 270}[k])


def run(path=None, textures=None, size=(820, 960)):
    root = tk.Tk()
    root.geometry('%dx%d' % size)
    try:
        ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.ico')
        if os.path.exists(ico):
            root.iconbitmap(ico)
    except tk.TclError:
        pass
    App(root, path, textures)
    root.mainloop()
