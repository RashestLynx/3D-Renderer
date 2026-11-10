"""Command line.

    modelview render  model.glb -o out.png [--views 0,45,180] [--size 640x900] [--textures map.json]
    modelview compare model.glb -o out.png --a orig.json --b new.json [--labels "Original,Upscaled"] [--views ...]
    modelview view    model.glb [--textures map.json]

A texture map is JSON {material name: image path}; it replaces each listed material's colour texture.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

from .gltf import load
from .raster import Camera, render


def _size(s):
    w, h = s.lower().split('x')
    return int(w), int(h)


def _font(n):
    for f in ('arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def _tex(path):
    return json.load(open(path)) if path else None


def _views(model, args):
    cams = [Camera(yaw=float(v), pitch=args.pitch, zoom=args.zoom) for v in args.views.split(',')]
    if args.closeup:
        lo, hi = model.bounds()
        h = hi[1] - lo[1]
        # the top of a standing character: its head
        target = ((lo[0] + hi[0]) / 2, hi[1] - h * 0.075, (lo[2] + hi[2]) / 2)
        cams.append(Camera(yaw=float(args.views.split(',')[0]) - 20, pitch=args.pitch, target=target, span=h * 0.17))
    return cams


def sheet(rows, labels, title, size):
    """rows: [[PIL images]] -> one labelled sheet."""
    f, fs = _font(26), _font(22)
    cw, ch = size
    n = max(len(r) for r in rows)
    out = Image.new('RGB', (cw * n + 10 * (n - 1), 50 + len(rows) * (ch + 42)), (18, 18, 22))
    d = ImageDraw.Draw(out)
    d.text((10, 10), title, fill='white', font=f)
    for r, (label, imgs) in enumerate(zip(labels, rows)):
        y = 50 + r * (ch + 42)
        d.text((10, y + 8), label, fill=(120, 220, 140) if r else (205, 205, 205), font=fs)
        for i, im in enumerate(imgs):
            out.paste(im, (i * (cw + 10), y + 40))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='modelview', description='Render and view glTF/GLB models.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('render', 'compare', 'view'):
        p = sub.add_parser(name)
        p.add_argument('model')
        p.add_argument('--views', default='0,45,180', help='camera yaws in degrees, comma separated')
        p.add_argument('--pitch', type=float, default=5.0)
        p.add_argument('--zoom', type=float, default=1.0)
        p.add_argument('--size', type=_size, default=(560, 860))
        p.add_argument('--closeup', action='store_true', help='add a close-up of the top (a character\'s head)')
        p.add_argument('--title', default='')
        if name != 'compare':
            p.add_argument('--textures', help='JSON {material: image path}')
        if name != 'view':
            p.add_argument('-o', '--out', required=True)
        if name == 'compare':
            p.add_argument('--a', required=True, help='texture map A (JSON)')
            p.add_argument('--b', required=True, help='texture map B (JSON)')
            p.add_argument('--labels', default='A,B')
    a = ap.parse_args(argv)
    title = a.title or os.path.basename(a.model)
    if a.cmd == 'view':
        from .viewer import run
        run(a.model, _tex(a.textures))
        return
    if a.cmd == 'render':
        m = load(a.model, _tex(a.textures))
        rows, labels = [[render(m, c, a.size) for c in _views(m, a)]], ['']
    else:
        rows = []
        for tm in (a.a, a.b):
            m = load(a.model, _tex(tm))
            rows.append([render(m, c, a.size) for c in _views(m, a)])
        labels = a.labels.split(',')
    sheet(rows, labels, title, a.size).save(a.out)
    print(a.out)


if __name__ == '__main__':
    main()
