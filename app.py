"""Entry point of modelview.exe.

    modelview.exe                       opens the real-time viewer (drop a model on the window)
    modelview.exe model.glb [map.json]  opens that model (drag & drop onto the exe / "Open with" work too)
    modelview.exe render|compare ...    command line, see `modelview.exe --help`
Falls back to the simple software-rendered window when the PC has no OpenGL 3.3.
"""
import json
import sys


def main():
    args = sys.argv[1:]
    if args and args[0] in ('render', 'compare', 'view', '-h', '--help'):
        from modelview.cli import main as cli
        cli(args)
        return
    path = next((a for a in args if a.lower().endswith(('.glb', '.gltf'))), None)
    tex = next((a for a in args if a.lower().endswith('.json')), None)
    textures = json.load(open(tex)) if tex else None
    try:
        from modelview.gpu_viewer import run
    except Exception:
        run = None
    if run is not None:
        try:
            run(path, textures)
            return
        except Exception as e:                      # no OpenGL 3.3: the software viewer still works
            err = e
    else:
        err = None
    from modelview.viewer import run as soft
    if err:
        print('OpenGL viewer unavailable (%s); using the software viewer' % err)
    soft(path, textures)


if __name__ == '__main__':
    main()
