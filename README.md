# modelview

A fast, simple 3D model viewer and renderer for Windows. Open a `.glb` / `.gltf` model, spin it around in real
time, compare texture sets, and save screenshots — one `.exe`, nothing to install.

![The real-time viewer](docs/viewer.png)

## Get it

**Run the exe:** build it once with `build.bat` (below) and use `dist\modelview.exe`. Then:

- double-click `modelview.exe` and **drag a model onto the window**, or press **O** to browse, or
- drag a `.glb` file **onto `modelview.exe`** itself, or right-click a `.glb` → *Open with* → `modelview.exe`.

Try it with the included `examples/demo.glb`.

## Controls

| Mouse | |
|---|---|
| Left-drag | Orbit around the model |
| Right-drag / middle-drag / Shift+left-drag | Pan |
| Wheel | Zoom |
| Double-click | Re-frame the model |

| Key | | Key | |
|---|---|---|---|
| **O** | Open a model | **W** | Wireframe |
| **F** | Frame the model | **T** | Textures on/off |
| **1 2 3 4** | Front / right / back / left | **N** | Normal maps on/off |
| **Space** | Turntable spin | **G** | Flip normal-map green (DirectX ↔ OpenGL maps) |
| **S** | Screenshot → `Pictures\modelview` | **L** | Lighting on/off |
| **B** | Change background | **H** / **F1** | Hide / show help |
| **Esc** | Quit | | |

Drop a **texture map** (`.json`, below) on the window to re-texture the open model.

## What it renders

- glTF 2.0 binary (`.glb`) and text (`.gltf` with external or embedded buffers and images)
- Skinned characters in their bind pose
- Base colour textures + factors, tangent-space **normal maps**, alpha cut-outs (hair, leaves, fences), two-sided faces
- Three-point lighting with a sky/ground ambient, mip-mapped anisotropic textures, 4× MSAA
- PNG / JPEG / TIFF / TGA / DDS-via-Pillow textures

If the PC has no OpenGL 3.3, the exe falls back to a slower software-rendered window.

## Texture maps

A texture map swaps or supplies textures per material, without touching the model file — handy for models whose
images live in another folder, or for comparing an original texture set with an upscaled one:

```json
{
  "body_material": "C:/textures/body_upscaled.png",
  "head_material": {"color": "C:/textures/head.png", "normal": "C:/textures/head_n.png"},
  "eye_glass": ""
}
```

An empty value hides that material (clear shells such as glass that would otherwise draw solid).

## Command line

The same exe (or `python -m modelview`) renders images without opening a window — software rendering, no GPU needed:

```bat
modelview.exe render  model.glb -o sheet.png --views 0,45,180 --closeup
modelview.exe compare model.glb -o compare.png --a original.json --b upscaled.json --labels "Original,Upscaled"
```

![A render sheet](docs/render_sheet.png)

Options: `--views` camera angles in degrees, `--size 560x860`, `--pitch`, `--zoom`, `--closeup` (adds a close-up of
the top of the model, e.g. a character's head), `--title`.

## As a Python library

```python
from modelview import load, render, Camera
model = load('model.glb', {'body_material': 'body.png'})
render(model, Camera(yaw=30, pitch=10), size=(800, 1000)).save('out.png')
```

## Build from source

```bat
pip install -r requirements.txt pyinstaller
build.bat                       :: -> dist\modelview.exe
python app.py examples\demo.glb :: or run without building
python -m pytest tests          :: tests (pip install pytest)
```

Requires Python 3.9+, numpy, Pillow, moderngl and pyglet.
