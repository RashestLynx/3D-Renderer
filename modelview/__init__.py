"""modelview - a small, dependency-light 3D model renderer and viewer (numpy + Pillow, Tk for the window).

    from modelview import load, render, Camera
    img = render(load('model.glb'), Camera(yaw=30))
"""
from .gltf import Material, Model, Part, load, load_image  # noqa: F401
from .raster import Camera, Style, render  # noqa: F401

__version__ = '0.1.0'
