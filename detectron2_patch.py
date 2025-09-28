"""
Patch for detectron2 compatibility issues
"""
import sys
import PIL.Image

# Add compatibility for PIL Image.LINEAR
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR

# Pre-import torch to ensure libraries are loaded
import torch

# Set library path for detectron2
import os
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = torch_lib_path

