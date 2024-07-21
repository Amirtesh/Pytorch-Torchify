# Torchify/__init__.py

# Import the classes from VisionNet and TabularNet
from .VisionNet import ImageModel
from .TabularNet import TabularModel

# Import the modules themselves
import Torchify.VisionNet
import Torchify.TabularNet

__all__ = ['ImageModel', 'TabularModel', 'VisionNet', 'TabularNet']
