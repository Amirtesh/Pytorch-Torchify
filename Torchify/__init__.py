# Torchify/__init__.py

# Import the classes from VisionNet and TabularNet
from .VisionNet import ImageClassificationModel
from .TabularNet import TabularModel

# Import the modules themselves
import Torchify.VisionNet
import Torchify.TabularNet

__all__ = ['ImageClassificationModel', 'TabularModel', 'VisionNet', 'TabularNet']