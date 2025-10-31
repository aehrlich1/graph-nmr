from .dgmc import DGMC
from .gin import GIN
from .gine import GINE
from .mlp import MLP
from .rel import RelCNN
from .spline import SplineCNN

__all__ = [
    "MLP",
    "GIN",
    "GINE",
    "SplineCNN",
    "RelCNN",
    "DGMC",
]
