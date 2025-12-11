from .dgmc import DGMC
from .gcn import GCN
from .gin import GIN
from .gine import GINE
from .gine_norm import GINEN
from .mlp import MLP
from .rel import RelCNN
from .spline import SplineCNN

__all__ = [
    "MLP",
    "GIN",
    "GINE",
    "GINEN",
    "GCN",
    "SplineCNN",
    "RelCNN",
    "DGMC",
]
