"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import HopfieldMemoryOutput, HopfieldShapeContract, HopfieldShapeError
from .functional import hopfield_retrieve
from .memory import HopfieldMemory

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "hopfield_retrieve",
]