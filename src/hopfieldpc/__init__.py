"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import HopfieldMemoryOutput, HopfieldShapeContract, HopfieldShapeError
from .energy import hopfield_energy
from .functional import hopfield_retrieve, multi_step_retrieve
from .memory import HopfieldMemory

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "hopfield_energy",
    "hopfield_retrieve",
    "multi_step_retrieve",
]