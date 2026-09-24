from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .api import HopfieldMemoryOutput, HopfieldShapeContract, validate_query_memory
from .functional import hopfield_retrieve


class HopfieldMemory(nn.Module):
    """PyTorch module wrapper for modern Hopfield retrieval.

    The module owns retrieval configuration and delegates tensor operations to
    the functional core. External memory passed to :meth:`forward` takes
    precedence over any internal memory.

    Shape contract:
        query:     [batch_size, dim]
        memory:    [num_memories, dim]
        retrieved: [batch_size, dim]
        weights:   [batch_size, num_memories]
    """

    def __init__(
        self,
        dim: int,
        memory_size: int | None = None,
        *,
        beta: float = 1.0,
        num_updates: int = 1,
        learnable_memory: bool = False,
        normalize: bool = False,
        return_diagnostics: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}.")

        if memory_size is not None and (
            not isinstance(memory_size, int) or memory_size <= 0
        ):
            raise ValueError(
                f"memory_size must be None or a positive integer, got {memory_size!r}."
            )

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta!r}.")

        if not isinstance(num_updates, int) or num_updates <= 0:
            raise ValueError(
                f"num_updates must be a positive integer, got {num_updates!r}."
            )

        if learnable_memory and memory_size is None:
            raise ValueError("learnable_memory=True requires memory_size.")

        self.dim = dim
        self.memory_size = memory_size
        self.beta = float(beta)
        self.num_updates = num_updates
        self.learnable_memory = bool(learnable_memory)
        self.normalize = bool(normalize)
        self.return_diagnostics = bool(return_diagnostics)

        if memory_size is None:
            self.memory = None
        else:
            init_memory = self._init_memory(memory_size, dim)

            if learnable_memory:
                self.memory = nn.Parameter(init_memory)
            else:
                self.register_buffer("memory", init_memory)

    @staticmethod
    def _init_memory(memory_size: int, dim: int) -> torch.Tensor:
        """Initialize internal memory vectors."""

        std = 1.0 / math.sqrt(dim)
        memory = torch.empty(memory_size, dim)
        nn.init.normal_(memory, mean=0.0, std=std)
        return memory

    def get_memory(self, memory: torch.Tensor | None = None) -> torch.Tensor:
        """Resolve external memory or internal memory."""

        if memory is not None:
            return memory

        if self.memory is None:
            raise ValueError(
                "No memory was provided. Pass memory to forward()/validate_inputs(), "
                "or initialize HopfieldMemory with memory_size."
            )

        return self.memory

    def validate_inputs(
        self,
        query: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> HopfieldShapeContract:
        """Validate query and memory against the module shape contract."""

        resolved_memory = self.get_memory(memory)
        return validate_query_memory(
            query=query,
            memory=resolved_memory,
            expected_dim=self.dim,
        )

    def _build_basic_diagnostics(
        self,
        *,
        query: torch.Tensor,
        memory: torch.Tensor,
        weights: torch.Tensor,
    ) -> dict[str, Any]:
        """Build a minimal diagnostics dictionary.

        Full entropy/top-k/distance diagnostics will be added in a later task.
        This minimal dictionary confirms retrieval settings and tensor shapes.
        """

        if not self.return_diagnostics:
            return {}

        return {
            "beta": self.beta,
            "num_updates": self.num_updates,
            "normalize": self.normalize,
            "query_shape": tuple(query.shape),
            "memory_shape": tuple(memory.shape),
            "weights_shape": tuple(weights.shape),
        }

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> HopfieldMemoryOutput:
        """Run one-step Hopfield retrieval.

        Returns:
            A tuple-compatible output containing ``retrieved``, ``weights``,
            and ``diagnostics``::

                retrieved, weights, diagnostics = module(query, memory)
        """

        resolved_memory = self.get_memory(memory)
        self.validate_inputs(query=query, memory=resolved_memory)

        if self.num_updates != 1:
            raise NotImplementedError(
                "HopfieldMemory currently supports one-step retrieval only. "
                "Multi-step retrieval will be implemented in a later task."
            )

        retrieved, weights = hopfield_retrieve(
            query=query,
            memory=resolved_memory,
            beta=self.beta,
            normalize=self.normalize,
        )

        diagnostics = self._build_basic_diagnostics(
            query=query,
            memory=resolved_memory,
            weights=weights,
        )

        return HopfieldMemoryOutput(
            retrieved=retrieved,
            weights=weights,
            diagnostics=diagnostics,
        )