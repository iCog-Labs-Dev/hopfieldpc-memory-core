from __future__ import annotations

import math

import torch
from torch import nn

from .api import HopfieldMemoryOutput, HopfieldShapeContract, validate_query_memory


class HopfieldMemory(nn.Module):
    """Standalone modern Hopfield memory module API skeleton.

    Task 1 defines the API and shape contract.
    Task 2 will implement actual retrieval.
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
        std = 1.0 / math.sqrt(dim)
        memory = torch.empty(memory_size, dim)
        nn.init.normal_(memory, mean=0.0, std=std)
        return memory

    def get_memory(self, memory: torch.Tensor | None = None) -> torch.Tensor:
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
        resolved_memory = self.get_memory(memory)
        return validate_query_memory(
            query=query,
            memory=resolved_memory,
            expected_dim=self.dim,
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> HopfieldMemoryOutput:
        self.validate_inputs(query=query, memory=memory)

        raise NotImplementedError(
            "Hopfield retrieval is not implemented yet. "
            "Task 2 will implement p = softmax(beta * query @ memory.T) "
            "and retrieved = p @ memory."
        )