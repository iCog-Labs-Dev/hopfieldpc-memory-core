import pytest
import torch
from torch import nn

from hopfieldpc import HopfieldMemory, HopfieldShapeError


def test_external_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)

    query = torch.randn(8, 4)
    memory = torch.randn(10, 4)

    contract = module.validate_inputs(query=query, memory=memory)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 10


def test_internal_fixed_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, memory_size=7, learnable_memory=False)

    query = torch.randn(8, 4)

    contract = module.validate_inputs(query=query)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 7
    assert isinstance(module.memory, torch.Tensor)
    assert not isinstance(module.memory, nn.Parameter)
    assert module.memory.requires_grad is False


def test_internal_learnable_memory_shape_contract() -> None:
    module = HopfieldMemory(dim=4, memory_size=7, learnable_memory=True)

    query = torch.randn(8, 4)

    contract = module.validate_inputs(query=query)

    assert contract.batch_size == 8
    assert contract.dim == 4
    assert contract.memory_size == 7
    assert isinstance(module.memory, nn.Parameter)
    assert module.memory.requires_grad is True


def test_missing_memory_raises_error() -> None:
    module = HopfieldMemory(dim=4)

    query = torch.randn(8, 4)

    with pytest.raises(ValueError, match="No memory was provided"):
        module.validate_inputs(query=query)


def test_query_must_be_2d() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(8, 4, 1)
    memory = torch.randn(10, 4)

    with pytest.raises(HopfieldShapeError, match="query must have shape"):
        module.validate_inputs(query=query, memory=memory)


def test_memory_must_be_2d() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(8, 4)
    memory = torch.randn(10, 4, 1)

    with pytest.raises(HopfieldShapeError, match="memory must have shape"):
        module.validate_inputs(query=query, memory=memory)


def test_query_and_memory_dim_must_match() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(8, 4)
    memory = torch.randn(10, 5)

    with pytest.raises(HopfieldShapeError, match="query and memory must share"):
        module.validate_inputs(query=query, memory=memory)


def test_query_dim_must_match_module_dim() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(8, 5)
    memory = torch.randn(10, 5)

    with pytest.raises(HopfieldShapeError, match="expected dim=4"):
        module.validate_inputs(query=query, memory=memory)


def test_invalid_init_values_raise_errors() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        HopfieldMemory(dim=0)

    with pytest.raises(ValueError, match="memory_size must be"):
        HopfieldMemory(dim=4, memory_size=0)

    with pytest.raises(ValueError, match="beta must be positive"):
        HopfieldMemory(dim=4, beta=0.0)

    with pytest.raises(ValueError, match="num_updates must be"):
        HopfieldMemory(dim=4, num_updates=0)

    with pytest.raises(ValueError, match="learnable_memory=True requires"):
        HopfieldMemory(dim=4, learnable_memory=True)