import pytest
import torch

from hopfieldpc import (
    HopfieldMemory,
    HopfieldMemoryOutput,
    HopfieldShapeError,
    hopfield_retrieve,
)


def test_memory_forward_output_contract_and_shapes() -> None:
    module = HopfieldMemory(dim=64, beta=2.0)
    query = torch.randn(8, 64)
    memory = torch.randn(20, 64)

    output = module(query, memory)
    retrieved, weights, diagnostics = output

    assert isinstance(output, HopfieldMemoryOutput)
    assert retrieved.shape == (8, 64)
    assert weights.shape == (8, 20)
    assert isinstance(diagnostics, dict)


def test_named_output_matches_unpacked_output() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(2, 4)
    memory = torch.randn(3, 4)

    output = module(query, memory)
    retrieved, weights, diagnostics = output

    assert output.retrieved is retrieved
    assert output.weights is weights
    assert output.diagnostics is diagnostics


def test_external_memory_takes_precedence_over_internal_memory() -> None:
    module = HopfieldMemory(dim=4, memory_size=3)
    query = torch.randn(2, 4)
    external_memory = torch.randn(5, 4)

    _, weights, diagnostics = module(query, external_memory)

    assert weights.shape == (2, 5)
    assert diagnostics["memory_shape"] == (5, 4)


def test_module_matches_functional_retrieval() -> None:
    module = HopfieldMemory(dim=4, beta=2.0)
    query = torch.randn(2, 4)
    memory = torch.randn(3, 4)

    expected_retrieved, expected_weights = hopfield_retrieve(
        query,
        memory,
        beta=2.0,
    )
    retrieved, weights, _ = module(query, memory)

    assert torch.allclose(retrieved, expected_retrieved)
    assert torch.allclose(weights, expected_weights)


def test_disabled_diagnostics_preserves_output_contract() -> None:
    module = HopfieldMemory(dim=4, return_diagnostics=False)
    query = torch.randn(2, 4)
    memory = torch.randn(3, 4)

    retrieved, weights, diagnostics = module(query, memory)

    assert retrieved.shape == (2, 4)
    assert weights.shape == (2, 3)
    assert diagnostics == {}


def test_forward_enforces_shape_contract() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(2, 5)
    memory = torch.randn(3, 5)

    with pytest.raises(HopfieldShapeError, match="expected dim=4"):
        module(query, memory)
