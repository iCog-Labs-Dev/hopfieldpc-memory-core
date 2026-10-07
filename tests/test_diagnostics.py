import math

import pytest
import torch

from hopfieldpc import HopfieldMemory
from hopfieldpc.diagnostics import (
    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    max_attention_weight,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
)


def test_attention_entropy_low_for_one_hot_weights() -> None:
    weights = torch.tensor([[1.0, 0.0, 0.0]])

    entropy = attention_entropy(weights)

    assert torch.allclose(entropy, torch.tensor([0.0]), atol=1e-6)


def test_attention_entropy_high_for_uniform_weights() -> None:
    weights = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])

    entropy = attention_entropy(weights)

    assert torch.allclose(entropy, torch.tensor([math.log(3.0)]), atol=1e-6)


def test_normalized_entropy_uniform_is_one() -> None:
    weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

    entropy = attention_entropy(weights, normalize=True)

    assert torch.allclose(entropy, torch.tensor([1.0]), atol=1e-6)


def test_topk_mass() -> None:
    weights = torch.tensor([[0.5, 0.3, 0.2]])

    assert torch.allclose(topk_mass(weights, k=1), torch.tensor([0.5]))
    assert torch.allclose(topk_mass(weights, k=2), torch.tensor([0.8]))
    assert torch.allclose(topk_mass(weights, k=10), torch.tensor([1.0]))


def test_topk_invalid_k_raises() -> None:
    weights = torch.tensor([[0.5, 0.5]])

    with pytest.raises(ValueError, match="k must be"):
        topk_mass(weights, k=0)

    with pytest.raises(ValueError, match="k must be"):
        topk_mass(weights, k=True)


def test_max_attention_weight_and_argmax() -> None:
    weights = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])

    assert torch.allclose(max_attention_weight(weights), torch.tensor([0.7, 0.8]))
    assert torch.equal(argmax_memory_index(weights), torch.tensor([1, 0]))


def test_retrieval_distance() -> None:
    retrieved = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    distance = retrieval_distance(retrieved, target)

    assert torch.allclose(distance, torch.tensor([1.0, 1.0]))


def test_retrieval_distance_shape_mismatch_raises() -> None:
    retrieved = torch.randn(2, 3)
    target = torch.randn(2, 4)

    with pytest.raises(ValueError, match="same shape"):
        retrieval_distance(retrieved, target)


def test_selected_memory_distance() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    retrieved = torch.tensor([[0.9, 0.1]])
    weights = torch.tensor([[0.8, 0.2]])

    distance = selected_memory_distance(retrieved, memory, weights)

    expected = torch.linalg.vector_norm(torch.tensor([[-0.1, 0.1]]), dim=-1)
    assert torch.allclose(distance, expected)


def test_build_retrieval_diagnostics_contains_expected_keys() -> None:
    query = torch.tensor([[0.9, 0.1]])
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    weights = torch.tensor([[0.8, 0.2]])
    retrieved = weights @ memory

    diagnostics = build_retrieval_diagnostics(
        query=query,
        memory=memory,
        retrieved=retrieved,
        weights=weights,
        beta=2.0,
        num_updates=1,
        normalize=False,
        top_k=2,
    )

    expected_keys = {
        "beta",
        "num_updates",
        "normalize",
        "query_shape",
        "memory_shape",
        "retrieved_shape",
        "weights_shape",
        "top_k",
        "entropy",
        "normalized_entropy",
        "top1_mass",
        "topk_mass",
        "max_attention_weight",
        "argmax_memory_index",
        "query_retrieved_distance",
        "selected_memory_distance",
        "entropy_mean",
        "top1_mass_mean",
        "topk_mass_mean",
        "selected_memory_distance_mean",
    }

    assert expected_keys.issubset(diagnostics.keys())


def test_hopfield_memory_module_returns_full_diagnostics() -> None:
    module = HopfieldMemory(dim=2, beta=5.0, diagnostics_top_k=2)
    query = torch.tensor([[0.9, 0.1]])
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    output = module(query, memory)

    assert "entropy" in output.diagnostics
    assert "top1_mass" in output.diagnostics
    assert "topk_mass" in output.diagnostics
    assert "selected_memory_distance" in output.diagnostics
    assert output.diagnostics["top_k"] == 2


def test_disabled_diagnostics_still_returns_empty_dict() -> None:
    module = HopfieldMemory(dim=2, return_diagnostics=False)
    query = torch.randn(2, 2)
    memory = torch.randn(3, 2)

    output = module(query, memory)

    assert output.diagnostics == {}


def test_diagnostics_are_detached_from_graph() -> None:
    module = HopfieldMemory(dim=2)
    query = torch.randn(2, 2, requires_grad=True)
    memory = torch.randn(3, 2, requires_grad=True)

    output = module(query, memory)

    assert output.retrieved.requires_grad is True
    assert output.diagnostics["entropy"].requires_grad is False