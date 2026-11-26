
"""Offline tests for raw and L2 Herding memory construction."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from benchmarks.mnist_herding import (
    build_herding_memory,
    herding,
)


def reference_herding(
    x: torch.Tensor,
    k: int,
    normalize_for_selection: bool,
) -> torch.Tensor:
    """Independent reference implementation returning original embeddings."""
    selection_x = (
        F.normalize(x, p=2, dim=1)
        if normalize_for_selection
        else x
    )

    class_mean = selection_x.mean(dim=0)
    selected_indices = []
    available = list(range(len(x)))

    for step in range(1, k + 1):
        errors = []

        for index in available:
            previous_sum = (
                selection_x[selected_indices].sum(dim=0)
                if selected_indices
                else torch.zeros_like(class_mean)
            )

            candidate_mean = (
                previous_sum + selection_x[index]
            ) / step

            error = (candidate_mean - class_mean).square().sum()
            errors.append(error.item())

        best_position = min(range(len(errors)), key=errors.__getitem__)
        selected_index = available.pop(best_position)
        selected_indices.append(selected_index)

    return x[torch.tensor(selected_indices, device=x.device)]


def make_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [4.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# Core Herding algorithm
# ---------------------------------------------------------------------


@pytest.mark.parametrize("normalize", [False, True])
def test_herding_matches_reference(normalize: bool) -> None:
    x = make_embeddings()

    actual = herding(
        x,
        3,
        normalize_for_selection=normalize,
    )

    expected = reference_herding(x, 3, normalize)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_herding_returns_requested_number_of_embeddings(
    normalize: bool,
) -> None:
    x = make_embeddings()

    selected = herding(
        x,
        3,
        normalize_for_selection=normalize,
    )

    assert selected.shape == (3, 2)


@pytest.mark.parametrize("normalize", [False, True])
def test_herding_selects_distinct_embeddings(normalize: bool) -> None:
    x = make_embeddings()

    selected = herding(
        x,
        4,
        normalize_for_selection=normalize,
    )

    assert torch.unique(selected, dim=0).shape[0] == 4


@pytest.mark.parametrize("normalize", [False, True])
def test_herding_does_not_modify_input(normalize: bool) -> None:
    x = make_embeddings()
    original = x.clone()

    herding(x, 3, normalize_for_selection=normalize)

    assert torch.equal(x, original)


def test_l2_herding_returns_original_raw_embeddings() -> None:
    x = make_embeddings()

    selected = herding(x, 3, normalize_for_selection=True)

    # The selected rows must come from the original, unnormalized input.
    assert all(
        any(torch.equal(row, original) for original in x)
        for row in selected
    )

    # At least one selected vector should retain a non-unit norm.
    assert not torch.allclose(
        selected.norm(dim=1),
        torch.ones(selected.shape[0]),
    )


def test_herding_k_equals_n_returns_all_embeddings() -> None:
    x = make_embeddings()

    selected = herding(x, len(x), normalize_for_selection=False)

    assert selected.shape == x.shape
    assert torch.equal(
        selected.sort(dim=0).values,
        x.sort(dim=0).values,
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_herding_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="shape"):
        herding(torch.randn(3, 2, 1), 2)


def test_herding_rejects_integer_embeddings() -> None:
    with pytest.raises(TypeError, match="floating"):
        herding(torch.tensor([[1, 2], [3, 4]]), 1)


def test_herding_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        herding(torch.empty(0, 3), 1)


@pytest.mark.parametrize("k", [0, -1, 6])
def test_herding_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="1 <= k <= N"):
        herding(make_embeddings(), k)


# ---------------------------------------------------------------------
# Class-wise memory construction
# ---------------------------------------------------------------------


def test_build_memory_has_correct_shapes_and_labels() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    memory, memory_labels, center = build_herding_memory(
        embeddings,
        labels,
        per_class=2,
        center=True,
        num_classes=2,
        normalize_for_selection=False,
    )

    assert memory.shape == (4, 2)
    assert memory_labels.tolist() == [0, 0, 1, 1]
    assert torch.allclose(center, embeddings.mean(dim=0))


def test_build_memory_without_center_returns_none() -> None:
    embeddings = torch.randn(12, 4)
    labels = torch.tensor([0] * 6 + [1] * 6)

    _, _, center = build_herding_memory(
        embeddings,
        labels,
        per_class=2,
        center=False,
        num_classes=2,
    )

    assert center is None


def test_build_memory_rejects_insufficient_class_samples() -> None:
    embeddings = torch.randn(5, 3)
    labels = torch.tensor([0, 0, 0, 1, 1])

    with pytest.raises(ValueError, match="Class 1"):
        build_herding_memory(
            embeddings,
            labels,
            per_class=3,
            center=True,
            num_classes=2,
        )


def test_build_memory_rejects_mismatched_sample_counts() -> None:
    with pytest.raises(ValueError, match="same number"):
        build_herding_memory(
            torch.randn(5, 3),
            torch.tensor([0, 0, 1]),
            per_class=1,
            center=True,
            num_classes=2,
        )


def test_build_memory_rejects_invalid_per_class() -> None:
    with pytest.raises(ValueError, match="per_class"):
        build_herding_memory(
            torch.randn(6, 3),
            torch.tensor([0, 0, 0, 1, 1, 1]),
            per_class=0,
            center=True,
            num_classes=2,
        )
