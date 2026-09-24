import torch

from hopfieldpc import hopfield_retrieve


def test_higher_beta_gives_sharper_retrieval() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = torch.tensor([[1.0, 0.0]])

    _, low_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=0.1)
    _, high_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    low_beta_top_weight = low_beta_weights.max(dim=-1).values.item()
    high_beta_top_weight = high_beta_weights.max(dim=-1).values.item()

    assert high_beta_top_weight > low_beta_top_weight
    assert high_beta_top_weight > 0.99


def test_low_beta_is_more_uniform_than_high_beta() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = torch.tensor([[1.0, 0.0]])

    _, low_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=0.01)
    _, high_beta_weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    low_beta_range = (
        low_beta_weights.max(dim=-1).values - low_beta_weights.min(dim=-1).values
    ).item()
    high_beta_range = (
        high_beta_weights.max(dim=-1).values - high_beta_weights.min(dim=-1).values
    ).item()

    assert high_beta_range > low_beta_range


def test_normalized_scoring_still_returns_original_memory_values() -> None:
    memory = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query = torch.tensor([[1.0, 0.0]])

    retrieved, weights = hopfield_retrieve(
        query=query,
        memory=memory,
        beta=10.0,
        normalize=True,
    )

    assert weights[0, 0] > 0.99
    assert retrieved[0, 0] > 9.9