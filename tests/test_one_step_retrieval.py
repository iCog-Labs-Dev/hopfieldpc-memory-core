import torch

from hopfieldpc import HopfieldMemory, hopfield_retrieve


def test_functional_retrieval_output_shapes() -> None:
    query = torch.randn(8, 4)
    memory = torch.randn(10, 4)

    retrieved, weights = hopfield_retrieve(query=query, memory=memory, beta=1.0)

    assert retrieved.shape == (8, 4)
    assert weights.shape == (8, 10)


def test_functional_weights_sum_to_one() -> None:
    query = torch.randn(8, 4)
    memory = torch.randn(10, 4)

    _, weights = hopfield_retrieve(query=query, memory=memory, beta=1.0)

    expected = torch.ones(8)
    assert torch.allclose(weights.sum(dim=-1), expected, atol=1e-6)


def test_module_forward_output_shapes() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)

    query = torch.randn(8, 4)
    memory = torch.randn(10, 4)

    output = module(query=query, memory=memory)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 10)
    assert isinstance(output.diagnostics, dict)


def test_nearest_memory_gets_highest_weight() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = torch.tensor([[0.95, 0.05]])

    _, weights = hopfield_retrieve(query=query, memory=memory, beta=5.0)

    top_index = int(weights.argmax(dim=-1).item())
    assert top_index == 0


def test_high_beta_retrieves_close_to_target_memory() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = torch.tensor([[0.95, 0.05]])

    retrieved, weights = hopfield_retrieve(query=query, memory=memory, beta=10.0)

    target = memory[0].unsqueeze(0)

    assert weights[0, 0] > 0.99
    assert torch.allclose(retrieved, target, atol=1e-2)


def test_module_uses_internal_fixed_memory() -> None:
    module = HopfieldMemory(dim=2, memory_size=3, beta=1.0, learnable_memory=False)

    query = torch.randn(4, 2)

    output = module(query=query)

    assert output.retrieved.shape == (4, 2)
    assert output.weights.shape == (4, 3)


def test_module_uses_internal_learnable_memory() -> None:
    module = HopfieldMemory(dim=2, memory_size=3, beta=1.0, learnable_memory=True)

    query = torch.randn(4, 2)

    output = module(query=query)

    assert output.retrieved.shape == (4, 2)
    assert output.weights.shape == (4, 3)
    assert module.memory.requires_grad is True


def test_retrieval_is_differentiable_for_query_and_memory() -> None:
    query = torch.randn(4, 3, requires_grad=True)
    memory = torch.randn(5, 3, requires_grad=True)

    retrieved, weights = hopfield_retrieve(query=query, memory=memory, beta=1.0)
    loss = retrieved.pow(2).mean() + weights.pow(2).mean()
    loss.backward()

    assert query.grad is not None
    assert memory.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(memory.grad).all()