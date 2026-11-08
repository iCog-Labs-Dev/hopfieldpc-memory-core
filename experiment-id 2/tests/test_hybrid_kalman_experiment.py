import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from hybrid_kalman_experiment import PrototypeUpdater


def test_hybrid_uses_accuracy_gated_momentum_and_updates_uncertainty():
    bank = PrototypeUpdater(1, 2, method="hybrid", gamma=0.9, process_noise=0.01,
                            prototypes_per_class=1)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.zeros(2, dtype=torch.long)
    bank.update(features, labels, 1.0)
    result = bank.update(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), labels, 0.0)

    assert 0 < result["gain"].item() < 1
    assert result["velocity"].norm().item() > 0
    assert bank.uncertainty.item() == pytest.approx(0.51, abs=1e-6)
    assert torch.isfinite(bank.values).all()
    assert not torch.isclose(bank.values.norm(), torch.ones(()))


def test_ablations_disable_one_component():
    for method in ("momentum_only", "kalman_only", "ema"):
        bank = PrototypeUpdater(1, 2, method=method, prototypes_per_class=1)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.zeros(2, dtype=torch.long)
        bank.update(features, labels, 1.0)
        result = bank.update(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), labels, 0.0)
        assert torch.isfinite(bank.values).all()
        assert torch.isfinite(result["gain"]).all()


def test_rejects_nonfinite_accuracy():
    bank = PrototypeUpdater(1, 2, method="hybrid", prototypes_per_class=1)
    with pytest.raises(ValueError, match="accuracy"):
        bank.update(torch.ones(1, 2), torch.zeros(1, dtype=torch.long), float("nan"))
