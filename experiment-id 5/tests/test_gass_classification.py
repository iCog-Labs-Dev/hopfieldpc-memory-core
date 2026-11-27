"""Digit-label retrieval and separation of fresh evidence, replay and evaluation."""
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gass_classification import classification_metrics, fit_memory
from gass import GASSBank, GASSConfig


def test_classification_uses_associated_labels_and_keeps_evaluation_frozen():
    # Key coordinate 0 identifies digit 4, coordinate 1 identifies digit 2.
    x = torch.eye(2).repeat(8, 1)
    y = torch.tensor([4, 2]).repeat(8)
    cfg = GASSConfig(read_score="cosine", write_score="cosine", retrieval_loss="squared",
                     window_size=16, fixed_capacity=True, beta=20)
    bank = GASSBank(10, 2, 2, 2, cfg, key_dim=2)
    with patch.object(bank, "online_update", wraps=bank.online_update) as fresh:
        with patch.object(bank, "refine_window", wraps=bank.refine_window) as replay:
            history = fit_memory(bank, x, y, epochs=3, batch_size=5, seed=7)
    assert sum(len(call.args[0]) for call in fresh.call_args_list) == len(y)
    assert sum(len(call.args[0]) for call in replay.call_args_list) == len(y) * 2
    assert [row["active_slots"] for row in history] == [2, 2, 2]

    state = {name: tensor.clone() for name, tensor in bank.named_buffers()}
    result = classification_metrics(bank.read, torch.eye(2), torch.tensor([4, 2]),
                                    batch_size=1, device=torch.device("cpu"))
    assert result["accuracy"] == 1
    assert result["confusion"][4][4] == result["confusion"][2][2] == 1
    assert result["accuracy_by_digit"][0] is None
    for name, tensor in bank.named_buffers():
        assert torch.equal(tensor, state[name])


def test_confusion_rows_are_true_digits_and_final_partial_batch_counts():
    features = torch.nn.functional.one_hot(torch.tensor([1, 1, 4]), 10).float()
    result = classification_metrics(torch.nn.Identity(), features, torch.tensor([1, 2, 4]),
                                    batch_size=2, device=torch.device("cpu"))
    assert result["count"] == 3
    assert result["accuracy"] == 2 / 3
    assert result["support_by_digit"][2] == result["confusion"][2][1] == 1
    assert result["accuracy_by_digit"][2] == 0.
