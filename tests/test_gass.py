"""Exercise the public GASS API with a geometry and loss supplied by a caller."""
import torch

from hopfieldpc import GASSBank, GASSConfig
from hopfieldpc.gass import KeyGeometry, RetrievalLoss


class BoundedSignalGeometry(KeyGeometry):
    def project_keys(self, keys):
        return keys.clamp(-2, 2)

    def read_similarity(self, queries, keys):
        return -(queries[:, None] - keys).abs().mean(-1)

    def write_similarity(self, queries, keys):
        return -self.distance(queries, keys)


def signal_loss(targets, retrieved, scale):
    return torch.nn.functional.smooth_l1_loss(retrieved, targets, reduction="none").mean(-1)


def test_custom_geometry_and_loss_through_the_memory_lifecycle():
    queries = torch.linspace(-3, 3, 30).reshape(15, 2)
    targets = queries[:, :1].sin()
    cfg = GASSConfig(beta=2., min_child_mass=1., patience=1, edit_budget=1)
    bank = GASSBank(1, 2, 1, 4, cfg, key_dim=2, geometry=BoundedSignalGeometry(),
                    loss=RetrievalLoss(signal_loss))
    assert bank.initialize_from_embeddings(queries, targets)

    active = bank.mask
    scores = -(queries[:, None] - bank.keys[active]).abs().mean(-1)
    expected = (cfg.beta_r * scores + bank.weights[active].log()).softmax(-1) @ bank.values[active]
    torch.testing.assert_close(bank.read(queries), expected)

    write = bank.online_update(queries, targets + .2)
    assert write["after"] <= write["fresh_before"]
    replays = bank.refine_window(queries, targets, sweeps=2)
    assert len(replays) == 2
    assert replays[-1]["after"] <= replays[0]["before"]
    edits = bank.adapt_structure(queries, targets)
    assert edits["trials"]
    assert edits["after"] <= edits["before"]
    assert bank.keys[bank.mask].abs().max() <= 2
    assert 1 <= bank.active_count <= 4
    assert torch.isfinite(bank.read(queries)).all()
