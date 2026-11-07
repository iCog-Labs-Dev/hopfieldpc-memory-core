"""Capacity isolation, selected-alpha control parity, and durable epoch resume."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import admission_ablation as admission
import alpha_ablation as alpha
import capacity_ablation as study
from alpha_memory import ResidualEventWriter, apply_bank_batch, audit_alpha_log, make_alpha_bank
from mnist_online_memory import validate_config
from online_memory_artifacts import diagnostic_mode
from test_admission_ablation import fixture_model, toy_model
from test_alpha_ablation import reference_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu(monkeypatch):
    torch.set_num_threads(1)
    for module in (admission, alpha, study):
        monkeypatch.setattr(module, 'source_identity', lambda: {'commit': 'fixture', 'source_sha256': 'fixture'})
    for module in (alpha, study):
        monkeypatch.setattr(module, 'archive_source', lambda root, source: None)


def config_and_warm():
    cfg = validate_config({'num_classes': 2, 'max_memories_per_class': 1, 'ema_alpha': .03})
    warm = make_alpha_bank(cfg, 2, .03)
    warm.initialize_center(torch.zeros(2))
    return cfg, warm


def completed_step2(tmp_path, monkeypatch):
    reference = reference_fixture(tmp_path, monkeypatch)
    worker = alpha.run_study(reference, output=tmp_path/'alpha_worker')
    step2 = alpha.aggregate_studies([worker], output=tmp_path/'step2')
    monkeypatch.setattr(study, 'build_baseline_net', fixture_model)
    monkeypatch.setattr(study, 'load_mnist_tensors', admission.load_mnist_tensors)
    return step2


def test_capacity_initialization_transfers_only_center_into_independent_empty_buffers():
    cfg, warm = config_and_warm()
    banks = study.fresh_banks(cfg, [1, 2], 2, warm.state_dict())
    assert banks[1].values.shape == (2, 2) and banks[2].values.shape == (4, 2)
    assert all(not bank.ready and bank.version == 0 and not bank.metadata for bank in banks.values())
    admission.require_equal(banks[1].state_dict(), warm.state_dict())
    assert len({b.center.data_ptr() for b in banks.values()} | {warm.center.data_ptr()}) == 3
    assert banks[1].values.data_ptr() != banks[2].values.data_ptr()
    for bank in banks.values():
        assert bank.alpha == .03 and bank.beta == 5 and bank.threshold == .2
        assert not list(bank.parameters())
    warm.commit(warm.plan_candidate(torch.tensor([3., 0.]), 0, bootstrap=True), example_id=0, step=0, epoch=1)
    with pytest.raises(ValueError, match='empty'):
        study.fresh_banks(cfg, [1, 2], 2, warm.state_dict())


def test_larger_capacity_adds_a_distinct_mode_when_control_is_full_and_replays(tmp_path):
    cfg, warm = config_and_warm()
    banks = study.fresh_banks(cfg, [1, 2], 2, warm.state_dict())
    writers = {k: ResidualEventWriter(tmp_path/study.capacity_name(k), 3, 2) for k in banks}
    for writer in writers.values():
        writer.event(dict(type='center', center=[0., 0.]))
    model = toy_model()
    for batch, (x, y, ids) in enumerate([
            (torch.tensor([[3., 0.], [0., 3.]]), torch.tensor([0, 1]), torch.tensor([0, 1])),
            (torch.tensor([[3., 2.5]]), torch.tensor([0]), torch.tensor([2]))]):
        with diagnostic_mode(model):
            stats = {k: apply_bank_batch(model, bank, x, model.head(x).argmax(-1), y, ids,
                cfg=cfg, epoch=2, batch=batch, step=batch, seed=0, run_id='fixture', writer=writers[k])
                for k, bank in banks.items()}
    assert stats[1]['counts']['reason:class_capacity_reached'] == 1
    assert stats[2]['counts']['add'] == 1
    assert banks[1].occupancy() == [1, 1] and banks[2].occupancy() == [2, 1]
    assert torch.equal(banks[2].values[2], torch.tensor([3., 2.5]))
    for k, bank in banks.items():
        audit = audit_alpha_log(tmp_path/study.capacity_name(k), writers[k].flush(), study.bank_config(cfg, k),
                               bank, torch.arange(3), torch.tensor([0, 1, 0]))
        assert audit['replay_exact'] and audit['examples'] == 3


def test_different_bank_shapes_preserve_direct_loss_gradients_optimizer_and_rng():
    cfg, warm = config_and_warm()
    banks = study.fresh_banks(cfg, [1, 2], 2, warm.state_dict())
    model, direct = toy_model(), toy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    direct_optimizer = torch.optim.Adam(direct.parameters(), lr=.001)
    x, y = torch.tensor([[3., 0.], [0., 3.], [3., 2.5]]), torch.tensor([0, 1, 0])
    for step in range(3):
        before = torch.get_rng_state().clone()
        loss, _ = study.shared_step(model, optimizer, banks, x, y, torch.arange(3), cfg=cfg, writers={},
            epoch=2, batch=step, step=step, seed=0, run_id='fixture', costs={}, bank_costs={k: {} for k in banks})
        assert torch.equal(before, torch.get_rng_state())
        direct_optimizer.zero_grad(set_to_none=True)
        expected = F.cross_entropy(direct.head(direct.encoder(x)), y)
        expected.backward(); direct_optimizer.step()
        assert loss == float(expected.detach())
        admission.require_equal(model.state_dict(), direct.state_dict())
        admission.require_equal(optimizer.state_dict(), direct_optimizer.state_dict())
        for p, q in zip(model.parameters(), direct.parameters()):
            assert torch.equal(p.grad, q.grad)


def test_evaluation_is_read_only_and_reports_real_pairs_or_unavailable_banks():
    cfg, warm = config_and_warm()
    banks = study.fresh_banks(cfg, [1, 2], 2, warm.state_dict())
    model = toy_model(); model.train()
    x, y = torch.tensor([[3., 0.], [0., 3.], [3., 2.5]]), torch.tensor([0, 1, 0])
    missing = study.evaluate_banks(model, banks, x, y, cfg, control=1)
    assert missing['capacity_1']['accuracy'] is None
    assert missing['capacity_2']['paired_vs_control'] is None
    assert missing['capacity_2']['evaluated'] == 0 and missing['capacity_2']['fallback_count'] == 3
    with diagnostic_mode(model):
        for bank in banks.values():
            apply_bank_batch(model, bank, x[:2], torch.tensor([0, 1]), y[:2], torch.arange(2),
                             cfg=cfg, epoch=2, batch=0, step=0, seed=0, run_id='fixture')
    before = {k: copy.deepcopy(bank.state_dict()) for k, bank in banks.items()}
    rng = torch.get_rng_state().clone()
    measured = study.evaluate_banks(model, banks, x, y, cfg, control=1)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for k, bank in banks.items():
        admission.require_equal(before[k], bank.state_dict())
    pair = measured['capacity_2']['paired_vs_control']
    assert sum(pair[k] for k in ('helped', 'harmed', 'both_correct', 'both_wrong')) == len(y)
    assert abs(pair['delta_vs_control'] - (pair['helped']-pair['harmed'])/len(y)) < 1e-7


def test_synthetic_capacity_study_matches_selected_alpha_control_and_audits(tmp_path, monkeypatch):
    reference = completed_step2(tmp_path, monkeypatch)
    output = study.run_study(reference, output=tmp_path/'capacity')
    result = json.loads((output/'results.json').read_text())
    assert result['manifest']['capacities'] == [2, 4]
    assert result['manifest']['selected_alpha'] == json.loads((reference/'selection.json').read_text())['alpha']
    assert not result['manifest']['test_set_opened'] and result['manifest']['protected_artifacts_unchanged']
    seed = result['seeds'][0]
    assert seed['exact_control_bank_every_epoch'] and seed['exact_cnn_adam_every_epoch']
    assert all(a['replay_exact'] and a['examples'] == 80 for a in seed['audits'].values())
    assert set(r['method'] for r in seed['validation_results']) == {'direct', 'capacity_2', 'capacity_4', 'accuracy_guided', 'fixed_kmeans_reference'}
    assert all(r['paired_vs_control'] is not None for r in seed['validation_results'] if r['method'] == 'capacity_4')
    assert (output/'report.md').is_file() and (output/'validation_corruptions.png').is_file()
    combined = study.aggregate_studies([output], output=tmp_path/'combined')
    pooled = json.loads((combined/'results.json').read_text())
    assert pooled['selection'] == result['selection']
    assert Path(pooled['seeds'][0]['worker_seed_directory']).is_dir()
    with pytest.raises(ValueError, match='Duplicate'):
        study.aggregate_studies([output, output], output=tmp_path/'duplicate')
    assert not (tmp_path/'duplicate').exists()
    other = tmp_path/'worker_copy'
    shutil.copytree(output, other)
    manifest = json.loads((output/'manifest.json').read_text()); manifest['source'] = {'changed': True}
    (output/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source'):
        study.aggregate_studies([other, output], output=tmp_path/'bad_source')


def test_capacity_epoch_resume_preserves_all_banks_cnn_and_private_rng(tmp_path, monkeypatch):
    reference = completed_step2(tmp_path, monkeypatch)
    full = study.run_study(reference, output=tmp_path/'full')
    resumed = study.run_study(reference, output=tmp_path/'resumed', stop_after_epoch=2)
    study.run_study(output=resumed, resume=True)
    a = torch.load(full/'seed_0/checkpoints/epoch_3.pt', weights_only=False)
    b = torch.load(resumed/'seed_0/checkpoints/epoch_3.pt', weights_only=False)
    for key in ('model', 'optimizer', 'banks', 'history', 'shuffle_rng', 'split_ids', 'optimizer_step'):
        admission.require_equal(a[key], b[key], key)
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    assert np.array_equal(a['rng']['numpy'][1], b['rng']['numpy'][1])
    assert a['rng']['numpy'][0] == b['rng']['numpy'][0]
    assert a['rng']['numpy'][2:] == b['rng']['numpy'][2:]
    assert a['rng']['python'] == b['rng']['python']
    monkeypatch.setattr(study, 'source_identity', lambda: {'commit': 'changed', 'source_sha256': 'changed'})
    with pytest.raises(ValueError, match='source'):
        study.run_study(output=resumed, resume=True)


def test_capacity_low_disk_stop_is_resumable_before_first_checkpoint(tmp_path, monkeypatch):
    reference = completed_step2(tmp_path, monkeypatch)
    output = tmp_path/'disk_limited'
    with monkeypatch.context() as patch:
        patch.setattr(study.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
        with pytest.raises(OSError, match='reserve'):
            study.run_study(reference, output=output)
    assert json.loads((output/'manifest.json').read_text())['status'] == 'failed'
    study.run_study(output=output, resume=True)
    assert json.loads((output/'manifest.json').read_text())['status'] == 'complete'


def test_invalid_or_unaudited_step2_reference_is_rejected_before_output(tmp_path, monkeypatch):
    reference = completed_step2(tmp_path, monkeypatch)
    path = reference/'results.json'
    original = json.loads(path.read_text())
    invalid = copy.deepcopy(original)
    invalid['selection']['alpha'] = .99
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='selected alpha'):
        study.run_study(reference, output=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
    invalid = copy.deepcopy(original)
    invalid['seeds'][0]['audits'][alpha.alpha_name(original['selection']['alpha'])]['replay_exact'] = False
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='unaudited'):
        study.run_study(reference, output=tmp_path/'unaudited')
    assert not (tmp_path/'unaudited').exists()


@pytest.mark.parametrize('values', [[], [2], [2, 2], [0, 2], [-1, 2], [True, 2], [2., 4], [1, 4]])
def test_invalid_capacity_grids_fail(values):
    with pytest.raises(ValueError, match='Capacities'):
        study.validate_capacities(values, 2)


def test_capacity_selection_uses_clean_first_then_robustness_then_smaller_bank():
    def row(k, correct):
        return dict(method=study.capacity_name(k), kind='gaussian', level=0, accuracy=correct/100,
                    both_correct=correct, helped=0)
    seed = dict(validation_results=[row(8, 98), row(16, 97)],
                robust_auc={study.capacity_name(k)+':'+kind: auc for k, auc in [(8, .1), (16, .9)] for kind, _ in study.KINDS})
    assert study.select_capacity([seed], [8, 16])['capacity'] == 8
    seed['validation_results'][1].update(accuracy=.98, both_correct=98)
    assert study.select_capacity([seed], [8, 16])['capacity'] == 16
    seed['robust_auc'] = {key: .5 for key in seed['robust_auc']}
    assert study.select_capacity([seed], [8, 16])['capacity'] == 8
    for record in seed['validation_results']:
        record['accuracy'] = None
    assert study.select_capacity([seed], [8, 16])['capacity'] is None
