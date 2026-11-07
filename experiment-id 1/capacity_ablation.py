"""Step 3: online capacity comparison at the alpha selected in Step 2.

Independent banks share actual direct-only CNN training. The original capacity
must reproduce the selected Step 2 bank exactly. Validation only; no test data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import uuid

import numpy as np
import torch

from admission_ablation import CPU, require_equal, training_fingerprint
from alpha_ablation import alpha_name, archive_source, prediction_metrics, select_alpha, shared_step
from alpha_memory import VECTOR_ENCODING, ResidualEventWriter, audit_alpha_log, make_alpha_bank
from benchmarks.mnist_baseline import environment_info, load_mnist_tensors, make_corrupted_test_set, robust_auc
from diagnose_memory import file_hash, memory_health
from hopfieldpc.diagnostics import attention_entropy
from mnist_online_memory import EXPERIMENT_ROOT, build_baseline_net, checked_logits, extract, make_bank, validate_config
from online_memory import centered_retrieve
from online_memory_artifacts import atomic_json, diagnostic_mode, restore_rng, rng_state, save_checkpoint, source_identity, timed

VARIANT_ID = 'exp001_step03_all_correct_capacity_comparison'
DISK_RESERVE_BYTES = 128 * 1024**2
SELECTION_RULE = 'clean_correct_count; ties: mean of Gaussian/salt-pepper/occlusion AUC; smaller capacity'
KINDS = (('gaussian', 'gaussian_sigmas'), ('salt_pepper', 'salt_pepper_probs'), ('occlusion', 'occlusion_sizes'))


def capacity_name(capacity):
    return 'capacity_' + str(capacity)


def validate_capacities(values, control):
    if (not isinstance(values, list) or len(values) < 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values)
            or len(set(values)) != len(values) or control not in values):
        raise ValueError('Capacities must be unique positive integers, including the Step 2 control.')
    return list(values)


def bank_config(cfg, capacity):
    return cfg | {'max_memories_per_class': capacity}


def fresh_banks(cfg, capacities, dim, warm_state):
    """Transfer the frozen center, never resize or copy occupied memory slots."""
    extra = warm_state['_extra_state']
    if (extra['version'] != 0 or extra['next_id'] != 0 or extra['metadata']
            or not extra['center_initialized'] or (warm_state['entry_ids'] != -1).any()
            or (warm_state['labels'] != -1).any() or (warm_state['values'] != 0).any()):
        raise ValueError('Warm-up bank must be empty and have its frozen center initialized.')
    banks = {k: make_alpha_bank(bank_config(cfg, k), dim, cfg['ema_alpha']) for k in capacities}
    for bank in banks.values():
        bank.initialize_center(warm_state['center'])
        if list(bank.parameters()):
            raise ValueError('Memory must contain buffers without optimizer parameters.')
    return banks


@torch.no_grad()
def evaluate_banks(model, banks, images, labels, cfg, *, control, original=None, fixed=None):
    z = extract(model, images, cfg, CPU)
    snapshots = {capacity_name(k): b.snapshot() for k, b in banks.items()}
    if original is not None:
        snapshots['accuracy_guided'] = original.snapshot()
    predictions, diagnostics = defaultdict(list), defaultdict(lambda: defaultdict(list))
    with diagnostic_mode(model):
        for start in range(0, len(z), cfg['eval_batch_size']):
            query = z[start:start + cfg['eval_batch_size']]
            predictions['direct'].append(checked_logits(model.head(query)))
            for name, snapshot in snapshots.items():
                if snapshot.ready:
                    retrieved, weights = centered_retrieve(query, snapshot.values, snapshot.center,
                        beta=cfg['hopfield_beta'], num_updates=cfg['hopfield_num_updates'],
                        normalization_epsilon=cfg['normalization_epsilon'])
                    predictions[name].append(checked_logits(model.head(retrieved)))
                    diagnostics[name]['entropy'].append(attention_entropy(weights))
                    diagnostics[name]['top1_mass'].append(weights.max(-1).values)
            if fixed is not None:
                retrieved, _ = centered_retrieve(query, fixed['values'], fixed['center'],
                    beta=cfg['hopfield_beta'], num_updates=cfg['hopfield_num_updates'],
                    normalization_epsilon=cfg['normalization_epsilon'])
                predictions['fixed_kmeans_reference'].append(checked_logits(model.head(retrieved)))
    joined = {name: torch.cat(parts) for name, parts in predictions.items()}
    names = ['direct', *snapshots] + (['fixed_kmeans_reference'] if fixed is not None else [])
    metrics = {name: prediction_metrics(joined.get(name), joined['direct'], labels, cfg['num_classes']) for name in names}
    for name, fields in diagnostics.items():
        metrics[name].update({key: float(torch.cat(values).mean()) for key, values in fields.items()})
    reference = capacity_name(control)
    for capacity in banks:
        name = capacity_name(capacity)
        if name != reference:
            pair = None
            if name in joined and reference in joined:
                measured = prediction_metrics(joined[name], joined[reference], labels, cfg['num_classes'])
                pair = {key: measured[key] for key in ('helped', 'harmed', 'both_correct', 'both_wrong', 'mcnemar_p', 'evaluated')}
                pair.update(control_method=reference, delta_vs_control=measured['delta_vs_direct'])
            metrics[name]['paired_vs_control'] = pair
    return metrics


def protected_files(step2, reference, cfg, seeds):
    manifest = json.loads((step2/'manifest.json').read_text())
    paths = dict(manifest['protected_files'])
    for name in ('manifest.json', 'results.json'):
        paths[str((step2/name).resolve())] = file_hash(step2/name)
    for seed in seeds:
        folder = Path(reference[seed]['worker_seed_directory'])
        for epoch in range(cfg['warmup_epochs'] + 1, cfg['total_epochs'] + 1):
            path = folder/f'checkpoints/epoch_{epoch}.pt'
            paths[str(path.resolve())] = file_hash(path)
    return paths


def check_protected(paths):
    if any(file_hash(Path(path)) != digest for path, digest in paths.items()):
        raise ValueError('A protected original, Step 1, or Step 2 artifact changed.')


def train_seed(root, parent, reference, parent_cfg, cfg, capacities, seed, images, labels, source,
               *, resume=False, stop_after_epoch=None):
    folder = root/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    control = parent_cfg['max_memories_per_class']
    control_name, reference_name = capacity_name(control), alpha_name(cfg['ema_alpha'])
    warm = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{cfg["warmup_epochs"]}.pt', map_location='cpu', weights_only=False)
    require_equal(warm['config'], parent_cfg, 'parent.warmup_config')
    ids = {key: torch.as_tensor(warm['split_ids'][key], dtype=torch.long) for key in ('train', 'val')}
    all_ids = torch.cat(list(ids.values()))
    if len(all_ids) != len(all_ids.unique()) or int(all_ids.min()) < 0 or int(all_ids.max()) >= len(images):
        raise ValueError('Invalid or overlapping original split IDs.')
    coverage = torch.bincount(labels[ids['train']], minlength=cfg['num_classes'])
    if len(coverage) != cfg['num_classes'] or (coverage < cfg['fixed_reference']['prototypes_per_class']).any():
        raise ValueError('Training class coverage is insufficient.')
    latest = folder/'latest.json'
    continuing = resume and latest.is_file()
    state = torch.load(folder/json.loads(latest.read_text())['checkpoint'], map_location='cpu', weights_only=False) if continuing else warm
    if continuing:
        for key, value in (('source', source), ('config', cfg), ('capacities', capacities), ('variant_id', VARIANT_ID)):
            if state[key] != value:
                raise ValueError('Resume '+key+' differs.')
        require_equal(state['split_ids'], ids, 'resume.splits')
    model = build_baseline_net(seed)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    optimizer.load_state_dict(state['optimizer'])
    banks = fresh_banks(cfg, capacities, model.head.fc.in_features, warm['bank'])
    if continuing:
        for capacity, bank in banks.items():
            bank.load_state_dict(state['banks'][capacity_name(capacity)])
        expected = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{state["epoch"]}.pt', map_location='cpu', weights_only=False)
        for key in ('model', 'optimizer', 'shuffle_rng', 'optimizer_step'):
            require_equal(state[key], expected[key], 'resume.'+key)
        expected_control = torch.load(Path(reference[seed]['worker_seed_directory'])/f'checkpoints/epoch_{state["epoch"]}.pt', map_location='cpu', weights_only=False)
        require_equal(banks[control].state_dict(), expected_control['banks'][reference_name], 'resume.control_bank')
    shuffle = torch.Generator()
    shuffle.set_state(state['shuffle_rng'])
    restore_rng(state['rng'])
    costs = dict(state['costs']) if continuing else {}
    bank_costs = {k: dict(state['bank_costs'][capacity_name(k)]) if continuing else {} for k in capacities}
    writers = {k: ResidualEventWriter(folder/capacity_name(k), cfg['event_chunk_size'], cfg['write_chunk_size'],
        state['log_cursors'][capacity_name(k)] if continuing else None, initial_bank=banks[k]) for k in capacities}
    if not continuing:
        np.savez_compressed(folder/'split_ids.npz', **{key: value.numpy() for key, value in ids.items()})
        for capacity, writer in writers.items():
            writer.event(dict(type='center', center=banks[capacity].center.tolist(), alpha=cfg['ema_alpha'],
                              capacity=capacity, origin='original_warmup_train_mean'))
    history = list(state['history']) if continuing else []
    step = state['optimizer_step']
    previous_elapsed = state.get('elapsed_seconds', 0.) if continuing else 0.
    started = time.perf_counter()
    try:
        for epoch in range(state['epoch'] + 1, cfg['total_epochs'] + 1):
            model.train()
            order = torch.randperm(len(ids['train']), generator=shuffle)
            totals, actions = {k: Counter() for k in capacities}, {k: Counter() for k in capacities}
            weighted_loss = 0.
            print(f'capacity comparison seed {seed}: epoch {epoch}/{cfg["total_epochs"]}', flush=True)
            with timed(costs, 'shared_training_wall', CPU):
                for batch, start in enumerate(range(0, len(order), cfg['batch_size'])):
                    if batch % 50 == 0 and shutil.disk_usage(root).free < DISK_RESERVE_BYTES:
                        raise OSError('Free disk space is below the 128 MiB checkpoint/log reserve; resume after freeing space.')
                    source_ids = ids['train'][order[start:start + cfg['batch_size']]]
                    loss, results = shared_step(model, optimizer, banks, images[source_ids], labels[source_ids], source_ids,
                        cfg=cfg, writers=writers, epoch=epoch, batch=batch, step=step, seed=seed,
                        run_id=root.name, costs=costs, bank_costs=bank_costs)
                    step += 1
                    weighted_loss += loss * len(source_ids)
                    for capacity, stats in results.items():
                        actions[capacity].update(stats['counts'])
                        totals[capacity].update({key: value for key, value in stats.items() if key not in ('counts', 'local_diagnostics_available')})
                    if (batch + 1) % 50 == 0:
                        print(f'  seed {seed} epoch {epoch} batch {batch+1}: '+', '.join(
                            f'K={k} writes={bank.version}' for k, bank in banks.items())+
                            f', seconds={time.perf_counter()-started:.1f}', flush=True)
            expected = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{epoch}.pt', map_location='cpu', weights_only=False)
            for key, value in (('model', model.state_dict()), ('optimizer', optimizer.state_dict()),
                               ('shuffle_rng', shuffle.get_state()), ('optimizer_step', step)):
                require_equal(value, expected[key], key)
            old = torch.load(Path(reference[seed]['worker_seed_directory'])/f'checkpoints/epoch_{epoch}.pt', map_location='cpu', weights_only=False)
            require_equal(banks[control].state_dict(), old['banks'][reference_name], 'capacity_control_bank')
            old_history = next(h for h in reference[seed]['history'] if h['epoch'] == epoch)
            require_equal(dict(actions[control]), old_history['banks'][reference_name]['counts'], 'control.decisions')
            for key in ('examples', 'compared', 'direct_correct', 'hop_correct', 'fallback_count', 'in_sample_post_write_correct'):
                require_equal(totals[control][key], old_history['banks'][reference_name]['totals'].get(key, 0), 'control.'+key)
            with timed(costs, 'clean_validation', CPU):
                validation = evaluate_banks(model, banks, images[ids['val']], labels[ids['val']], cfg, control=control)
            require_equal(validation['direct'], old_history['validation']['direct'], 'validation.direct')
            require_equal(validation[control_name], old_history['validation'][reference_name], 'validation.control')
            history.append(dict(epoch=epoch, optimizer_step=step, loss=weighted_loss/len(ids['train']),
                banks={capacity_name(k): dict(counts=dict(actions[k]), totals=dict(totals[k]), storage=banks[k].storage(step)) for k in capacities},
                validation=validation, cnn_exact=True, adam_exact=True, shuffle_exact=True, control_capacity_bank_exact=True))
            cursors = {capacity_name(k): writer.flush() for k, writer in writers.items()}
            relative = f'checkpoints/epoch_{epoch}.pt'
            save_checkpoint(folder/relative, dict(variant_id=VARIANT_ID, source=source, config=cfg, capacities=capacities,
                model=model.state_dict(), optimizer=optimizer.state_dict(), banks={capacity_name(k): bank.state_dict() for k, bank in banks.items()},
                shuffle_rng=shuffle.get_state(), rng=rng_state(), epoch=epoch, optimizer_step=step, history=history,
                log_cursors=cursors, split_ids=ids, costs=costs, bank_costs={capacity_name(k): c for k, c in bank_costs.items()},
                elapsed_seconds=previous_elapsed + time.perf_counter() - started))
            atomic_json(latest, dict(checkpoint=relative, epoch=epoch, log_cursors=cursors))
            atomic_json(folder/'progress.json', dict(history=history))
            print(f'  seed {seed} epoch {epoch}: CNN/Adam and K={control} bank exact; validation '+
                  ', '.join(f'{name}={m["accuracy"]}' for name, m in validation.items()), flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < cfg['total_epochs']:
                return None
        final = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{cfg["total_epochs"]}.pt', map_location='cpu', weights_only=False)
        original = make_bank(parent_cfg, model.head.fc.in_features)
        original.load_state_dict(final['bank'])
        fixed = torch.load(parent/f'seed_{seed}/fixed_reference.pt', map_location='cpu', weights_only=False)
        rows, aucs = [], {}
        with timed(costs, 'validation_corruptions', CPU):
            for kind, key in KINDS:
                for level in cfg[key]:
                    corrupted = make_corrupted_test_set(images[ids['val']], seed=seed, kind=kind, level=level, stream=1)
                    measured = evaluate_banks(model, banks, corrupted, labels[ids['val']], cfg,
                                              control=control, original=original, fixed=fixed)
                    for method, metrics in measured.items():
                        if method in ('direct', control_name, 'accuracy_guided', 'fixed_kmeans_reference'):
                            old_method = reference_name if method == control_name else method
                            old_row = next(r for r in reference[seed]['validation_results'] if r['method'] == old_method and r['kind'] == kind and r['level'] == level)
                            for field in ('accuracy', 'confusion', 'helped', 'harmed', 'evaluated', 'fallback_count'):
                                require_equal(metrics[field], old_row[field], 'reference_validation.'+method+'.'+field)
                        rows.append(dict(seed=seed, method=method, kind=kind, level=level, **metrics))
                for method in measured:
                    values = [r['accuracy'] for r in rows if r['kind'] == kind and r['method'] == method]
                    aucs[method+':'+kind] = robust_auc(cfg[key], values) if all(v is not None for v in values) else None
        cursors = {capacity_name(k): writer.flush() for k, writer in writers.items()}
        audits = {}
        with timed(costs, 'independent_mutation_audits', CPU):
            for capacity, bank in banks.items():
                print(f'auditing seed {seed} capacity {capacity}', flush=True)
                audit = audit_alpha_log(folder/capacity_name(capacity), cursors[capacity_name(capacity)],
                                        bank_config(cfg, capacity), bank, ids['train'], labels)
                if audit['examples'] != len(ids['train']) * (cfg['total_epochs'] - cfg['warmup_epochs']):
                    raise AssertionError('Example logs are incomplete.')
                audits[capacity_name(capacity)] = audit
        result = dict(seed=seed, history=history, validation_results=rows, robust_auc=aucs, audits=audits,
            banks={capacity_name(k): bank.storage(step) for k, bank in banks.items()},
            memory_health={capacity_name(k): memory_health(model.head, bank.snapshot().values, bank.snapshot().labels) if bank.ready else None for k, bank in banks.items()},
            costs=costs, bank_costs={capacity_name(k): c for k, c in bank_costs.items()}, log_cursors=cursors,
            elapsed_seconds=previous_elapsed + time.perf_counter() - started,
            exact_cnn_adam_every_epoch=True, exact_control_bank_every_epoch=True)
        atomic_json(folder/'results.json', result)
        return result
    except Exception as error:
        atomic_json(folder/'failure.json', dict(error=type(error).__name__, message=str(error), optimizer_step=step,
            uncommitted_log_cursors={capacity_name(k): writer.flush() for k, writer in writers.items()}))
        raise


def select_capacity(results, capacities):
    ranked = []
    for capacity in capacities:
        name = capacity_name(capacity)
        rows = [next(r for r in s['validation_results'] if r['method'] == name and r['kind'] == 'gaussian' and r['level'] == 0) for s in results]
        if any(row['accuracy'] is None for row in rows):
            continue
        correct = sum(row['helped'] + row['both_correct'] for row in rows)
        auc = statistics.mean(s['robust_auc'][name+':'+kind] for s in results for kind, _ in KINDS)
        ranked.append((correct, auc, -capacity, capacity))
    if not ranked:
        return dict(capacity=None, reason='No capacity has a complete bank across every reported seed.')
    winner = max(ranked)
    return dict(capacity=winner[-1], clean_correct=winner[0], secondary_mean_auc=winner[1],
                criterion=SELECTION_RULE, validation_selection_only=True)


def reports(root, results, manifest):
    selection = select_capacity(results, manifest['capacities'])
    atomic_json(root/'results.json', dict(manifest=manifest, seeds=results, selection=selection))
    atomic_json(root/'selection.json', selection)
    rows = [row for seed in results for row in seed['validation_results']]
    fields = ['seed', 'method', 'kind', 'level', 'accuracy', 'delta_vs_direct', 'evaluated', 'fallback_count',
              'helped', 'harmed', 'both_correct', 'both_wrong', 'mcnemar_p', 'entropy', 'top1_mass',
              'control_helped', 'control_harmed', 'control_delta', 'control_mcnemar_p']
    with (root/'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader()
        for row in rows:
            pair = row.get('paired_vs_control') or {}
            writer.writerow(row | dict(control_helped=pair.get('helped'), control_harmed=pair.get('harmed'),
                control_delta=pair.get('delta_vs_control'), control_mcnemar_p=pair.get('mcnemar_p')))
    methods = ['direct', *[capacity_name(k) for k in manifest['capacities']], 'accuracy_guided', 'fixed_kmeans_reference']
    lines = ['# Step 3: capacity comparison', '',
        f'Validation only. Alpha={manifest["selected_alpha"]}; final epoch retained. Clean accuracy is the primary criterion.',
        'Each original-capacity bank exactly reproduces the selected Step 2 bank at every epoch.',
        'Actual CNN training is shared across independent banks; memory cannot affect its loss.',
        '', '| Method | '+' | '.join('Seed '+str(s['seed']) for s in results)+' | Mean clean accuracy |',
        '|---|'+'---:|'*(len(results)+1)]
    for method in methods:
        values = [next(r['accuracy'] for r in s['validation_results'] if r['kind'] == 'gaussian' and r['level'] == 0 and r['method'] == method) for s in results]
        available = [v for v in values if v is not None]
        lines.append('| '+method+' | '+' | '.join(f'{100*v:.3f}%' if v is not None else 'unavailable' for v in values)+' | '+
                     (f'{100*statistics.mean(available):.3f}% (n={len(available)})' if available else 'unavailable')+' |')
    lines += ['', '## Normalized validation robustness AUC', '', '| Method | Gaussian | Salt-and-pepper | Occlusion |', '|---|---:|---:|---:|']
    for method in methods:
        values = [[s['robust_auc'][method+':'+kind] for s in results if s['robust_auc'][method+':'+kind] is not None] for kind, _ in KINDS]
        lines.append('| '+method+' | '+' | '.join(f'{statistics.mean(v):.6f}' if v else 'unavailable' for v in values)+' |')
    lines += ['', f'Selected capacity: {selection["capacity"]}. This is validation selection, not a new test result.',
        'Higher robustness AUC is better. Per-class accuracy, confusion, attention, per-seed paired comparisons against direct and the capacity control, writes, costs, and storage are in the structured metrics.',
        'The fixed reference retains its original count and post-training center. Adaptive centers remain frozen at warm-up; actual adaptive occupancy is reported.',
        'Immediate per-write predictions are unavailable for all banks; pre-write snapshot and batch post-write diagnostics remain enabled.',
        '', '[Structured metrics](results.json) | [CSV](results.csv) | [Manifest](manifest.json) | [Selection](selection.json)']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (kind, _) in zip(axes, KINDS):
        levels = sorted({r['level'] for r in rows if r['kind'] == kind})
        for method in methods:
            values = [[r['accuracy'] for r in rows if r['kind'] == kind and r['level'] == level and r['method'] == method and r['accuracy'] is not None] for level in levels]
            ax.plot(levels, [100*statistics.mean(v) if v else float('nan') for v in values], '-o', label=method)
        ax.set(title='Validation: '+kind, xlabel='Corruption level', ylabel='Accuracy (%)'); ax.legend(fontsize=6); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(root/'validation_corruptions.png', dpi=140); plt.close(fig)


def run_study(step2=None, *, capacities=None, seeds=None, output=None, resume=False, stop_after_epoch=None):
    prior = None
    if resume:
        if output is None or step2 is not None or capacities is not None or seeds is not None:
            raise ValueError('Resume requires only its existing output directory.')
        prior = json.loads((output/'manifest.json').read_text())
        if prior['variant_id'] != VARIANT_ID:
            raise ValueError('Wrong resume variant.')
        step2, capacities, seeds = Path(prior['step2_run']), prior['capacities'], prior['seeds']
    if step2 is None:
        raise ValueError('A completed Step 2 comparison is required.')
    step2 = Path(step2).resolve()
    reference_data = json.loads((step2/'results.json').read_text())
    old_manifest = json.loads((step2/'manifest.json').read_text())
    require_equal(reference_data['manifest'], old_manifest, 'reference.manifest')
    if (old_manifest['variant_id'] not in ('exp001_step02_all_correct_alpha_sweep', 'exp001_step02_all_correct_alpha_sweep_aggregate')
            or old_manifest['status'] != 'complete' or not old_manifest['validation_only'] or old_manifest['test_set_opened']):
        raise ValueError('Step 2 must be a completed, validation-only alpha comparison.')
    alpha = reference_data['selection']['alpha']
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha)
            or not 0 < alpha <= 1 or alpha not in old_manifest['alphas']):
        raise ValueError('Step 2 has no valid selected alpha.')
    require_equal(reference_data['selection'], select_alpha(reference_data['seeds'], old_manifest['alphas']), 'reference.selection')
    parent = Path(old_manifest['parent_run']).resolve()
    parent_cfg = validate_config(json.loads((parent/'manifest.json').read_text())['config'])
    require_equal(parent_cfg, old_manifest['config'], 'parent.config')
    cfg = validate_config(parent_cfg | {'ema_alpha': alpha})
    control = parent_cfg['max_memories_per_class']
    capacities = validate_capacities([control, control*2] if capacities is None else capacities, control)
    reference = {s['seed']: s for s in reference_data['seeds']}
    seeds = sorted(reference) if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s not in reference for s in seeds):
        raise ValueError('Seeds must be a unique nonempty subset of Step 2 seeds.')
    if stop_after_epoch is not None and (isinstance(stop_after_epoch, bool) or not isinstance(stop_after_epoch, int)
            or not cfg['warmup_epochs'] < stop_after_epoch <= cfg['total_epochs']):
        raise ValueError('Invalid stopping epoch.')
    for seed in seeds:
        result = reference[seed]
        audit = result['audits'][alpha_name(alpha)]
        if (not result['exact_cnn_adam_every_epoch'] or not result['exact_alpha_0p1_every_epoch']
                or not audit['replay_exact'] or not audit['independent_assignment_and_ema_verified']):
            raise ValueError('Step 2 reference is unaudited.')
    torch.set_num_threads(cfg['num_threads'])
    source = source_identity()
    protected = protected_files(step2, reference, cfg, seeds)
    check_protected(protected)
    images, labels = load_mnist_tensors(cfg['data_dir'], train=True)
    fingerprint = training_fingerprint(images, labels)
    if fingerprint != old_manifest['training_dataset_sha256']:
        raise ValueError('Official training data differs from Step 2.')
    if prior:
        for key, value in (('source', source), ('protected_files', protected), ('training_dataset_sha256', fingerprint), ('config', cfg)):
            if prior[key] != value:
                raise ValueError('Resume '+key+' differs.')
    root = Path(output or EXPERIMENT_ROOT/'results/variants'/('capacity_'+uuid.uuid4().hex)).resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        archive_source(root, source)
    manifest = dict(variant_id=VARIANT_ID, step2_run=str(step2), parent_run=str(parent), config=cfg, parent_cnn_config=parent_cfg,
        capacities=capacities, control_capacity=control, selected_alpha=alpha, seeds=seeds, source=source, protected_files=protected,
        bank_settings={capacity_name(k): dict(ema_alpha=alpha, max_memories_per_class=k, admission_policy='all_direct_correct',
            beta=cfg['hopfield_beta'], merge_distance_threshold=cfg['merge_distance_threshold'], normalization_epsilon=cfg['normalization_epsilon']) for k in capacities},
        training_dataset_sha256=fingerprint, validation_only=True, test_set_opened=False, environment=environment_info(CPU),
        local_write_diagnostics=False, log_vector_encoding=VECTOR_ENCODING, disk_reserve_bytes=DISK_RESERVE_BYTES,
        shared_actual_cnn_training=True, backbone_feature_cache_used=False, selection_rule=SELECTION_RULE, status='running')
    atomic_json(root/'manifest.json', manifest)
    atomic_json(root/'resolved_settings.json', {key: manifest[key] for key in ('variant_id', 'config', 'parent_cnn_config', 'bank_settings', 'capacities', 'control_capacity', 'selected_alpha', 'seeds', 'selection_rule')})
    print('Capacity comparison output: '+str(root), flush=True)
    completed = []
    try:
        for seed in seeds:
            file = root/f'seed_{seed}/results.json'
            result = json.loads(file.read_text()) if resume and file.is_file() else train_seed(root, parent, reference,
                parent_cfg, cfg, capacities, seed, images, labels, source, resume=resume, stop_after_epoch=stop_after_epoch)
            if result is None:
                atomic_json(root/'manifest.json', manifest | dict(status='incomplete', last_seed=seed))
                return root
            completed.append(result)
        check_protected(protected)
        manifest |= dict(status='complete', protected_artifacts_unchanged=True, exact_cnn_adam_and_control_every_epoch=True, all_mutation_audits_passed=True)
        reports(root, completed, manifest)
        atomic_json(root/'manifest.json', manifest)
    except Exception as error:
        atomic_json(root/'manifest.json', manifest | dict(status='failed', error=type(error).__name__, message=str(error)))
        raise
    print('Capacity comparison complete: '+str(root), flush=True)
    return root


def aggregate_studies(runs, *, output=None):
    if not runs:
        raise ValueError('At least one worker is required.')
    runs = [Path(p).resolve() for p in runs]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in runs]
    first = manifests[0]
    if any(m['variant_id'] != VARIANT_ID or m['status'] != 'complete' for m in manifests):
        raise ValueError('Every capacity worker must be complete.')
    for manifest in manifests[1:]:
        for key in ('source', 'config', 'capacities', 'control_capacity', 'selected_alpha', 'bank_settings', 'log_vector_encoding',
                    'training_dataset_sha256', 'parent_run', 'step2_run', 'selection_rule'):
            if manifest[key] != first[key]:
                raise ValueError('Incompatible worker '+key)
    protected, required, completed, origins = {}, set(), {}, []
    for root, manifest in zip(runs, manifests):
        required.update(manifest['seeds'])
        for path, digest in manifest['protected_files'].items():
            if path in protected and protected[path] != digest:
                raise ValueError('Incompatible protected artifact hashes.')
            protected[path] = digest
        origin = dict(run_directory=str(root), manifest_sha256=file_hash(root/'manifest.json'), completed_seeds=[])
        for seed in manifest['seeds']:
            if seed in completed:
                raise ValueError('Duplicate completed seed: '+str(seed))
            file = root/f'seed_{seed}/results.json'
            result = json.loads(file.read_text())
            cfg = manifest['config']
            if (result['seed'] != seed or not result['exact_cnn_adam_every_epoch'] or not result['exact_control_bank_every_epoch']
                    or set(result['audits']) != {capacity_name(k) for k in manifest['capacities']}
                    or any(not a['replay_exact'] or not a['independent_assignment_and_ema_verified'] for a in result['audits'].values())
                    or [h['epoch'] for h in result['history']] != list(range(cfg['warmup_epochs']+1, cfg['total_epochs']+1))):
                raise ValueError('A seed is incomplete or unaudited.')
            with np.load(file.parent/'split_ids.npz', allow_pickle=False) as ids:
                expected = len(ids['train']) * (cfg['total_epochs']-cfg['warmup_epochs'])
            if any(a['examples'] != expected for a in result['audits'].values()):
                raise ValueError('A seed has incomplete exposure logs.')
            result['worker_seed_directory'] = str(file.parent)
            completed[seed] = result
            origin['completed_seeds'].append(dict(seed=seed, result_sha256=file_hash(file)))
        origins.append(origin)
    if set(completed) != required:
        raise ValueError('Missing completed seeds.')
    check_protected(protected)
    root = Path(output or EXPERIMENT_ROOT/'results/variants'/('capacity_comparison_'+uuid.uuid4().hex)).resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = first | dict(variant_id=VARIANT_ID+'_aggregate', artifact_type='completed_seed_aggregate', seeds=sorted(completed),
        source_runs=origins, protected_files=protected, aggregation_source=source_identity(), status='complete',
        selection_scope='All reported seeds; per-worker selections are not used.')
    atomic_json(root/'manifest.json', manifest)
    atomic_json(root/'resolved_settings.json', {key: manifest[key] for key in ('variant_id', 'config', 'parent_cnn_config', 'bank_settings', 'capacities', 'control_capacity', 'selected_alpha', 'seeds', 'selection_rule')})
    reports(root, [completed[s] for s in sorted(completed)], manifest)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--step2-run', type=Path)
    group.add_argument('--resume', type=Path)
    group.add_argument('--aggregate', type=Path, nargs='+')
    parser.add_argument('--capacities', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    if args.resume and args.output:
        parser.error('--resume already identifies its output directory')
    if args.aggregate:
        if args.capacities or args.seeds or args.stop_after_epoch is not None:
            parser.error('--aggregate combines completed workers without training')
        print(aggregate_studies(args.aggregate, output=args.output))
    else:
        run_study(args.step2_run, capacities=args.capacities, seeds=args.seeds,
                  output=args.resume or args.output, resume=bool(args.resume), stop_after_epoch=args.stop_after_epoch)
