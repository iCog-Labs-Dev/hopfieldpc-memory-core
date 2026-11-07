"""Step 2: controlled online alpha sweep, clean validation accuracy first.

One actual CNN training trajectory feeds independent detached banks. This is
equivalent to training one CNN per alpha because memory cannot affect its loss.
Alpha=0.1 must reproduce every Step 1 bank checkpoint exactly. No test data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import shutil
import time
import uuid
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from admission_ablation import CPU, parent_fingerprints, require_equal, training_fingerprint
from alpha_memory import VECTOR_ENCODING, ResidualEventWriter, apply_bank_batch, audit_alpha_log, make_alpha_bank
from benchmarks.mnist_baseline import environment_info, load_mnist_tensors, make_corrupted_test_set, robust_auc
from benchmarks.mnist_prototype_retrieval import mcnemar_test
from diagnose_memory import file_hash, memory_health
from hopfieldpc.diagnostics import attention_entropy
from mnist_online_memory import (
    EXPERIMENT_ROOT, build_baseline_net, checked_logits, extract, make_bank, validate_config,
)
from online_memory import centered_retrieve, correctness_categories
from online_memory_artifacts import (
    EventWriter, atomic_json, diagnostic_mode, restore_rng, rng_state, save_checkpoint, source_identity, timed,
)

VARIANT_ID = "exp001_step02_all_correct_alpha_sweep"
DEFAULT_ALPHAS = [0.03, 0.1, 0.3]
CONTROL = 0.1


def validate_alphas(values):
    if (not isinstance(values, list) or len(values) < 2
            or any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or not 0 < a <= 1 for a in values)
            or len(set(values)) != len(values) or CONTROL not in values):
        raise ValueError("Alphas must be unique finite values in (0,1], including the 0.1 control.")
    return [float(a) for a in values]


def alpha_name(alpha):
    return "alpha_"+str(float(alpha))


def alpha_directory(alpha):
    return alpha_name(alpha).replace('.', 'p')


def archive_source(root, source):
    repo = EXPERIMENT_ROOT.parent
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=repo).decode().split('\0')
    names = sorted(n for n in set(names)-{''} if (repo/n).is_file())
    digest = hashlib.sha256()
    archive = root/'training_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            data = (repo/name).read_bytes()
            digest.update(name.encode()); digest.update(data)
            z.writestr(name, data)
    if digest.hexdigest() != source['source_sha256']:
        raise ValueError('Source changed while creating the source archive.')
    atomic_json(root/'training_source.json', dict(source=source, archive_sha256=file_hash(archive), files=len(names)))


def shared_step(model, optimizer, banks, images, labels, ids, *, cfg, writers,
                epoch, batch, step, seed, run_id, costs, bank_costs):
    optimizer.zero_grad(set_to_none=True)
    with timed(costs, 'cnn_forward', CPU):
        z = model.encoder(images)
        logits = model.head(z)
        direct = checked_logits(logits).detach()
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite loss or embeddings.')
    queries = z.detach().clone()
    results = {}
    for alpha, bank in banks.items():
        with diagnostic_mode(model):
            results[alpha] = apply_bank_batch(model, bank, queries, direct, labels, ids, cfg=cfg,
                            epoch=epoch, batch=batch, step=step, seed=seed, run_id=run_id,
                            writer=writers.get(alpha), costs=bank_costs[alpha])
    with timed(costs, 'cnn_backward_optimizer', CPU):
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite CNN gradients.')
        optimizer.step()
    return float(loss.detach()), results


def prediction_metrics(pred, direct, labels, classes):
    if pred is None:
        return dict(accuracy=None, evaluated=0, fallback_count=len(labels), delta_vs_direct=None,
                    mcnemar_p=None, helped=0, harmed=0, both_correct=0, both_wrong=0,
                    per_class_accuracy=[None]*classes, per_class_denominator=[0]*classes,
                    confusion=None, status='incomplete_bank')
    correct = pred == labels
    categories = Counter(correctness_categories(direct, pred, labels))
    denom = torch.bincount(labels, minlength=classes)
    numer = torch.bincount(labels[correct], minlength=classes)
    confusion = torch.bincount(labels*classes+pred, minlength=classes**2).reshape(classes, classes)
    accuracy = float(correct.float().mean())
    return dict(accuracy=accuracy, evaluated=len(labels), fallback_count=0,
                delta_vs_direct=accuracy-float((direct == labels).float().mean()),
                mcnemar_p=mcnemar_test(direct == labels, correct)['p'],
                **{k: categories[k] for k in ('helped', 'harmed', 'both_correct', 'both_wrong')},
                per_class_accuracy=[int(n)/int(d) if d else None for n, d in zip(numer, denom)],
                per_class_denominator=denom.tolist(), confusion=confusion.tolist(), status='evaluated')


@torch.no_grad()
def evaluate_banks(model, banks, images, labels, cfg, *, original=None, fixed=None):
    z = extract(model, images, cfg, CPU)
    snapshots = {alpha_name(a): b.snapshot() for a, b in banks.items()}
    if original is not None:
        snapshots['accuracy_guided'] = original.snapshot()
    predictions, diagnostics = defaultdict(list), defaultdict(lambda: defaultdict(list))
    with diagnostic_mode(model):
        for start in range(0, len(z), cfg['eval_batch_size']):
            query = z[start:start+cfg['eval_batch_size']]
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
                retrieved, _ = centered_retrieve(query, fixed['values'], fixed['center'], beta=cfg['hopfield_beta'],
                            num_updates=cfg['hopfield_num_updates'], normalization_epsilon=cfg['normalization_epsilon'])
                predictions['fixed_kmeans_reference'].append(checked_logits(model.head(retrieved)))
    joined = {name: torch.cat(parts) for name, parts in predictions.items()}
    names = ['direct', *snapshots] + (['fixed_kmeans_reference'] if fixed is not None else [])
    metrics = {name: prediction_metrics(joined.get(name), joined['direct'], labels, cfg['num_classes']) for name in names}
    for name, fields in diagnostics.items():
        metrics[name].update({k: float(torch.cat(v).mean()) for k, v in fields.items()})
    return metrics


def protected_files(parent, step1, reference, cfg, seeds):
    paths = {str((parent/name).resolve()): digest for name, digest in parent_fingerprints(parent, cfg, seeds).items()}
    for name in ('manifest.json', 'results.json'):
        paths[str((step1/name).resolve())] = file_hash(step1/name)
    for seed in seeds:
        folder = Path(reference[seed]['worker_seed_directory'])
        for epoch in range(cfg['warmup_epochs']+1, cfg['total_epochs']+1):
            path = folder/f'checkpoints/epoch_{epoch}.pt'
            paths[str(path.resolve())] = file_hash(path)
    return paths


def train_seed(root, parent, reference, cfg, alphas, seed, images, labels, source, *, resume=False, stop_after_epoch=None):
    folder = root/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    warm = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{cfg["warmup_epochs"]}.pt', map_location='cpu', weights_only=False)
    if warm['config'] != cfg or warm['bank']['_extra_state']['version'] != 0:
        raise ValueError('Parent warm-up is incompatible or already contains memory.')
    ids = {k: torch.as_tensor(warm['split_ids'][k], dtype=torch.long) for k in ('train', 'val')}
    all_ids = torch.cat(list(ids.values()))
    if len(all_ids) != len(all_ids.unique()) or int(all_ids.min()) < 0 or int(all_ids.max()) >= len(images):
        raise ValueError('Invalid or overlapping original split IDs.')
    coverage = torch.bincount(labels[ids['train']], minlength=cfg['num_classes'])
    if len(coverage) != cfg['num_classes'] or (coverage < cfg['fixed_reference']['prototypes_per_class']).any():
        raise ValueError('Training class coverage is insufficient.')
    latest = folder/'latest.json'
    continuation = resume and latest.is_file()
    state = torch.load(folder/json.loads(latest.read_text())['checkpoint'], map_location='cpu', weights_only=False) if continuation else warm
    if continuation:
        if state['source'] != source or state['config'] != cfg or state['alphas'] != alphas or state['variant_id'] != VARIANT_ID:
            raise ValueError('Resume source, config, alpha grid, or variant differs.')
        require_equal(state['split_ids'], ids, 'resume.splits')
    model = build_baseline_net(seed)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    optimizer.load_state_dict(state['optimizer'])
    banks = {a: make_alpha_bank(cfg, model.head.fc.in_features, a) for a in alphas}
    for a, bank in banks.items():
        bank.load_state_dict(state['banks'][alpha_name(a)] if continuation else warm['bank'])
        if not bank.center_initialized or list(bank.parameters()):
            raise ValueError('Bank must be centered buffers without optimizer parameters.')
    shuffle = torch.Generator()
    shuffle.set_state(state['shuffle_rng'])
    restore_rng(state['rng'])
    if continuation:
        expected = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{state["epoch"]}.pt', map_location='cpu', weights_only=False)
        require_equal(model.state_dict(), expected['model'], 'resume.CNN')
        require_equal(optimizer.state_dict(), expected['optimizer'], 'resume.Adam')
        require_equal(shuffle.get_state(), expected['shuffle_rng'], 'resume.shuffle')
    costs = dict(state['costs']) if continuation else {}
    bank_costs = {a: dict(state['bank_costs'][alpha_name(a)]) if continuation else {} for a in alphas}
    writers = {a: ResidualEventWriter(folder/alpha_directory(a), cfg['event_chunk_size'], cfg['write_chunk_size'],
                              state['log_cursors'][alpha_name(a)] if continuation else None,initial_bank=banks[a]) for a in alphas}
    if not continuation:
        np.savez_compressed(folder/'split_ids.npz', **{k: v.numpy() for k, v in ids.items()})
        for a in alphas:
            writers[a].event(dict(type='center', center=banks[a].center.tolist(), alpha=a, origin='original_warmup_train_mean'))
    history = list(state['history']) if continuation else []
    step = state['optimizer_step']
    previous_elapsed = state.get('elapsed_seconds', 0.) if continuation else 0.
    started = time.perf_counter()
    try:
        for epoch in range(state['epoch']+1, cfg['total_epochs']+1):
            model.train()
            order = torch.randperm(len(ids['train']), generator=shuffle)
            totals = {a: Counter() for a in alphas}
            actions = {a: Counter() for a in alphas}
            weighted_loss = 0.
            print(f'alpha sweep seed {seed}: epoch {epoch}/{cfg["total_epochs"]}', flush=True)
            with timed(costs, 'shared_training_wall', CPU):
                for batch, start in enumerate(range(0, len(order), cfg['batch_size'])):
                    if batch % 50 == 0 and shutil.disk_usage(root).free < 64*1024**2:
                        raise OSError('Free disk space is below the 64 MiB checkpoint/log reserve; resume after freeing space.')
                    source_ids = ids['train'][order[start:start+cfg['batch_size']]]
                    loss, batch_results = shared_step(model, optimizer, banks, images[source_ids], labels[source_ids], source_ids,
                                cfg=cfg, writers=writers, epoch=epoch, batch=batch, step=step, seed=seed,
                                run_id=root.name, costs=costs, bank_costs=bank_costs)
                    step += 1
                    weighted_loss += loss*len(source_ids)
                    for a, stats in batch_results.items():
                        actions[a].update(stats['counts'])
                        totals[a].update({k: v for k, v in stats.items() if k not in ('counts', 'local_diagnostics_available')})
                    if (batch+1) % 50 == 0:
                        print(f'  seed {seed} epoch {epoch} batch {batch+1}: '+
                              ', '.join(f'a={a:g} writes={b.version}' for a,b in banks.items())+
                              f', seconds={time.perf_counter()-started:.1f}', flush=True)
            expected = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{epoch}.pt', map_location='cpu', weights_only=False)
            require_equal(model.state_dict(), expected['model'], 'CNN')
            require_equal(optimizer.state_dict(), expected['optimizer'], 'Adam')
            require_equal(shuffle.get_state(), expected['shuffle_rng'], 'shuffle')
            if step != expected['optimizer_step']:
                raise AssertionError('Optimizer progress differs.')
            control = torch.load(Path(reference[seed]['worker_seed_directory'])/f'checkpoints/epoch_{epoch}.pt', map_location='cpu', weights_only=False)
            require_equal(banks[CONTROL].state_dict(), control['bank'], 'alpha_0.1_positive_control')
            control_history = next(h for h in reference[seed]['history'] if h['epoch'] == epoch)
            require_equal(dict(actions[CONTROL]), control_history['counts'], 'positive_control.decisions')
            for name in ('examples', 'compared', 'direct_correct', 'hop_correct', 'fallback_count', 'in_sample_post_write_correct'):
                require_equal(totals[CONTROL][name], control_history['totals'].get(name, 0), 'positive_control.'+name)
            with timed(costs, 'clean_validation', CPU):
                validation = evaluate_banks(model, banks, images[ids['val']], labels[ids['val']], cfg)
            require_equal(validation['direct'], control_history['validation']['direct'], 'validation.direct')
            require_equal(validation[alpha_name(CONTROL)], control_history['validation']['all_direct_correct'], 'validation.alpha_0.1')
            metrics = dict(epoch=epoch, optimizer_step=step, loss=weighted_loss/len(ids['train']),
                           banks={alpha_name(a): dict(counts=dict(actions[a]), totals=dict(totals[a]), storage=banks[a].storage(step)) for a in alphas},
                           validation=validation, cnn_exact=True, adam_exact=True, shuffle_exact=True, alpha_0p1_bank_exact=True)
            history.append(metrics)
            cursors = {alpha_name(a): writers[a].flush() for a in alphas}
            relative = f'checkpoints/epoch_{epoch}.pt'
            save_checkpoint(folder/relative, dict(variant_id=VARIANT_ID, source=source, config=cfg, alphas=alphas,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), banks={alpha_name(a): b.state_dict() for a,b in banks.items()},
                    shuffle_rng=shuffle.get_state(), rng=rng_state(), epoch=epoch, optimizer_step=step, history=history,
                    log_cursors=cursors, split_ids=ids, costs=costs, bank_costs={alpha_name(a): v for a,v in bank_costs.items()},
                    elapsed_seconds=previous_elapsed+time.perf_counter()-started))
            atomic_json(latest, dict(checkpoint=relative, epoch=epoch, log_cursors=cursors))
            atomic_json(folder/'progress.json', dict(history=history))
            print(f'  seed {seed} epoch {epoch}: CNN/Adam and alpha=.1 bank exact; validation '+
                  ', '.join(f'{name}={m["accuracy"]}' for name,m in validation.items()), flush=True)
            if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < cfg['total_epochs']:
                return None
        final_state = torch.load(parent/f'seed_{seed}/checkpoints/epoch_{cfg["total_epochs"]}.pt', map_location='cpu', weights_only=False)
        require_equal(model.state_dict(), final_state['model'], 'final.CNN')
        require_equal(optimizer.state_dict(), final_state['optimizer'], 'final.Adam')
        original = make_bank(cfg, model.head.fc.in_features)
        original.load_state_dict(final_state['bank'])
        fixed = torch.load(parent/f'seed_{seed}/fixed_reference.pt', map_location='cpu', weights_only=False)
        rows, aucs = [], {}
        with timed(costs, 'validation_corruptions', CPU):
            for kind, key in (('gaussian','gaussian_sigmas'),('salt_pepper','salt_pepper_probs'),('occlusion','occlusion_sizes')):
                for level in cfg[key]:
                    corrupted = make_corrupted_test_set(images[ids['val']], seed=seed, kind=kind, level=level, stream=1)
                    scored = evaluate_banks(model, banks, corrupted, labels[ids['val']], cfg, original=original, fixed=fixed)
                    for method, value in scored.items():
                        if method in ('direct', 'accuracy_guided', 'fixed_kmeans_reference', alpha_name(CONTROL)):
                            old_method = 'all_direct_correct' if method == alpha_name(CONTROL) else method
                            old = next(r for r in reference[seed]['validation_results'] if r['kind']==kind and r['level']==level and r['method']==old_method)
                            for field in ('accuracy','confusion','helped','harmed','evaluated','fallback_count'):
                                require_equal(value[field], old[field], 'reference_validation.'+method+'.'+field)
                        rows.append(dict(seed=seed, method=method, kind=kind, level=level, **value))
                for method in scored:
                    values = [r['accuracy'] for r in rows if r['kind']==kind and r['method']==method]
                    aucs[method+':'+kind] = robust_auc(cfg[key], values) if all(v is not None for v in values) else None
        cursors = {alpha_name(a): writers[a].flush() for a in alphas}
        audits = {}
        with timed(costs, 'independent_mutation_audits', CPU):
            for a in alphas:
                print(f'auditing seed {seed} alpha {a:g}', flush=True)
                audit = audit_alpha_log(folder/alpha_directory(a), cursors[alpha_name(a)], cfg, banks[a], ids['train'], labels)
                if audit['examples'] != len(ids['train'])*(cfg['total_epochs']-cfg['warmup_epochs']):
                    raise AssertionError('Example logs are incomplete.')
                audits[alpha_name(a)] = audit
        result = dict(seed=seed, history=history, validation_results=rows, robust_auc=aucs, audits=audits,
                      banks={alpha_name(a): b.storage(step) for a,b in banks.items()},
                      memory_health={alpha_name(a): memory_health(model.head,b.snapshot().values,b.snapshot().labels) if b.ready else None for a,b in banks.items()},
                      costs=costs, bank_costs={alpha_name(a): c for a,c in bank_costs.items()}, log_cursors=cursors,
                      elapsed_seconds=previous_elapsed+time.perf_counter()-started,
                      exact_cnn_adam_every_epoch=True, exact_alpha_0p1_every_epoch=True)
        atomic_json(folder/'results.json', result)
        return result
    except Exception as error:
        atomic_json(folder/'failure.json', dict(error=type(error).__name__, message=str(error), optimizer_step=step,
                    uncommitted_log_cursors={alpha_name(a): w.flush() for a,w in writers.items()}))
        raise


def select_alpha(results, alphas):
    ranked = []
    for alpha in alphas:
        name = alpha_name(alpha)
        rows = [next(r for r in s['validation_results'] if r['method']==name and r['kind']=='gaussian' and r['level']==0) for s in results]
        if any(r['accuracy'] is None for r in rows):
            continue
        correct = sum(r['both_correct']+r['helped'] for r in rows)
        noise = statistics.mean(s['robust_auc'][name+':'+k] for s in results for k in ('gaussian','salt_pepper','occlusion'))
        ranked.append((correct, noise, -abs(alpha-CONTROL), -alpha, alpha))
    if not ranked:
        return dict(alpha=None, reason='No alpha has a complete bank across every seed.')
    winner = max(ranked)
    return dict(alpha=winner[-1], clean_correct=winner[0], secondary_mean_auc=winner[1],
                criterion='Maximize total clean validation correct; ties: mean of three corruption AUCs, then closest to 0.1, then smaller alpha.',
                validation_selection_only=True)


def reports(root, results, manifest):
    selection = select_alpha(results, manifest['alphas'])
    atomic_json(root/'results.json', dict(manifest=manifest, seeds=results, selection=selection))
    atomic_json(root/'selection.json', selection)
    rows = [r for s in results for r in s['validation_results']]
    with (root/'results.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['seed','method','kind','level','accuracy','delta_vs_direct','evaluated','fallback_count','helped','harmed','both_correct','both_wrong','mcnemar_p']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    methods = ['direct','accuracy_guided',*[alpha_name(a) for a in manifest['alphas']], 'fixed_kmeans_reference']
    def clean(s, name):
        return next(r['accuracy'] for r in s['validation_results'] if r['kind']=='gaussian' and r['level']==0 and r['method']==name)
    lines = ['# Step 2: alpha comparison', '', 'Validation only. Clean accuracy is the primary selection criterion; noise robustness breaks ties.',
             'Independent banks share an actual direct-CE CNN training trajectory. No backbone feature cache is used.',
             'CNN/Adam/shuffle and the alpha=0.1 bank were checked exactly against prior checkpoints at every epoch.',
             'Every exposure and mutation is logged. Per-write local prediction diagnostics are explicitly unavailable; batch snapshot/post-write diagnostics remain enabled.',
             '', '| Method | '+ ' | '.join('Seed '+str(s['seed']) for s in results)+' | Mean clean accuracy |',
             '|---|'+'---:|'*(len(results)+1)]
    for method in methods:
        values = [clean(s,method) for s in results]
        available = [v for v in values if v is not None]
        lines.append('| '+method+' | '+' | '.join('unavailable' if v is None else f'{100*v:.3f}%' for v in values)+
                     ' | '+(f'{100*statistics.mean(available):.3f}% (n={len(available)})' if available else 'unavailable')+' |')
    lines += ['', '## Normalized validation robustness AUC', '', '| Method | Gaussian | Salt-and-pepper | Occlusion |', '|---|---:|---:|---:|']
    for method in methods:
        values = [[s['robust_auc'][method+':'+k] for s in results if s['robust_auc'][method+':'+k] is not None] for k in ('gaussian','salt_pepper','occlusion')]
        lines.append('| '+method+' | '+' | '.join(f'{statistics.mean(v):.6f}' if v else 'unavailable' for v in values)+' |')
    lines += ['', 'Higher AUC is better; this is accuracy integrated over corruption strength, not ROC AUC.',
              '', f'Selected alpha: {selection["alpha"]}. This selection uses validation data; no new test result or across-seed significance claim is made.',
              '', '[Structured metrics](results.json) | [CSV](results.csv) | [Manifest](manifest.json) | [Selection](selection.json)']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for ax,kind in zip(axes, ('gaussian','salt_pepper','occlusion')):
        levels = sorted({r['level'] for r in rows if r['kind']==kind})
        for method in methods:
            values = []
            for level in levels:
                available = [r['accuracy'] for r in rows if r['kind']==kind and r['level']==level and r['method']==method and r['accuracy'] is not None]
                values.append(statistics.mean(available)*100 if available else float('nan'))
            ax.plot(levels,values,marker='o',label=method)
        ax.set(title='Validation: '+kind,xlabel='Corruption level',ylabel='Accuracy (%)');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'validation_corruptions.png',dpi=140);plt.close(fig)


def run_study(step1=None, *, alphas=None, seeds=None, output=None, resume=False, stop_after_epoch=None):
    prior = None
    if resume:
        if output is None or step1 is not None or alphas is not None or seeds is not None:
            raise ValueError('Resume requires only its existing output directory.')
        prior = json.loads((output/'manifest.json').read_text())
        if prior['variant_id'] != VARIANT_ID:
            raise ValueError('Wrong resume variant.')
        step1, alphas, seeds = Path(prior['step1_run']), prior['alphas'], prior['seeds']
    if step1 is None:
        raise ValueError('A completed Step 1 comparison is required.')
    step1 = step1.resolve()
    reference_data = json.loads((step1/'results.json').read_text())
    reference = {s['seed']: s for s in reference_data['seeds']}
    reference_manifest = json.loads((step1/'manifest.json').read_text())
    if reference_manifest['status'] != 'complete' or not reference_manifest['validation_only']:
        raise ValueError('Step 1 must be complete and validation-only.')
    parent = Path(reference_manifest['parent_run']).resolve()
    cfg = validate_config(json.loads((parent/'manifest.json').read_text())['config'])
    if cfg != reference_manifest['config'] or cfg['ema_alpha'] != CONTROL:
        raise ValueError('Parent config or alpha=0.1 control is incompatible.')
    alphas = validate_alphas(DEFAULT_ALPHAS if alphas is None else alphas)
    seeds = sorted(reference) if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s,bool) or not isinstance(s,int) or s not in reference for s in seeds):
        raise ValueError('Seeds must be a unique nonempty subset of Step 1 seeds.')
    if stop_after_epoch is not None and not cfg['warmup_epochs'] < stop_after_epoch <= cfg['total_epochs']:
        raise ValueError('Invalid stopping epoch.')
    for seed in seeds:
        if not reference[seed]['audit']['replay_exact'] or not reference[seed]['cnn_exact_every_epoch']:
            raise ValueError('Step 1 reference is unaudited.')
    torch.set_num_threads(cfg['num_threads'])
    source = source_identity()
    protected = protected_files(parent, step1, reference, cfg, seeds)
    images, labels = load_mnist_tensors(cfg['data_dir'], train=True)
    fingerprint = training_fingerprint(images, labels)
    if fingerprint != reference_manifest['training_dataset_sha256']:
        raise ValueError('Official training data differs from Step 1.')
    if prior:
        for key,value in (('source',source),('protected_files',protected),('training_dataset_sha256',fingerprint),('config',cfg)):
            if prior[key] != value:
                raise ValueError(f'Resume {key} differs.')
    root = (output or EXPERIMENT_ROOT/'results'/'variants'/f'alpha_{uuid.uuid4().hex}').resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        archive_source(root, source)
    manifest = dict(variant_id=VARIANT_ID, step1_run=str(step1), parent_run=str(parent), config=cfg,
                    alphas=alphas, seeds=seeds, source=source, protected_files=protected,
                    bank_settings={alpha_name(a): dict(ema_alpha=a, admission_policy='all_direct_correct',
                            max_memories_per_class=cfg['max_memories_per_class'], beta=cfg['hopfield_beta'],
                            merge_distance_threshold=cfg['merge_distance_threshold'],
                            normalization_epsilon=cfg['normalization_epsilon']) for a in alphas},
                    training_dataset_sha256=fingerprint, validation_only=True, test_set_opened=False,
                    environment=environment_info(CPU), local_write_diagnostics=False,
                    log_vector_encoding=VECTOR_ENCODING,
                    disk_reserve_bytes=64*1024**2,
                    shared_actual_cnn_training=True, backbone_feature_cache_used=False,
                    selection_rule='clean_correct_count; ties: mean of Gaussian/salt-pepper/occlusion AUC; closest to 0.1; smaller alpha',
                    status='running')
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'resolved_settings.json',dict(variant_id=VARIANT_ID,base_cnn_config=cfg,
                bank_settings=manifest['bank_settings'],seeds=seeds,selection_rule=manifest['selection_rule']))
    print('Alpha sweep output: '+str(root),flush=True)
    completed = []
    try:
        for seed in seeds:
            file = root/f'seed_{seed}/results.json'
            result = json.loads(file.read_text()) if resume and file.is_file() else train_seed(root,parent,reference,cfg,alphas,seed,images,labels,source,
                                        resume=resume,stop_after_epoch=stop_after_epoch)
            if result is None:
                atomic_json(root/'manifest.json',manifest|dict(status='incomplete',last_seed=seed))
                return root
            completed.append(result)
        if any(file_hash(Path(path)) != digest for path,digest in protected.items()):
            raise AssertionError('A protected original or Step 1 artifact changed.')
        manifest |= dict(status='complete', protected_artifacts_unchanged=True,
                         exact_cnn_adam_and_alpha_0p1_every_epoch=True,
                         all_mutation_audits_passed=True)
        reports(root,completed,manifest)
        atomic_json(root/'manifest.json',manifest)
    except Exception as error:
        atomic_json(root/'manifest.json',manifest|dict(status='failed',error=type(error).__name__,message=str(error)))
        raise
    print('Alpha sweep complete: '+str(root),flush=True)
    return root


def aggregate_studies(runs, *, output=None):
    """Combine unique completed seed results; recompute selection over all seeds."""
    if not runs:
        raise ValueError('At least one worker run is required.')
    runs = [Path(p).resolve() for p in runs]
    manifests = [json.loads((p/'manifest.json').read_text()) for p in runs]
    first = manifests[0]
    if any(m.get('variant_id') != VARIANT_ID for m in manifests):
        raise ValueError('Wrong alpha worker variant.')
    for manifest in manifests[1:]:
        for key in ('source','config','alphas','bank_settings','log_vector_encoding','training_dataset_sha256','parent_run','step1_run','selection_rule'):
            if manifest[key] != first[key]:
                raise ValueError('Incompatible worker '+key)
    protected, required, completed, origins = {}, set(), {}, []
    for root,manifest in zip(runs,manifests):
        required.update(manifest['seeds'])
        for path,digest in manifest['protected_files'].items():
            if path in protected and protected[path] != digest:
                raise ValueError('Incompatible protected artifact hashes.')
            protected[path] = digest
        origin = dict(run_directory=str(root), manifest_sha256=file_hash(root/'manifest.json'), completed_seeds=[])
        for seed in manifest['seeds']:
            file = root/f'seed_{seed}/results.json'
            if not file.is_file():
                continue
            if seed in completed:
                raise ValueError('Duplicate completed seed: '+str(seed))
            result = json.loads(file.read_text())
            if (result['seed'] != seed or not result['exact_cnn_adam_every_epoch'] or not result['exact_alpha_0p1_every_epoch']
                    or set(result['audits']) != {alpha_name(a) for a in first['alphas']}
                    or any(not a['replay_exact'] or not a['independent_assignment_and_ema_verified'] for a in result['audits'].values())):
                raise ValueError('A seed has not passed its required audits.')
            cfg = first['config']
            if [h['epoch'] for h in result['history']] != list(range(cfg['warmup_epochs']+1,cfg['total_epochs']+1)):
                raise ValueError('A seed has an incomplete epoch history.')
            with np.load(root/f'seed_{seed}/split_ids.npz',allow_pickle=False) as ids:
                expected = len(ids['train'])*(cfg['total_epochs']-cfg['warmup_epochs'])
            if any(a['examples'] != expected for a in result['audits'].values()):
                raise ValueError('A seed has incomplete exposure logs.')
            result['worker_seed_directory'] = str(file.parent)
            completed[seed] = result
            origin['completed_seeds'].append(dict(seed=seed,result_sha256=file_hash(file)))
        origins.append(origin)
    if set(completed) != required:
        raise ValueError('Missing completed seeds: '+str(sorted(required-set(completed))))
    if any(file_hash(Path(p)) != digest for p,digest in protected.items()):
        raise ValueError('A protected original/reference artifact changed.')
    root = (output or EXPERIMENT_ROOT/'results'/'variants'/f'alpha_comparison_{uuid.uuid4().hex}').resolve()
    root.mkdir(parents=True,exist_ok=False)
    manifest = {**first, 'variant_id':VARIANT_ID+'_aggregate', 'artifact_type':'completed_seed_aggregate',
                'seeds':sorted(completed), 'protected_files':protected, 'source_runs':origins,
                'aggregation_source':source_identity(), 'status':'complete',
                'protected_artifacts_unchanged':True, 'all_mutation_audits_passed':True,
                'exact_cnn_adam_and_alpha_0p1_every_epoch':True,
                'selection_scope':'All reported seeds; per-worker selections are not used.'}
    atomic_json(root/'manifest.json',manifest)
    atomic_json(root/'resolved_settings.json',dict(variant_id=manifest['variant_id'],base_cnn_config=manifest['config'],
                bank_settings=manifest['bank_settings'],seeds=manifest['seeds'],selection_rule=manifest['selection_rule']))
    reports(root,[completed[s] for s in sorted(completed)],manifest)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--step1-run',type=Path)
    group.add_argument('--resume',type=Path)
    group.add_argument('--aggregate',type=Path,nargs='+')
    parser.add_argument('--alphas',type=float,nargs='+')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stop-after-epoch',type=int)
    args = parser.parse_args()
    if args.resume and args.output:
        parser.error('--resume already identifies the output directory')
    if args.aggregate:
        if args.alphas or args.seeds or args.stop_after_epoch is not None:
            parser.error('--aggregate combines completed worker seeds without training')
        print(aggregate_studies(args.aggregate,output=args.output))
    else:
        run_study(args.step1_run,alphas=args.alphas,seeds=args.seeds,output=args.resume or args.output,
                  resume=bool(args.resume),stop_after_epoch=args.stop_after_epoch)
