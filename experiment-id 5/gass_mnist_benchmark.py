"""Paired evaluation of GASS on MNIST. Run via mnist_autoencoder_ui_gass.py --benchmark.

Every arm sees the same detached pre-optimizer logits. Only the direct autoencoder
learns from pixel targets. Validation selects temperatures; official test images
are read once after training. A nearest-centroid digit probe is fit on clean
training logits, independently of every memory. Its accuracy is a diagnostic.
"""

import json
import os
import sys
import time
from dataclasses import asdict, replace

import torch
from torch.nn import functional as F

from mnist_reconstruction.model import AutoencoderVariant, build_sample_corruption, digit_overlap
from benchmarks.mnist_baseline import configure_threads, load_mnist, resolve_device
from gass_autoencoder import ReferenceVariant as GASSVariant, comparison_config
from gass import ReferenceBank as GASSBank
from hopfieldpc.ema import ema_step
from hopfieldpc.functional import read_from_logits, unit_rows as unit
from gass.geometry import association_loss


class ClusteringControl(GASSVariant):
    """Controls share initial capacity and data. Each implements its own write rule."""

    def __init__(self, rule, cfg, seed, device, settings):
        super().__init__(cfg, seed, device, replace(settings, read_score="cosine", write_score="cosine",
                         fixed_capacity=True, momentum=False, separation=None))
        self.rule = rule
        self.mass = None
        self.gradient_optimizer = None

    def initialize_from(self, z):
        bank = self.bank
        if not bank.initialize_from_embeddings(z, seed=self.seed):
            return
        self.mass = torch.zeros_like(bank.weights)
        self.mass[bank.mask] = bank.weights[bank.mask] * len(z)
        if self.rule == "Gradient":
            # Optimise the associative read on detached observations. The encoder is shared.
            self.centers = torch.nn.Parameter(bank.values.detach().clone())
            self.gradient_optimizer = torch.optim.Adam([self.centers], lr=self.optimizer.param_groups[0]["lr"])

    def write(self, z):
        bank = self.bank
        if self.rule == "Frozen":
            return
        slots = bank.mask.nonzero().flatten()
        if self.rule == "Gradient":
            with torch.enable_grad():
                centers = self.centers[slots]
                logits = bank.weights[slots].log() + self.settings.beta_r * (unit(z) @ unit(centers).T)
                reconstruction, _ = read_from_logits(logits, centers)
                self.gradient_optimizer.zero_grad(set_to_none=True)
                F.mse_loss(reconstruction, z).backward()
                self.gradient_optimizer.step()
                bank.values[slots] = self.centers[slots].detach()
                bank.keys[slots] = unit(bank.values[slots])
            return
        angular = unit(z) @ bank.keys[slots].T
        if self.rule == "Soft EMA":
            r = (bank.weights[slots].log() + self.settings.beta_w * angular).softmax(1)
        else:
            if self.rule == "Hard assignment":
                winners = torch.cdist(z, bank.values[slots]).argmin(1)
            else:
                winners = angular.argmax(1)
            r = F.one_hot(winners, len(slots)).to(z)
        mass = r.sum(0)
        occupied = mass > 0
        active = slots[occupied]
        weights = r[:, occupied]
        n = mass[occupied]
        observation = weights.T @ z / n[:, None]
        if self.rule == "Soft EMA":
            bank.values[active] = ema_step(bank.values[active], observation, alpha=.1)
            bank.keys[active] = unit(ema_step(bank.keys[active], unit(weights.T @ unit(z)), alpha=.1))
        else:
            previous = self.mass[active]
            bank.values[active] = (previous[:, None]*bank.values[active] + n[:, None]*observation) / (previous+n)[:, None]
            bank.keys[active] = unit(previous[:, None]*bank.keys[active] + weights.T @ unit(z))
        self.mass[active] += n
        bank.weights[slots] = self.mass[slots] / self.mass[slots].sum()

    def cluster_edits(self, z):
        """Hard spherical split–merge control, screened by overlap and cluster distortion."""
        if self.rule != "Split-merge clustering":
            return
        bank = self.bank
        slots = bank.mask.nonzero().flatten()
        gram = bank.keys[slots] @ bank.keys[slots].T
        gram.fill_diagonal_(-torch.inf)
        if len(slots) > bank.minimum and gram.max() >= self.settings.merge_cosine:
            pair = int(gram.argmax())
            a, b = int(slots[pair // len(slots)]), int(slots[pair % len(slots)])
            edit = bank._merge(bank.layout, a, b)
            if edit is not None:
                bank.keys.copy_(edit.layout.keys)
                bank.values.copy_(edit.layout.values)
                bank.weights.copy_(edit.layout.weights)
                self.mass[a] += self.mass[b]
                self.mass[b] = 0
        slots = bank.mask.nonzero().flatten()
        free = (~bank.mask).nonzero().flatten()
        if not len(free):
            return
        scores = unit(z) @ bank.keys[slots].T
        best, labels = scores.max(1)
        r = F.one_hot(labels, len(slots)).to(z)
        mass = r.sum(0)
        distortion = ((1-best)[:, None]*r).sum(0) / mass.clamp_min(1)
        supported = (mass >= 2*self.settings.min_child_mass) & (distortion > 1-self.settings.merge_cosine)
        if not supported.any():
            return
        col = int(distortion.masked_fill(~supported, -torch.inf).argmax())
        parent, child = int(slots[col]), int(free[0])
        members = z[labels == col]
        directions = unit(members)
        first = (directions @ bank.keys[parent]).argmin()
        second = (directions @ directions[first]).argmin()
        group = (directions @ directions[torch.stack((first, second))].T).argmax(1)
        counts = torch.bincount(group, minlength=2)
        if counts.min() < self.settings.min_child_mass:
            return
        old_mass, old_weight = self.mass[parent].clone(), bank.weights[parent].clone()
        for index, slot in enumerate((parent, child)):
            bank.values[slot] = members[group == index].mean(0)
            bank.keys[slot] = unit(directions[group == index].sum(0))
            fraction = counts[index] / len(group)
            self.mass[slot], bank.weights[slot] = old_mass*fraction, old_weight*fraction


def _variants(args, cfg, seed, device):
    variant_class = getattr(args, "gass_variant_class", GASSVariant)
    direct = AutoencoderVariant("Direct", cfg, seed, device)
    arms = {"Direct": direct, "EMA": AutoencoderVariant("EMA", cfg, seed, device),
            "Dynamic": AutoencoderVariant("Dynamic", cfg, seed, device)}
    settings = args.gass
    grid = args.validation_grid or [replace(settings,beta=b) for b in dict.fromkeys(args.beta_grid or [settings.beta])]
    candidates = {}
    for index, point in enumerate(grid):
        name = "GASS" if len(grid) == 1 else f"GASS candidate {index+1}"
        metric = args.metric if "mahalanobis" in (point.read_score,point.write_score) else None
        arms[name] = variant_class(cfg, seed, device, point, metric)
        candidates[name] = point
    if args.suite in ("ablations", "all"):
        variants = {
            "GASS scalar": {"momentum": False}, "GASS velocity": {"momentum": True},
            "GASS association only": {"value_aware": False}, "GASS value-aware": {"value_aware": True},
            "GASS shared cosine": {"read_score": "cosine", "write_score": "cosine", "read_beta": None, "write_beta": None},
            "GASS mixed": {"read_score": "dice", "write_score": "cosine", "write_beta": settings.beta*.5},
            "GASS Euclidean": {"read_score": "euclidean", "write_score": "euclidean"},
            "GASS fixed": {"fixed_capacity": True}, "GASS uniform": {"uniform_weights": True},
            "GASS no read loss": {"read_weight": 0.}, "GASS heuristic": {"heuristic_edits": True},
            "GASS Dice loss": {"retrieval_loss": "dice"},
        }
        if hasattr(variant_class, "benchmark_ablations"):
            variants = variant_class.benchmark_ablations(variants)
        for name, overrides in variants.items():
            altered = replace(settings, **overrides)
            metric = args.metric if "mahalanobis" in (altered.read_score, altered.write_score) else None
            arms[name] = variant_class(cfg, seed, device, altered, metric)
    if args.suite in ("baselines", "all"):
        for rule in ("Hard assignment", "Frozen", "Gradient", "Soft EMA", "Spherical clustering", "Split-merge clustering"):
            arms[rule] = ClusteringControl(rule, cfg, seed, device, settings)
    for arm in arms.values():
        arm.net, arm.optimizer = direct.net, direct.optimizer
    return direct, arms, candidates


@torch.no_grad()
def _encode(direct, images, batch_size, device):
    direct.net.eval()
    return torch.cat([direct.net.output_logits(direct.encode(batch.to(device))).flatten(1) for batch in images.split(batch_size)])


@torch.no_grad()
def _write(arm, z, task_score):
    if arm.bank is None or not arm.bank.ready:
        return
    if isinstance(arm, ClusteringControl):
        arm.write(z)
    elif arm.name == "EMA":
        arm._ema_write(z)
    else:
        arm.bank.online_update(z, task_score=task_score)


@torch.no_grad()
def _finish(arm, reference, epoch):
    bank = arm.bank
    if bank is None:
        return {}
    if not bank.ready:
        if isinstance(arm, ClusteringControl):
            arm.initialize_from(reference)
        elif not isinstance(arm, GASSVariant) or epoch >= arm.settings.warmup_epochs:
            bank.initialize_from_embeddings(reference, seed=arm.seed)
        return {}
    if isinstance(arm, ClusteringControl):
        arm.cluster_edits(reference)
    elif isinstance(arm, GASSVariant):
        if (epoch-arm.settings.warmup_epochs) % arm.settings.structural_interval == 0:
            return bank.adapt_structure(reference, epoch=epoch)
    return {}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_process_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                                                     "nonpaged", "pagefile", "peak_pagefile")]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        handle = ctypes.c_void_p(-1)  # Current-process pseudo handle
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
        return c.peak_rss if ok else None
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss*1024)


@torch.no_grad()
def _tracking(reference, settings, steps, seed, metric, bank_class=GASSBank):
    """Measure soft-assignment estimation error against known moving MNIST-logit means."""
    k = min(4, len(reference))
    bank = bank_class(reference.shape[1], k, k, k, settings, metric=metric).to(reference)
    if not bank.initialize_from_embeddings(reference, seed=seed):
        return {"skipped": "too few distinct feasible seed keys"}
    rng = torch.Generator(device=reference.device).manual_seed(seed+42000)
    def normal(x):
        return torch.randn(x.shape, generator=rng, device=x.device, dtype=x.dtype)
    truth = bank.values.clone() + bank.uncertainty.sqrt()[:,None]*normal(bank.values)
    variance = bank.scale * settings.observation_noise
    drift = (bank.scale * settings.process_noise * settings.dt).sqrt()
    history = []
    for _ in range(steps):
        truth = truth + drift * normal(truth)
        identities = torch.randint(k, (32,), generator=rng, device=truth.device)
        batch = truth[identities]
        observation = batch + variance.sqrt() * normal(batch)
        # The filter gets noisy observations only; latent slot identities are evaluation data.
        bank.online_update(observation)
        error = (bank.values - truth).square()
        history.append({"tracking_mse": float(error.mean()), "mean_filter_variance": float(bank.uncertainty.mean()),
                        "mse_over_variance": float((error/bank.uncertainty[:,None]).mean()),
                        "coordinate_95_coverage": float((error <= 1.96**2 * bank.uncertainty[:,None]).float().mean())})
    return {"model": "Gaussian random walk on MNIST-logit means, fixed initial slot identities", "history": history}


@torch.no_grad()
def _metrics(arm, queries, clean_logits, images, labels, centroids, batch_size, *, outputs=None):
    if outputs is None:
        outputs = torch.cat([arm.retrieve_logits(chunk) for chunk in queries.split(batch_size)])
    target = images.flatten(1).to(outputs)
    labels = labels.to(outputs.device)
    error = (outputs.sigmoid() - target).square().mean(1)
    direct_error = (queries.sigmoid() - target).square().mean(1)
    prediction = torch.cdist(outputs, centroids).argmin(1)
    correct = prediction == labels
    per_digit = {str(i): float(error[labels == i].mean()) for i in range(10) if (labels == i).any()}
    result = {"pixel_mse": float(error.mean()), "probe_accuracy": float(correct.float().mean()),
              "helped_fraction": float((error < direct_error).float().mean()),
              "harmed_fraction": float((error > direct_error).float().mean()), "mse_by_digit": per_digit,
              "worst_digit_mse": max(per_digit.values())}
    if isinstance(arm, GASSVariant) and arm.bank.ready:
        variance = getattr(arm.bank, "retrieval_scale", arm.bank.scale * arm.settings.observation_noise)
        result["associative_loss"] = float(association_loss(arm.settings.retrieval_loss, clean_logits, outputs, variance).mean())
    return result


def _run_seed(args, data, cfg, seed, device):
    train, labels, official, official_labels = data
    # Fixed validation identities across seeds; never read the official test split during selection.
    split = torch.randperm(len(train), generator=torch.Generator().manual_seed(917))
    nval = min(256 if args.quick else 5000, len(train)//5)
    validation, training = split[:nval], split[nval:]
    rng = torch.Generator().manual_seed(seed)
    chosen = training[torch.randperm(len(training), generator=rng)]
    training = chosen[:2000] if args.quick else chosen
    direct, arms, candidates = _variants(args, cfg, seed, device)
    windows = [args.gass.window_size, *(a.settings.window_size for a in arms.values() if isinstance(a, GASSVariant))]
    if min(len(training), min(windows)) <= cfg.memory.initial_memories:
        raise ValueError("The training reference window needs more observations than the initial slot count.")
    ref = training[:max(windows)]
    times = {name: 0. for name in arms}
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    val_images = train[validation]
    val_noisy = build_sample_corruption(val_images, args.noise, args.noise_level, seed+11000)
    untrained = [a for a in arms.values() if isinstance(a,GASSVariant) and not isinstance(a,ClusteringControl)
                 and a.settings.warmup_epochs == 0]
    if untrained and args.stream == "iid":
        init = _encode(direct, train[ref], cfg.eval_batch_size, device)
        for arm in untrained:
            arm.bank.initialize_from_embeddings(init[:arm.settings.window_size], seed=seed)
    seen = torch.zeros(len(train), dtype=torch.bool)
    best_digit = {name: {} for name in arms}
    final_val = {}
    for epoch in range(1, cfg.epochs+1):
        order = training[torch.randperm(len(training), generator=rng)]
        if args.stream == "by-digit":
            groups = min(5, cfg.epochs)
            group = groups*(epoch-1)//cfg.epochs
            order = order[labels[order]*groups//10 == group]
        elif args.stream == "rare-digit":
            rare = (labels[order] == 9).nonzero().flatten()
            keep = labels[order] != 9
            keep[rare[::10]] = True
            order = order[keep]
        # Structural data only comes from digits admitted to the stream by this epoch.
        seen[order] = True
        scope = training[seen[training]]
        epoch_ref = scope[:max(windows)]
        if epoch == 1 and untrained and args.stream != "iid":
            init = _encode(direct, train[epoch_ref], cfg.eval_batch_size, device)
            for arm in untrained:
                arm.bank.initialize_from_embeddings(init[:arm.settings.window_size], seed=seed)
        sums = dict.fromkeys(arms, 0.)
        score_sum = 0.
        direct.net.train()
        for ids in order.split(cfg.batch_size):
            images = train[ids].to(device)
            direct.optimizer.zero_grad(set_to_none=True)
            raw = direct.net.output_logits(direct.encode(images))
            loss = F.mse_loss(raw.sigmoid(), images)
            z = raw.detach().flatten(1)
            quality = digit_overlap(raw.detach().sigmoid(), images).item()
            score_sum += quality * len(ids)
            loss.backward()
            direct.optimizer.step()
            for name, arm in arms.items():
                _sync(device)
                start = time.perf_counter()
                with torch.no_grad():
                    output = arm.retrieve_logits(z).sigmoid().reshape_as(images)
                    sums[name] += float(F.mse_loss(output, images)) * len(ids)
                    _write(arm, z, quality)
                _sync(device)
                times[name] += time.perf_counter()-start
        reference = _encode(direct, train[epoch_ref], cfg.eval_batch_size, device)
        clean = _encode(direct, val_images, cfg.eval_batch_size, device)
        noisy = _encode(direct, val_noisy, cfg.eval_batch_size, device)
        # Training-only nearest-class-mean probe, fitted to the same representation for every arm.
        probe_ids = scope[:min(len(scope), 2048)]
        probe = _encode(direct, train[probe_ids], cfg.eval_batch_size, device)
        probe_labels = labels[probe_ids].to(device)
        centers = torch.stack([probe[probe_labels == i].mean(0) if (probe_labels == i).any()
                               else torch.full_like(probe[0], 1e10) for i in range(10)])
        epoch_stats = {}
        for name, arm in arms.items():
            _sync(device)
            start = time.perf_counter()
            window = reference[:arm.settings.window_size if isinstance(arm,GASSVariant) else args.gass.window_size]
            if name == "Dynamic" and arm.bank.ready:
                stats = arm.bank.adapt_structure(window, epoch=epoch, task_score=score_sum/len(order), allow_spawn=epoch<cfg.epochs)
            else:
                stats = _finish(arm, window, epoch)
            _sync(device)
            times[name] += time.perf_counter()-start
            val = _metrics(arm, noisy, clean, val_images, labels[validation], centers, cfg.eval_batch_size)
            final_val[name] = val["pixel_mse"]
            prev = best_digit[name]
            for digit, mse in val["mse_by_digit"].items():
                prev[digit] = min(prev.get(digit, mse), mse)
            val["mean_digit_forgetting_mse"] = sum(val["mse_by_digit"][d]-prev[d] for d in prev)/len(prev)
            trials = stats.get("trials", [])
            epoch_stats[name] = {"prewrite_mse": sums[name]/len(order), "validation_noisy": val,
                                 "active": arm.bank.active_count if arm.bank is not None else 0,
                                 "edits": trials, "rejected_fraction": sum(not t["accepted"] for t in trials)/len(trials) if trials else None}
            if hasattr(arm.bank, "last_write"):
                epoch_stats[name].update(last_write=arm.bank.last_write, refinement=stats.get("refinement", []))
        history.append(epoch_stats)
    winner = min(candidates, key=final_val.get)
    count = min(len(official), 512 if args.quick else len(official))
    images, test_labels = official[:count], official_labels[:count]
    noise = build_sample_corruption(images, args.noise, args.noise_level, seed+22000)
    clean = _encode(direct, images, cfg.eval_batch_size, device)
    noisy = _encode(direct, noise, cfg.eval_batch_size, device)
    metrics = {}
    for name, arm in arms.items():
        if name in candidates and name != winner:
            continue
        public = "GASS" if name == winner else name
        metrics[public] = {}
        readouts = {}
        for view, z in (("clean", clean), ("noisy", noisy)):
            with torch.no_grad():
                readouts[view] = torch.cat([arm.retrieve_logits(batch) for batch in z.split(cfg.eval_batch_size)])
            metrics[public][view] = _metrics(arm, z, clean, images, test_labels, centers, cfg.eval_batch_size,
                                             outputs=readouts[view])
        if hasattr(arm, "paired_logit_rms"):
            metrics[public]["perturbation"] = arm.paired_logit_rms(clean, noisy, readouts["clean"], readouts["noisy"])
        metrics[public]["bank_seconds"] = times[name]
        metrics[public]["bank_buffer_bytes"] = sum(x.numel()*x.element_size() for x in arm.bank.buffers()) if arm.bank is not None else 0
    selected = candidates[winner]
    tracking_metric = args.metric if "mahalanobis" in (selected.read_score, selected.write_score) else None
    tracking = (_tracking(reference, selected, args.tracking_steps, seed, tracking_metric, type(arms[winner].bank))
                if args.tracking_steps else None)
    return {"seed": seed, "train_count": len(training), "validation_count": nval, "test_count": count,
            "selected_beta": selected.beta, "selected_settings": asdict(selected), "tracking": tracking,
            "validation_candidates": {k: {"settings": asdict(v), "pixel_mse": final_val[k]} for k,v in candidates.items()},
            "epochs": history, "test": metrics, "wall_seconds": time.perf_counter()-started,
            "process_peak_rss_bytes": _peak_process_bytes(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None}


def _interval(values, draws):
    x = torch.tensor(values, dtype=torch.float64)
    bounds = torch.quantile(x[draws].mean(1), torch.tensor([.025, .975], dtype=x.dtype)).tolist() if len(x)>1 else None
    return {"mean": float(x.mean()), "seed_bootstrap_95": bounds}


def run_benchmark(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    seeds = args.seeds or list(range(args.seed, args.seed+3))
    overrides = {"beta_read": args.similarity_beta} if args.similarity_beta is not None else {}
    cfg = replace(comparison_config(args, args.quick, overrides), epochs=args.epochs, batch_size=args.batch_size,
                  eval_batch_size=max(64, args.batch_size), learning_rate=args.learning_rate)
    configure_threads(cfg.num_threads)
    device = resolve_device(args.device)
    data = load_mnist(args.data_dir)
    runs = []
    for seed in seeds:
        run = _run_seed(args, data, cfg, seed, device)
        runs.append(run)
        print(f"Seed {seed} finished in {run['wall_seconds']:.1f}s", flush=True)
    draws = torch.randint(len(runs), (4000,len(runs)), generator=torch.Generator().manual_seed(9))
    summary = {}
    for name in runs[0]["test"]:
        summary[name] = {}
        for view in ("clean", "noisy"):
            summary[name][view] = {metric: _interval([r["test"][name][view][metric] for r in runs], draws)
                                   for metric in ("pixel_mse", "probe_accuracy", "worst_digit_mse", "helped_fraction", "harmed_fraction")}
            summary[name][view]["paired_mse_delta_vs_direct"] = _interval([
                r["test"][name][view]["pixel_mse"]-r["test"]["Direct"][view]["pixel_mse"] for r in runs], draws)
        print(f"{name:24s} clean MSE {summary[name]['clean']['pixel_mse']['mean']:.6f}   "
              f"noisy MSE {summary[name]['noisy']['pixel_mse']['mean']:.6f}")
    result = {"suite": args.suite, "stream": args.stream, "gass": asdict(args.gass), "quick": args.quick,
              "epochs": cfg.epochs, "batch_size": cfg.batch_size, "noise": args.noise, "noise_level": args.noise_level,
              "capacity": {"min": cfg.memory.min_active_memories, "initial": cfg.memory.initial_memories,
                           "max": cfg.memory.max_memories}, "beta_grid": args.beta_grid,
              "learning_rate": cfg.learning_rate, "baseline_beta_read": cfg.memory.beta_read,
              "device": str(device), "seeds": seeds, "summary": summary, "runs": runs}
    if args.output is not None:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Metrics saved to {args.output}")
    return result
