"""Paired MNIST comparison of ordinary GASS and retrieval-constrained ARKU."""
import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from marc_mnist.cli import parse_args
from marc_mnist.model import GASSVariant
from benchmarks.mnist_baseline import configure_threads, load_mnist, resolve_device
from gass_autoencoder import comparison_config
from gass_mnist_benchmark import _encode, _metrics
from mnist_reconstruction.model import AutoencoderVariant, build_sample_corruption


@dataclass(frozen=True)
class Profile:
    paired: bool
    retrieval_margin: float | None
    marc_window: str = "minibatch"
    bank_init: str = "random"


def comparison_profiles(comparison):
    strict = 1 / (8 * 784)
    if comparison == "initialization":
        profiles = {}
        for label, paired, margin in (("clean", False, 0.), ("paired", True, 0.), ("strict_paired", True, strict)):
            for init in ("random", "warm-up"):
                profiles[f"{label}_{init}"] = Profile(paired, margin, "reference", init)
        return profiles
    profiles = {"current_paired": Profile(True, None), "marc_paired": Profile(True, 0.),
                "strict_paired": Profile(True, strict)}
    if comparison == "reference":
        profiles.update(marc_reference=Profile(True, 0., "reference"), strict_reference=Profile(True, strict, "reference"))
    else:
        profiles.update(current_clean=Profile(False, None), marc_clean=Profile(False, 0.))
    return profiles


@torch.no_grad()
def window_loss(bank, queries, targets):
    state = bank._active_layout(bank.layout)
    logits = bank._read_logits(queries.double(), state.keys, state.weights)
    output = logits.softmax(1) @ state.values
    return float(bank.retrieval_loss(targets.double(), output, bank.retrieval_scale).mean())


@torch.no_grad()
def test_metrics(arm, cues, clean_logits, images, labels, centers):
    metrics = _metrics(arm, cues, clean_logits, images, labels, centers, 256)
    if arm.bank is not None:
        bank = arm.bank
        keys, values, weights = (buf[bank.mask] for buf in (bank.keys, bank.values, bank.weights))
        top_slot = bank._read_logits(cues, keys, weights).argmax(1)
        closest = torch.cdist(images.flatten(1).to(values), values.sigmoid()).argmin(1)
        metrics.update(closest_slot_agreement=float((top_slot == closest).float().mean()), active=len(keys))
    return metrics


def run_seed(args, cfg, device, train, labels, test, test_labels, seed):
    direct = AutoencoderVariant("Direct", cfg, seed, device)
    profiles = args.profiles
    arms = {name: GASSVariant(cfg, seed, device, replace(args.gass, retrieval_margin=p.retrieval_margin,
                                                          marc_window=p.marc_window, bank_init=p.bank_init))
            for name, p in profiles.items()}
    compare_reference = any(arm.uses_reference_for_marc for arm in arms.values())
    for arm in arms.values():
        arm.net = direct.net
        arm.optimizer = direct.optimizer
    generator = torch.Generator().manual_seed(seed)
    window = train[:args.gass.window_size]
    noisy_window = build_sample_corruption(window, "gaussian", .3, 9000 + seed)
    counts = {name: Counter() for name in arms}
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        noisy_train = build_sample_corruption(train, "gaussian", .3, 30000 + 1000 * seed + epoch)
        direct.net.train()
        for ids in torch.randperm(len(train), generator=generator).split(cfg.batch_size):
            images = train[ids].to(device)
            direct.optimizer.zero_grad(set_to_none=True)
            raw = direct.net.output_logits(direct.encode(images))
            clean = raw.detach().flatten(1)
            with torch.no_grad():
                noisy = direct.net.output_logits(direct.encode(noisy_train[ids].to(device))).flatten(1)
                if compare_reference:
                    ref_noisy, ref_clean = _encode(direct, torch.cat((noisy_window, window)), 256, device).chunk(2)
                    direct.net.train()
            (raw.sigmoid() - images).square().mean().backward()
            direct.optimizer.step()
            for name, arm in arms.items():
                if not arm.bank.ready:
                    continue
                profile = profiles[name]
                cues = noisy if profile.paired else clean
                before = window_loss(arm.bank, cues, clean)
                if compare_reference:
                    anchor = (ref_noisy if profile.paired else ref_clean, ref_clean)
                    before_anchor = window_loss(arm.bank, *anchor)
                arm.bank.online_update(cues, clean, retrieval_reference=anchor if arm.uses_reference_for_marc else None)
                change = window_loss(arm.bank, cues, clean) - before
                counts[name]["writes"] += 1
                counts[name]["writes_increasing_loss"] += int(change > 1e-10)
                if compare_reference:
                    anchor_change = window_loss(arm.bank, *anchor) - before_anchor
                    counts[name]["writes_increasing_reference_loss"] += int(anchor_change > 1e-10)
                protected_change = anchor_change if arm.uses_reference_for_marc else change
                if profile.retrieval_margin is not None and protected_change > 1e-10:
                    raise AssertionError(f"{name} increased protected retrieval loss by {protected_change}")
        clean = _encode(direct, window, 256, device)
        noisy = _encode(direct, noisy_window, 256, device)
        for name, arm in arms.items():
            cues = noisy if profiles[name].paired else clean
            if not arm.bank.ready:
                if epoch >= arm.settings.warmup_epochs:
                    if not arm.bank.initialize_from_embeddings(cues, clean, seed=seed):
                        raise RuntimeError(f"Could not seed {name} from the training reference")
                continue
            report = arm.bank.adapt_structure(cues, clean, epoch=epoch)
            for edit in report["trials"]:
                if edit["accepted"]:
                    regressed = edit["read_after"] > edit["read_before"] + 1e-10
                    counts[name]["accepted_edits"] += 1
                    counts[name]["edits_increasing_loss"] += int(regressed)
                    if profiles[name].retrieval_margin is not None and regressed:
                        raise AssertionError(f"{name} accepted a retrieval-increasing edit")
        if epoch % 4 == 0:
            print(f"Seed {seed}, epoch {epoch}/{cfg.epochs}", flush=True)

    # Only the completed models see official test images. Fit the digit probe on training labels.
    features = _encode(direct, train, 256, device)
    labels = labels.to(device)
    centers = torch.stack([features[labels == digit].mean(0) for digit in range(10)])
    test_logits = _encode(direct, test, 256, device)
    noisy_test = build_sample_corruption(test, "gaussian", .3, 22000 + seed)
    cues = _encode(direct, noisy_test, 256, device)
    metrics = {name: test_metrics(arm, cues, test_logits, test, test_labels, centers)
               for name, arm in {"direct": direct, **arms}.items()}
    print(f"Seed {seed} noisy MSE " + ", ".join(f"{n}={m['pixel_mse']:.5f}" for n, m in metrics.items()), flush=True)
    return {"seed": seed, "test": metrics, "training_counts": counts, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--comparison", choices=("proposal", "reference", "initialization"), default="proposal")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.epochs < 1 or len(set(options.seeds)) != len(options.seeds):
        parser.error("Use positive epochs and distinct seeds.")
    args = parse_args(["--bank-init", "random", "--initial-memories", "20", "--min-active-memories", "2",
                       "--max-memories", "80", "--window-size", "256", "--gass-read-beta", "80",
                       "--patience", "2", "--merge-threshold", ".90", "--support-fraction", ".10",
                       "--min-child-mass", "1", "--edit-budget", "5"])
    cfg = replace(comparison_config(args, False, {}), epochs=options.epochs, batch_size=128, learning_rate=.001)
    args.profiles = comparison_profiles(options.comparison)
    device = resolve_device(options.device)
    configure_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, labels, test, test_labels = load_mnist(args.data_dir)
    ids = torch.randperm(len(train), generator=torch.Generator().manual_seed(917))[768:4864]
    report = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
              "torch": str(torch.__version__), "epochs": cfg.epochs, "seeds": options.seeds,
              "train_count": len(ids), "test_count": len(test), "training_batch": cfg.batch_size,
              "learning_rate": cfg.learning_rate, "noise_sigma": .3, "initial_slots": 20, "slot_limits": [2, 80],
              "comparison": options.comparison, "settings": asdict(args.gass),
              "profiles": {n: asdict(p) for n, p in args.profiles.items()}, "runs": []}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in options.seeds:
        report["runs"].append(run_seed(args, cfg, device, train[ids], labels[ids], test, test_labels, seed))
        temporary = options.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(options.output)


if __name__ == "__main__":
    main()
