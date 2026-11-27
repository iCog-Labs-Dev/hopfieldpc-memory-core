"""Supervised digit classification with a frozen encoder and continuous GASS reads."""
from dataclasses import asdict

import torch
from torch.nn.functional import one_hot

from benchmarks.mnist_baseline import NUM_CLASSES, build_baseline_net, encode, make_splits, train_baseline
from gass import GASSBank


@torch.no_grad()
def classification_metrics(predict, features, labels, *, batch_size, device):
    """Rows of the confusion matrix are true digits and columns are predictions."""
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for batch, target in zip(features.split(batch_size), labels.split(batch_size)):
        prediction = predict(batch.to(device)).argmax(1).cpu()
        counts = torch.bincount(target.cpu() * NUM_CLASSES + prediction, minlength=NUM_CLASSES**2)
        confusion += counts.reshape_as(confusion)
    support = confusion.sum(1)
    per_digit = [correct / count if count else None for correct, count in zip(confusion.diag().tolist(), support.tolist())]
    return {"accuracy": int(confusion.diag().sum()) / len(labels), "count": len(labels),
            "accuracy_by_digit": per_digit, "support_by_digit": support.tolist(), "confusion": confusion.tolist()}


def fit_memory(bank, features, labels, *, epochs, batch_size, seed, on_epoch_end=None):
    """Only the first pass assimilates measurements. Later passes refine cached evidence."""
    generator = torch.Generator().manual_seed(seed)
    targets = one_hot(labels, num_classes=NUM_CLASSES).to(dtype=bank.values.dtype)
    reference_ids = torch.randperm(len(labels), generator=generator)[:bank.cfg.window_size]
    reference, reference_targets = features[reference_ids], targets[reference_ids]
    if not bank.initialize_from_embeddings(reference, reference_targets, seed=seed):
        raise ValueError("The training reference window has too few distinct feasible keys to seed the bank.")
    history = []
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(len(labels), generator=generator).split(batch_size):
            if epoch == 1:
                bank.online_update(features[indices], targets[indices])
            else:
                bank.refine_window(features[indices], targets[indices])
        report = {}
        if epoch % bank.cfg.structural_interval == 0:
            report = bank.adapt_structure(reference, reference_targets, epoch=epoch)
        row = {"epoch": epoch, "active_slots": bank.active_count,
               "accepted_edits": [t["operation"] for t in report.get("trials", []) if t["accepted"]]}
        if on_epoch_end is not None:
            row["validation"] = on_epoch_end()
        history.append(row)
    return history


def run_classification(args, data, device):
    splits = make_splits(data, seed=args.seed, val_size=256 if args.quick else 5000,
                         train_subset=2000 if args.quick else None, test_subset=512 if args.quick else None)
    if min(len(splits.train_labels), args.gass.window_size) < args.initial_memories:
        raise ValueError("The training reference window must contain at least as many images as initial memories.")
    bank = GASSBank(NUM_CLASSES, args.initial_memories, args.min_active_memories, args.max_memories,
                    args.gass, key_dim=args.embedding_dim, metric=args.metric).to(device)
    net = build_baseline_net(seed=args.seed, embed_dim=args.embedding_dim)
    train_baseline(net, splits.train_images, splits.train_labels, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed, device=device)
    net.eval().requires_grad_(False)
    encode_args = {"device": device, "batch_size": args.batch_size}
    train_features = encode(net.encoder, splits.train_images, **encode_args)
    val_features = encode(net.encoder, splits.val_images, **encode_args)
    def validate():
        return classification_metrics(bank.read, val_features, splits.val_labels, **encode_args)

    history = fit_memory(bank, train_features, splits.train_labels, epochs=args.memory_epochs,
                         batch_size=args.batch_size, seed=args.seed, on_epoch_end=validate)
    test_features = encode(net.encoder, splits.test_images, **encode_args)
    test = {name: classification_metrics(read, test_features, splits.test_labels, **encode_args)
            for name, read in (("direct", net.head), ("gass", bank.read))}
    return {"seed": args.seed, "encoder_epochs": args.epochs, "memory_epochs": args.memory_epochs,
            "embedding_dim": args.embedding_dim, "train_count": len(train_features),
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "capacity": {"min": bank.minimum, "initial": bank.initial, "max": len(bank.weights)},
            "gass_settings": asdict(bank.cfg), "history": history, "test": test}
