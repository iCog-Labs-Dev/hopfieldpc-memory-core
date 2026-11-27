"""Experiment 5: MNIST classification by continuous key-to-label GASS retrieval."""
import argparse
import json
import math
from pathlib import Path

from gass_options import ROOT, add_memory_options, load_metric, memory_settings
from benchmarks.mnist_baseline import configure_threads, load_mnist, resolve_device
from gass_classification import run_classification
from gass import GASSConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = GASSConfig(read_score="cosine", write_score="cosine", retrieval_loss="squared")
    add_memory_options(parser, defaults)
    parser.set_defaults(initial_memories=20, min_active_memories=2, max_memories=80)
    parser.add_argument("--quick", action="store_true", help="Use 2,000 training, 256 validation and 512 test images")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised encoder training epochs")
    parser.add_argument("--memory-epochs", type=int, default=3, help="GASS passes over the frozen training embeddings")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, help="Write settings, validation history and test confusion matrices to JSON")
    args = parser.parse_args(argv)
    if min(args.epochs, args.memory_epochs, args.batch_size, args.embedding_dim) < 1:
        parser.error("Epochs, batch size and embedding dimension must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not 1 <= args.min_active_memories <= args.initial_memories <= args.max_memories:
        parser.error("Require 1 <= min-active-memories <= initial-memories <= max-memories.")
    if args.window_size < args.initial_memories or args.mixture_floor * args.max_memories >= 1:
        parser.error("Check that the window can seed every initial slot and that mixture-floor is below 1/max-memories.")
    try:
        args.gass = memory_settings(args, defaults)
        args.metric = load_metric(args.metric)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    configure_threads(2)
    result = run_classification(args, load_mnist(args.data_dir), resolve_device(args.device))
    for row in result["history"]:
        print(f"Memory epoch {row['epoch']}   active slots {row['active_slots']}   "
              f"validation accuracy {row['validation']['accuracy']:.2%}")
    print("Digit     Direct     GASS")
    direct, gass = result["test"]["direct"], result["test"]["gass"]
    for digit, (a, b) in enumerate(zip(direct["accuracy_by_digit"], gass["accuracy_by_digit"])):
        if a is not None:
            print(f"{digit:>5}   {a:8.2%} {b:8.2%}")
    print(f"Test    {direct['accuracy']:8.2%} {gass['accuracy']:8.2%}   ({gass['count']} images)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
