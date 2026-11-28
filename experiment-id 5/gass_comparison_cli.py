"""Command-line configuration for the two GASS reconstruction comparisons."""
import json
import math
from dataclasses import replace
from pathlib import Path

from gass_options import ROOT, add_memory_options, load_metric, memory_settings
from gass_autoencoder import GASSVariant, ReferenceVariant
from gass import GASSConfig, ReferenceConfig
from mnist_reconstruction.model import argument_parser


def build_comparison_parser(description, defaults, *, bank_init_choices=("warm-up",)):
    parser = argument_parser(description, bank_init_choices=bank_init_choices)
    parser.set_defaults(data_dir=str(ROOT / "data"))
    parser.add_argument("--beta-read", dest="similarity_beta", type=float, help="Override the baseline read temperature")
    add_memory_options(parser, defaults)
    parser.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true", help="Run paired MNIST validation and test evaluation without Tk")
    parser.add_argument("--suite", choices=("comparison", "ablations", "baselines", "all"), default="comparison")
    parser.add_argument("--seeds", type=int, nargs="+", help="Independent benchmark seeds (default: seed and next two)")
    parser.add_argument("--beta-grid", type=float, nargs="+", help="Select a shared GASS temperature by validation noisy-image MSE")
    parser.add_argument("--validation-grid", type=Path, help="JSON list of GASSConfig overrides to compare on validation data")
    parser.add_argument("--tracking-steps", type=int, default=0,
                        help="Measure uncertainty on a controlled trajectory anchored to MNIST training logits")
    parser.add_argument("--stream", choices=("iid", "by-digit", "rare-digit"), default="iid",
                        help="Benchmark stationary, changing digit groups, or an underrepresented nine")
    parser.add_argument("--output", type=Path, help="Save benchmark metrics, edit history and confidence intervals to JSON")
    return parser


def parse_comparison_args(parser, argv, defaults):
    args = parser.parse_args(argv)
    # An explicit shared temperature (including a grid) overrides the profile's read temperature.
    use_profile_read_beta = args.gass_beta is None and args.gass_read_beta is None and not args.beta_grid
    if args.gass_beta is None:
        args.gass_beta = defaults.beta
    if use_profile_read_beta:
        args.gass_read_beta = defaults.read_beta
    try:
        args.gass = memory_settings(args, defaults)
        args.metric = load_metric(args.metric)
        if args.seeds and len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Benchmark seeds must be distinct.")
        if args.beta_grid and (any(not math.isfinite(b) or b <= 0 for b in args.beta_grid)
                               or args.gass_read_beta is not None or args.gass_write_beta is not None):
            raise ValueError("Use positive finite grid temperatures and leave the per-score overrides unset.")
        if args.validation_grid is not None:
            if args.beta_grid:
                raise ValueError("Choose one validation grid.")
            grid = json.loads(args.validation_grid.read_text(encoding="utf-8-sig"))
            if not isinstance(grid, list) or not grid or not all(isinstance(point, dict) for point in grid):
                raise ValueError("The validation grid must be a nonempty JSON array of setting objects.")
            if use_profile_read_beta:
                for point in grid:
                    if "beta" in point:
                        point.setdefault("read_beta", None)
            args.validation_grid = [replace(args.gass, **point) for point in grid]
        if args.tracking_steps < 0:
            raise ValueError("Tracking steps must be nonnegative.")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.similarity_beta is not None and not (math.isfinite(args.similarity_beta) and args.similarity_beta > 0):
        parser.error("--beta-read must be finite and positive")
    return args


def comparison_args(argv, *, safeguarded, description):
    defaults = GASSConfig() if safeguarded else ReferenceConfig()
    parser = build_comparison_parser(description, defaults)
    args = parse_comparison_args(parser, argv, defaults)
    args.gass_variant_class = GASSVariant if safeguarded else ReferenceVariant
    return args


def run_comparison(args, *, ui_type=None):
    if args.benchmark:
        from gass_mnist_benchmark import run_benchmark
        return run_benchmark(args)

    import tkinter as tk
    from gass_ui import GASSComparisonUI, ReferenceComparisonUI

    root = tk.Tk()
    if ui_type is None:
        ui_type = GASSComparisonUI if isinstance(args.gass, GASSConfig) else ReferenceComparisonUI
    ui_type(root, args)
    root.mainloop()
