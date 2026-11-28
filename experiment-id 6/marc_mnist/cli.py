"""Experiment 6 MNIST reconstruction with optional retrieval protection."""
from gass_comparison_cli import build_comparison_parser, parse_comparison_args, run_comparison
from .model import GASS_BANK_INIT_CHOICES, GASSVariant, ReconstructionConfig


def parse_args(argv=None):
    defaults = ReconstructionConfig()
    parser = build_comparison_parser(__doc__, defaults, bank_init_choices=GASS_BANK_INIT_CHOICES)
    parser.set_defaults(epochs=20, initial_memories=20, min_active_memories=2, max_memories=80,
                        gass_beta=None)
    parser.add_argument("--marc-window", choices=("minibatch", "reference"), default=defaults.marc_window,
                        help="Use reference to protect the structural training images on every write. "
                             "This enables monotonic MARC unless a strict retrieval margin is supplied.")
    parser.add_argument("--retrieval-margin", type=float, default=defaults.retrieval_margin,
                        help="Use zero for monotonic MARC or a positive margin for strict value updates.")
    args = parse_comparison_args(parser, argv, defaults)
    args.gass_variant_class = GASSVariant
    return args


def main():
    args = parse_args()
    if args.benchmark:
        return run_comparison(args)
    from .ui import GASSComparisonUI
    return run_comparison(args, ui_type=GASSComparisonUI)
