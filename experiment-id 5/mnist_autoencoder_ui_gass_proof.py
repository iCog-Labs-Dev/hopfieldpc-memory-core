"""Continuous safeguarded GASS for the MNIST autoencoder (proposal §§8–18).

Read detached decoder logits before assimilation. The default image profile uses
sigmoid Dice, pixelwise Bernoulli KL and scalar trust. Fixed-window optimization
and frozen-bank recall never count replay as fresh measurement evidence.
"""

from gass_comparison_cli import comparison_args, run_comparison


def parse_args(argv=None):
    return comparison_args(argv, safeguarded=True, description=__doc__)


if __name__ == "__main__":
    run_comparison(parse_args())
