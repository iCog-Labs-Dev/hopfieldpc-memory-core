"""Compare Direct, EMA, previous Dynamic and GASS on clean and corrupted MNIST.

GASS self-associates detached decoder logits. The default uses cosine scores,
Gaussian value evidence and a scalar Kalman filter. Association geometry, write
mode and retrieval loss can be selected independently for the MNIST comparison.
"""

from gass_comparison_cli import comparison_args, run_comparison


def parse_args(argv=None):
    return comparison_args(argv, safeguarded=False, description=__doc__)


if __name__ == "__main__":
    run_comparison(parse_args())
