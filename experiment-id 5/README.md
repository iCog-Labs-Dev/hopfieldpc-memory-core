# Experiment 5 — GASS on MNIST

The full MNIST run has a [results report](RESULTS.md).

Run these commands from the repository root with the project Python environment active.

```powershell
python "experiment-id 5/experiment.py" --quick --epochs 2 --memory-epochs 3 --gass-read-beta 80
python "experiment-id 5/mnist_autoencoder_ui_gass_proof.py" --epochs 20 --window-size 256 --initial-memories 20 --min-active-memories 2 --max-memories 80 --patience 2 --merge-threshold 0.90 --support-fraction 0.10 --min-child-mass 1 --edit-budget 5 --gass-read-beta 80
```

The classification experiment trains a supervised MNIST encoder for `--epochs` epochs and then freezes it. GASS associates continuous embeddings with ten-dimensional one-hot digit targets using cosine scores and squared retrieval loss by default. An independent test image is classified by taking the largest component of the bank read. A linear head on the same encoder provides the direct baseline.

`--memory-epochs` controls passes over the training embeddings. The first pass updates trust from fresh evidence. Later passes refine the stored bank. Validation and test reads keep the bank fixed. The output includes per-digit accuracy for both classifiers. Add `--output "experiment-id 5/results/classification.json"` to save the settings, validation history and confusion matrices. Rows are true digits and columns are predicted digits.

The safeguarded reconstruction UI retains its sigmoid-Dice and Bernoulli-KL defaults. Its `--benchmark` option runs the paired reconstruction comparison headlessly. The original GASS comparison is available through `mnist_autoencoder_ui_gass.py`.

Within Experiment 5, `from gass import GASSBank, GASSConfig` selects the MNIST defaults. The experiment supplies the similarity functions, key projections and reconstruction or classification loss. [Experiment 5 profiles](gass/geometry.py)

Other experiments can import `GASSBank`, `GASSConfig`, `KeyGeometry` and `RetrievalLoss` from `hopfieldpc.gass`. The shared bank implements trust and variance updates, fixed-window descent, structural proposals and the optional stream movement budget. [Shared bank](../src/hopfieldpc/gass/bank.py) · [Runnable custom-geometry example](../tests/test_gass.py)

Experiment 4's revised launcher uses the same MNIST autoencoder and comparison UI. [Autoencoder](mnist_reconstruction/model.py) · [Comparison UI](mnist_reconstruction/ui.py)

The Lean project contains the proofs for the safeguarded proposal. [Lean setup and theorem index](lean/README.md)
