# Experiment 6 — Retrieval-protected GASS on MNIST

This experiment compares ordinary GASS with monotonic and strict retrieval checks (MARC), random and warm-up initialization, and minibatch and fixed-reference checks. The reconstruction UI also lets you put memory at the encoder bottleneck and inspect its stored features during training. The recorded benchmark scores use decoder-output memory. [Results and reproduction commands](RESULTS.md)

## Run from the repository root

```powershell
.\.venv\Scripts\python.exe "experiment-id 6/mnist_autoencoder_ui_gass_proof.py"
.\.venv\Scripts\python.exe "experiment-id 6/experiment.py" --comparison reference --device cuda --seeds 0 1 2 --output "experiment-id 6/results/marc_reference_window_mnist.json"
.\.venv\Scripts\python.exe -m pytest -q "experiment-id 6/tests"
```

The UI launcher also supports `--benchmark` for the headless reconstruction comparison. The dedicated benchmark accepts `--comparison proposal`, `reference` or `initialization`. Each JSON file records its settings, seeds and per-run measurements.

The safeguarded reconstruction UI defaults to 20 epochs, a 256-image reference window and 20 starting memories (minimum 2, maximum 80). It uses patience 2, merge threshold 0.90, support fraction 0.10, minimum child mass 1 and up to 5 accepted edits per structural pass. Sigmoid-Dice reads use β = 80 and writes use β = 20, with Bernoulli-KL retrieval loss.

In the safeguarded UI, set **GASS initialization** and **Window size (training images)** before pressing **Start Training**. **warm-up** seeds the bank from the selected layer's training features after the configured warm-up (one epoch by default). **random** starts the first batch with seeded Gaussian prototypes. The window sets how many training images are sampled for structural checks and warm-up seeding. You can also preset the controls with `--bank-init random --window-size 256`.

**GASS readout layer** selects the memory position for the next UI run. **Decoder output** (default) stores 784 pixel logits. **Encoder bottleneck** stores 64-dimensional encoder outputs and passes the recalled vector through the decoder. Warm-up and MARC use training features from the selected layer. The bottleneck starts with cosine reads and writes and squared retrieval loss, so its MARC checks measure error in the encoding. The preview reports image-space error and the gallery decodes stored vectors into digits. Each layer remembers its selected scores, retrieval loss and number of frozen key iterations while the UI is open.

In the **Memory banks** tab, switch **GASS prototype view** at any time, including during training. **Stored values** shows the 64 bottleneck features in index order on an 8×8 grid. **Decoded image** (default) shows the same slot after decoding to 28×28 pixel logits. Both views use the same signed brightness scale. With output-layer memory, both show the same 784 stored pixel logits. Switching views redraws the latest preview immediately.

Retrieval compares the input with each slot's key and blends the values using those match scores and the slot weights. With random initialization and Dice for both reads and writes, the first trial step for learning keys is scaled by the number of coordinates in the score. Backtracking shrinks the step until it passes the objective and feasibility checks, or keeps the previous keys if every trial fails.

The safeguarded launcher accepts `--retrieval-margin 0` for monotonic MARC and a positive margin for strict value updates. Its reconstruction UI trains on clean self-associations. The [paired retrieval benchmark](experiment.py) also tests noisy training cues against clean targets. See the [comparison and reproduction command](RESULTS.md#retrieval-protected-proposal-on-noisy-mnist).

Choose **MARC checks → reference** before starting training to use the same **Window size** training images to judge every write and structural edit. The command-line equivalent is `--marc-window reference --window-size 256`. Reference mode enables monotonic MARC by default and honors a positive `--retrieval-margin` for strict value updates. The bank still learns from each fresh minibatch. The decoder re-encodes the reference images for each write, and that snapshot stays fixed while the candidate is checked. Decoder training can change the reference loss between writes.

The experiment-specific code lives in [marc_mnist](marc_mnist). It imports the [Experiment 5 MNIST model and comparison tools](<../experiment-id 5/README.md>) and the [shared GASS memory and MARC solver](../src/hopfieldpc/gass).
