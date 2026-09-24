# hopfieldpc-memory-core

Standalone modern Hopfield memory core for the HopfieldPC project.

<p align="justify">
This repository implements and validates the core Hopfield memory mechanism before integration into FabricPC. It focuses on beta-controlled associative retrieval, tensor-shape validation, diagnostics, synthetic benchmarks, and unit tests.

## Setup

Clone the repository:

```bash
git clone https://github.com/iCog-Labs-Dev/hopfieldpc-memory-core.git
cd hopfieldpc-memory-core
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import hopfieldpc; print('hopfieldpc import OK')"
```

## Current Implementation Stage

Current task:

```text
Task 2: Implement one-step Hopfield retrieval
```

Implemented so far:

```text
src/hopfieldpc/api.py
src/hopfieldpc/functional.py
src/hopfieldpc/memory.py
src/hopfieldpc/__init__.py
tests/test_api_contract.py
tests/test_one_step_retrieval.py
tests/test_beta_behavior.py
```

The module now validates tensor shapes and performs one-step modern Hopfield retrieval.

## Expected Tensor Contract

```text
query:     [batch_size, dim]
memory:    [num_memories, dim]
retrieved: [batch_size, dim]
weights:   [batch_size, num_memories]
```

## Core Retrieval Formula

```python
scores = beta * query @ memory.T
weights = softmax(scores)
retrieved = weights @ memory
```

Equivalent mathematical form:

```text
p = softmax(β qMᵀ)
h_memory = pM
```

## Run Tests

Run all tests:

```bash
pytest -q
```

Run only the API contract tests:

```bash
pytest tests/test_api_contract.py -q
```

Run only one-step retrieval tests:

```bash
pytest tests/test_one_step_retrieval.py tests/test_beta_behavior.py -q
```

## Next Planned Tasks

```text
Task 3: Add memory modes and prepare module extension points.
Task 4: Add optional multi-step retrieval.
Task 5: Add diagnostics such as entropy, top-k mass, and retrieval distance.
Task 6: Add synthetic retrieval benchmarks.
```
