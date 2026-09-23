# hopfieldpc-memory-core

Standalone modern Hopfield memory core for the HopfieldPC project.

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
Task 1: Define HopfieldMemory API and tensor shape contract
```

Implemented so far:

```text
src/hopfieldpc/api.py
src/hopfieldpc/memory.py
src/hopfieldpc/__init__.py
tests/test_api_contract.py
```

This task defines the API and validates tensor shapes. The actual retrieval equation will be implemented in the next task.

## Expected Tensor Contract

```text
query:     [batch_size, dim]
memory:    [num_memories, dim]
retrieved: [batch_size, dim]
weights:   [batch_size, num_memories]
```

## Basic Import Check

```python
from hopfieldpc import HopfieldMemory

module = HopfieldMemory(dim=4, memory_size=8, beta=1.0)
print(module)
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

## Planned Core Retrieval Formula

The next implementation step will add the Hopfield retrieval computation:

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
