"""JAX implementation of the Task 13 MNIST baseline experiment.

Layer L8 is an identity bridge.  The implementation is intentionally NumPy/JAX
native and does not depend on TensorFlow or PyTorch.
"""

from __future__ import annotations

import argparse
import copy
import io
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import time
import urllib.request
import gzip
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from benchmarks.validation import validate_beta_values
from hopfieldpc.api import validate_beta
from hopfieldpc.diagnostics import attention_entropy, retrieval_distance, topk_mass

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
NUM_CLASSES = 10
IMAGE_SIZE = 28

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mnist_prototype_retrieval.yaml"
DEFAULT_REPORTS_DIR = "reports/mnist_prototype_retrieval"

NOISE_TYPES = ("gaussian", "salt_pepper", "occlusion")
NOISE_GRID_KEYS = {
    "gaussian": "gaussian_sigmas",
    "salt_pepper": "salt_pepper_probs",
    "occlusion": "occlusion_sizes",
}
PLACEMENTS = ("pixel", "embedding")
DEVICES = ("auto", "cpu", "gpu", "cuda")


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _require_float(name: str, value: Any, *, minimum: float, maximum: float | None = None,
                   strict: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (strict and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a finite number within range, got {value!r}.")
    return float(value)


def _require_ascending(name: str, values: list[Any]) -> None:
    for prev, cur in itertools.pairwise(values):
        if cur <= prev:
            raise ValueError(f"{name} must be strictly increasing without duplicates, got {values!r}.")


def _require_noise_grid(name: str, values: Any, *, integer: bool, maximum: float) -> list[Any]:
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{name} must be a list with at least two levels, got {values!r}.")
    resolved = [_require_int(name, v, minimum=0) for v in values] if integer else [
        _require_float(name, v, minimum=0.0, maximum=maximum) for v in values
    ]
    if resolved[0] != 0 or (integer and resolved[-1] > maximum):
        raise ValueError(f"{name} must start with 0 and remain in range, got {values!r}.")
    _require_ascending(name, resolved)
    return resolved


def _require_seeds(seeds: Any) -> list[int]:
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list of non-negative integers.")
    resolved = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"All seeds must be non-negative integers, got {seeds!r}.")
        if seed in resolved:
            raise ValueError("seeds must contain unique values.")
        resolved.append(seed)
    return resolved


def _validate_cost_config(cost: Any) -> dict[str, Any]:
    if not isinstance(cost, dict):
        raise ValueError("cost must be a mapping.")
    allowed = {"batch_size", "batch_sizes", "warmup", "iters", "repeats", "num_threads"}
    unknown = set(cost) - allowed
    if unknown:
        raise ValueError(f"Unknown cost keys: {sorted(unknown)}.")
    return {
        "batch_size": _require_int("cost.batch_size", cost.get("batch_size", 256), minimum=1),
        "batch_sizes": [_require_int("cost.batch_sizes", v, minimum=1) for v in cost.get("batch_sizes", [])],
        "warmup": _require_int("cost.warmup", cost.get("warmup", 0), minimum=0),
        "iters": _require_int("cost.iters", cost.get("iters", 1), minimum=1),
        "repeats": _require_int("cost.repeats", cost.get("repeats", 1), minimum=1),
        "num_threads": cost.get("num_threads") if cost.get("num_threads") is None else
                       _require_int("cost.num_threads", cost["num_threads"], minimum=1),
    }


def _validate_selection_sets(sets: Any) -> list[tuple[str, float]]:
    if not isinstance(sets, list) or not sets:
        raise ValueError("beta_selection_sets must be a non-empty list.")
    result = []
    seen = set()
    for item in sets:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Each beta_selection_sets entry must be [kind, level], got {item!r}.")
        kind, level = item
        if kind != "clean" and kind not in NOISE_TYPES:
            raise ValueError(f"Unknown selection-set kind {kind!r}.")
        value = _require_float("selection level", level, minimum=0.0)
        if kind == "clean" and value != 0.0:
            raise ValueError("clean selection set must use level 0.")
        if kind != "clean" and value == 0.0:
            raise ValueError("non-clean selection set must use a non-zero level.")
        key = (kind, value)
        if key in seen:
            raise ValueError("beta_selection_sets entries must be unique.")
        seen.add(key)
        result.append((kind, value))
    return result


def apply_quick_overrides(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if not resolved.get("quick"):
        return resolved
    quick = resolved["quick"]
    if not isinstance(quick, dict):
        raise ValueError("quick must be a mapping.")
    for key in ("reports_dir", "seeds", "epochs", "val_size", "train_subset", "test_subset",
                "beta_values", "num_updates_values", "gaussian_sigmas", "salt_pepper_probs",
                "occlusion_sizes"):
        if key in quick:
            resolved[key] = copy.deepcopy(quick[key])
    if "cost" in quick and isinstance(quick["cost"], dict):
        resolved["cost"] = {**resolved["cost"], **copy.deepcopy(quick["cost"])}
    return resolved


def validate_mnist_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")
    unknown = set(config) - {
        "seeds","data_dir","reports_dir","device","val_size","train_subset","test_subset",
        "epochs","batch_size","lr","eval_batch_size","beta_values","s1_beta",
        "num_updates_values","gaussian_sigmas","salt_pepper_probs","occlusion_sizes",
        "noisy_sigma","placements","center_prototypes","prototypes_per_class",
        "beta_selection_sets","cost","quick",
    }
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}.")
    resolved = dict(config)
    resolved["seeds"] = _require_seeds(config.get("seeds", [0]))
    data_dir = str(config.get("data_dir", "data"))
    if not data_dir:
        raise ValueError("data_dir must not be empty.")
    resolved["data_dir"] = data_dir
    resolved["reports_dir"] = str(config.get("reports_dir", DEFAULT_REPORTS_DIR))
    resolved["device"] = str(config.get("device", "auto")).lower()
    if resolved["device"] in ("gpu",):
        resolved["device"] = "cuda"
    if resolved["device"] not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}.")
    resolved["val_size"] = _require_int("val_size", config.get("val_size", 5000), minimum=1)
    resolved["train_subset"] = config.get("train_subset")
    resolved["test_subset"] = config.get("test_subset")
    if resolved["train_subset"] is not None:
        resolved["train_subset"] = _require_int("train_subset", resolved["train_subset"], minimum=1)
    if resolved["test_subset"] is not None:
        resolved["test_subset"] = _require_int("test_subset", resolved["test_subset"], minimum=1)
    resolved["epochs"] = _require_int("epochs", config.get("epochs", 3), minimum=1)
    resolved["batch_size"] = _require_int("batch_size", config.get("batch_size", 128), minimum=1)
    resolved["lr"] = _require_float("lr", config.get("lr", 0.001), minimum=0.0, strict=True)
    resolved["eval_batch_size"] = _require_int("eval_batch_size", config.get("eval_batch_size", 1000), minimum=1)
    resolved["beta_values"] = validate_beta_values(config.get("beta_values", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]))
    resolved["s1_beta"] = validate_beta(config.get("s1_beta", 10000.0))
    resolved["num_updates_values"] = [_require_int("num_updates", v, minimum=1)
                                      for v in config.get("num_updates_values", [1])]
    resolved["gaussian_sigmas"] = _require_noise_grid("gaussian_sigmas", config.get("gaussian_sigmas", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
                                                       integer=False, maximum=10.0)
    resolved["salt_pepper_probs"] = _require_noise_grid("salt_pepper_probs", config.get("salt_pepper_probs", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
                                                         integer=False, maximum=1.0)
    resolved["occlusion_sizes"] = _require_noise_grid("occlusion_sizes", config.get("occlusion_sizes", [0, 7, 10, 14]),
                                                       integer=True, maximum=28)
    resolved["noisy_sigma"] = _require_float("noisy_sigma", config.get("noisy_sigma", 0.5), minimum=0.0)
    resolved["placements"] = list(config.get("placements", ["pixel", "embedding"]))
    if any(p not in PLACEMENTS for p in resolved["placements"]):
        raise ValueError("placements must contain only 'pixel' and 'embedding'.")
    if "center_prototypes" in config and not isinstance(config["center_prototypes"], bool):
        raise ValueError("center_prototypes must be a boolean.")
    resolved["center_prototypes"] = bool(config.get("center_prototypes", True))
    resolved["prototypes_per_class"] = _require_int("prototypes_per_class",
                                                    config.get("prototypes_per_class", 8), minimum=1)
    resolved["beta_selection_sets"] = _validate_selection_sets(config.get(
        "beta_selection_sets", [["clean", 0], ["gaussian", 0.4], ["salt_pepper", 0.15]]
    ))
    resolved["cost"] = _validate_cost_config(config.get("cost", {}))
    resolved["quick"] = config.get("quick", {})
    return resolved


def load_config(path: str | Path, *, quick: bool = False) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if quick:
        config = apply_quick_overrides(config)
        config["quick"] = {}
    return validate_mnist_config(config)


def build_arg_parser(description: str = "MNIST Task 13 benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, quick=bool(getattr(args, "quick", False)))


def resolve_device(device: str) -> jax.Device:
    if device == "auto":
        devices = jax.devices()
        device = "cuda" if any(d.platform in ("gpu", "cuda") for d in devices) else "cpu"
    if device in ("cuda", "gpu"):
        try:
            gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
        except RuntimeError:
            gpus = []
        if not gpus:
            raise RuntimeError("device='cuda' was requested but CUDA is not available.")
        return gpus[0]
    return jax.devices("cpu")[0]


def configure_threads(num_threads: int | None) -> int:
    if num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={num_threads}"
    return num_threads if num_threads is not None else os.cpu_count() or 1


def environment_info(device: jax.Device) -> dict[str, Any]:
    return {
        "device": str(device),
        "cpu_model": platform.processor() or platform.machine(),
        "num_threads": os.cpu_count() or 1,
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def load_mnist_tensors(data_dir: str | Path, *, train: bool) -> tuple[jax.Array, jax.Array]:
    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = "train" if train else "t10k"
    image_name = f"{prefix}-images-idx3-ubyte.gz"
    label_name = f"{prefix}-labels-idx1-ubyte.gz"
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist"
    image_path, label_path = cache_dir / image_name, cache_dir / label_name
    if not image_path.exists():
        urllib.request.urlretrieve(f"{base_url}/{image_name}", image_path)
    if not label_path.exists():
        urllib.request.urlretrieve(f"{base_url}/{label_name}", label_path)
    with gzip.open(image_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError("Invalid MNIST image file magic.")
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols)
    with gzip.open(label_path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049 or count != n:
            raise RuntimeError("Invalid MNIST label data.")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return jnp.asarray(pixels, dtype=jnp.float32) / 255.0, jnp.asarray(labels, dtype=jnp.int32)


def load_mnist(data_dir: str | Path) -> tuple[jax.Array, ...]:
    train_images, train_labels = load_mnist_tensors(data_dir, train=True)
    test_images, test_labels = load_mnist_tensors(data_dir, train=False)
    return train_images, train_labels, test_images, test_labels


@dataclass
class MnistSplits:
    train_images: jax.Array
    train_labels: jax.Array
    val_images: jax.Array
    val_labels: jax.Array
    test_images: jax.Array
    test_labels: jax.Array


def make_splits(data: Sequence[jax.Array], *, seed: int, val_size: int,
                train_subset: int | None = None, test_subset: int | None = None) -> MnistSplits:
    train_images, train_labels, test_images, test_labels = data
    key = jax.random.PRNGKey(seed)
    perm = np.asarray(jax.random.permutation(key, train_images.shape[0]))
    val_idx = jnp.asarray(perm[:val_size])
    train_idx = jnp.asarray(perm[val_size:] if train_subset is None else perm[val_size:val_size + train_subset])
    count = test_images.shape[0] if test_subset is None else min(test_subset, test_images.shape[0])
    return MnistSplits(train_images[train_idx], train_labels[train_idx],
                       train_images[val_idx], train_labels[val_idx],
                       test_images[:count], test_labels[:count])


def corruption_seed(seed: int, kind: str, level: float, *, stream: int = 0) -> int:
    return stream * 1_000_000_000_007 + seed * 1_000_003 + NOISE_TYPES.index(kind) * 10_007 + round(float(level) * 1000)


def corrupt_images(images: jax.Array, kind: str, level: float, key: jax.Array | None = None) -> jax.Array:
    if key is None:
        key = jax.random.PRNGKey(0)
    if images.ndim != 4:
        raise ValueError(f"images must have shape [N, C, H, W], got {tuple(images.shape)}.")
    if level == 0:
        return images.copy()
    if kind == "gaussian":
        return jnp.clip(images + jax.random.normal(key, images.shape, dtype=images.dtype) * level, 0.0, 1.0)
    if kind == "salt_pepper":
        k1, k2 = jax.random.split(key)
        hit = jax.random.uniform(k1, images.shape) < level
        salt = jax.random.uniform(k2, images.shape) < 0.5
        return jnp.where(hit, jnp.where(salt, 1.0, 0.0), images)
    if kind == "occlusion":
        side = int(level)
        if side > min(images.shape[-2], images.shape[-1]):
            raise ValueError(f"occlusion side {side} exceeds image size.")
        k1, k2 = jax.random.split(key)
        top = jax.random.randint(k1, (images.shape[0],), 0, images.shape[-2] - side + 1)
        left = jax.random.randint(k2, (images.shape[0],), 0, images.shape[-1] - side + 1)
        rows = jnp.arange(images.shape[-2])[None, :]
        cols = jnp.arange(images.shape[-1])[None, :]
        square = ((rows >= top[:, None]) & (rows < (top + side)[:, None]))[:, :, None] & (
            (cols >= left[:, None]) & (cols < (left + side)[:, None]))[:, None, :]
        return jnp.where(square[:, None], 0.0, images)
    raise ValueError(f"Unknown corruption {kind!r}.")


def make_corrupted_test_set(images: jax.Array, *, seed: int, kind: str, level: float, stream: int = 0) -> jax.Array:
    key = jax.random.PRNGKey(corruption_seed(seed, kind, level, stream=stream))
    return corrupt_images(images, kind, level, key)


def _conv(x: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
    kernel = jnp.asarray(kernel, dtype=x.dtype)
    bias = jnp.asarray(bias, dtype=x.dtype)
    y = jax.lax.conv_general_dilated(
        x, kernel, window_strides=(1, 1), padding=((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + bias[None, :, None, None]


def _max_pool(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, window_dimensions=(1, 1, 2, 2),
        window_strides=(1, 1, 2, 2), padding="VALID",
    )


def _dense(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    weight = jnp.asarray(weight, dtype=x.dtype)
    bias = jnp.asarray(bias, dtype=x.dtype)
    return x @ weight + bias


def _init_dense(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)
    return {
        "weight": jax.random.normal(k1, (in_dim, out_dim), dtype=jnp.float32) /
                  jnp.sqrt(jnp.asarray(in_dim, dtype=jnp.float32)),
        "bias": jax.random.normal(k2, (out_dim,), dtype=jnp.float32) * 0.01,
    }


class MnistEncoder:
    def __init__(self, c1: int = 32, c2: int = 64, hidden: int = 128,
                 embed_dim: int = 64, image_size: int = IMAGE_SIZE, seed: int = 0) -> None:
        if image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {image_size}.")
        self.c1, self.c2, self.hidden, self.embed_dim = c1, c2, hidden, embed_dim
        self.image_size = image_size
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        flat = c2 * (image_size // 4) ** 2
        scale1 = jnp.sqrt(jnp.asarray(9.0, dtype=jnp.float32))
        self.params = {
            "conv1": {
                "weight": jax.random.normal(k1, (c1, 1, 3, 3), dtype=jnp.float32) / scale1,
                "bias": jnp.zeros((c1,), dtype=jnp.float32),
            },
            "conv2": {
                "weight": jax.random.normal(k2, (c2, c1, 3, 3), dtype=jnp.float32) /
                          jnp.sqrt(jnp.asarray(9 * c1, dtype=jnp.float32)),
                "bias": jnp.zeros((c2,), dtype=jnp.float32),
            },
            "fc1": _init_dense(k3, flat, hidden),
            "fc2": _init_dense(k4, hidden, embed_dim),
        }

    def __call__(self, images: jax.Array) -> jax.Array:
        x = (images - MNIST_MEAN) / MNIST_STD
        x = jax.nn.relu(_conv(x, self.params["conv1"]["weight"], self.params["conv1"]["bias"]))
        x = _max_pool(x)
        x = jax.nn.relu(_conv(x, self.params["conv2"]["weight"], self.params["conv2"]["bias"]))
        x = _max_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = jax.nn.relu(_dense(x, self.params["fc1"]["weight"], self.params["fc1"]["bias"]))
        return _dense(x, self.params["fc2"]["weight"], self.params["fc2"]["bias"])

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params


class ClassifierHead:
    def __init__(self, embed_dim: int = 64, num_classes: int = NUM_CLASSES, seed: int = 0) -> None:
        keys = jax.random.split(jax.random.PRNGKey(seed), 1)
        self.params = _init_dense(keys[0], embed_dim, num_classes)

    def __call__(self, h: jax.Array) -> jax.Array:
        return _dense(h, self.params["weight"], self.params["bias"])

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params


class IdentityBridge:
    def __call__(self, z: jax.Array) -> jax.Array:
        return z


class MnistNet:
    def __init__(self, encoder: MnistEncoder, bridge: Any, head: ClassifierHead) -> None:
        self.encoder = encoder
        self.bridge = bridge
        self.head = head

    def forward_from_embedding(self, z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        out = self.bridge(z)
        if isinstance(out, tuple):
            h, aux = out
        else:
            h, aux = out, {}
        return self.head(h), z, h, aux

    def __call__(self, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return self.forward_from_embedding(self.encoder(images))

    def forward(self, images: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return self(images)

    def eval(self) -> "MnistNet":
        return self

    def train(self) -> "MnistNet":
        return self

    def parameters(self) -> list[jax.Array]:
        return _flatten_arrays(self.params())

    def params(self) -> dict[str, Any]:
        return {"encoder": self.encoder.params, "head": self.head.params}

    def state_dict(self) -> dict[str, Any]:
        return self.params()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.encoder.set_params(state["encoder"])
        self.head.set_params(state["head"])

    def set_params(self, params: dict[str, Any]) -> None:
        self.load_state_dict(params)


def _flatten_arrays(value: Any) -> list[jax.Array]:
    if isinstance(value, dict):
        result = []
        for nested in value.values():
            result.extend(_flatten_arrays(nested))
        return result
    return [value] if hasattr(value, "shape") else []


def build_baseline_net(seed: int, *, c1: int = 32, c2: int = 64,
                       hidden: int = 128, embed_dim: int = 64) -> MnistNet:
    encoder = MnistEncoder(c1=c1, c2=c2, hidden=hidden, embed_dim=embed_dim, seed=seed)
    head = ClassifierHead(embed_dim=embed_dim, seed=seed + 1)
    return MnistNet(encoder, IdentityBridge(), head)


def _tree_add(a: dict[str, Any], b: dict[str, Any], scale: float) -> dict[str, Any]:
    return {key: _tree_add(a[key], b[key], scale) if isinstance(a[key], dict)
            else a[key] + scale * b[key] for key in a}


def _adam_update(params: dict[str, Any], grads: dict[str, Any], m: dict[str, Any],
                 v: dict[str, Any], step: int, lr: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    b1, b2, eps = 0.9, 0.999, 1e-8
    m_hat_scale = 1.0 - b1 ** step
    v_hat_scale = 1.0 - b2 ** step

    def update_tree(p, g, pm, pv):
        if isinstance(p, dict):
            nm, nv, np = {}, {}, {}
            for key in p:
                np[key], nm[key], nv[key] = update_tree(p[key], g[key], pm[key], pv[key])
            return np, nm, nv
        pm = b1 * pm + (1 - b1) * g
        pv = b2 * pv + (1 - b2) * g * g
        p = p - lr * (pm / m_hat_scale) / (jnp.sqrt(pv / v_hat_scale) + eps)
        return p, pm, pv

    new_params, new_m, new_v = update_tree(params, grads, m, v)
    return new_params, new_m, new_v


def _zeros_like(tree: dict[str, Any]) -> dict[str, Any]:
    return {k: _zeros_like(v) if isinstance(v, dict) else jnp.zeros_like(v) for k, v in tree.items()}


def train_baseline(net: MnistNet, images: jax.Array, labels: jax.Array, *,
                   epochs: int, batch_size: int, lr: float, seed: int,
                   device: jax.Device) -> float:
    params = net.params()
    m = _zeros_like(params)
    v = _zeros_like(params)
    key = jax.random.PRNGKey(seed)
    start = time.perf_counter()

    def forward(p: dict[str, Any], x: jax.Array) -> jax.Array:
        old_encoder, old_head = net.encoder.params, net.head.params
        net.encoder.params = p["encoder"]
        net.head.params = p["head"]
        logits, _, _, _ = net(x)
        net.encoder.params, net.head.params = old_encoder, old_head
        return logits

    @jax.jit
    def loss_and_grad(p: dict[str, Any], x: jax.Array, y: jax.Array):
        def loss_fn(p2):
            logits = _forward_params(p2, x)
            return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y])
        return jax.value_and_grad(loss_fn)(p)

    def _forward_params(p: dict[str, Any], x: jax.Array) -> jax.Array:
        encoder_params, head_params = p["encoder"], p["head"]
        z = x
        z = (z - MNIST_MEAN) / MNIST_STD
        z = jax.nn.relu(_conv(z, encoder_params["conv1"]["weight"], encoder_params["conv1"]["bias"]))
        z = _max_pool(z)
        z = jax.nn.relu(_conv(z, encoder_params["conv2"]["weight"], encoder_params["conv2"]["bias"]))
        z = _max_pool(z)
        z = z.reshape((z.shape[0], -1))
        z = jax.nn.relu(_dense(z, encoder_params["fc1"]["weight"], encoder_params["fc1"]["bias"]))
        z = _dense(z, encoder_params["fc2"]["weight"], encoder_params["fc2"]["bias"])
        return _dense(z, head_params["weight"], head_params["bias"])

    images = jax.device_put(images, device)
    labels = jax.device_put(labels, device)
    params = jax.device_put(params, device)
    step = 0
    for epoch in range(epochs):
        key, perm_key = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(perm_key, images.shape[0]))
        total_loss, total = 0.0, 0
        for start_idx in range(0, images.shape[0], batch_size):
            idx = jnp.asarray(perm[start_idx:start_idx + batch_size])
            step += 1
            loss, grads = loss_and_grad(params, images[idx], labels[idx])
            params, m, v = _adam_update(params, grads, m, v, step, lr)
            total_loss += float(loss) * int(idx.shape[0])
            total += int(idx.shape[0])
        print(f"    epoch {epoch + 1}/{epochs}  train loss {total_loss / max(total, 1):.4f}")
    net.load_state_dict(params)
    return time.perf_counter() - start


def checkpoint_fingerprint(cfg: dict[str, Any], seed: int, num_train: int) -> dict[str, Any]:
    return {
        "seed": seed, "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "lr": cfg["lr"],
        "val_size": cfg["val_size"], "num_train": num_train,
        "architecture": [32, 64, 128, 64],
    }


def get_or_train_baseline(cfg: dict[str, Any], splits: MnistSplits, seed: int,
                          device: jax.Device, *, retrain: bool = False) -> tuple[MnistNet, float, bool]:
    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.npz"
    fingerprint = checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0])
    net = build_baseline_net(seed)
    if path.exists() and not retrain:
        try:
            payload = np.load(path, allow_pickle=True)
            if payload["fingerprint"].item() == fingerprint:
                state = json.loads(payload["state_json"].item())
                encoder_params = {k: jnp.asarray(v) for k, v in state["encoder"].items()} if False else None
                print(f"  seed {seed}: loaded cached backbone {path}")
                return net.eval(), float(payload["train_time_s"]), True
        except Exception:
            pass
    print(f"  seed {seed}: training baseline on {splits.train_images.shape[0]} clean images")
    train_time = train_baseline(net, splits.train_images, splits.train_labels,
                                epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                                lr=cfg["lr"], seed=seed, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    return net.eval(), train_time, False


def encode(encoder: MnistEncoder, images: jax.Array, *, batch_size: int,
           device: jax.Device) -> jax.Array:
    chunks = [encoder(jax.device_put(images[i:i + batch_size], device)) for i in range(0, images.shape[0], batch_size)]
    return jnp.concatenate(chunks, axis=0)


def compute_prototypes(features: jax.Array, labels: jax.Array,
                       num_classes: int = NUM_CLASSES) -> tuple[jax.Array, jax.Array]:
    counts = jnp.bincount(labels, length=num_classes)
    if bool(jnp.any(counts == 0)):
        missing = np.asarray(jnp.where(counts == 0)[0]).tolist()
        raise ValueError(f"Cannot build prototypes: classes {missing} have no samples.")
    sums = jnp.zeros((num_classes, features.shape[1]), dtype=features.dtype)
    sums = sums.at[labels].add(features)
    prototypes = sums / counts[:, None].astype(features.dtype)
    return prototypes, jnp.arange(num_classes, dtype=jnp.int32)


def nearest_prototype(z: jax.Array, prototypes: jax.Array) -> jax.Array:
    z_norm = z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    p_norm = prototypes / jnp.maximum(jnp.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12)
    return jnp.argmax(z_norm @ p_norm.T, axis=-1)


def forward_embeddings(net: MnistNet, z: jax.Array, *, batch_size: int,
                       device: jax.Device) -> dict[str, jax.Array]:
    batches = [net.forward_from_embedding(jax.device_put(z[i:i + batch_size], device))
               for i in range(0, z.shape[0], batch_size)]
    out = {
        "head_pred": jnp.argmax(jnp.concatenate([b[0] for b in batches], axis=0), axis=-1),
        "z": jnp.concatenate([b[1] for b in batches], axis=0),
        "h": jnp.concatenate([b[2] for b in batches], axis=0),
    }
    auxes = [b[3] for b in batches if b[3]]
    if auxes and "weights" in auxes[0]:
        out["weights"] = jnp.concatenate([b[3]["weights"] for b in batches], axis=0)
        out["proto_pred"] = jnp.concatenate([b[3]["proto_pred"] for b in batches], axis=0)
        if "class_mass" in auxes[0]:
            out["class_mass"] = jnp.concatenate([b[3]["class_mass"] for b in batches], axis=0)
    return out


def forward_images(net: MnistNet, images: jax.Array, *, batch_size: int,
                   device: jax.Device) -> dict[str, jax.Array]:
    batches = [net(jax.device_put(images[i:i + batch_size], device))
               for i in range(0, images.shape[0], batch_size)]
    logits = jnp.concatenate([b[0] for b in batches], axis=0)
    out = {
        "head_pred": jnp.argmax(logits, axis=-1),
        "z": jnp.concatenate([b[1] for b in batches], axis=0),
        "h": jnp.concatenate([b[2] for b in batches], axis=0),
    }
    if batches[0][3]:
        aux = batches[0][3]
        out["weights"] = jnp.concatenate([b[3]["weights"] for b in batches], axis=0)
        out["proto_pred"] = jnp.concatenate([b[3]["proto_pred"] for b in batches], axis=0)
        if "class_mass" in aux:
            out["class_mass"] = jnp.concatenate([b[3]["class_mass"] for b in batches], axis=0)
    return out


def retrieval_confusion(labels: jax.Array, retrieved: jax.Array,
                        num_classes: int = NUM_CLASSES) -> jax.Array:
    flat = labels * num_classes + retrieved
    return jnp.bincount(flat, length=num_classes * num_classes).reshape(num_classes, num_classes)


def summarize_outputs(outputs: dict[str, jax.Array], labels: jax.Array,
                      prototypes: jax.Array, memory_labels: jax.Array
                      ) -> tuple[dict[str, float | None], dict[str, jax.Array], jax.Array]:
    weights = outputs.get("weights")
    if weights is None:
        proto_pred = memory_labels[nearest_prototype(outputs["z"], prototypes)]
    else:
        proto_pred = outputs["proto_pred"]
    head_correct = outputs["head_pred"] == labels
    proto_correct = proto_pred == labels
    dist = retrieval_distance(outputs["h"], prototypes[labels])
    metrics = {
        "head_acc": float(jnp.mean(head_correct.astype(jnp.float32))),
        "proto_acc": float(jnp.mean(proto_correct.astype(jnp.float32))),
        "dist_to_true_proto": float(jnp.mean(dist)),
        "entropy_norm": None, "top1_weight": None, "true_proto_weight": None,
    }
    if weights is not None:
        metrics["entropy_norm"] = float(jnp.mean(attention_entropy(weights, normalize=True)))
        metrics["top1_weight"] = float(jnp.mean(topk_mass(weights, k=1)))
        if "class_mass" in outputs:
            true_weight = outputs["class_mass"][jnp.arange(labels.shape[0]), labels]
        else:
            true_mask = (memory_labels[None, :] == labels[:, None]).astype(weights.dtype)
            true_weight = jnp.sum(weights * true_mask, axis=-1)
        metrics["true_proto_weight"] = float(jnp.mean(true_weight))
    confusion = retrieval_confusion(labels, proto_pred, num_classes=prototypes.shape[0])
    return metrics, {"head": head_correct, "proto": proto_correct}, confusion


def robust_auc(levels: Sequence[float], accuracies: Sequence[float]) -> float:
    if len(levels) != len(accuracies) or len(levels) < 2:
        raise ValueError("robust_auc needs matching level/accuracy lists of length >= 2.")
    area = sum((levels[i + 1] - levels[i]) * (accuracies[i + 1] + accuracies[i]) / 2.0
               for i in range(len(levels) - 1))
    return area / (levels[-1] - levels[0])


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, statistics.stdev(values) if len(values) > 1 else 0.0


def count_trainable_params(model: Any) -> int:
    if hasattr(model, "params"):
        return sum(int(x.size) for x in _flatten_arrays(model.params()))
    return 0


def count_buffer_values(model: Any, *, floating_only: bool = False) -> int:
    if hasattr(model, "prototypes"):
        values = [model.prototypes, model.memory_labels]
        return sum(int(x.size) for x in values if not floating_only or
                   jnp.issubdtype(x.dtype, jnp.floating))
    return 0


def model_size_bytes(model: Any) -> int:
    state = getattr(model, "state_dict", lambda: {})()
    total = sum(int(x.size) * int(np.dtype(np.asarray(x).dtype).itemsize)
                for x in _flatten_arrays(state))
    return total


def analytical_macs(net: MnistNet) -> list[tuple[str, int]]:
    c1, c2 = net.encoder.c1, net.encoder.c2
    flat = c2 * (net.encoder.image_size // 4) ** 2
    result = [
        ("conv1", c1 * 9 * 28 * 28),
        ("conv2", c2 * c1 * 9 * 14 * 14),
        ("fc1", flat * net.encoder.hidden),
        ("fc2", net.encoder.hidden * net.encoder.embed_dim),
        ("head", net.encoder.embed_dim * NUM_CLASSES),
    ]
    if hasattr(net.bridge, "macs_per_image"):
        result.append(("hopfield", net.bridge.macs_per_image()))
    return result


def hook_macs(net: MnistNet, *, batch_size: int = 2) -> int:
    return sum(macs for _, macs in analytical_macs(net))


def median_iqr(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(arr)), "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))}


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def measure_cost(model: Any, batch_size: int, device: jax.Device, *,
                 warmup: int = 10, iters: int = 100,
                 image_size: int = IMAGE_SIZE) -> dict[str, Any]:
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, 1, image_size, image_size))
    x = jax.device_put(x, device)
    for _ in range(warmup):
        model(x)
    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        jax.block_until_ready(model(x)[0]) if hasattr(model(x), "__len__") else None
        times_ms.append((time.perf_counter() - start) * 1000.0)
    p50 = _percentile(times_ms, 0.50)
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "batch_size": batch_size,
        "latency_ms_p50": p50,
        "latency_ms_p95": _percentile(times_ms, 0.95),
        "latency_ms_mean": mean_ms,
        "latency_per_image_us": p50 * 1000.0 / batch_size,
        "throughput_img_s": batch_size / (mean_ms / 1000.0),
        "peak_mem_mb": max(0.001, batch_size * image_size * image_size * 4 / 2**20),
        "peak_mem_source": "activation upper bound",
    }


def noise_grids(cfg: dict[str, Any]) -> dict[str, list[float]]:
    return {kind: cfg[NOISE_GRID_KEYS[kind]] for kind in NOISE_TYPES}


def copy_net(net: Any) -> Any:
    return copy.deepcopy(net)


def time_full_evaluation(model: Any, images: jax.Array, *, batch_size: int,
                         device: jax.Device) -> float:
    start = time.perf_counter()
    forward_images(model, images, batch_size=batch_size, device=device)
    return time.perf_counter() - start


def run_baseline_only(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    rows = []
    for seed in cfg["seeds"]:
        splits = make_splits(data, seed=seed, val_size=cfg["val_size"],
                             train_subset=cfg["train_subset"], test_subset=cfg["test_subset"])
        net = build_baseline_net(seed)
        train_baseline(net, splits.train_images, splits.train_labels,
                       epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                       lr=cfg["lr"], seed=seed, device=device)
        z_train = encode(net.encoder, splits.train_images, batch_size=cfg["eval_batch_size"], device=device)
        prototypes, labels = compute_prototypes(z_train, splits.train_labels)
        for kind in ("clean", *NOISE_TYPES):
            levels = [0.0] if kind == "clean" else noise_grids(cfg)[kind]
            for level in levels:
                corrupted = splits.test_images if level == 0 else make_corrupted_test_set(
                    splits.test_images, seed=seed, kind=kind, level=level
                )
                outputs = forward_images(net, corrupted, batch_size=cfg["eval_batch_size"], device=device)
                metrics, _, _ = summarize_outputs(outputs, splits.test_labels, prototypes, labels)
                rows.append({"seed": seed, "kind": kind, "level": level, **metrics})
    out_dir = Path(cfg["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_results.json").write_text(
        json.dumps({"config": cfg, "environment": environment_info(device), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Baseline reports saved to: {out_dir}")
    return {"rows": rows}


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser("Task 13 MNIST JAX baseline")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run_baseline_only(cfg, retrain=args.retrain)


if __name__ == "__main__":
    main()
