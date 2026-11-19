"""CIFAR-10 hyperparameter configuration for Experiment 4.

Mirrors ``config.py`` while setting CIFAR-10 embedding dimension, capacity budget,
optimizer regularization, data augmentation, and RGB corruption grids, and converts
memory settings into validated ``GlobalDynamicConfig`` instances via ``MemoryHyperparameters``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from benchmarks.cifar10_baseline import CIFAR10_EMBED_DIM, CIFAR10_HIDDEN_DIM
from config import MANUAL_FALLBACKS, MemoryHyperparameters, _resolve_var

# ---------------------------------------------------------------------------
# Core Design Hyperparameters (self-calibrating math defaults via AUTO_CALIBRATE)
# ---------------------------------------------------------------------------
AUTO_CALIBRATE: bool = True

DIM: int = CIFAR10_EMBED_DIM  # D = 128 matching Task 13 Cifar10Encoder
MAX_MEMORIES: int = 80  # N_max (matches 8 prototypes/class x 10 classes = 80 budget)
INITIAL_MEMORIES: int = 10  # K_init
MIN_ACTIVE_MEMORIES: int = 2  # K_global_min
WARMUP_EPOCHS: int = 2  # E_warmup (2 epochs on CIFAR-10 so RGB Conv features form semantic clusters)

BETA_READ: float | None = None
BETA_WRITE: float | None = None
NUM_UPDATES: int = 1

GAMMA_MOM: float = 0.9
GAMMA_JOHN: float = 1.0
INITIAL_UNCERTAINTY: float | None = None
UNCERTAINTY_MIN: float | None = None
UNCERTAINTY_MAX: float | None = None
PROCESS_NOISE: float = 0.01
OBSERVATION_NOISE: float = 0.05

TAU_SIM: float | None = None
LAMBDA_SIM: float | None = None

TAU_COAL: float = 0.95
COALESCENCE_PATIENCE: int = 2

TAU_SUPPORT: float | None = None
TAU_EFF: float | None = None
ALPHA_N: float | None = None
ALPHA_S: float | None = None
PRUNE_PATIENCE: int = 2

TAU_UNCERT: float | None = None
N_KISH_MIN: float | None = None
SPLIT_SCALE: float | None = None
LAMBDA_DISP: float = 2.0
LAMBDA_SHARP: float = 0.5
STRUCTURAL_BASE_NOISE: float | None = None

EPS_STAB: float = 1e-8
EPS_NORM: float = 1e-8

ONLINE_UPDATE_MODE: str = "batch"
STAT_CHUNK_SIZE: int = 2048

# ---------------------------------------------------------------------------
# CIFAR-10 Training & Evaluation Variables
# ---------------------------------------------------------------------------
COMPARISON_VARIANTS: tuple[str, ...] = (
    "fixed_capacity",
    "dynamic_adaptive",
)
ABLATIONS: tuple[str, ...] = COMPARISON_VARIANTS
SEEDS: tuple[int, ...] = (0, 1, 2)
DEVICE: str = "auto"
DATA_DIR: str = "data"
EPOCHS: int = 10
BATCH_SIZE: int = 128
EVAL_BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.001
WEIGHT_DECAY: float = 1e-4
ENCODER_CHANNELS: tuple[int, int, int] = (32, 64, 128)
ENCODER_HIDDEN: int = CIFAR10_HIDDEN_DIM
CONV_DROPOUT: float = 0.2
DENSE_DROPOUT: float = 0.3
VAL_SIZE: int = 5000
TRAIN_SUBSET: int | None = None
TEST_SUBSET: int | None = None
NUM_THREADS: int = 4
OUTPUT_PATH: Path = Path("experiment-id 4/results/cifar10_global_dynamic.json")
CORRUPTIONS: dict[str, list[float | int]] = {
    "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
    "occlusion": [0, 7, 10, 14],
}

# ---------------------------------------------------------------------------
# Quick Preset Override Variables
# ---------------------------------------------------------------------------
QUICK_MAX_MEMORIES: int | None = None
QUICK_INITIAL_MEMORIES: int | None = None
QUICK_MIN_ACTIVE_MEMORIES: int | None = None
QUICK_WARMUP_EPOCHS: int | None = 1
QUICK_TAU_SUPPORT: float | None = None
QUICK_TAU_EFF: float | None = None
QUICK_N_KISH_MIN: float | None = None
QUICK_EPOCHS: int = 3
QUICK_VAL_SIZE: int = 500
QUICK_TRAIN_SUBSET: int | None = 2000
QUICK_TEST_SUBSET: int | None = 1000
QUICK_OUTPUT_PATH: Path = Path("experiment-id 4/results/cifar10_global_dynamic_quick.json")
QUICK_CORRUPTIONS: dict[str, list[float | int]] = {
    "gaussian": [0.0, 0.3, 0.5, 1.0],
    "salt_pepper": [0.0, 0.1, 0.3],
    "occlusion": [0, 7, 14],
}


@dataclass(frozen=True)
class Cifar10ExperimentConfig:
    """Complete Experiment 4 CIFAR-10 training, comparison, and evaluation configuration."""

    dataset: str = "cifar10"
    ablations: tuple[str, ...] = ABLATIONS
    seeds: tuple[int, ...] = SEEDS
    device: str = DEVICE
    data_dir: str = DATA_DIR
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    encoder_channels: tuple[int, int, int] = ENCODER_CHANNELS
    encoder_hidden: int = ENCODER_HIDDEN
    conv_dropout: float = CONV_DROPOUT
    dense_dropout: float = DENSE_DROPOUT
    val_size: int = VAL_SIZE
    train_subset: int | None = TRAIN_SUBSET
    test_subset: int | None = TEST_SUBSET
    num_threads: int = NUM_THREADS
    output_path: Path = OUTPUT_PATH
    memory: MemoryHyperparameters = field(
        default_factory=lambda: MemoryHyperparameters(
            dim=DIM,
            max_memories=MAX_MEMORIES,
            initial_memories=INITIAL_MEMORIES,
            min_active_memories=MIN_ACTIVE_MEMORIES,
            warmup_epochs=WARMUP_EPOCHS,
        )
    )
    corruptions: dict[str, list[float | int]] = field(
        default_factory=lambda: {k: list(v) for k, v in CORRUPTIONS.items()}
    )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["ablations"] = list(self.ablations)
        raw["seeds"] = list(self.seeds)
        raw["encoder_channels"] = list(self.encoder_channels)
        raw["output_path"] = str(self.output_path)
        return raw


def _build_cifar10_memory(*, quick: bool, auto_calibrate: bool) -> MemoryHyperparameters:
    _ = MANUAL_FALLBACKS  # shared fallback registry from config.py
    return MemoryHyperparameters(
        dim=DIM,
        max_memories=(
            QUICK_MAX_MEMORIES if (quick and QUICK_MAX_MEMORIES is not None) else MAX_MEMORIES
        ),
        initial_memories=(
            QUICK_INITIAL_MEMORIES
            if (quick and QUICK_INITIAL_MEMORIES is not None)
            else INITIAL_MEMORIES
        ),
        min_active_memories=(
            QUICK_MIN_ACTIVE_MEMORIES
            if (quick and QUICK_MIN_ACTIVE_MEMORIES is not None)
            else MIN_ACTIVE_MEMORIES
        ),
        warmup_epochs=(
            QUICK_WARMUP_EPOCHS if (quick and QUICK_WARMUP_EPOCHS is not None) else WARMUP_EPOCHS
        ),
        beta_read=_resolve_var(BETA_READ, "beta_read", auto_calibrate=auto_calibrate),
        beta_write=_resolve_var(BETA_WRITE, "beta_write", auto_calibrate=auto_calibrate),
        num_updates=NUM_UPDATES,
        gamma_mom=GAMMA_MOM,
        gamma_john=GAMMA_JOHN,
        initial_uncertainty=_resolve_var(
            INITIAL_UNCERTAINTY, "initial_uncertainty", auto_calibrate=auto_calibrate
        ),
        uncertainty_min=_resolve_var(
            UNCERTAINTY_MIN, "uncertainty_min", auto_calibrate=auto_calibrate
        ),
        uncertainty_max=_resolve_var(
            UNCERTAINTY_MAX, "uncertainty_max", auto_calibrate=auto_calibrate
        ),
        process_noise=PROCESS_NOISE,
        observation_noise=OBSERVATION_NOISE,
        tau_sim=_resolve_var(TAU_SIM, "tau_sim", auto_calibrate=auto_calibrate),
        lambda_sim=_resolve_var(LAMBDA_SIM, "lambda_sim", auto_calibrate=auto_calibrate),
        tau_coal=TAU_COAL,
        coalescence_patience=COALESCENCE_PATIENCE,
        tau_support=_resolve_var(
            TAU_SUPPORT,
            "tau_support",
            auto_calibrate=auto_calibrate,
            quick_override=QUICK_TAU_SUPPORT if quick else None,
        ),
        tau_eff=_resolve_var(
            TAU_EFF,
            "tau_eff",
            auto_calibrate=auto_calibrate,
            quick_override=QUICK_TAU_EFF if quick else None,
        ),
        alpha_n=_resolve_var(ALPHA_N, "alpha_n", auto_calibrate=auto_calibrate),
        alpha_s=_resolve_var(ALPHA_S, "alpha_s", auto_calibrate=auto_calibrate),
        prune_patience=PRUNE_PATIENCE,
        tau_uncert=_resolve_var(TAU_UNCERT, "tau_uncert", auto_calibrate=auto_calibrate),
        n_kish_min=_resolve_var(
            N_KISH_MIN,
            "n_kish_min",
            auto_calibrate=auto_calibrate,
            quick_override=QUICK_N_KISH_MIN if quick else None,
        ),
        split_scale=_resolve_var(SPLIT_SCALE, "split_scale", auto_calibrate=auto_calibrate),
        lambda_disp=LAMBDA_DISP,
        lambda_sharp=LAMBDA_SHARP,
        structural_base_noise=_resolve_var(
            STRUCTURAL_BASE_NOISE, "structural_base_noise", auto_calibrate=auto_calibrate
        ),
        eps_stab=EPS_STAB,
        eps_norm=EPS_NORM,
        online_update_mode=ONLINE_UPDATE_MODE,
        stat_chunk_size=STAT_CHUNK_SIZE,
        auto_calibrate=auto_calibrate,
    )


def get_cifar10_config(
    *,
    quick: bool = False,
    auto_calibrate: bool | None = None,
    ablations: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    online_update_mode: str | None = None,
    output_path: Path | None = None,
    memory_overrides: dict[str, Any] | None = None,
    experiment_overrides: dict[str, Any] | None = None,
) -> Cifar10ExperimentConfig:
    """Build Cifar10ExperimentConfig from module variables with optional overrides."""

    resolved_auto = AUTO_CALIBRATE if auto_calibrate is None else auto_calibrate
    mem = _build_cifar10_memory(quick=quick, auto_calibrate=resolved_auto)
    mem_updates: dict[str, Any] = {}
    if online_update_mode is not None:
        mem_updates["online_update_mode"] = online_update_mode
    if memory_overrides:
        mem_updates.update(memory_overrides)
    if mem_updates:
        mem = replace(mem, **mem_updates)

    source_corruptions = QUICK_CORRUPTIONS if quick else CORRUPTIONS
    cfg = Cifar10ExperimentConfig(
        ablations=ablations if ablations is not None else ABLATIONS,
        seeds=seeds if seeds is not None else SEEDS,
        device=DEVICE,
        data_dir=DATA_DIR,
        epochs=QUICK_EPOCHS if quick else EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        encoder_channels=ENCODER_CHANNELS,
        encoder_hidden=ENCODER_HIDDEN,
        conv_dropout=CONV_DROPOUT,
        dense_dropout=DENSE_DROPOUT,
        val_size=QUICK_VAL_SIZE if quick else VAL_SIZE,
        train_subset=QUICK_TRAIN_SUBSET if quick else TRAIN_SUBSET,
        test_subset=QUICK_TEST_SUBSET if quick else TEST_SUBSET,
        num_threads=NUM_THREADS,
        output_path=(
            output_path
            if output_path is not None
            else (QUICK_OUTPUT_PATH if quick else OUTPUT_PATH)
        ),
        memory=mem,
        corruptions={k: list(v) for k, v in source_corruptions.items()},
    )
    if experiment_overrides:
        cfg = replace(cfg, **experiment_overrides)
    return cfg


get_config = get_cifar10_config
