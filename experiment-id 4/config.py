"""Hyperparameter configuration for Experiment 4.

Defines core design variables, mathematically self-calibrating defaults, and quick/full
MNIST experiment presets, and converts them into validated ``GlobalDynamicConfig``
instances for ``hopfieldpc.GlobalDynamicHopfieldBank``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from hopfieldpc.api import GlobalDynamicConfig

# ---------------------------------------------------------------------------
# Core Design Hyperparameters (4 primary knobs + self-calibrating math defaults)
# ---------------------------------------------------------------------------
AUTO_CALIBRATE: bool = True  # Derive None thresholds analytically from (D, M, K_active, Q, R_n, tau_coal)

DIM: int = 64  # D
MAX_MEMORIES: int = 80  # N_max
INITIAL_MEMORIES: int = 10  # K_init
MIN_ACTIVE_MEMORIES: int = 2  # K_global_min (>= 2 keeps softmax competition non-degenerate)
WARMUP_EPOCHS: int = 1  # E_warmup

# Hopfield temperatures (None -> beta_read = sqrt(D) + ln(K_active), beta_write = 1.5*sqrt(D) + ln(K_active))
BETA_READ: float | None = None
BETA_WRITE: float | None = None
NUM_UPDATES: int = 1

# Online Kalman-momentum & John Factor
# None initial_uncertainty -> warm-up per-dimension embedding variance P_0
# None uncertainty_min / max -> (1e-4 * P_0, 10 * P_0)
GAMMA_MOM: float = 0.9
GAMMA_JOHN: float = 1.0
INITIAL_UNCERTAINTY: float | None = None
UNCERTAINTY_MIN: float | None = None
UNCERTAINTY_MAX: float | None = None
PROCESS_NOISE: float = 0.01  # Q (relative to P_0)
OBSERVATION_NOISE: float = 0.05  # R_n (relative to P_0)

# Similarity repulsion & Gram coalescence
# None tau_sim -> tau_coal - 0.10; None lambda_sim -> 2e-4 * mean(||M - mu||^2)
TAU_SIM: float | None = None
LAMBDA_SIM: float | None = None

TAU_COAL: float = 0.95
COALESCENCE_PATIENCE: int = 2  # T_coal

# Temporally smoothed support & pruning
# None tau_support / tau_eff -> self-scales with M / K_active
# None alpha_n / alpha_s -> 1 - 1 / PRUNE_PATIENCE
TAU_SUPPORT: float | None = None
TAU_EFF: float | None = None
ALPHA_N: float | None = None
ALPHA_S: float | None = None
PRUNE_PATIENCE: int = 2  # T_prune

# Structural uncertainty & spectral fission
# None tau_uncert -> DARE steady-state P_inf + single-mode dispersion equilibrium from tau_coal
# None n_kish_min -> min(2*sqrt(D), 0.10 * M / K_active)
# None split_scale -> places newborn children at cosine similarity ~ tau_sim
# None structural_base_noise -> couples R_0 = R_n
TAU_UNCERT: float | None = None
N_KISH_MIN: float | None = None
SPLIT_SCALE: float | None = None
LAMBDA_DISP: float = 2.0
LAMBDA_SHARP: float = 0.5
STRUCTURAL_BASE_NOISE: float | None = None

# Manual fallback values used only when AUTO_CALIBRATE is False and a variable above is None
MANUAL_FALLBACKS: dict[str, float] = {
    "beta_read": 10.0,
    "beta_write": 15.0,
    "initial_uncertainty": 1.0,
    "uncertainty_min": 1e-4,
    "uncertainty_max": 10.0,
    "tau_sim": 0.85,
    "lambda_sim": 0.05,
    "tau_support": 50.0,
    "tau_eff": 10.0,
    "alpha_n": 0.5,
    "alpha_s": 0.5,
    "tau_uncert": 0.044,
    "n_kish_min": 25.0,
    "split_scale": 0.50,
    "structural_base_noise": 0.05,
}

EPS_STAB: float = 1e-8
EPS_NORM: float = 1e-8

# Execution & batching controls
ONLINE_UPDATE_MODE: str = "batch"  # "batch" | "sequential"
STAT_CHUNK_SIZE: int = 2048

# ---------------------------------------------------------------------------
# Experiment Training & Evaluation Variables
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
VAL_SIZE: int = 5000
TRAIN_SUBSET: int | None = None
TEST_SUBSET: int | None = None
NUM_THREADS: int = 4
OUTPUT_PATH: Path = Path("experiment-id 4/results/global_dynamic.json")
CORRUPTIONS: dict[str, list[float | int]] = {
    "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "salt_pepper": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
    "occlusion": [0, 7, 10, 14],
}

# ---------------------------------------------------------------------------
# Quick Preset Override Variables (None inherits from main config variables)
# ---------------------------------------------------------------------------
QUICK_MAX_MEMORIES: int | None = None
QUICK_INITIAL_MEMORIES: int | None = None
QUICK_MIN_ACTIVE_MEMORIES: int | None = None
QUICK_WARMUP_EPOCHS: int | None = None
QUICK_TAU_SUPPORT: float | None = None
QUICK_TAU_EFF: float | None = None
QUICK_N_KISH_MIN: float | None = None
QUICK_EPOCHS: int = 3
QUICK_VAL_SIZE: int = 500
QUICK_TRAIN_SUBSET: int | None = 2000
QUICK_TEST_SUBSET: int | None = 1000
QUICK_OUTPUT_PATH: Path = Path("experiment-id 4/results/global_dynamic_quick.json")
QUICK_CORRUPTIONS: dict[str, list[float | int]] = {
    "gaussian": [0.0, 0.5, 1.0],
    "salt_pepper": [0.0, 0.2],
    "occlusion": [0, 7],
}


@dataclass(frozen=True)
class MemoryHyperparameters:
    """Validated memory hyperparameters for Experiment 4."""

    # Architecture & capacity
    dim: int = DIM
    max_memories: int = MAX_MEMORIES
    initial_memories: int = INITIAL_MEMORIES
    min_active_memories: int = MIN_ACTIVE_MEMORIES
    warmup_epochs: int = WARMUP_EPOCHS

    # Hopfield read / write temperatures & retrieval steps
    beta_read: float | None = BETA_READ
    beta_write: float | None = BETA_WRITE
    num_updates: int = NUM_UPDATES

    # Online Kalman-momentum & John Factor
    gamma_mom: float = GAMMA_MOM
    gamma_john: float = GAMMA_JOHN
    initial_uncertainty: float | None = INITIAL_UNCERTAINTY
    uncertainty_min: float | None = UNCERTAINTY_MIN
    uncertainty_max: float | None = UNCERTAINTY_MAX
    process_noise: float = PROCESS_NOISE
    observation_noise: float = OBSERVATION_NOISE

    # Online similarity repulsion (Sec. 5.7)
    tau_sim: float | None = TAU_SIM
    lambda_sim: float | None = LAMBDA_SIM

    # Epoch-end redundancy coalescence (Sec. 6.4)
    tau_coal: float = TAU_COAL
    coalescence_patience: int = COALESCENCE_PATIENCE

    # Temporally smoothed support & starvation pruning (Sec. 6.1, 6.3)
    tau_support: float | None = TAU_SUPPORT
    tau_eff: float | None = TAU_EFF
    alpha_n: float | None = ALPHA_N
    alpha_s: float | None = ALPHA_S
    prune_patience: int = PRUNE_PATIENCE

    # Structural uncertainty & spectral fission (Sec. 6.2, 6.5)
    tau_uncert: float | None = TAU_UNCERT
    n_kish_min: float | None = N_KISH_MIN
    split_scale: float | None = SPLIT_SCALE
    lambda_disp: float = LAMBDA_DISP
    lambda_sharp: float = LAMBDA_SHARP
    structural_base_noise: float | None = STRUCTURAL_BASE_NOISE

    # Numerical stability & execution
    eps_stab: float = EPS_STAB
    eps_norm: float = EPS_NORM
    online_update_mode: str = ONLINE_UPDATE_MODE
    stat_chunk_size: int = STAT_CHUNK_SIZE
    auto_calibrate: bool = AUTO_CALIBRATE

    def to_global_dynamic_config(self, ablation_mode: str = "dynamic_adaptive") -> GlobalDynamicConfig:
        """Build a validated GlobalDynamicConfig for the requested memory variant."""

        return GlobalDynamicConfig(
            dim=self.dim,
            max_memories=self.max_memories,
            initial_memories=self.initial_memories,
            min_active_memories=self.min_active_memories,
            warmup_epochs=self.warmup_epochs,
            beta_read=self.beta_read,
            beta_write=self.beta_write,
            num_updates=self.num_updates,
            gamma_mom=self.gamma_mom,
            gamma_john=self.gamma_john,
            initial_uncertainty=self.initial_uncertainty,
            uncertainty_min=self.uncertainty_min,
            uncertainty_max=self.uncertainty_max,
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
            tau_sim=self.tau_sim,
            lambda_sim=self.lambda_sim,
            tau_coal=self.tau_coal,
            coalescence_patience=self.coalescence_patience,
            tau_support=self.tau_support,
            tau_eff=self.tau_eff,
            alpha_n=self.alpha_n,
            alpha_s=self.alpha_s,
            prune_patience=self.prune_patience,
            tau_uncert=self.tau_uncert,
            n_kish_min=self.n_kish_min,
            split_scale=self.split_scale,
            lambda_disp=self.lambda_disp,
            lambda_sharp=self.lambda_sharp,
            structural_base_noise=self.structural_base_noise,
            eps_stab=self.eps_stab,
            eps_norm=self.eps_norm,
            ablation_mode=ablation_mode,
            stat_chunk_size=self.stat_chunk_size,
            auto_calibrate=self.auto_calibrate,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete Experiment 4 training, comparison, and evaluation configuration."""

    ablations: tuple[str, ...] = ABLATIONS
    seeds: tuple[int, ...] = SEEDS
    device: str = DEVICE
    data_dir: str = DATA_DIR
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    val_size: int = VAL_SIZE
    train_subset: int | None = TRAIN_SUBSET
    test_subset: int | None = TEST_SUBSET
    num_threads: int = NUM_THREADS
    output_path: Path = OUTPUT_PATH
    memory: MemoryHyperparameters = field(default_factory=MemoryHyperparameters)
    corruptions: dict[str, list[float | int]] = field(
        default_factory=lambda: {k: list(v) for k, v in CORRUPTIONS.items()}
    )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["ablations"] = list(self.ablations)
        raw["seeds"] = list(self.seeds)
        raw["output_path"] = str(self.output_path)
        return raw


def _resolve_var(
    primary: float | None,
    key: str,
    *,
    auto_calibrate: bool,
    quick_override: float | None = None,
) -> float | None:
    val = quick_override if quick_override is not None else primary
    if val is not None:
        return val
    return None if auto_calibrate else MANUAL_FALLBACKS[key]


def _build_memory_from_module_vars(*, quick: bool, auto_calibrate: bool) -> MemoryHyperparameters:
    return MemoryHyperparameters(
        dim=DIM,
        max_memories=(
            QUICK_MAX_MEMORIES
            if (quick and QUICK_MAX_MEMORIES is not None)
            else MAX_MEMORIES
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
            QUICK_WARMUP_EPOCHS
            if (quick and QUICK_WARMUP_EPOCHS is not None)
            else WARMUP_EPOCHS
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


def get_config(
    *,
    quick: bool = False,
    auto_calibrate: bool | None = None,
    ablations: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    online_update_mode: str | None = None,
    output_path: Path | None = None,
    memory_overrides: dict[str, Any] | None = None,
    experiment_overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Build ExperimentConfig from module config variables with optional overrides."""

    resolved_auto = AUTO_CALIBRATE if auto_calibrate is None else auto_calibrate
    mem = _build_memory_from_module_vars(quick=quick, auto_calibrate=resolved_auto)
    mem_updates: dict[str, Any] = {}
    if online_update_mode is not None:
        mem_updates["online_update_mode"] = online_update_mode
    if memory_overrides:
        mem_updates.update(memory_overrides)
    if mem_updates:
        mem = replace(mem, **mem_updates)

    source_corruptions = QUICK_CORRUPTIONS if quick else CORRUPTIONS
    cfg = ExperimentConfig(
        ablations=ablations if ablations is not None else ABLATIONS,
        seeds=seeds if seeds is not None else SEEDS,
        device=DEVICE,
        data_dir=DATA_DIR,
        epochs=QUICK_EPOCHS if quick else EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        val_size=QUICK_VAL_SIZE if quick else VAL_SIZE,
        train_subset=QUICK_TRAIN_SUBSET if quick else TRAIN_SUBSET,
        test_subset=QUICK_TEST_SUBSET if quick else TEST_SUBSET,
        num_threads=NUM_THREADS,
        output_path=output_path if output_path is not None else (QUICK_OUTPUT_PATH if quick else OUTPUT_PATH),
        memory=mem,
        corruptions={k: list(v) for k, v in source_corruptions.items()},
    )
    if experiment_overrides:
        cfg = replace(cfg, **experiment_overrides)
    return cfg
