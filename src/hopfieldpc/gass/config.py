"""Validated settings for the reference and safeguarded continuous memory banks."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceConfig:
    beta: float = 20.0
    observation_noise: float = 1.0  # r0, relative to warm-up coordinate variance
    initial_variance: float = 1.0  # p0
    process_noise: float = 0.01  # q per minibatch
    mixture_rate: float = 0.1
    read_weight: float = 1.0
    capacity_cost: float = 0.1
    margin: float = 0.001
    momentum: bool = False
    damping: float = 0.9
    window_size: int = 512
    edit_budget: int = 3
    patience: int = 2
    merge_cosine: float = 0.95  # Threshold on the key geometry's pair affinity.
    min_child_mass: float = 2.0
    read_beta: float | None = None
    write_beta: float | None = None
    value_aware: bool = True
    birth_weight: float | None = None
    separation: float | None = None
    guard_writes: bool = False
    guard_read_loss: bool = False
    uniform_weights: bool = False
    fixed_capacity: bool = False
    heuristic_edits: bool = False
    initial_covariance: tuple[float, float, float] | None = None
    process_covariance: tuple[float, float, float] | None = None
    dt: float = 1.
    support_fraction: float = .05

    def __post_init__(self):
        counts = (self.window_size, self.edit_budget, self.patience)
        if any(type(x) is not int for x in counts):
            raise ValueError("Window size, edit budget and patience must be integers.")
        if not all(math.isfinite(x) for x in vars(self).values() if isinstance(x, (float, int))):
            raise ValueError("GASS settings must be finite.")
        if not 0 <= self.support_fraction <= 1:
            raise ValueError("The support threshold must be a fraction of equal-share mass.")
        if min(self.beta_r, self.beta_w, self.dt) <= 0:
            raise ValueError("Temperatures and the time step must be positive.")
        if self.birth_weight is not None and not 0 < self.birth_weight < 1:
            raise ValueError("Birth mixture weight must be strictly between zero and one.")
        if self.separation is not None and not -1 < self.separation < 1:
            raise ValueError("The maximum pair affinity must be strictly between -1 and 1.")
        if min(self.beta, self.observation_noise, self.initial_variance, self.capacity_cost,
               self.margin, self.min_child_mass) <= 0 or min(self.process_noise, self.read_weight) < 0:
            raise ValueError("Invalid GASS noise, temperature, support or objective settings.")
        if not 0 < self.mixture_rate < 1 or not 0 <= self.damping < 1 or not -1 < self.merge_cosine < 1:
            raise ValueError("Invalid mixture rate, damping or merge affinity threshold.")
        for matrix in (self.initial_covariance, self.process_covariance):
            if matrix is not None:
                pp, pv, vv = matrix
                if not all(map(math.isfinite, matrix)) or min(pp, vv) < 0 or pp*vv < pv*pv:
                    raise ValueError("Velocity covariance must be positive semidefinite.")
        if self.initial_covariance is not None:
            pp, pv, vv = self.initial_covariance
            if pp*vv <= pv*pv:
                raise ValueError("Initial velocity covariance must be positive definite.")
        qvv = self.process_noise if self.process_covariance is None else self.process_covariance[2]
        if self.momentum and qvv == self.damping == 0:
            raise ValueError("The velocity filter requires positive process noise or nonzero damping.")
        if min(self.window_size, self.patience) < 1 or self.edit_budget < 0:
            raise ValueError("Invalid reference-window size, patience or edit budget.")

    @property
    def beta_r(self):
        return self.beta if self.read_beta is None else self.read_beta

    @property
    def beta_w(self):
        return self.beta if self.write_beta is None else self.write_beta


@dataclass(frozen=True)
class GASSConfig(ReferenceConfig):
    guard_writes: bool = True
    logit_bound: float = 16.  # Absolute coordinate bound on stored values and velocity.
    variance_min: float = .001
    variance_max: float = 100.
    trust_min: float = .001
    trust_max: float = 10.
    mixture_floor: float = 1e-6
    descent_margin: float = 1e-4
    line_search_steps: int = 20
    recall_steps: int = 0
    refinement_sweeps: int = 0
    stream_budget: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.variance_min <= self.variance_max or not 0 < self.trust_min <= self.trust_max:
            raise ValueError("Variance and trust bounds must be positive and ordered.")
        if min(self.logit_bound, self.descent_margin) <= 0 or not 0 < self.mixture_floor < 1:
            raise ValueError("Invalid logit bound, descent margin or mixture floor.")
        for count in (self.line_search_steps, self.recall_steps, self.refinement_sweeps):
            if type(count) is not int or count < 0:
                raise ValueError("Iteration budgets must be nonnegative integers.")
        if self.line_search_steps == 0:
            raise ValueError("Specify at least one line-search attempt.")
        if self.heuristic_edits or not self.guard_writes:
            raise ValueError("Safeguarded writes and edits require objective gates.")
        if self.stream_budget is not None and (not math.isfinite(self.stream_budget) or self.stream_budget < 0):
            raise ValueError("The lifetime stream budget must be finite and nonnegative.")
