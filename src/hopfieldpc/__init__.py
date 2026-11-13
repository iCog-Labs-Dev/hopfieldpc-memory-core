"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import (
    UPDATE_METHODS,
    BankSnapshot,
    CandidateProposal,
    HopfieldMemoryOutput,
    HopfieldShapeContract,
    HopfieldShapeError,
    KalmanMomentumStepOutput,
    positive_int,
    validate_beta,
    validate_normalization_epsilon,
    validate_query_memory,
    validate_update_method,
)
from .diagnostics import (
    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    correctness_categories,
    max_attention_weight,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
    topk_recall,
)
from .ema import (
    EMAOnlineBank,
    EMAPrototypeUpdater,
    ema_step,
)
from .energy import hopfield_energy
from .functional import (
    centered_retrieve,
    hopfield_retrieve,
    kmeans_prototypes,
    multi_step_retrieve,
)
from .kalman import (
    HybridOnlineBank,
    HybridPrototypeUpdater,
    KalmanOnlineBank,
    KalmanPrototypeUpdater,
    hybrid_kalman_momentum_step,
    kalman_gain,
    kalman_step,
    kalman_uncertainty_update,
)
from .memory import HopfieldMemory
from .momentum import (
    MomentumOnlineBank,
    MomentumPrototypeUpdater,
    momentum_step,
    momentum_velocity_update,
)
from .online import (
    CachedOccupancyBank,
    OnlineHopfieldBank,
)
from .updates import (
    METHODS,
    BasePrototypeUpdater,
    PrototypeUpdater,
    kalman_momentum_step,
)

__all__ = [
    "BankSnapshot",
    "BasePrototypeUpdater",
    "CachedOccupancyBank",
    "CandidateProposal",
    "EMAOnlineBank",
    "EMAPrototypeUpdater",
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "HybridOnlineBank",
    "HybridPrototypeUpdater",
    "KalmanMomentumStepOutput",
    "KalmanOnlineBank",
    "KalmanPrototypeUpdater",
    "METHODS",
    "MomentumOnlineBank",
    "MomentumPrototypeUpdater",
    "OnlineHopfieldBank",
    "PrototypeUpdater",
    "UPDATE_METHODS",
    "argmax_memory_index",
    "attention_entropy",
    "build_retrieval_diagnostics",
    "centered_retrieve",
    "correctness_categories",
    "ema_step",
    "hopfield_energy",
    "hopfield_retrieve",
    "hybrid_kalman_momentum_step",
    "kalman_gain",
    "kalman_momentum_step",
    "kalman_step",
    "kalman_uncertainty_update",
    "kmeans_prototypes",
    "max_attention_weight",
    "momentum_step",
    "momentum_velocity_update",
    "multi_step_retrieve",
    "positive_int",
    "retrieval_distance",
    "selected_memory_distance",
    "topk_mass",
    "topk_recall",
    "validate_beta",
    "validate_normalization_epsilon",
    "validate_query_memory",
    "validate_update_method",
]