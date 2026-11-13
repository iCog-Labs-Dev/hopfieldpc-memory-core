"""Explicit, accuracy-gated memory buffers; never optimizer-trained parameters.

The caller freezes batch eligibility before calling ``plan_candidate`` / ``commit``.
Retrieval has no label argument. Labels constrain writes only.

Core implementations now reside in ``hopfieldpc`` and are re-exported here for
backward compatibility with Experiment 1 scripts and tests.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hopfieldpc.api import (
    BankSnapshot,
    CandidateProposal,
    positive_int,
    validate_beta,
    validate_normalization_epsilon,
    validate_query_memory,
)
from hopfieldpc.diagnostics import correctness_categories
from hopfieldpc.functional import centered_retrieve, hopfield_retrieve
from hopfieldpc.online import OnlineHopfieldBank

__all__ = [
    "BankSnapshot",
    "CandidateProposal",
    "OnlineHopfieldBank",
    "centered_retrieve",
    "correctness_categories",
    "hopfield_retrieve",
    "positive_int",
    "validate_beta",
    "validate_normalization_epsilon",
    "validate_query_memory",
]
