"""Shared GASS command-line settings for the Experiment 5 entry points."""
from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gass import GASSConfig
from gass.geometry import loss_names, score_names


def add_memory_options(parser, defaults):
    parser.add_argument("--gass-beta", type=float, default=defaults.beta, help="Shared GASS inverse temperature")
    parser.add_argument("--gass-read-beta", type=float)
    parser.add_argument("--gass-write-beta", type=float)
    parser.add_argument("--read-score", choices=score_names(), default=defaults.read_score)
    parser.add_argument("--write-score", choices=score_names(), default=defaults.write_score)
    parser.add_argument("--similarity-only", action="store_true", help="Omit value evidence from assignment (chi=0)")
    parser.add_argument("--retrieval-loss", choices=loss_names(), default=defaults.retrieval_loss)
    parser.add_argument("--key-radius", type=float, default=1., help="Dot-product key norm bound")
    parser.add_argument("--metric", type=Path, help="Fixed SPD Mahalanobis matrix (.npy); otherwise fit at warm-up")
    parser.add_argument("--split-scale", type=float, default=1., help="Euclidean split displacement in leading standard deviations")
    parser.add_argument("--split-angles", type=float, nargs="+", help="Cosine split proposal angles in radians")
    parser.add_argument("--support-fraction", type=float, default=defaults.support_fraction)
    parser.add_argument("--birth-weight", type=float)
    parser.add_argument("--separation", type=float, help="Maximum pair affinity allowed at every commit")
    parser.add_argument("--guard-writes", action="store_true", default=defaults.guard_writes, help="Require full batch objective nonincrease for writes")
    parser.add_argument("--guard-read-loss", action="store_true", help="Require nonincreasing retrieval loss for edits")
    parser.add_argument("--uniform-weights", action="store_true", help="Mixture-weight ablation")
    parser.add_argument("--fixed-capacity", action="store_true", help="Disable GASS structural edits")
    parser.add_argument("--heuristic-edits", action="store_true", help="Ablation: bypass the objective acceptance gate")
    parser.add_argument("--structural-interval", type=int, default=1)
    parser.add_argument("--min-active-memories", type=int)
    parser.add_argument("--initial-memories", type=int)
    parser.add_argument("--max-memories", type=int)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--merge-threshold", type=float, default=defaults.merge_cosine)
    parser.add_argument("--min-child-mass", type=float, default=defaults.min_child_mass)
    parser.add_argument("--dt", type=float, default=1.)
    parser.add_argument("--initial-covariance", nargs=3, type=float, metavar=("PP","PV","VV"))
    parser.add_argument("--process-covariance", nargs=3, type=float, metavar=("PP","PV","VV"))

    parser.add_argument("--observation-noise", type=float, default=defaults.observation_noise, help="GASS r0, in warm-up variance units")
    parser.add_argument("--initial-variance", type=float, default=defaults.initial_variance, help="GASS p0, in warm-up variance units")
    parser.add_argument("--process-noise", type=float, default=defaults.process_noise, help="GASS q per minibatch")
    parser.add_argument("--mixture-rate", type=float, default=defaults.mixture_rate)
    parser.add_argument("--read-weight", type=float, default=defaults.read_weight)
    parser.add_argument("--capacity-cost", type=float, default=defaults.capacity_cost)
    parser.add_argument("--margin", type=float, default=defaults.margin, help="Minimum structural objective improvement")
    parser.add_argument("--momentum", action="store_true", help="Use the position–velocity Kalman filter for GASS")
    parser.add_argument("--damping", type=float, default=defaults.damping)
    parser.add_argument("--window-size", type=int, default=defaults.window_size, help="Shared training reference window size")
    parser.add_argument("--edit-budget", type=int, default=defaults.edit_budget, help="Maximum accepted GASS edits per epoch")
    if isinstance(defaults, GASSConfig):
        add_safeguards(parser, defaults)


def add_safeguards(parser, cfg):
    fields = {
        "logit_bound": "Absolute image key and value bound",
        "variance_min": "Minimum observation variance in warm-up variance units",
        "variance_max": "Maximum observation variance in warm-up variance units",
        "trust_min": "Minimum predicted trust in warm-up variance units",
        "trust_max": "Maximum predicted trust in warm-up variance units",
        "mixture_floor": "Smallest active mixture probability",
        "descent_margin": "Positive sufficient-decrease coefficient for candidate steps",
        "line_search_steps": "Maximum projected key or weight backtracking attempts",
        "recall_steps": "Frozen key iterations before value readout (cosine or tied dot; zero for one-step GASS)",
        "refinement_sweeps": "Extra fixed-window continuous sweeps per structural check, without new measurements",
    }
    for field, help_text in fields.items():
        default = getattr(cfg, field)
        parser.add_argument("--" + field.replace("_", "-"), type=type(default), default=default, help=help_text)
    parser.add_argument("--stream-budget", type=float,
                        help="Optional lifetime movement bound B on keys, values, weights and R. "
                             "Each fresh update, replay sweep or structural pass can move by at most B/2^(n+1).")


def memory_settings(args, defaults):
    aliases = {"beta": "gass_beta", "read_beta": "gass_read_beta", "write_beta": "gass_write_beta",
               "merge_cosine": "merge_threshold"}
    values = {field.name: getattr(args, aliases.get(field.name, field.name), getattr(defaults, field.name))
              for field in fields(defaults)}
    values["value_aware"] = not args.similarity_only
    return replace(defaults, **values)


def load_metric(path):
    if path is None:
        return None
    import numpy as np
    return torch.from_numpy(np.load(path, allow_pickle=False))
