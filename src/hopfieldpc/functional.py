"""Functional Hopfield retrieval operations.

Implements the pure one-step modern Hopfield retrieval equation:

    weights = softmax(beta * query @ memory.T)
    retrieved = weights @ memory

and optional multi-step retrieval, which repeats the one-step update:

    q_{t+1} = HopfieldRetrieval(q_t, M)

The function in this file is intentionally independent from nn.Module so it can
be tested directly and reused by HopfieldMemory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, overload

import torch
import torch.nn.functional as F

from .api import (
    positive_int,
    validate_beta,
    validate_normalization_epsilon,
    validate_query_memory,
)


def _normalize_for_scores(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize query and memory only for similarity scoring.

    The retrieved vector is still computed from the original memory vectors.
    This keeps normalization from changing the memory values themselves.
    """

    query_norm = F.normalize(query, p=2, dim=-1, eps=eps)
    memory_norm = F.normalize(memory, p=2, dim=-1, eps=eps)
    return query_norm, memory_norm


def hopfield_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-step modern Hopfield retrieval.

    Args:
        query:
            Query tensor with shape [batch_size, dim].
        memory:
            Memory bank tensor with shape [num_memories, dim].
            Each row is one stored pattern.
        beta:
            Finite positive retrieval sharpness. Larger beta produces sharper
            memory selection.
        normalize:
            If True, query and memory are L2-normalized for similarity scoring.
            The retrieved vector is still computed from the original memory.

    Returns:
        retrieved:
            Retrieved memory vectors with shape [batch_size, dim].
        weights:
            Softmax memory weights with shape [batch_size, num_memories].

    Formula:
        scores = beta * query @ memory.T
        weights = softmax(scores)
        retrieved = weights @ memory
    """

    beta = validate_beta(beta)

    validate_query_memory(query=query, memory=memory)

    input_dtype = query.dtype
    working_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )
    query_work = query.to(dtype=working_dtype)
    memory_work = memory.to(dtype=working_dtype)

    if normalize:
        query_for_scores, memory_for_scores = _normalize_for_scores(
            query_work, memory_work
        )
    else:
        query_for_scores, memory_for_scores = query_work, memory_work

    similarities = torch.matmul(query_for_scores, memory_for_scores.transpose(0, 1))
    # Center before beta scaling to prevent positive overflow without changing softmax.
    similarities = similarities - similarities.amax(dim=-1, keepdim=True)
    scores = beta * similarities
    weights_work = torch.softmax(scores, dim=-1)
    retrieved_work = torch.matmul(weights_work, memory_work)

    return (
        retrieved_work.to(dtype=input_dtype),
        weights_work.to(dtype=input_dtype),
    )


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: ...


def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    num_updates: int = 1,
    normalize: bool = False,
    return_states: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
):
    """Run repeated Hopfield retrieval: q0 -> q1 -> ... -> qT.

    Args:
        query:
            Initial query q0 with shape [batch_size, dim].
        memory:
            Memory bank with shape [num_memories, dim].
        beta:
            Retrieval sharpness (positive).
        num_updates:
            Number of retrieval steps T (positive integer). Default 1
            reproduces plain one-step retrieval.
        normalize:
            Passed to every one-step retrieval (scoring-only normalization).
        return_states:
            If True, also return the list [q0, q1, ..., qT] for debugging.

    Returns:
        retrieved:
            Final state qT with shape [batch_size, dim].
        weights:
            Softmax weights from the last step, shape [batch_size, num_memories].
        states:
            Only if return_states=True: list of T + 1 tensors, q0 first.
    """

    if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    state = query
    states: list[torch.Tensor] | None = [query] if return_states else None
    weights: torch.Tensor | None = None

    for _ in range(num_updates):
        state, weights = hopfield_retrieve(
            query=state,
            memory=memory,
            beta=beta,
            normalize=normalize,
        )
        if states is not None:
            states.append(state)

    # num_updates > 0 is validated above, so the loop ran at least once.
    assert weights is not None

    if states is not None:
        return state, weights, states
    return state, weights


@torch.no_grad()
def centered_retrieve(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    *,
    beta: float = 5.0,
    num_updates: int = 1,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score centered directions and return a convex combination of RAW values."""

    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if center.shape != (query.shape[1],):
        raise ValueError("center must have shape [dim].")
    q, m = query - center, values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m).all():
        raise FloatingPointError("Nonfinite centered query or memory.")
    qnorm, mnorm = q.norm(dim=-1), m.norm(dim=-1)
    if (
        not torch.isfinite(qnorm).all()
        or not torch.isfinite(mnorm).all()
        or (mnorm <= eps).any()
    ):
        raise FloatingPointError(
            "Invalid centered query norm or degenerate memory key."
        )
    positive_int("num_updates", num_updates)
    views = F.normalize(m, dim=-1, eps=eps)
    raw = query
    for _ in range(num_updates):
        query_view = F.normalize(raw - center, dim=-1, eps=eps)
        _, weights = hopfield_retrieve(
            query_view, views, beta=beta, normalize=False
        )
        raw = weights @ values
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite retrieval output.")
    return raw, weights


def kmeans_prototypes(
    x: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator,
    iters: int = 25,
) -> torch.Tensor:
    """Seeded k-means++ / Lloyd on rows of x [N, D]; returns centroids [k, D]."""

    if k >= x.shape[0]:
        raise ValueError(
            f"k ({k}) must be smaller than the number of points ({x.shape[0]})."
        )
    first = torch.randint(0, x.shape[0], (1,), generator=generator)
    centroids = x[first]
    while centroids.shape[0] < k:
        d2 = torch.cdist(x, centroids).min(dim=1).values.pow(2)
        nxt = torch.multinomial(d2 / d2.sum(), 1, generator=generator)
        centroids = torch.cat([centroids, x[nxt]])
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        for j in range(k):
            members = x[assign == j]
            if members.shape[0] > 0:
                centroids[j] = members.mean(dim=0)
    return centroids


@torch.no_grad()
def masked_centered_retrieve(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 5.0,
    num_updates: int = 1,
    normalization_epsilon: float = 1e-8,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve raw prototypes using centered cosine or a domain-specific score.

    similarity(query, active_values) returns [batch_size, active_count] scores,
    with larger scores indicating better matches. Each read applies beta and
    softmax to those scores and mixes the original active values.
    """

    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [num_memories].")
    active_mask = mask.bool()
    if not active_mask.any():
        raise ValueError("At least one prototype must be active for retrieval.")

    active_values = values[active_mask]
    if not torch.isfinite(query).all() or not torch.isfinite(active_values).all():
        raise FloatingPointError("Nonfinite query or active memory.")
    if similarity is None:
        views = F.normalize(active_values - center, dim=-1, eps=eps)

    raw = query
    weights = torch.zeros(
        query.shape[0],
        values.shape[0],
        dtype=query.dtype,
        device=query.device,
    )
    for _ in range(num_updates):
        if similarity is None:
            scores = F.normalize(raw - center, dim=-1, eps=eps) @ views.T
        else:
            scores = similarity(raw, active_values)
            if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
                raise TypeError("similarity must return a floating-point tensor.")
            if scores.shape != (query.shape[0], active_values.shape[0]):
                raise ValueError("similarity must return scores of shape [batch_size, active_count].")
            scores = scores.to(query)
        active_weights = torch.softmax(beta * scores, dim=-1)
        raw = active_weights @ active_values
        weights[:, active_mask] = active_weights
    if not torch.isfinite(raw).all() or not torch.isfinite(weights).all():
        raise FloatingPointError("Nonfinite masked retrieval output.")
    return raw, weights


@torch.no_grad()
def principal_split_direction(
    members: torch.Tensor,
    fallback_dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute first principal component direction of assigned embeddings S_{j,n}."""

    if members.shape[0] >= 2:
        centered = members - members.mean(dim=0, keepdim=True)
        cov_scale = centered.norm(dim=-1).mean()
        if torch.isfinite(cov_scale) and cov_scale > eps:
            try:
                _, s, vh = torch.linalg.svd(centered, full_matrices=False)
                direction = vh[0]
                scale = s[0] / math.sqrt(max(members.shape[0] - 1, 1))
                if torch.isfinite(direction).all() and scale > eps:
                    return F.normalize(direction, dim=-1, eps=eps) * scale
            except RuntimeError:
                pass
    unit = torch.zeros(
        fallback_dim, dtype=members.dtype, device=members.device
    )
    unit[0] = 1.0
    return unit


@torch.no_grad()
def centered_cosine_similarity(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    *,
    normalization_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute centered cosine similarity matrix s_ij = z_tilde_i^T M_tilde_j."""

    validate_query_memory(query, values)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if center.shape != (query.shape[1],):
        raise ValueError("center must have shape [dim].")
    q_norm = F.normalize(query - center, dim=-1, eps=eps)
    m_norm = F.normalize(values - center, dim=-1, eps=eps)
    return q_norm @ m_norm.T


@torch.no_grad()
def _active_write_responsibilities(
    query: torch.Tensor,
    active_values: torch.Tensor,
    center: torch.Tensor,
    beta_write: float,
    normalization_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = query - center
    m = active_values - center
    if not torch.isfinite(q).all() or not torch.isfinite(m).all():
        raise FloatingPointError("Nonfinite centered query or active memory.")
    queries = F.normalize(q, dim=-1, eps=normalization_epsilon)
    keys = F.normalize(m, dim=-1, eps=normalization_epsilon)
    similarities = queries @ keys.T
    logits = beta_write * similarities
    return torch.softmax(logits - logits.amax(dim=-1, keepdim=True), dim=-1), similarities


@torch.no_grad()
def masked_global_responsibilities(
    query: torch.Tensor,
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta_write: float = 5.0,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global write responsibilities r_ij over all active slots."""

    beta = validate_beta(beta_write)
    eps = validate_normalization_epsilon(normalization_epsilon)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [num_memories].")
    active_mask = mask.bool()
    if not active_mask.any():
        raise ValueError("At least one prototype must be active.")

    weights, scores = _active_write_responsibilities(query, values[active_mask], center, beta, eps)
    similarities = scores.new_zeros((query.shape[0], len(values)))
    responsibilities = torch.zeros_like(similarities)
    similarities[:, active_mask] = scores
    responsibilities[:, active_mask] = weights
    return responsibilities, similarities


@torch.no_grad()
def masked_gram_matrix(
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalization_epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized active prototype keys M_tilde and Gram matrix G = M_tilde M_tilde^T."""

    eps = validate_normalization_epsilon(normalization_epsilon)
    if values.ndim != 2 or center.shape != (values.shape[1],):
        raise ValueError("values must be [N, D] and center must be [D].")
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("mask must have shape [N].")
    active = mask.bool()
    keys = torch.zeros_like(values)
    slots = active.nonzero().flatten()
    active_keys = F.normalize(values[slots] - center, dim=-1, eps=eps)
    keys[slots] = active_keys
    active_gram = active_keys @ active_keys.T
    gram = active_gram.new_zeros((len(values), len(values)))
    gram[slots[:, None], slots] = active_gram
    return keys, gram


@torch.no_grad()
def online_similarity_penalty_step(
    values: torch.Tensor,
    center: torch.Tensor,
    mask: torch.Tensor,
    *,
    tau_sim: float = 0.85,
    lambda_sim: float = 0.05,
    normalization_epsilon: float = 1e-8,
    scale_by_norm: bool = False,
) -> tuple[torch.Tensor, float]:
    """Apply Section 5.7 masked online similarity repulsion to detached prototype values."""

    eps = validate_normalization_epsilon(normalization_epsilon)
    active = mask.bool()
    updated = values.detach().clone()
    active_updated, loss = _active_similarity_penalty_step(
        values[active], center, tau_sim, lambda_sim, eps, scale_by_norm,
    )
    updated[active] = active_updated
    return updated, loss


@torch.no_grad()
def _active_similarity_penalty_step(
    active_values: torch.Tensor,
    center: torch.Tensor,
    tau_sim: float,
    lambda_sim: float,
    eps: float,
    scale_by_norm: bool,
) -> tuple[torch.Tensor, float]:
    if lambda_sim < 0.0 or not math.isfinite(lambda_sim):
        raise ValueError("lambda_sim must be finite and nonnegative.")
    if lambda_sim == 0.0 or len(active_values) < 2:
        return active_values, 0.0
    eff_lambda = float(lambda_sim)
    if scale_by_norm:
        mean_sq_norm = float(
            (active_values - center).pow(2).sum(dim=-1).mean().item()
        )
        eff_lambda = eff_lambda * max(mean_sq_norm, eps)

    with torch.enable_grad():
        m_var = active_values.detach().requires_grad_(True)
        keys = F.normalize(m_var - center.detach(), dim=-1, eps=eps)
        gram = keys @ keys.T
        diagonal = torch.eye(len(keys), dtype=torch.bool, device=keys.device)
        excess = torch.relu(gram - tau_sim).masked_fill(diagonal, 0)
        loss = 0.5 * excess.pow(2).sum()
        (grad,) = torch.autograd.grad(loss, m_var)

    return active_values - eff_lambda * grad, float(loss.detach().item())


def split_eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascending eigenvalues and leading direction of a symmetric split matrix."""
    # Concentrated responsibilities can produce many zero eigenvalues. CUDA
    # 11.8's eigh can fail on these matrices. Use LAPACK for the epoch-level
    # split and copy back only the spectrum and the leading direction.
    solver_input = matrix.cpu() if matrix.is_cuda else matrix
    values, vectors = torch.linalg.eigh(solver_input)
    return values.to(matrix.device), vectors[:, -1].contiguous().to(matrix.device)


@torch.no_grad()
def weighted_spectral_fission(
    embeddings: torch.Tensor | None = None,
    weights_a: torch.Tensor | None = None,
    *,
    split_scale: float | None = 0.25,
    center: torch.Tensor | None = None,
    tau_sim: float = 0.85,
    eps: float = 1e-8,
    weighted_mean: torch.Tensor | None = None,
    cov_matrix: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Section 6.5 responsibility-weighted spectral fission along leading PC of Sigma_j."""

    if weighted_mean is None or cov_matrix is None:
        if embeddings is None or weights_a is None:
            raise ValueError(
                "Provide either (embeddings, weights_a) or (weighted_mean, cov_matrix)."
            )
        w = weights_a.to(dtype=embeddings.dtype, device=embeddings.device).clamp_min(0.0)
        w_sum = float(w.sum().item())
        if w_sum <= eps:
            mean = embeddings.mean(dim=0)
            dim = embeddings.shape[1]
            cov = torch.zeros(dim, dim, dtype=embeddings.dtype, device=embeddings.device)
        else:
            w = w / w_sum
            mean = (w.unsqueeze(-1) * embeddings).sum(dim=0)
            centered = embeddings - mean.unsqueeze(0)
            weighted_centered = torch.sqrt(w).unsqueeze(-1) * centered
            cov = weighted_centered.T @ weighted_centered
    else:
        mean = weighted_mean
        cov = cov_matrix

    sym_cov = 0.5 * (cov + cov.T)
    eigvals, direction = split_eigensystem(sym_cov)
    # eigh returns ascending eigenvalues; reverse to descending order.
    eigvals_desc = torch.flip(eigvals, dims=[0]).clamp_min(0.0)
    result = _spectral_children(mean, eigvals_desc, direction, split_scale, center, tau_sim, eps)
    result["covariance"] = sym_cov
    return result


@torch.no_grad()
def sample_space_fission(
    embeddings: torch.Tensor,
    weights_a: torch.Tensor,
    *,
    split_scale: float | None = 0.25,
    center: torch.Tensor | None = None,
    tau_sim: float = 0.85,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute the same principal split from the sample Gram matrix for wide vectors."""
    weights = weights_a.to(embeddings).clamp_min(0)
    mass = weights.sum()
    if mass.item() <= eps:
        mean = embeddings.mean(0)
        eigenvalues = embeddings.new_zeros(1)
        direction = torch.zeros_like(mean)
    else:
        weights = weights / mass
        mean = weights @ embeddings
        centered = weights.sqrt().unsqueeze(1) * (embeddings - mean)
        eigenvalues, sample_direction = split_eigensystem(centered @ centered.T)
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        # Lift the leading sample-space eigenvector back into feature space.
        direction = centered.T @ sample_direction
        if eigenvalues[0] > 0:
            direction = direction / eigenvalues[0].sqrt()
    return _spectral_children(mean, eigenvalues, direction, split_scale, center, tau_sim, eps)


def _spectral_children(
    mean: torch.Tensor,
    eigvals_desc: torch.Tensor,
    u1: torch.Tensor,
    split_scale: float | None,
    center: torch.Tensor | None,
    tau_sim: float,
    eps: float,
) -> dict[str, Any]:
    lambda1 = float(eigvals_desc[0].item())
    lambda2 = float(eigvals_desc[1].item()) if eigvals_desc.numel() > 1 else 0.0
    trace_cov = float(eigvals_desc.sum().item())
    anisotropy = lambda1 / (trace_cov + eps)
    sigma1 = math.sqrt(max(lambda1, 0.0))
    if split_scale is None:
        centered_ref = mean - center if center is not None else mean
        eff_split_scale = auto_split_scale(
            float(centered_ref.norm().item()),
            sigma1,
            tau_sim=tau_sim,
            eps=eps,
        )
    else:
        eff_split_scale = float(split_scale)
    delta = eff_split_scale * sigma1 * u1
    child_a = mean + delta
    child_b = mean - delta

    return {
        "mean": mean,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "trace": trace_cov,
        "anisotropy": anisotropy,
        "sigma1": sigma1,
        "u1": u1,
        "split_scale": eff_split_scale,
        "delta": delta,
        "displacement_norm": float(delta.norm().item()),
        "child_a": child_a,
        "child_b": child_b,
    }


def auto_hopfield_betas(dim: int, active_count: int) -> tuple[float, float]:
    """Analytical inverse temperatures scaled by sphere dimension sqrt(D) and log(K_active)."""

    root_d = math.sqrt(max(dim, 1))
    log_k = math.log(max(active_count, 2))
    beta_read = root_d + log_k
    beta_write = 1.5 * root_d + log_k
    return beta_read, beta_write


def auto_support_thresholds(
    num_samples: int,
    active_count: int,
    dim: int,
) -> tuple[float, float, float]:
    """Self-scaling (tau_support, tau_eff, n_kish_min) from sample count M and K_active."""

    k = max(active_count, 1)
    fair_share = float(num_samples) / float(k)
    tau_support = 0.05 * fair_share
    # Concentrated small cluster has S_j <= N_j; diffuse floor scales as 1 / K_active.
    tau_eff = tau_support * min(0.5, 2.0 / float(max(k, 2)))
    n_kish_min = max(2.0, min(2.0 * math.sqrt(max(dim, 1)), 0.10 * fair_share))
    return tau_support, tau_eff, n_kish_min


def auto_split_scale(
    centered_parent_norm: float,
    sigma1: float,
    *,
    tau_sim: float = 0.85,
    eps: float = 1e-8,
) -> float:
    """Analytical split scale placing newborn children at cosine similarity ~ tau_sim < tau_coal."""

    ts = max(0.0, min(0.999, float(tau_sim)))
    angular_factor = math.sqrt((1.0 - ts) / (1.0 + ts))
    raw_scale = (float(centered_parent_norm) / (float(sigma1) + eps)) * angular_factor
    return max(0.1, min(2.0, raw_scale))


def auto_smoothing_alpha(prune_patience: int) -> float:
    """EMA support smoothing coefficient coupled to starvation patience T_prune: 1 - 1/T_prune."""

    return 1.0 - 1.0 / float(max(int(prune_patience), 1))


