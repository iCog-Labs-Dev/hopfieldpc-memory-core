
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from benchmarks import mnist_prototype_retrieval as benchmark


def herding(
    x: torch.Tensor,
    k: int,
    *,
    normalize_for_selection: bool = True,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Select k distinct embeddings whose mean approximates the class mean.

    Args:
        x:
            Original embeddings with shape [N, D].
        k:
            Number of exemplars to select.
        normalize_for_selection:
            If True, use L2-normalized embeddings for selection.
            If False, use original embeddings for selection.
        return_indices:
            If True, return (selected_indices, selected_embeddings).
            If False (default), return selected_embeddings.

    Returns:
        selected_embeddings:
            Corresponding original raw embeddings of shape [k, D].
            Or (selected_indices, selected_embeddings) if return_indices=True.

    Note:
        Normalization affects selection only.
        The returned embeddings are always original raw embeddings.
    """

    if x.ndim != 2:
        raise ValueError("x must have shape [N, D].")

    if not x.is_floating_point():
        raise TypeError("x must contain floating-point embeddings.")

    n = x.shape[0]

    if n == 0:
        raise ValueError("x must contain at least one embedding.")

    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= N.")

    # 1. Choose the representation used during selection.
    selection_x = (
        F.normalize(x, p=2, dim=1)
        if normalize_for_selection
        else x
    )

    # 2. Calculate the class mean.
    class_mean = selection_x.mean(dim=0)

    # 3. Initialize the running sum of selected embeddings.
    selected_sum = torch.zeros_like(class_mean)

    # Track which candidates are still available.
    available = torch.ones(
        n,
        dtype=torch.bool,
        device=x.device,
    )

    selected_indices = []

    # 4. Iteratively select k exemplars.
    for step in range(1, k + 1):

        # Get indices of candidates not yet selected.
        candidate_indices = available.nonzero(
            as_tuple=True
        )[0]

        candidates = selection_x[candidate_indices]

        # Calculate the hypothetical mean if each candidate
        # were added to the previously selected embeddings.
        candidate_means = (
            selected_sum.unsqueeze(0) + candidates
        ) / step

        # Calculate squared distance from the class mean.
        squared_errors = (
            candidate_means - class_mean
        ).square().sum(dim=1)

        # Select the candidate producing the smallest error.
        best_index = candidate_indices[
            squared_errors.argmin()
        ]

        selected_indices.append(best_index)

        # Update the running sum.
        selected_sum += selection_x[best_index]

        # Prevent selecting this sample again.
        available[best_index] = False

    # 5. Convert selected indices into a tensor.
    selected_indices_tensor = torch.stack(selected_indices) if selected_indices else torch.empty(0, dtype=torch.long, device=x.device)

    # 6. Retrieve the original raw embeddings.
    selected_embeddings = x[selected_indices_tensor]

    if return_indices:
        return selected_indices_tensor, selected_embeddings
    return selected_embeddings


def build_herding_memory(
    z_train: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    center: bool,
    num_classes: int,
    normalize_for_selection: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Build a memory bank by applying Herding independently per class."""

    if z_train.ndim != 2:
        raise ValueError("z_train must have shape [N, D].")

    if labels.ndim != 1:
        raise ValueError("labels must have shape [N].")

    if z_train.shape[0] != labels.shape[0]:
        raise ValueError(
            "z_train and labels must contain the same number of samples."
        )

    if per_class < 1:
        raise ValueError("per_class must be at least 1.")

    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")

    memory_blocks = []
    label_blocks = []

    for class_id in range(num_classes):

        # Extract original embeddings belonging to this class.
        class_embeddings = z_train[labels == class_id]

        if class_embeddings.shape[0] < per_class:
            raise ValueError(
                f"Class {class_id} has only "
                f"{class_embeddings.shape[0]} samples, "
                f"but {per_class} are required."
            )

        # Select exemplars and retrieve their original embeddings.
        selected_embeddings = herding(
            class_embeddings,
            per_class,
            normalize_for_selection=normalize_for_selection,
        )

        memory_blocks.append(selected_embeddings)

        # Assign the corresponding class label to each exemplar.
        label_blocks.append(
            torch.full(
                (selected_embeddings.shape[0],),
                class_id,
                dtype=torch.long,
                device=selected_embeddings.device,
            )
        )

    # Combine selected raw embeddings from all classes.
    memory = torch.cat(memory_blocks, dim=0)

    # Combine their corresponding labels.
    memory_labels = torch.cat(label_blocks, dim=0)

    # Keep the existing global raw embedding mean for Hopfield centering.
    center_offset = z_train.mean(dim=0) if center else None

    return memory, memory_labels, center_offset


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA operations before measuring elapsed time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_herding_memory_build(
    run: benchmark.BaselineRun,
    *,
    per_class: int,
    center: bool,
    num_classes: int,
    normalize_for_selection: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    float,
]:
    """Measure Herding memory construction time."""

    _sync_if_cuda(run.device)

    start = time.perf_counter()

    memory, memory_labels, center_offset = build_herding_memory(
        run.z_train,
        run.splits.train_labels,
        per_class=per_class,
        center=center,
        num_classes=num_classes,
        normalize_for_selection=normalize_for_selection,
    )

    _sync_if_cuda(run.device)

    elapsed = time.perf_counter() - start

    return memory, memory_labels, center_offset, elapsed
