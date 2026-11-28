"""Box-constrained retrieval-aware value step (proposal equations 49–56)."""
import torch


def protected_value_step(values, direction, precision, read_gradient, read_curvature, bound, storage_dtype):
    """Return a storage-representable displacement satisfying both quadratic bounds."""
    def displacement(multiplier):
        step = (direction - multiplier * read_gradient) / (precision + multiplier * read_curvature)[:, None]
        return (values + step).clamp(-bound, bound) - values

    def retrieval_bound(step):
        return (step * read_gradient).sum() + 0.5 * (read_curvature * step.square().sum(1)).sum()

    step = displacement(0.)
    if retrieval_bound(step) > 0:
        if (read_curvature > 0).all():
            minimum = (values - read_gradient / read_curvature[:, None]).clamp(-bound, bound) - values
            if retrieval_bound(minimum) >= 0:
                return torch.zeros_like(values)
        low, high = 0., 1.
        for _ in range(64):
            if retrieval_bound(displacement(high)) <= 0:
                break
            low, high = high, 2 * high
        else:
            return torch.zeros_like(values)
        for _ in range(32):
            mid = (low + high) / 2
            if retrieval_bound(displacement(mid)) <= 0:
                high = mid
            else:
                low = mid
        step = displacement(high)

    # A rounded point on the constraint boundary can fall just outside.
    for fraction in (1., 0.999999, 0.5):
        coordinates = (values + fraction * step).to(storage_dtype).to(values.dtype)
        if coordinates.abs().max() > bound:
            continue
        stored = coordinates - values
        total_bound = 0.5 * (precision * stored.square().sum(1)).sum() - (direction * stored).sum()
        if total_bound <= 0 and retrieval_bound(stored) <= 0:
            return stored
    return torch.zeros_like(values)
