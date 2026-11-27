"""The retrieval term supplied by an experiment to the GASS objective."""
import torch
from torch import nn


class RetrievalLoss(nn.Module):
    """Wrap a loss(targets, retrieved, scale) returning one nonnegative loss per row.

    Optionally supply gradient_curvature(targets, retrieved, scale) returning
    the per-row output gradient and a scalar curvature upper bound. Otherwise
    GASS differentiates the loss and backtracks against the full objective.
    """

    def __init__(self, loss, gradient_curvature=None):
        super().__init__()
        self.loss = loss
        self.gradient_curvature = gradient_curvature

    def forward(self, targets, retrieved, scale):
        rows = self.loss(targets, retrieved, scale)
        if rows.shape != targets.shape[:-1]:
            raise ValueError("The retrieval loss must return one value per row.")
        return rows

    def local_model(self, targets, retrieved, scale):
        if self.gradient_curvature is not None:
            return self.gradient_curvature(targets, retrieved, scale)
        with torch.enable_grad():
            output = retrieved.detach().requires_grad_()
            loss = self(targets, output, scale).sum()
            grad = torch.autograd.grad(loss, output, allow_unused=True)[0] if loss.requires_grad else None
        return (torch.zeros_like(output) if grad is None else grad), 1.
