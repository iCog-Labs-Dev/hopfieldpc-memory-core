"""MNIST autoencoder model and memory comparison arms shared by experiments 4 and 5."""
from __future__ import annotations

import argparse
import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from benchmarks.mnist_baseline import IMAGE_SIZE, MNIST_MEAN, MNIST_STD, corrupt_images
from hopfieldpc import GlobalDynamicHopfieldBank, ema_step, hopfield_retrieve
from hopfieldpc.api import KalmanMomentumStepOutput
from hopfieldpc.kalman import auto_spawning_uncertainty_threshold, john_effective_noise, responsibility_kalman_gain


class RevisedHopfieldBank(GlobalDynamicHopfieldBank):
    """Use the initial variance as the noise unit and accumulate evidence per sample."""

    def resolve_variance_scale(self) -> float:
        return self.resolve_initial_uncertainty()

    def _relative_noises(self, task_score: float | torch.Tensor | None) -> tuple[float, float]:
        residual = self._observation_residual(task_score)
        baseline = self.cfg.structural_base_noise
        return residual, residual if baseline is None else baseline

    def resolve_noises(self, task_score=None) -> tuple[float, float, float]:
        """Return process, observation, and baseline structural noise in variance units."""
        s = self.resolve_variance_scale()
        residual, baseline = self._relative_noises(task_score)
        return s * self.cfg.process_noise, s * residual, s * baseline

    def resolve_structural_base_noise(self, task_score=None) -> float:
        return self._relative_noises(task_score)[1]

    def _gain_stabilizer(self) -> float:
        return self.resolve_variance_scale() * self.cfg.eps_stab

    def _minibatch_step(
        self, current, target, velocity, uncertainty, responsibilities, total_mass,
        *, observation_noise, process_noise,
    ) -> tuple[KalmanMomentumStepOutput, torch.Tensor]:
        gamma = self.cfg.gamma_john
        powered_mass = total_mass if gamma == 1.0 else responsibilities.pow(gamma).sum(dim=0)
        weighted_prior = uncertainty * powered_mass
        gain = responsibility_kalman_gain(weighted_prior, observation_noise, eps=self._gain_stabilizer())
        retention = self.cfg.gamma_mom ** powered_mass.unsqueeze(-1)
        velocity = retention * velocity + (1.0 - retention) * (target - current)
        value = current + gain.unsqueeze(-1) * velocity
        # Q is a per-sample rate, so the whole powered mass contributes to diffusion.
        posterior = ((1.0 - gain) * uncertainty + powered_mass * process_noise).clamp(
            *self.resolve_uncertainty_bounds()
        )
        effective_noise = john_effective_noise(
            responsibilities, observation_noise, gamma_john=gamma, eps=self.cfg.eps_stab,
        )
        return KalmanMomentumStepOutput(value, velocity, posterior, gain), effective_noise.mean()

    def resolve_tau_uncert(self, task_score=None) -> float:
        if self.cfg.tau_uncert is not None:
            return float(self.cfg.tau_uncert)
        residual, baseline = self._relative_noises(task_score)
        # Apply the floor in relative units, then convert the entire threshold to variance.
        relative_threshold = auto_spawning_uncertainty_threshold(
            self.cfg.process_noise, residual, structural_base_noise=baseline,
            lambda_disp=self.cfg.lambda_disp, tau_coal=self.cfg.tau_coal,
        )
        return self.resolve_variance_scale() * relative_threshold


class MnistAutoencoder(nn.Module):
    """Small convolutional MNIST autoencoder."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * (IMAGE_SIZE // 4) ** 2),
            nn.ReLU(),
        )
        self.reconstruct_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder((images - MNIST_MEAN) / MNIST_STD)
        return z, self.decode(z)

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.output_logits(embedding).sigmoid()

    def output_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(embedding).view(
            -1, 32, IMAGE_SIZE // 4, IMAGE_SIZE // 4
        )
        return self.reconstruct_conv(decoded)


class AutoencoderVariant:
    """One autoencoder with direct output or prototype memory."""

    model_class = MnistAutoencoder
    bank_class = RevisedHopfieldBank
    output_dim = IMAGE_SIZE ** 2

    def __init__(
        self,
        name: str,
        cfg: object,
        seed: int,
        device: torch.device,
    ) -> None:
        self.seed = seed
        self.name = name
        torch.manual_seed(seed)
        self.net = self.model_class(embed_dim=cfg.memory.dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.bank: GlobalDynamicHopfieldBank | None = None
        if name != "Direct":
            mode = "fixed_capacity" if name in ("Fixed", "EMA") else "dynamic_adaptive"
            memory_cfg = replace(cfg.memory, dim=self.output_dim).to_global_dynamic_config(
                ablation_mode=mode
            )
            self.bank = self.bank_class(
                memory_cfg,
                online_update_mode=cfg.memory.online_update_mode,
            ).to(device)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        z = self.net.encoder((images - MNIST_MEAN) / MNIST_STD)
        return z

    @torch.no_grad()
    def training_embeddings(self, images: torch.Tensor, cfg: object, device: torch.device) -> torch.Tensor:
        features = []
        self.net.eval()
        for start in range(0, len(images), cfg.eval_batch_size):
            batch = images[start : start + cfg.eval_batch_size].to(device)
            features.append(self.net.output_logits(self.encode(batch)).flatten(1))
        return torch.cat(features, dim=0)

    def initialize_bank(self, images: torch.Tensor, cfg: object, device: torch.device) -> None:
        if self.bank is not None:
            self.bank.initialize_from_embeddings(self.training_embeddings(images, cfg, device), seed=self.seed)

    def finish_bank_epoch(self, images, cfg, device, epoch, score):
        """Initialize after warm-up and run the fixed or dynamic bank's epoch update."""
        if self.bank is None:
            return {}
        if not self.bank.ready:
            self.initialize_bank(images, cfg, device)
            return {}
        if self.name == "EMA":
            return {}
        features = self.training_embeddings(images, cfg, device)
        return self.bank.adapt_structure(
            features, epoch=epoch, task_score=score,
            allow_spawn=self.name == "Dynamic" and epoch < cfg.epochs,
        )

    @torch.no_grad()
    def prototype_images(self, slots: tuple[int, ...] | None = None) -> dict[int, Image.Image]:
        if self.bank is None or not self.bank.ready:
            return {}
        if slots is None:
            slots = tuple(self.bank.mask.nonzero().flatten().tolist())
        if not slots:
            return {}
        return vector_images(self.bank.values[list(slots)], slots)

    @torch.no_grad()
    def _read_output(self, logits: torch.Tensor) -> torch.Tensor:
        return self.retrieve_logits(logits).sigmoid()

    def retrieve_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.bank is None or not self.bank.ready:
            return logits
        beta, _ = self.bank.resolve_betas()
        retrieved, _ = hopfield_retrieve(logits.flatten(1), self.bank.values[self.bank.mask], beta=beta)
        return retrieved.reshape_as(logits)

    def train_batch_metrics(self, images: torch.Tensor, device: torch.device) -> tuple[float, float]:
        """Train the direct reconstruction and write detached pre-update decoder logits."""
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        logits = self.net.output_logits(self.encode(images))
        direct_image = logits.sigmoid()
        loss = F.mse_loss(direct_image, images)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            score = digit_overlap(direct_image.detach(), images).item()
            if self.bank is None or not self.bank.ready:
                return loss.item(), score
            candidates = logits.detach()
            output_mse = F.mse_loss(self._read_output(candidates), images)
            if self.name == "EMA":
                self._ema_write(candidates.flatten(1))
            else:
                self.bank.online_update(candidates.flatten(1), task_score=score)
            return output_mse.item(), score

    @torch.no_grad()
    def _ema_write(self, logits: torch.Tensor) -> None:
        """Apply ema.py's nearest-prototype minibatch update to a single unlabeled pool."""
        bank = self.bank
        slots = bank.mask.nonzero().flatten()
        closest = torch.cdist(logits, bank.values[slots]).argmin(dim=1)
        counts = torch.bincount(closest, minlength=len(slots))
        sums = logits.new_zeros((len(slots), bank.dim))
        sums.index_add_(0, closest, logits)
        assigned = counts > 0
        updated_slots = slots[assigned]
        means = sums[assigned] / counts[assigned, None]
        bank.values[updated_slots] = ema_step(bank.values[updated_slots], means)
        bank.version += 1

    @torch.no_grad()
    def generate(self, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        self.net.eval()
        images = images.to(device)
        reconstruction, z = self.reconstruct(images)
        return reconstruction.cpu(), z.cpu()

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(images)
        return self._read_output(self.net.output_logits(z)), z


def digit_overlap(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-image soft Dice score for grayscale digit strokes in [0, 1]."""
    p, t = prediction.flatten(1), target.flatten(1)
    mass = p.square().sum(dim=1) + t.square().sum(dim=1)
    overlap = 2 * (p * t).sum(dim=1)
    score = overlap / mass.clamp_min(torch.finfo(p.dtype).tiny)
    return torch.where(mass > 0, score, 1.0).clamp(0, 1).mean()


def build_sample_corruption(
    images: torch.Tensor,
    kind: str,
    level: float,
    seed: int,
) -> torch.Tensor:
    """Apply a deterministic corruption to a sample batch."""
    generator = torch.Generator().manual_seed(seed)
    return corrupt_images(images, kind, level, generator)


def vector_images(values: torch.Tensor, slots: tuple[int, ...]) -> dict[int, Image.Image]:
    """Lay out raw coordinates row by row with a fixed signed brightness scale."""
    width = math.isqrt(values.shape[1] - 1) + 1
    height = (values.shape[1] + width - 1) // width
    values = values.detach()
    signed = values / (1 + values.abs())
    pixels = ((signed + 1) * 127.5).round().to(torch.uint8)
    pixels = F.pad(pixels, (0, width * height - values.shape[1]), value=128)
    grids = pixels.reshape(-1, height, width).cpu().numpy()
    return {slot: Image.fromarray(grid) for slot, grid in zip(slots, grids)}


def argument_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--quick", action="store_true", help="Use 2,000 training images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bank-init", choices=("warm-up",), default="warm-up",
                        help="Initialize prototypes after the first epoch")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--noise", choices=("gaussian", "salt_pepper", "occlusion"), default="gaussian")
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-dir", default="data")
    return parser
