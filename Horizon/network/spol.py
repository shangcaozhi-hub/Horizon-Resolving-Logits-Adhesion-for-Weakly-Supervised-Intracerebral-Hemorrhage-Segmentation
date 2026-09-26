import math
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from einops import rearrange
from backbone import resnet
from network.GMM import GaussianMixture


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def dilated_skull_mask(skull_mask, reference):
    """Resize the skull mask to CAM resolution and return its inside region."""
    batch = reference.shape[0]
    if skull_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    mask = skull_mask.detach().to(device=reference.device, dtype=torch.float32)
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[:2] != (batch, 1):
        raise ValueError("skull_mask must be [B,H,W], [B,1,H,W] or [B,1,1,H,W]")
    mask = F.interpolate(mask, size=reference.shape[-2:], mode="nearest")
    return F.max_pool2d(mask.float(), kernel_size=7, stride=1, padding=3) > 0


def _positive_ints(values, name):
    values = (values,) if isinstance(values, int) else tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(dict.fromkeys(values))


def _enqueue(queue, count, samples):
    """Prepend real observations; never fit unfilled zero entries."""
    samples = samples.detach().to(queue)
    if samples.ndim != 2 or samples.shape[1] != queue.shape[1]:
        raise ValueError("Bank samples must have shape [N, feat_dim]")
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Non-finite prototype bank features")
    n = min(samples.shape[0], queue.shape[0])
    if n:
        queue.copy_(torch.cat((samples[:n], queue[:-n].clone()), dim=0))
        count.fill_(min(int(count.item()) + n, queue.shape[0]))
    return n


class clusterSupportBank(nn.Module):
    """Class-conditional balanced OT over image prototypes; class 0 is background.

    Uses soft Sinkhorn transport directly, without the old Gumbel resampling.
    Queues, valid lengths and EMA prototypes are all checkpoint buffers.
    """
    def __init__(self, feat_dim, bank_size, num_prototypes, gamma=0.99,
                 epsilon=0.05, sinkhorn_iterations=50):
        super().__init__()
        _positive_ints((feat_dim, bank_size, num_prototypes, sinkhorn_iterations), "OT dimensions")
        if not 0 <= gamma < 1 or epsilon <= 0:
            raise ValueError("Require 0 <= gamma < 1 and epsilon > 0")
        self.feat_dim, self.bank_size = feat_dim, bank_size
        self.num_prototypes, self.gamma = num_prototypes, gamma
        self.epsilon, self.sinkhorn_iterations = epsilon, sinkhorn_iterations
        for kind in ("fore", "back"):
            self.register_buffer(f"{kind}_support_bank", torch.zeros(bank_size, feat_dim))
            self.register_buffer(f"{kind}_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("context_prototypes", F.normalize(torch.randn(2, num_prototypes, feat_dim), dim=-1))
        self.register_buffer("initialized", torch.zeros(2, dtype=torch.bool))
        self.register_buffer("num_updates", torch.zeros(2, dtype=torch.long))

    @torch.no_grad()
    def update_bank(self, fore_inpro, back_inpro):
        _enqueue(self.fore_support_bank, self.fore_count, fore_inpro)
        _enqueue(self.back_support_bank, self.back_count, back_inpro)

    def _transport(self, logits):
        # Log-domain balancing avoids exp overflow/underflow.
        log_q = logits.float().t() / self.epsilon
        k, n = log_q.shape
        log_q = log_q - torch.logsumexp(log_q.flatten(), dim=0)
        for _ in range(self.sinkhorn_iterations):
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) - math.log(k)
            log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) - math.log(n)
        return (log_q + math.log(n)).exp().t()

    @torch.no_grad()
    def update_context_protos(self):
        for cls, kind in enumerate(("back", "fore")):
            count = int(getattr(self, f"{kind}_count").item())
            if count < self.num_prototypes:
                continue
            features = l2_normalize(getattr(self, f"{kind}_support_bank")[:count])
            old = self.context_prototypes[cls]
            if not self.initialized[cls]:
                indices = torch.linspace(0, count - 1, self.num_prototypes, device=features.device).long()
                old.copy_(features[indices])
                self.initialized[cls] = True
            transport = self._transport(features @ old.t())
            centers = l2_normalize(transport.t() @ features)
            updated = l2_normalize(self.gamma * old + (1 - self.gamma) * centers)
            if not torch.isfinite(updated).all():
                raise FloatingPointError("Non-finite OT prototypes")
            old.copy_(updated)
            self.num_updates[cls] += 1


'''class PrototypeSupportBank(nn.Module):
    """Independent GMM bank, with equal class component counts by default.

    foreground_components is an explicit override for loading legacy checkpoints.
    Update both classes whenever bank_size new paired observations have arrived.
    update_interval is retained for API compatibility; complete windows trigger updates.
    Each update uses only valid entries currently retained in each class queue.
    Each trigger refits EM from K-means, matches means with Hungarian assignment,
    then applies gamma to normalized sufficient statistics, not parameters.
    history_mass/history_mean/history_m2 jointly represent those statistics in
    centered form. Each retained queue snapshot is one minibatch; full covariance
    and diagonal covariance use the same recursion. Unsupported components may
    lose mixture weight naturally; no component revival or step cap is applied.
    """
    def __init__(self, feat_dim, bank_size, num_prototypes, num_components=3, *,
                 update_interval=0, init_iterations=100, foreground_components=None,
                 covariance_type="full", momentum_gamma=0.9, history_decay=None,
                 online_statistics=True):
        super().__init__()
        _positive_ints((feat_dim, bank_size, num_prototypes, num_components, init_iterations), "GMM dimensions")
        foreground_components = num_components if foreground_components is None else foreground_components
        _positive_ints((foreground_components,), "foreground_components")
        if bank_size < max(2, num_components, foreground_components):
            raise ValueError("bank_size must cover both classes' component counts and at least two samples")
        if not isinstance(update_interval, int) or update_interval < 0:
            raise ValueError("update_interval must be a nonnegative integer")
        self.feat_dim, self.bank_size = feat_dim, bank_size
        self.num_classes, self.num_prototypes = 2, num_prototypes
        self.num_components = num_components
        self.foreground_components = foreground_components
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be diag or full")
        self.covariance_type = covariance_type
        if not 0 <= momentum_gamma < 1:
            raise ValueError("momentum_gamma must be in [0, 1)")
        self.online_statistics = online_statistics
        if online_statistics:
            # Kept only so old checkpoints can still be loaded strictly; it is
            # deliberately not used by the online EM update below.
            self.register_buffer("history_decay", torch.tensor(
                1.0 if history_decay is None else history_decay, dtype=torch.float64))
            self.register_buffer("decay_reference", torch.tensor(1000., dtype=torch.float64))
        self.fit_diagnostics = {}
        self.update_interval, self.init_iterations = update_interval, init_iterations
        self.momentum_gamma = float(momentum_gamma)
        for i in range(self.num_classes):
            k = foreground_components if i == 1 else num_components
            self.register_buffer(f"queue{i}", torch.zeros(bank_size, feat_dim))
            self.register_buffer(f"queue_ptr{i}", torch.zeros(1, dtype=torch.long))
            self.register_buffer(f"queue_count{i}", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"fit_steps{i}", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"fit_count{i}", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"gmm_weights_{i}", torch.full((k,), 1 / k))
            self.register_buffer(f"gmm_means_{i}", torch.zeros(k, feat_dim))
            covariance = torch.ones(k, feat_dim) if covariance_type == "diag" else torch.eye(feat_dim).repeat(k, 1, 1)
            self.register_buffer(f"gmm_covariances_{i}", covariance)
            if online_statistics:
                self.register_buffer(f"history_mass{i}", torch.zeros(k, dtype=torch.float64))
                self.register_buffer(f"history_mean{i}", torch.zeros(k, feat_dim, dtype=torch.float64))
                self.register_buffer(f"history_m2{i}", torch.zeros_like(covariance, dtype=torch.float64))
                self.register_buffer(f"history_initialized{i}", torch.zeros((), dtype=torch.bool))
                self.register_buffer(f"pending_mass{i}", torch.zeros(k, dtype=torch.float64))
                self.register_buffer(f"pending_mean{i}", torch.zeros(k, feat_dim, dtype=torch.float64))
                self.register_buffer(f"pending_m2{i}", torch.zeros_like(covariance, dtype=torch.float64))
                self.register_buffer(f"processed_count{i}", torch.zeros((), dtype=torch.long))

    @staticmethod
    @torch.no_grad()
    def select_samples(
            pseudo_label,
            hie_fea,
            num_components=1,
    ):
        """Sample half the batch foreground and equally many background pixels."""
        _positive_ints((num_components,), "num_components")

        features = hie_fea.detach().reshape(
            -1, hie_fea.shape[-1]
        ).float()
        labels = pseudo_label.reshape(-1).to(features.device)

        if features.shape[0] != labels.numel():
            raise ValueError(
                "Pseudo labels and features have different spatial sizes"
            )

        foreground_indices = torch.where(labels == 1)[0]
        background_indices = torch.where(labels == 0)[0]

        # 没有可靠前景时，两类都不写入
        n = min(foreground_indices.numel() // 2, background_indices.numel())
        if n == 0:
            empty = features.new_empty((0, features.shape[-1]))
            return [empty, empty]

        # 整个 batch 统一随机采样前景

        selected_foreground = foreground_indices[
            torch.randperm(
                foreground_indices.numel(),
                device=foreground_indices.device,
            )[:n]]

        # 背景数量跟随实际采到的前景数量

        selected_background = background_indices[
            torch.randperm(
                background_indices.numel(),
                device=background_indices.device,
            )[:n]]

        return [
            features[selected_background],
            features[selected_foreground],
        ]

    @torch.no_grad()
    def update_bank(self, pseudo_label, hie_fea):
        self.update_samples(self.select_samples(pseudo_label, hie_fea, self.num_components))

    @torch.no_grad()
    def update_samples(self, samples):
        if not self.online_statistics:
            raise RuntimeError("Online GMM requires historical statistics")
        if len(samples) != 2 or len(samples[0]) != len(samples[1]):
            raise ValueError("Expected equal background/foreground sample counts")

        for i, values in enumerate(samples):
            _enqueue(
                getattr(self, f"queue{i}"),
                getattr(self, f"queue_count{i}"),
                values,
            )
            getattr(self, f"queue_ptr{i}").add_(len(values))
            getattr(self, f"fit_steps{i}").add_(1)

        self.update_statistics()

    @staticmethod
    def _regularize(covariance):
        if covariance.ndim == 2:
            if not torch.isfinite(covariance).all():
                raise FloatingPointError("Non-finite diagonal GMM variance")
            return covariance.clamp_min(1e-4)
        covariance = (covariance + covariance.transpose(-1, -2)) * 0.5
        eye = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        scale = covariance.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1.0)
        for factor in (1e-5, 1e-4, 1e-3, 1e-2):
            candidate = covariance + (factor * scale)[..., None, None] * eye
            _, info = torch.linalg.cholesky_ex(candidate)
            if torch.isfinite(candidate).all() and (info == 0).all():
                return candidate
        raise FloatingPointError("GMM covariance is not positive definite")

    @staticmethod
    def _merge_statistics(mass_a, mean_a, m2_a, mass_b, mean_b, m2_b):
        """Merge weighted centered moments without subtracting large squared sums."""
        total = mass_a + mass_b
        safe_total = total.clamp_min(torch.finfo(total.dtype).tiny)
        delta = mean_b - mean_a
        mean = mean_a + (mass_b / safe_total)[:, None] * delta
        between = delta.square() if m2_a.ndim == 2 else delta.unsqueeze(-1) * delta.unsqueeze(-2)
        shape = (total.numel(),) + (1,) * (m2_a.ndim - 1)
        m2 = m2_a + m2_b + (mass_a * (mass_b / safe_total)).reshape(shape) * between
        return total, mean, m2

    def _statistics_buffers(self, cls, destination):
        """mass: component mass; mean: weighted mean; m2: centered second moment.

        Pending uses raw queue soft counts. History mass sums to one; history m2
        has the same normalization, so m2 / mass recovers component covariance.
        """
        return tuple(getattr(self, f"{destination}_{field}{cls}")
                     for field in ("mass", "mean", "m2"))


    @torch.no_grad()
    def _publish_statistics(self, cls):
        mass, mean, m2 = self._statistics_buffers(cls, "history")
        means = getattr(self, f"gmm_means_{cls}")
        covs = getattr(self, f"gmm_covariances_{cls}")
        supported = mass > 0
        if not supported.any() or not all(torch.isfinite(x).all() for x in (mass, mean, m2)):
            raise FloatingPointError("Invalid historical GMM statistics")
        shape = (int(supported.sum()),) + (1,) * (m2.ndim - 1)
        covariance = m2[supported] / mass[supported].reshape(shape)
        means[supported] = mean[supported].to(means)
        # Regularization affects published covariance, not accumulated raw moments.
        covs[supported] = self._regularize(covariance.to(covs))
        getattr(self, f"gmm_weights_{cls}").copy_((mass / mass.sum()).to(means))

    @torch.no_grad()
    def _fit_queue_statistics(self, cls, match):
        """Fresh K-means/EM, followed by old-to-new minimum-distance assignment."""
        count = int(getattr(self, f"queue_count{cls}"))
        queue = getattr(self, f"queue{cls}")[:count]
        old_means = getattr(self, f"gmm_means_{cls}")
        gmm = GaussianMixture(old_means.shape[0], self.feat_dim,
                              covariance_type=self.covariance_type,
                              init_params="kmeans").to(queue.device)
        gmm.fit(queue, n_iter=self.init_iterations)
        mass = gmm.pi.detach().reshape(-1).double()
        mean = gmm.mu.detach().squeeze(0).double()
        covariance = gmm.var.detach().squeeze(0).double()
        if not all(torch.isfinite(x).all() for x in (mass, mean, covariance)):
            raise FloatingPointError("Non-finite refitted GMM parameters")
        permutation = torch.arange(mass.numel(), device=queue.device)
        costs = mass.new_zeros(mass.numel())
        if match:
            distance = (old_means.double()[:, None] - mean[None]).square().sum(-1)
            rows, cols = linear_sum_assignment(distance.cpu().numpy())
            rows = torch.as_tensor(rows, device=queue.device)
            permutation[rows] = torch.as_tensor(cols, device=queue.device)
            costs = distance[torch.arange(mass.numel(), device=queue.device), permutation]
        mass = mass[permutation] / mass.sum() * count
        mean, covariance = mean[permutation], covariance[permutation]
        m2 = covariance * mass.reshape((-1,) + (1,) * (covariance.ndim - 1))
        diagnostics = dict(em_iterations=gmm.n_iter_, em_converged=gmm.converged_,
                           matched_components=permutation.tolist(), matching_costs=costs.tolist())
        return (mass, mean, m2), diagnostics

    @torch.no_grad()
    def _initialize_class(self, cls):
        """Seed normalized history directly from a fresh fitted mixture."""
        statistics, diagnostics = self._fit_queue_statistics(cls, match=False)
        mass, mean, m2 = statistics
        count = int(getattr(self, f"queue_count{cls}"))
        history = self._statistics_buffers(cls, "history")
        total = mass.sum()
        for buffer, value in zip(history, (mass / total, mean, m2 / total)):
            buffer.copy_(value)
        getattr(self, f"history_initialized{cls}").fill_(True)
        getattr(self, f"processed_count{cls}").fill_(count)
        self.fit_diagnostics[cls] = dict(mode="initial_em", statistics_source="fitted_queue",
                                        num_samples=count, gamma=self.momentum_gamma,
                                        current_mass=mass.tolist(), previous_mass=[0.] * mass.numel(),
                                        history_mass=history[0].tolist(), component_steps=[1.] * mass.numel(),
                                        **diagnostics)

    @torch.no_grad()
    def _commit_class(self, cls, history, pending):
        """Normalized sufficient-statistic EMA with component-specific steps.

        Conditional moments use the responsibility-weighted EMA step without
        clipping. _merge_statistics includes the between-mean covariance term.
        """
        current_mass, current_mean, current_m2 = pending
        if (not all(torch.isfinite(value).all() for value in pending)
                or (current_mass < 0).any()):
            raise FloatingPointError("Invalid online EM minibatch statistics")
        normalizer = current_mass.sum()
        if normalizer <= 0:
            return
        reseeded = not bool(getattr(self, f"history_initialized{cls}").item())
        if reseeded:
            # Old alpha/gamma checkpoints have stale history means/moments.
            # Start a coherent recursion from their published mixture instead.
            weights = getattr(self, f"gmm_weights_{cls}").double()
            means = getattr(self, f"gmm_means_{cls}").double()
            covariance = getattr(self, f"gmm_covariances_{cls}").double()
            mass = weights / weights.sum()
            shape = (-1,) + (1,) * (covariance.ndim - 1)
            for buffer, value in zip(history, (mass, means, covariance * mass.reshape(shape))):
                buffer.copy_(value)
            getattr(self, f"history_initialized{cls}").fill_(True)
        previous_mass = history[0].clone()
        gamma = self.momentum_gamma
        proportion = current_mass / normalizer
        new_mass = gamma * previous_mass + (1 - gamma) * proportion
        tiny = torch.finfo(current_mass.dtype).tiny
        step = (1 - gamma) * proportion / new_mass.clamp_min(tiny)
        shape = (-1,) + (1,) * (current_m2.ndim - 1)
        old_covariance = history[2] / previous_mass.clamp_min(tiny).reshape(shape)
        current_covariance = current_m2 / current_mass.clamp_min(tiny).reshape(shape)
        # Merge conditional distributions, including the between-mean correction.
        _, mean, covariance = self._merge_statistics(
            1 - step, history[1], (1 - step).reshape(shape) * old_covariance,
            step, current_mean, step.reshape(shape) * current_covariance)
        history[0].copy_(new_mass)
        history[1].copy_(mean)
        history[2].copy_(covariance * new_mass.reshape(shape))
        self._publish_statistics(cls)
        count = int(getattr(self, f"queue_count{cls}"))
        getattr(self, f"processed_count{cls}").add_(count)
        self.fit_diagnostics[cls] = dict(mode="matched_em", statistics_source="fitted_queue",
                                        num_samples=count, em_iterations=0,
                                        gamma=gamma, current_coefficient=1 - gamma,
                                        component_steps=step.tolist(),
                                        history_mass_normalized=True, history_reseeded=reseeded,
                                        previous_mass=previous_mass.tolist(),
                                        current_mass=current_mass.tolist(),
                                        history_mass=history[0].tolist())

    @torch.no_grad()
    def update_statistics(self):
        """On a foreground queue trigger, update both classes together.

        First update: K-means/EM -> queue moments -> initial GMM.
        Later updates: fresh EM -> Hungarian matching -> sufficient-statistic EMA.
        Arrival counters schedule updates; only queued features contribute mass.
        """
        if not self.online_statistics:
            raise RuntimeError("Legacy bank has no online history; initialize a new bank for training")
        # Background arrivals alone never change published GMM parameters.
        if self.queue_ptr1.item() < self.bank_size:
            return
        initializing = all(getattr(self, f"fit_count{i}").item() == 0 for i in range(self.num_classes))
        if initializing and any(int(getattr(self, f"queue_count{i}")) < max(2, getattr(self, f"gmm_weights_{i}").numel())
                                for i in range(self.num_classes)):
            return
        for i in range(self.num_classes):
            history = self._statistics_buffers(i, "history")
            pending = self._statistics_buffers(i, "pending")
            if initializing:
                self._initialize_class(i)
            else:
                # Both classes use the same bounded-queue rule. Discarded features
                # never contribute to this update's mass, mean or covariance.
                for buffer in pending:
                    buffer.zero_()
                statistics, diagnostics = self._fit_queue_statistics(i, match=True)
                for buffer, value in zip(pending, statistics):
                    buffer.copy_(value)
                self._commit_class(i, history, pending)
                self.fit_diagnostics[i].update(diagnostics)
            if initializing:
                # The initialization window becomes the previous window.
                self._publish_statistics(i)
            for buffer in pending:
                buffer.zero_()
            getattr(self, f"queue_ptr{i}").zero_()
            getattr(self, f"fit_steps{i}").zero_()
            getattr(self, f"fit_count{i}").add_(1)

    @torch.no_grad()
    def sample_prototypes(self):
        """Sample mixture components by weight, then x = mean + L @ noise.

        Full covariance preserves channel correlations. Factor each component
        once before gathering sampled factors; do not factor duplicate matrices
        for every sampled prototype. Diagonal banks remain supported explicitly.
        """
        if any(getattr(self, f"fit_count{i}").item() == 0 for i in range(self.num_classes)):
            raise RuntimeError("Prototype bank is not fitted for every class yet")
        prototypes = []
        for i in range(self.num_classes):
            weights = getattr(self, f"gmm_weights_{i}")
            means = getattr(self, f"gmm_means_{i}")
            covariance = getattr(self, f"gmm_covariances_{i}")
            if (not torch.isfinite(weights).all() or (weights < 0).any()
                    or weights.sum() <= 0 or not torch.isfinite(means).all()
                    or not torch.isfinite(covariance).all()):
                raise FloatingPointError("Invalid GMM parameters before prototype sampling")
            indices = torch.multinomial(weights, self.num_prototypes, replacement=True)
            noise = torch.randn(self.num_prototypes, self.feat_dim, device=means.device, dtype=means.dtype)
            if self.covariance_type == "diag":
                if (covariance <= 0).any():
                    raise FloatingPointError("GMM diagonal variances must be positive")
                perturbation = covariance[indices].sqrt() * noise
            else:
                factors = torch.linalg.cholesky(covariance)
                perturbation = torch.bmm(factors[indices], noise.unsqueeze(-1)).squeeze(-1)
            samples = means[indices] + perturbation
            if not torch.isfinite(samples).all():
                raise FloatingPointError("Non-finite sampled GMM prototypes")
            prototypes.append(samples)
        return torch.stack(prototypes)'''


class PrototypeSupportBank(nn.Module):
    """First-window EM initialization, then responsibility-based online GMM."""
    def __init__(self, feat_dim, bank_size, num_prototypes, num_components=5, *,
                 init_iterations=50, foreground_components=None, covariance_type="full",
                 momentum_gamma=0.9, covariance_shrinkage=0.25):
        super().__init__()
        _positive_ints((feat_dim, bank_size, num_prototypes, num_components, init_iterations), "GMM dimensions")
        foreground_components = num_components if foreground_components is None else foreground_components
        _positive_ints((foreground_components,), "foreground_components")
        if bank_size < max(2, num_components, foreground_components): raise ValueError("bank_size too small")
        if covariance_type not in ("diag", "full"): raise ValueError("covariance_type must be diag or full")
        if not 0 <= momentum_gamma < 1: raise ValueError("momentum_gamma must be in [0,1)")
        if not 0 <= covariance_shrinkage <= 1: raise ValueError("covariance_shrinkage must be in [0,1]")

        self.feat_dim, self.bank_size = feat_dim, bank_size
        self.num_classes, self.num_prototypes = 2, num_prototypes
        self.num_components, self.foreground_components = num_components, foreground_components
        self.covariance_type, self.momentum_gamma = covariance_type, float(momentum_gamma)
        self.covariance_shrinkage, self.init_iterations = float(covariance_shrinkage), init_iterations
        self.online_statistics, self.fit_diagnostics = True, {}

        for i in range(2):
            k = foreground_components if i == 1 else num_components
            self.register_buffer(f"queue{i}", torch.zeros(bank_size, feat_dim))
            self.register_buffer(f"queue_ptr{i}", torch.zeros(1, dtype=torch.long))
            self.register_buffer(f"queue_count{i}", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"fit_count{i}", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"gmm_weights_{i}", torch.full((k,), 1 / k))
            self.register_buffer(f"gmm_means_{i}", torch.zeros(k, feat_dim))
            cov = torch.ones(k, feat_dim) if covariance_type == "diag" else torch.eye(feat_dim).repeat(k, 1, 1)
            self.register_buffer(f"gmm_covariances_{i}", cov)
            self.register_buffer(f"history_mass{i}", torch.zeros(k, dtype=torch.float64))
            self.register_buffer(f"history_mean{i}", torch.zeros(k, feat_dim, dtype=torch.float64))
            self.register_buffer(f"history_m2{i}", torch.zeros_like(cov, dtype=torch.float64))
            self.register_buffer(f"pending_mass{i}", torch.zeros(k, dtype=torch.float64))
            self.register_buffer(f"pending_mean{i}", torch.zeros(k, feat_dim, dtype=torch.float64))
            self.register_buffer(f"pending_m2{i}", torch.zeros_like(cov, dtype=torch.float64))
            self.register_buffer(f"processed_count{i}", torch.zeros((), dtype=torch.long))

    @staticmethod
    @torch.no_grad()
    def select_samples(pseudo_label, hie_fea, pixels_per_background=3, background_trials=3):
        B, N, D = hie_fea.shape
        H, W = pseudo_label.shape[-2:]
        if H * W != N: raise ValueError("Pseudo labels and features have different spatial sizes")
        features = hie_fea.detach().float().reshape(B, H, W, D)
        labels = pseudo_label.detach().reshape(B, H, W).to(features.device)
        foreground_samples, background_samples = [], []

        def sample_background(feat, label):
            coords = torch.nonzero(label == 0, as_tuple=False)
            if coords.numel() == 0: return None
            for _ in range(background_trials):
                y, x = coords[torch.randint(coords.shape[0], (1,), device=coords.device)].squeeze(0)
                y, x = int(y), int(x)
                local_feat = feat[max(0,y-1):min(H,y+2), max(0,x-1):min(W,x+2)]
                local_label = label[max(0,y-1):min(H,y+2), max(0,x-1):min(W,x+2)]
                candidates = local_feat[local_label == 0]
                if candidates.shape[0] >= pixels_per_background:
                    idx = torch.randperm(candidates.shape[0], device=feat.device)[:pixels_per_background]
                    return F.normalize(candidates[idx].mean(0), p=2, dim=0)
            return None

        for b in range(B):
            fg_coords = torch.nonzero(labels[b] == 1, as_tuple=False)
            if fg_coords.numel() == 0: continue
            y, x = fg_coords[torch.randint(fg_coords.shape[0], (1,), device=fg_coords.device)].squeeze(0)
            fg, bg = features[b, y, x], sample_background(features[b], labels[b])
            if bg is None: continue
            foreground_samples.append(fg); background_samples.append(bg)

        if not foreground_samples:
            empty = hie_fea.new_empty((0, D))
            return [empty, empty]
        return [torch.stack(background_samples), torch.stack(foreground_samples)]

    @torch.no_grad()
    def update_samples(self, samples):
        if len(samples) != 2 or len(samples[0]) != len(samples[1]): raise ValueError("Expected equal BG/FG samples")
        for i, values in enumerate(samples):
            stored = _enqueue(getattr(self, f"queue{i}"), getattr(self, f"queue_count{i}"), values)
            getattr(self, f"queue_ptr{i}").add_(stored)
        self.update_statistics()

    @staticmethod
    def _regularize(cov):
        if cov.ndim == 2:
            if not torch.isfinite(cov).all(): raise FloatingPointError("Non-finite diagonal variance")
            return cov.clamp_min(1e-4)
        cov = (cov + cov.transpose(-1, -2)) * 0.5
        eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        scale = cov.diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(1.0)
        for factor in (1e-5, 1e-4, 1e-3, 1e-2):
            candidate = cov + (factor * scale)[..., None, None] * eye
            _, info = torch.linalg.cholesky_ex(candidate)
            if torch.isfinite(candidate).all() and (info == 0).all(): return candidate
        raise FloatingPointError("GMM covariance is not positive definite")

    def _shrink_covariance(self, cov):
        if cov.ndim == 2 or self.covariance_shrinkage == 0: return cov
        diag = torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1))
        return (1 - self.covariance_shrinkage) * cov + self.covariance_shrinkage * diag

    @staticmethod
    def _merge_statistics(ma, ua, m2a, mb, ub, m2b):
        total = ma + mb
        safe = total.clamp_min(torch.finfo(total.dtype).tiny)
        delta = ub - ua
        mean = ua + (mb / safe)[:, None] * delta
        between = delta.square() if m2a.ndim == 2 else delta.unsqueeze(-1) * delta.unsqueeze(-2)
        shape = (total.numel(),) + (1,) * (m2a.ndim - 1)
        m2 = m2a + m2b + (ma * mb / safe).reshape(shape) * between
        return total, mean, m2

    def _statistics_buffers(self, cls, dst):
        return tuple(getattr(self, f"{dst}_{x}{cls}") for x in ("mass", "mean", "m2"))

    def _weighted_statistics(self, x, resp):
        x, resp = x.double(), resp.double()
        mass = resp.sum(0)
        mean = resp.T @ x / mass.clamp_min(torch.finfo(mass.dtype).tiny)[:, None]
        centered = x[:, None] - mean
        m2 = ((resp[..., None] * centered.square()).sum(0) if self.covariance_type == "diag"
              else torch.einsum("nk,nkd,nke->kde", resp, centered, centered))
        return mass, mean, m2

    @torch.no_grad()
    def _accumulate_statistics(self, cls, values, dst):
        weights, means = getattr(self, f"gmm_weights_{cls}"), getattr(self, f"gmm_means_{cls}")
        cov = getattr(self, f"gmm_covariances_{cls}")
        target = self._statistics_buffers(cls, dst)

        if self.covariance_type == "diag":
            logdet = cov.log().sum(-1)
        else:
            chol = torch.linalg.cholesky(cov)
            logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        logw = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()[None]

        for chunk in values.split(512):
            if not len(chunk): continue
            diff = chunk[:, None] - means
            if self.covariance_type == "diag":
                distance = (diff.square() / cov).sum(-1)
            else:
                solved = torch.linalg.solve_triangular(chol, diff.permute(1, 2, 0), upper=False)
                distance = solved.square().sum(1).T
            resp = (logw - 0.5 * (distance + logdet[None])).softmax(-1)
            if not torch.isfinite(resp).all(): raise FloatingPointError("Non-finite GMM responsibilities")
            merged = self._merge_statistics(*target, *self._weighted_statistics(chunk, resp))
            for buf, value in zip(target, merged): buf.copy_(value)

    @torch.no_grad()
    def _publish_statistics(self, cls):
        mass, mean, m2 = self._statistics_buffers(cls, "history")
        means, covs = getattr(self, f"gmm_means_{cls}"), getattr(self, f"gmm_covariances_{cls}")
        supported = mass > 0
        shape = (int(supported.sum()),) + (1,) * (m2.ndim - 1)
        cov = m2[supported] / mass[supported].reshape(shape)
        cov = self._regularize(self._shrink_covariance(cov.to(covs)))
        means[supported] = mean[supported].to(means); covs[supported] = cov
        getattr(self, f"gmm_weights_{cls}").copy_((mass / mass.sum()).to(means))

    @torch.no_grad()
    def _initialize_class(self, cls):
        count = int(getattr(self, f"queue_count{cls}"))
        queue = getattr(self, f"queue{cls}")[:count]
        weights = getattr(self, f"gmm_weights_{cls}")
        gmm = GaussianMixture(weights.numel(), self.feat_dim, covariance_type=self.covariance_type,
                              init_params="kmeans").to(queue.device)
        gmm.fit(queue, n_iter=self.init_iterations)

        mass = gmm.pi.detach().reshape_as(weights).double()
        mass = mass / mass.sum()
        mean = gmm.mu.detach().squeeze(0).double()
        cov = gmm.var.detach().reshape_as(getattr(self, f"gmm_covariances_{cls}")).double()
        shape = (-1,) + (1,) * (cov.ndim - 1)
        history = self._statistics_buffers(cls, "history")
        for buf, value in zip(history, (mass, mean, cov * mass.reshape(shape))): buf.copy_(value)

        getattr(self, f"processed_count{cls}").fill_(count)
        self._publish_statistics(cls)
        self.fit_diagnostics[cls] = dict(mode="initial_em", num_samples=count, em_iterations=gmm.n_iter_,
                                         em_converged=gmm.converged_, gamma=self.momentum_gamma,
                                         current_mass=(mass * count).tolist(), history_mass=mass.tolist())

    @torch.no_grad()
    def _commit_class(self, cls, history, pending):
        current_mass, current_mean, current_m2 = pending
        normalizer = current_mass.sum()
        if normalizer <= 0: return

        previous_mass = history[0].clone()
        gamma = self.momentum_gamma
        proportion = current_mass / normalizer
        new_mass = gamma * previous_mass + (1 - gamma) * proportion
        tiny = torch.finfo(current_mass.dtype).tiny
        step = (1 - gamma) * proportion / new_mass.clamp_min(tiny)
        shape = (-1,) + (1,) * (current_m2.ndim - 1)

        old_cov = history[2] / previous_mass.clamp_min(tiny).reshape(shape)
        cur_cov = current_m2 / current_mass.clamp_min(tiny).reshape(shape)
        _, mean, cov = self._merge_statistics(
            1 - step, history[1], (1 - step).reshape(shape) * old_cov,
            step, current_mean, step.reshape(shape) * cur_cov)

        history[0].copy_(new_mass); history[1].copy_(mean); history[2].copy_(cov * new_mass.reshape(shape))
        self._publish_statistics(cls)
        count = int(getattr(self, f"queue_count{cls}"))
        getattr(self, f"processed_count{cls}").add_(count)
        self.fit_diagnostics[cls] = dict(mode="online_em", num_samples=count, gamma=gamma,
                                         previous_mass=previous_mass.tolist(), current_mass=current_mass.tolist(),
                                         current_proportion=proportion.tolist(), component_steps=step.tolist(),
                                         history_mass=new_mass.tolist())

    @torch.no_grad()
    def update_statistics(self):
        if self.queue_ptr1.item() < self.bank_size: return
        initializing = all(getattr(self, f"fit_count{i}").item() == 0 for i in range(2))

        for i in range(2):
            history, pending = self._statistics_buffers(i, "history"), self._statistics_buffers(i, "pending")
            if initializing:
                self._initialize_class(i)
            else:
                for buf in pending: buf.zero_()
                count = int(getattr(self, f"queue_count{i}"))
                self._accumulate_statistics(i, getattr(self, f"queue{i}")[:count], "pending")
                self._commit_class(i, history, pending)

            for buf in pending: buf.zero_()
            getattr(self, f"queue_ptr{i}").zero_()
            getattr(self, f"fit_count{i}").add_(1)

    @torch.no_grad()
    def sample_prototypes(self):
        if any(getattr(self, f"fit_count{i}").item() == 0 for i in range(2)):
            raise RuntimeError("Prototype bank is not fitted")

        prototypes = []
        for i in range(2):
            weights, means = getattr(self, f"gmm_weights_{i}"), getattr(self, f"gmm_means_{i}")
            cov = getattr(self, f"gmm_covariances_{i}")
            indices = torch.multinomial(weights, self.num_prototypes, replacement=True)
            noise = torch.randn(self.num_prototypes, self.feat_dim, device=means.device, dtype=means.dtype)

            if self.covariance_type == "diag":
                perturb = cov[indices].sqrt() * noise
            else:
                factors = torch.linalg.cholesky(cov)
                perturb = torch.bmm(factors[indices], noise.unsqueeze(-1)).squeeze(-1)

            samples = F.normalize(means[indices] + perturb, p=2, dim=-1)
            prototypes.append(samples)

        return torch.stack(prototypes)


class SPOL_pretrain(nn.Module):
    PROTOTYPE_CHANNELS = {"out2": (256, 63), "out3": (512, 63), "out4": (1024, 63), "out": (128, 128)}

    def __init__(self, num_cls=1, num_prototype=20, *,
                 num_components=(1, 2, 3, 4, 5, 10), bank_sizes=(500, 1000, 2000), thresholds=(0.8,),
                 train_cluster=True, cluster_num_prototypes=50, gmm_init_iterations=50, pretrained=True,
                 component_bank_size=1000, capacity_components=5, cluster_bank_size=1000,
                 foreground_components=None, gmm_covariance_type="full",
                 gmm_momentum_gamma=0.9, gmm_covariance_shrinkage=0.25,
                 single_bank_config=None):
        super().__init__()
        if num_cls != 1: raise ValueError("SPOL prototype banks currently support binary classification only")

        self.num_cls, self.test, self.prototype_update_enabled = num_cls, False, False
        self.component_counts = _positive_ints(num_components, "num_components")
        self.bank_sizes = _positive_ints(bank_sizes, "bank_sizes")
        self.thresholds = tuple(dict.fromkeys(float(t) for t in thresholds))

        self.resnet50 = resnet.resnet50(pretrained=pretrained, strides=(2,2,2,1), dilations=(1,1,1,1))
        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1, self.stage2 = nn.Sequential(self.resnet50.layer1), nn.Sequential(self.resnet50.layer2)
        self.stage3, self.stage4 = nn.Sequential(self.resnet50.layer3), nn.Sequential(self.resnet50.layer4)

        self.fc51, self.bn51, self.fc52 = nn.Conv2d(2048,128,1), nn.BatchNorm2d(128), nn.Conv2d(128,2048,1)
        self.fc41, self.bn41, self.fc42 = nn.Conv2d(1024,128,1), nn.BatchNorm2d(128), nn.Conv2d(128,1024,1)
        self.fc31, self.bn31, self.fc32 = nn.Conv2d(512,128,1), nn.BatchNorm2d(128), nn.Conv2d(128,512,1)
        self.linear5, self.bn5 = nn.Conv2d(2048,128,1), nn.BatchNorm2d(128)
        self.linear4, self.bn4 = nn.Conv2d(1024,128,1), nn.BatchNorm2d(128)
        self.linear3, self.bn3 = nn.Conv2d(512,128,1), nn.BatchNorm2d(128)
        self.classifier = nn.Conv2d(128, num_cls, 1)

        self.feature_dim = 3 + sum(v[1] for v in self.PROTOTYPE_CHANNELS.values())
        self.scale_projections = nn.ModuleDict({n: nn.Conv2d(i, o, 1, bias=False) for n,(i,o) in self.PROTOTYPE_CHANNELS.items()})
        self.fusion_linear = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        with torch.no_grad(): nn.init.eye_(self.fusion_linear.weight)

        self.bank_configs, self.bank_groups, self.cluster_bank_configs = {}, {}, {}
        ref_t = self.thresholds[0]
        if single_bank_config is None:
            configs = [("components", ref_t, component_bank_size, k) for k in self.component_counts]
            configs += [("capacity", ref_t, s, capacity_components) for s in self.bank_sizes]
            configs += [("threshold", t, component_bank_size, capacity_components) for t in self.thresholds]
        else:
            threshold, size, components = single_bank_config
            configs = [("components", float(threshold), int(size), int(components))]
            self.thresholds = (float(threshold),)

        for group, threshold, size, components in configs:
            name = ("PrototypeSupportBank_" + format(threshold, "g").replace(".", "")
                    if group == "components" and size == component_bank_size and components == capacity_components
                    else f"PrototypeSupportBank_{group}_t{format(threshold,'g').replace('.','p')}_b{size}_c{components}")
            self.add_module(name, PrototypeSupportBank(
                self.feature_dim, size, num_prototype, components, init_iterations=gmm_init_iterations,
                foreground_components=foreground_components, covariance_type=gmm_covariance_type,
                momentum_gamma=gmm_momentum_gamma, covariance_shrinkage=gmm_covariance_shrinkage))
            self.bank_configs[name], self.bank_groups[name] = (threshold, size, components), group

        if train_cluster:
            self.add_module("clusterSupportBank", clusterSupportBank(self.feature_dim, cluster_bank_size, cluster_num_prototypes))
            self.cluster_bank_configs["clusterSupportBank"] = cluster_bank_size

        self.drop_out = nn.Dropout2d(0.1)

    def fuse_prototype_features(self, x, out2, out3, out4, out):
        size, blocks = out3.shape[-2:], []
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        for name, feature in (("out2", out2), ("out3", out3), ("out4", out4), ("out", out)):
            projected = feature if name == "out" else self.scale_projections[name](feature)
            if projected.shape[-2:] != size: projected = F.interpolate(projected, size=size, mode="bilinear", align_corners=False)
            blocks.append(F.normalize(projected, p=2, dim=1))
        blocks.append(x)
        fused = rearrange(torch.cat(blocks, dim=1), "b d h w -> b (h w) d")
        return F.normalize(F.gelu(self.fusion_linear(fused)), p=2, dim=-1)

    def extract_backbone_features(self, x):
        out1 = self.stage0(x); out2 = self.stage1(out1); out3 = self.stage2(out2); out4 = self.stage3(out3); out5 = self.stage4(out4)
        return out2, out3, out4, out5

    def fuse_classification_features(self, out3, out4, out5):
        out5a = F.relu(self.bn51(self.fc51(self.drop_out(out5.mean((2,3),keepdim=True)))), inplace=True)
        out4a = F.relu(self.bn41(self.fc41(self.drop_out(out4.mean((2,3),keepdim=True)))), inplace=True)
        out3a = F.relu(self.bn31(self.fc31(self.drop_out(out3.mean((2,3),keepdim=True)))), inplace=True)
        vector = out5a * out4a * out3a
        out5_, out4_, out3_ = torch.sigmoid(self.fc52(vector))*out5, torch.sigmoid(self.fc42(vector))*out4, torch.sigmoid(self.fc32(vector))*out3
        out3_ = F.relu(self.bn3(self.linear3(out3_)), inplace=True)
        out4_ = F.interpolate(F.relu(self.bn4(self.linear4(out4_)), inplace=True), size=out3.shape[-2:], mode="bilinear", align_corners=True)
        out5_ = F.interpolate(F.relu(self.bn5(self.linear5(out5_)), inplace=True), size=out3.shape[-2:], mode="bilinear", align_corners=True)
        return out5_ * out4_ * out3_

    def forward_features(self, x):
        out2, out3, out4, out5 = self.extract_backbone_features(x)
        out = self.fuse_classification_features(out3, out4, out5)
        cam = self.classifier(out)
        scores = F.adaptive_max_pool2d(cam, 1)
        hie_fea = self.fuse_prototype_features(x, out2.detach(), out3.detach(), out4.detach(), out)
        return scores, cam, hie_fea, out

    def forward(self, x, label=None, skull_mask=None):
        scores, cam, hie_fea, out = self.forward_features(x)
        if self.training and not self.test and self.prototype_update_enabled:
            if label is None: raise ValueError("Image-level labels required for prototype updates")
            devices = [x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.update_prototype_banks(cam.detach(), hie_fea.detach(), label, skull_mask)
        return (scores, cam, hie_fea, out) if self.test else (scores, cam)

    @torch.no_grad()
    def update_prototype_banks(self, cam, hie_fea, label, skull_mask=None):
        cam, hie_fea = cam.detach().float(), hie_fea.detach().float()
        batch, _, height, width = cam.shape
        labels = label.detach().to(cam.device).reshape(-1)
        if labels.numel() != batch: raise ValueError("Expected one binary label per image")

        positive = labels == 1
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, 1) + 1e-5)
        sampling_mask = dilated_skull_mask(skull_mask, norm_cam)

        if self.cluster_bank_configs:
            foreground = (norm_cam > 0.1) & positive[:,None,None,None] & sampling_mask
            background = ((norm_cam < 0.9) | ~positive[:,None,None,None]) & sampling_mask

            def pool(mask):
                weights = mask.flatten(2).transpose(1,2).float()
                counts = weights.sum(1)
                valid = counts[:,0] > 0
                return ((weights * hie_fea).sum(1) / counts.clamp_min(1))[valid]

            fore_samples, back_samples = pool(foreground), pool(background)
            for name in self.cluster_bank_configs:
                bank = getattr(self, name)
                bank.update_bank(fore_samples, back_samples); bank.update_context_protos()

        for threshold in self.thresholds:
            pseudo_label = torch.full_like(norm_cam, -1, dtype=torch.long)
            pseudo_label[(norm_cam == 0) & positive[:,None,None,None] & sampling_mask] = 0
            pseudo_label[(norm_cam > threshold) & positive[:,None,None,None] & sampling_mask] = 1
            samples = PrototypeSupportBank.select_samples(pseudo_label, hie_fea)
            for name, config in self.bank_configs.items():
                if config[0] == threshold: getattr(self, name).update_samples(samples)

    @torch.no_grad()
    def gmm_fit_report(self):
        """Return JSON-safe diagnostics for every GMM bank."""
        report = {}
        for name, config in self.bank_configs.items():
            bank = getattr(self, name)
            classes = []
            for cls, label in enumerate(("background", "foreground")):
                weights = getattr(bank, f"gmm_weights_{cls}").detach().double().cpu()
                covariance = getattr(bank, f"gmm_covariances_{cls}").detach()
                finite = bool(torch.isfinite(weights).all() and torch.isfinite(covariance).all())
                if finite and covariance.ndim == 2:
                    positive_definite = bool((covariance > 0).all())
                elif finite:
                    _, info = torch.linalg.cholesky_ex(covariance)
                    positive_definite = bool((info == 0).all())
                else:
                    positive_definite = False
                diagnostics = bank.fit_diagnostics.get(cls, {})
                classes.append({
                    "label": label,
                    "weights": weights.tolist(),
                    "active_components": int((weights >= 0.01).sum()),
                    "effective_components": float(torch.exp(
                        -(weights * weights.clamp_min(1e-12).log()).sum())),
                    "fitted": bool(getattr(bank, f"fit_count{cls}").item() > 0),
                    "fit_count": int(getattr(bank, f"fit_count{cls}").item()),
                    "queue_count": int(getattr(bank, f"queue_count{cls}").item()),
                    "processed_samples": int(getattr(bank, f"processed_count{cls}").item()),
                    "history_mass": getattr(bank, f"history_mass{cls}").detach().cpu().tolist(),
                    "finite": finite,
                    "covariance_positive_definite": positive_definite,
                    "fit_diagnostics": diagnostics,
                })
            report[name] = {
                "group": self.bank_groups[name],
                "threshold": float(config[0]),
                "bank_size": int(config[1]),
                "num_components": int(config[2]),
                "covariance_type": bank.covariance_type,
                "covariance_shrinkage": bank.covariance_shrinkage,
                "momentum_gamma": bank.momentum_gamma,
                "classes": classes,
            }
        return report


@register_model
def build_pretrain(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    return SPOL_pretrain(pretrained=pretrained, **kwargs)
