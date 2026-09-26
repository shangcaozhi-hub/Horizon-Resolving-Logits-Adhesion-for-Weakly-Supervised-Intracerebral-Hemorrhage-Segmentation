"""PPC with pseudo-label hard mining or label-free CAM weighting."""
import math
import torch
import torch.nn.functional as F


def sample_ppc(label, logits, *, mode="hard", temperature=0.1,
               pixel_sample_prob=0.5, negative_sample_prob=0.5,
               hard_pixel_fraction=0.6, hard_negative_fraction=0.6,
               skip_hardest_fraction=0.1, cam_logits=None,
               valid_mask=None,
               foreground_positive="last", num_components=1):
    """Choose pseudo-label hard mining or label-free weighted PPC.

    logits: [pixels, 2, prototypes]. All prototypes are treated alike regardless
    of their origin for negative sampling. The last num_components scores are
    center similarities: background positives use their max, foreground their mean.
    foreground_positive is accepted for compatibility but no longer selects targets.
    Sampling probabilities select floor(probability * candidate_count) uniformly
    without replacement (at least one pixel for a positive pixel probability).
    In hard mode, random/mined overlap is retained: a 0.5 pixel probability draws
    a random half plus the own-class positive-similarity rank interval [10%, 60%).
    Both classes rank pixels by maximum own-class prototype similarity, lowest
    first. Center targets do not change hard-pixel mining.
    Fractions use each class's full candidate count.
    Weighted mode ignores labels and evaluates both class directions at every
    pixel using all opposite-class prototypes. CAM confidence is obtained by
    per-image ReLU-max normalization. Prototype uncertainty is the linear value
    1-|sim_bg-sim_fg|/2 for cosine similarities. Their detached product weights
    pixels with confident CAM predictions but ambiguous prototype assignments.
    pixel_sample_prob accepts one value or (background, foreground).
    Losses are averaged within each present class, then across present classes.
    """
    if logits.ndim != 3 or logits.shape[1] != 2 or logits.shape[2] == 0:
        raise ValueError("Expected prototype logits [N, 2, P] with P > 0")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode not in ("hard", "weighted"):
        raise ValueError("mode must be hard or weighted")
    if foreground_positive not in ("last", "mean"):
        raise ValueError("foreground_positive must be last or mean")
    num_pixels, _, num_prototypes = logits.shape
    if (isinstance(num_components, bool) or not isinstance(num_components, int)
            or not 1 <= num_components <= num_prototypes):
        raise ValueError("num_components must be an integer in [1, P]")
    device = logits.device
    if mode == "weighted":
        if cam_logits is None or cam_logits.numel() != num_pixels:
            raise ValueError("Weighted PPC requires one aligned CAM logit per pixel")
        if cam_logits.ndim != 4 or cam_logits.shape[1] != 1:
            raise ValueError("Weighted PPC expects CAM logits [B, 1, H, W]")
        values = logits.float()
        background_positive = values[:, 0, :].amax(-1)
        foreground_positive_score = values[:, 1, -num_components:].mean(-1)
        targets = torch.zeros(num_pixels, dtype=torch.long, device=device)
        background_loss = F.cross_entropy(
            torch.cat((background_positive[:, None], values[:, 1]), dim=-1) / temperature,
            targets, reduction="none")
        foreground_loss = F.cross_entropy(
            torch.cat((foreground_positive_score[:, None], values[:, 0]), dim=-1) / temperature,
            targets, reduction="none")
        with torch.no_grad():
            '''positive_cam = F.relu(cam_logits.detach().float())
            probability = positive_cam / positive_cam.amax(
                dim=(2, 3), keepdim=True).clamp_min(1e-6)'''
            probability = torch.sigmoid(cam_logits.detach().float())
            probability = probability.reshape(-1)
            class_similarity = values.detach().amax(-1)
            prototype_gap = (class_similarity[:, 0] - class_similarity[:, 1]).abs()
            prototype_uncertainty = (1.0 - prototype_gap / 2.0).clamp(0.0, 1.0)
            foreground_confidence = (2.0 * probability - 1.0).clamp_min(0.0)
            background_confidence = (1.0 - 2.0 * probability).clamp_min(0.0)
            foreground_weight = foreground_confidence * prototype_uncertainty
            background_weight = background_confidence * prototype_uncertainty
            if valid_mask is not None:
                if valid_mask.numel() != num_pixels:
                    raise ValueError("Weighted PPC valid mask must align with CAM pixels")
                background_weight *= valid_mask.detach().reshape(-1).to(
                    device=device, dtype=background_weight.dtype)
        numerator = (foreground_weight * foreground_loss
                     + background_weight * background_loss).sum()
        denominator = (foreground_weight + background_weight).sum()
        return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)

    if label is None:
        raise ValueError("Hard PPC requires pseudo labels")
    label = label.detach().reshape(-1).to(device)
    if label.numel() != num_pixels:
        raise ValueError("Pixel labels and prototype logits must be aligned")
    probs = ((pixel_sample_prob,) * 2 if isinstance(pixel_sample_prob, (float, int))
             else tuple(pixel_sample_prob))
    if len(probs) != 2:
        raise ValueError("pixel_sample_prob must be a scalar or a pair")
    fractions = (*probs, negative_sample_prob, hard_pixel_fraction,
                 hard_negative_fraction, skip_hardest_fraction)
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in fractions):
        raise ValueError("Sampling fractions must be finite and in [0, 1]")

    class_losses = []
    for cls, sample_prob in enumerate(probs):
        candidates = torch.where(label == cls)[0]
        if candidates.numel() == 0 or sample_prob == 0:
            continue

        with torch.no_grad():
            count = candidates.numel()
            random_count = max(int(count * sample_prob), 1)
            random_rows = candidates[torch.randperm(count, device=device)[:random_count]]

            # Hard pixel: 与负类任一 prototype 的最大相似度越高，越困难
            opposite_score = logits[candidates, 1 - cls].detach().amax(-1)
            order = opposite_score.argsort(descending=True)
            if cls == 0:
                lower = int(count * skip_hardest_fraction)
                upper = int(count * hard_pixel_fraction)
            else:
                lower = int(count * skip_hardest_fraction * 0.1)
                upper = int(count * (hard_pixel_fraction - skip_hardest_fraction))
            hard_rows = candidates[order[lower:upper]]
            # Concatenation intentionally retains random/hard overlap twice.
            rows = torch.cat((random_rows, hard_rows))

        values = logits[rows].float()
        centers = values[:, cls, -num_components:]
        positive = values[:, cls].amax(-1) if cls == 0 else centers.mean(-1)
        opposite = values[:, 1 - cls]

        random_negative_count = int(num_prototypes * negative_sample_prob)
        random_indices = torch.randperm(num_prototypes, device=device)[:random_negative_count]
        random_negatives = opposite[:, random_indices]

        hard_count = (max(int(num_prototypes * hard_negative_fraction), 1)
                      if hard_negative_fraction else 0)
        hard_negatives = opposite[:, :0]
        if hard_count:
            hard_indices = opposite.detach().topk(hard_count, dim=-1).indices
            # hard_indices = hard_indices[:, int(num_prototypes * skip_hardest_fraction):]
            hard_negatives = opposite.gather(1, hard_indices)
        negatives = torch.cat((hard_negatives, random_negatives), dim=-1)
        if negatives.shape[1] == 0:
            continue

        contrasts = torch.cat((positive[:, None], negatives), dim=-1)
        targets = torch.zeros(rows.numel(), dtype=torch.long, device=device)
        class_losses.append(F.cross_entropy(contrasts / temperature, targets))

    return (torch.stack(class_losses).mean() if class_losses
            else logits.reshape(-1)[:0].sum())
