import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils
from network.ppc_loss import sample_ppc


def _normalize_cam(cam):
    positive = F.relu(cam)
    max = (F.adaptive_max_pool2d(positive, (1, 1)) + 1e-5).detach()
    return positive / max


def _minmax_normalize_cam(cam):
    minimum = (-F.adaptive_max_pool2d(-cam, (1, 1))).detach()
    maximum = F.adaptive_max_pool2d(cam, (1, 1)).detach()
    value_range = (maximum - minimum).clamp_min(1e-5)
    return (cam - minimum) / value_range


def _refined_label(proto_mask, fore_mask, above_mask):
    uncertain = proto_mask | above_mask
    label = torch.zeros_like(proto_mask, dtype=torch.long)
    label[uncertain] = -1
    label[proto_mask & fore_mask] = 1
    return label, uncertain


def _forward_view(model, image, label=None, skull_mask=None):
    scores, cam, features = model.cam_phase(image, label=label, skull_mask=skull_mask)
    proto_logits, sim, _, proto_mask = model.proto_phase(features, cam.shape[-2:])
    fore_mask, above_mask = model.cam_masks(cam, label=label)
    return scores, cam, proto_logits, sim, proto_mask, fore_mask, above_mask


def compute_horizon_losses(model, image, label, skull_mask=None, flip_dims=None):
    """CLS + 0.1*PPC + 0.1*CVC + C2P, including view-1 PPC anchoring."""
    if flip_dims is None:
        horizontal = random.random() < 0.5
        vertical = random.random() < 0.5
        if not horizontal:
            vertical = True
        flip_dims = tuple(dim for dim, enabled in ((-1, horizontal), (-2, vertical)) if enabled)
    view2 = torch.flip(image, flip_dims)
    scores1, cam1, logits1, sim1, proto1, fore1, above1 = _forward_view(
        model, image, label, skull_mask)
    # The reference updates banks only from the first view.
    scores2, cam2, _, sim2, proto2, fore2, above2 = _forward_view(model, view2)
    label = label.reshape(-1).float()
    cls_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(scores1.reshape(-1).float(), label) +
        F.binary_cross_entropy_with_logits(scores2.reshape(-1).float(), label))

    # Negative images contribute background pixels in both views.
    positive_images = (label != 0).reshape(-1, 1, 1, 1)
    proto1 = proto1 & positive_images
    above1 = above1 & positive_images
    uncertain1 = proto1 | above1
    options = dict(getattr(model, "ppc_options", {}))
    options["foreground_positive"] = "mean" if model.bank_group == "ot" else "last"
    if options.get("mode", "hard") == "weighted":
        # Label-free weighted PPC evaluates both class directions at every pixel.
        valid_mask = None
        if skull_mask is not None:
            valid_mask = F.interpolate(
                skull_mask.float(), size=cam1.shape[-2:], mode="nearest") > 0
        ppc_loss = sample_ppc(
            None, logits1, cam_logits=cam1, valid_mask=valid_mask, **options)
    else:
        # Both pseudo-label maps supervise view-1 logits, as in the original engine.
        refined1, _ = _refined_label(proto1, fore1, above1)
        refined2, _ = _refined_label(torch.flip(proto2, flip_dims),
                                    torch.flip(fore2, flip_dims),
                                    torch.flip(above2, flip_dims))
        ppc_loss = 0.5 * (
            sample_ppc(refined1, logits1, **options) +
            sample_ppc(refined2, logits1, **options))

    abnormal = label == 1
    # With any positive image, average spatial maxima over the entire batch.
    bias = ((cam1 - uncertain1.float() * 1e3)[abnormal].amax(dim=(2, 3)).mean()

            if abnormal.any() else cam1.new_zeros(()))
    biased1 = _normalize_cam(cam1 - bias)

    aligned_sim2 = torch.flip(sim2, flip_dims)

    proto_cam1 = _normalize_cam(sim1[:, 1:2] - sim1[:, :1])
    proto_cam2 = _normalize_cam(aligned_sim2[:, 1:2] - aligned_sim2[:, :1])
    mask = (proto_cam1 > 0) & (proto_cam2 > 0)
    cvc_loss = (proto_cam2 - proto_cam1).abs()[mask].mean() if mask.any() else proto_cam1.new_zeros(())
    # cvc_loss = (proto_cam2 - proto_cam1).abs().mean() # + (biased2 - biased1).abs().mean()

    positive_mask = positive_images.reshape(-1).bool()

    # normal forward: bias participates in gradient
    q_live = biased1[positive_mask]
    # same forward value, but bias gradient is blocked
    biased1_stop_bias = _normalize_cam(cam1 - bias.detach())
    q_stop = biased1_stop_bias[positive_mask]
    # prototype supervision
    p = proto_cam1[positive_mask].detach()
    eps = 1e-6

    # mask before clamp
    # valid_mask = ((p > 0) | (q_live.detach() > 0)).to(q_live.dtype)

    p = p.clamp(eps, 1.0 - eps)
    q_live = q_live.clamp(eps, 1.0 - eps)
    q_stop = q_stop.clamp(eps, 1.0 - eps)

    kl_map = (p * (torch.log(p) - torch.log(q_live)) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q_stop)))

    c2p_loss = kl_map.mean() # (kl_map * valid_mask).sum() / (valid_mask.sum() + eps)
    loss = cls_loss + 0.1 * ppc_loss + 0.1 * cvc_loss + 1 * c2p_loss
    return dict(loss=loss, cls_loss=cls_loss, ppc_loss=ppc_loss,
                cvc_loss=cvc_loss, c2p_loss=c2p_loss)


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    set_training_mode=True):
    """Same call signature and GradScaler protocol as engine_spol."""
    model.train(set_training_mode)
    model.require_bank_ready()
    # Horizon online-bank experiments update the selected bank from the first
    # epoch when --update-bank is enabled; inference keeps this flag false.
    model.prototype_update_enabled = bool(getattr(model, "update_bank", False))
    max_norm = getattr(model, "grad_clip", None)
    if max_norm is not None and (not math.isfinite(max_norm) or max_norm <= 0):
        raise ValueError("grad_clip must be None or finite and positive")
    logger = utils.MetricLogger(delimiter="  ")
    logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    progress_path = getattr(model, "progress_log_path", None)
    if progress_path is not None:
        progress_path = Path(progress_path)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    for batch, (samples, targets, mask) in enumerate(logger.log_every(data_loader, 100, f"Epoch:{epoch}"), 1):
        image = samples.to(device, non_blocking=True)
        label = targets.to(device, non_blocking=True)
        # Match current SPOL: dataset support masks precede augmentation.
        skull_mask = F.max_pool2d(mask.float(), 5, stride=1, padding=2)
        losses = compute_horizon_losses(model, image, label, skull_mask)
        values = {name: value.detach().item() for name, value in losses.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Non-finite Horizon loss at epoch={epoch}, batch={batch}: {values}")

        optimizer.zero_grad()
        loss_scaler.scale(losses["loss"]).backward()
        loss_scaler.unscale_(optimizer)
        grad_norm = None
        if max_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
        loss_scaler.step(optimizer)
        loss_scaler.update()
        if image.is_cuda:
            torch.cuda.synchronize(image.device)

        lr = optimizer.param_groups[0]["lr"]
        logger.update(**values, lr=lr)
        if progress_path is not None:
            report = dict(epoch=int(epoch), batch=batch, num_batches=len(data_loader),
                          **values, lr=lr, grad_clip=max_norm,
                          grad_norm_before_clip=grad_norm)
            if batch == 1 or batch % 100 == 0 or batch == len(data_loader):
                report["bank"] = model.selected_bank_status()
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report) + "\n")
    logger.synchronize_between_processes()
    return {name: meter.global_avg for name, meter in logger.meters.items()}


@torch.no_grad()
def evaluate_cls(model, loader, device):
    """Use the SPOL classification metric definition, without sampling prototypes."""
    model.eval()
    predictions, labels = [], []
    logger = utils.MetricLogger(delimiter="  ")
    for image, label, *_ in logger.log_every(loader, 20, "Cls Val:"):
        image = image.to(device, non_blocking=True)
        scores, _, _ = model.cam_phase(image)
        predictions.append(scores.sigmoid().reshape(-1).cpu())
        labels.append(label.reshape(-1).cpu())
    if not labels:
        raise ValueError("Classification validation loader is empty")
    pred = (torch.cat(predictions).numpy() > 0.5).astype(np.float32)
    truth = torch.cat(labels).numpy()
    return dict(acc=accuracy_score(truth, pred),
                precision=precision_score(truth, pred, zero_division=0),
                recall=recall_score(truth, pred, zero_division=0),
                f1_score=f1_score(truth, pred, zero_division=0))


@torch.no_grad()
def evaluate_dice(model, loader, device, threshold=0.5):
    """Preserve Horizon normalized-CAM prediction and positive-slice evaluation.

    The original 'dice' field held 1-Dice. Return the actual Dice score for the
    SPOL main loop, where a larger score selects the best checkpoint.
    """
    model.eval()
    score_sum, image_count = 0.0, 0
    logger = utils.MetricLogger(delimiter="  ")
    for sample in logger.log_every(loader, 20, "Dice Val:"):
        image = sample["image"].to(device, non_blocking=True)
        mask = sample["mask"] if "mask" in sample else sample["label"]
        mask = mask.to(device).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("Expected binary segmentation masks [B,H,W] or [B,1,H,W]")
        _, cam, _ = model.cam_phase(image)
        # Keep the reference order: threshold the low-resolution CAM, then resize.
        pred = F.interpolate((_normalize_cam(cam) > threshold).float(),
                             mask.shape[-2:], mode="bilinear", align_corners=False)
        valid = mask.sum(dim=(1, 2, 3)) > 0
        score = 0.0
        if valid.any():
            p, target = pred[valid], mask[valid]
            inter = (p * target).sum()
            score = float((2 * inter + 1e-6) / (p.sum() + target.sum() + 1e-6))
        # Retain the reference's pooled positive masks and batch-size weighting.
        score_sum += score * image.shape[0]
        image_count += image.shape[0]
    return score_sum / image_count if image_count else 0.0