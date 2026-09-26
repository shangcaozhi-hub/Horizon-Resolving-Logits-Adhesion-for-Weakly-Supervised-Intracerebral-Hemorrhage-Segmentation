import numpy as np
from medpy import metric
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
clist = cm.get_cmap('Reds', 256)
import pingouin as pg
import pandas as pd
from scipy.stats import wasserstein_distance
from pathlib import Path
import torch


def d_prime(fg_logits, bg_logits, eps=1e-8):
    """
    Signal detection d-prime.

    Larger d' -> better foreground/background separation.
    """
    fg = np.asarray(fg_logits).flatten()
    bg = np.asarray(bg_logits).flatten()

    mu_fg = fg.mean()
    mu_bg = bg.mean()

    var_fg = fg.var(ddof=1)
    var_bg = bg.var(ddof=1)

    pooled_std = np.sqrt(
        0.5 * (var_fg + var_bg)
    )

    d = (mu_fg - mu_bg) / (pooled_std + eps)

    return float(d)


def calc_wasserstein(fg_logits, bg_logits):
    fg = np.asarray(fg_logits, dtype=np.float32)
    bg = np.asarray(bg_logits, dtype=np.float32)
    return wasserstein_distance(fg, bg)


def bhattacharyya_distance(fg_logits, bg_logits, bins=150, eps=1e-12):
    """
    Bhattacharyya distance between foreground and background logit distributions.

    Args:
        fg_logits: 1D numpy array
        bg_logits: 1D numpy array
        bins: histogram bins

    Returns:
        float: small -> more separated
    """
    fg = np.asarray(fg_logits).flatten()
    bg = np.asarray(bg_logits).flatten()

    # 使用相同的 logit 范围和 bin
    min_v = min(fg.min(), bg.min())
    max_v = max(fg.max(), bg.max())

    hist_fg, edges = np.histogram(
        fg,
        bins=bins,
        range=(min_v, max_v)
    )

    hist_bg, _ = np.histogram(
        bg,
        bins=edges
    )

    # 转换为概率分布
    p_fg = hist_fg / (hist_fg.sum() + eps)
    p_bg = hist_bg / (hist_bg.sum() + eps)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p_fg * p_bg))

    return float(bc)


def calculate_icc(gt_volume_list, pred_volume_list):

    """
    ICC(2,1): two-way random effects, absolute agreement
    """

    gt = np.asarray(gt_volume_list)
    pred = np.asarray(pred_volume_list)

    # 构造long format
    data = pd.DataFrame({
        "case": np.arange(len(gt)).tolist() * 2,
        "rater": ["GT"] * len(gt) + ["Prediction"] * len(pred),
        "volume": np.concatenate([gt, pred])
    })

    icc = pg.intraclass_corr(
        data=data,
        targets="case",
        raters="rater",
        ratings="volume"
    )

    # ICC(2,1)
    result = icc[icc["Type"] == "ICC2"]

    return result["ICC"].values[0]


def calculate_rve(gt_volume_list, pred_volume_list, thresholds):
    """
    gt_volume_list: ground truth volume list
    pred_volume_list: prediction volume list
    thresholds: dict
        {
            "small_medium": xxx,
            "medium_large": xxx
        }

    return:
        overall, small, medium, large RVE
    """

    gt = np.asarray(gt_volume_list, dtype=np.float32)
    pred = np.asarray(pred_volume_list, dtype=np.float32)

    # Relative Volume Error (%)
    rve = (np.abs(pred - gt) / (gt + 1e-8)) * 100

    # 根据GT volume划分
    small_mask = gt < thresholds["small_medium"]

    medium_mask = (
        (gt >= thresholds["small_medium"]) &
        (gt < thresholds["medium_large"])
    )

    large_mask = gt >= thresholds["medium_large"]

    return np.median(rve), np.median(rve[small_mask]), np.median(rve[medium_mask]), np.median(rve[large_mask])


def split_volume_tertiles(volumes):
    """
    volumes: list or np.array, 每个病例的体积/voxel数量
    return:
        labels: 每个病例对应 small / medium / large
        thresholds: small-medium 和 medium-large 的分界值
    """
    volumes = np.asarray(volumes)
    n = len(volumes)

    # 从小到大排序后的索引
    sorted_idx = np.argsort(volumes)

    labels = np.empty(n, dtype=object)

    # 三等分边界
    one_third = n // 3
    two_third = 2 * n // 3

    small_idx = sorted_idx[:one_third]
    medium_idx = sorted_idx[one_third:two_third]
    large_idx = sorted_idx[two_third:]

    labels[small_idx] = "small"
    labels[medium_idx] = "medium"
    labels[large_idx] = "large"

    thresholds = {
        "small_medium": volumes[sorted_idx[one_third]],
        "medium_large": volumes[sorted_idx[two_third]]
    }

    return labels, thresholds


def calculate_auprc(pred_prob, gt_mask):
    """
    Pixel/voxel-level AUPRC

    Args:
        pred_prob:
            sigmoid probability map
            shape: [D,H,W]

        gt_mask:
            binary mask
            shape: [D,H,W]

    Returns:
        AUPRC
    """

    pred_prob = pred_prob.reshape(-1)
    gt_mask = gt_mask.reshape(-1)

    auprc = average_precision_score(
        gt_mask,
        pred_prob
    )

    return auprc


def calc_auc(pred_prob, gt_mask):
    """
    Find optimal threshold based on Youden index.

    Args:
        pred_prob:
            sigmoid probability map [D,H,W]

        gt_mask:
            binary mask [D,H,W]

    Returns:
        best_threshold
        auc
        best_tpr
        best_fpr
    """

    y_score = pred_prob.reshape(-1)
    y_true = gt_mask.reshape(-1)

    # ROC-AUC
    auc = roc_auc_score(
        y_true,
        y_score
    )

    # ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score
    )

    # Youden index
    youden = tpr - fpr

    idx = np.argmax(youden)

    best_threshold = thresholds[idx]

    return auc, best_threshold


def calculate_iou(a, b, epsilon=1e-5):
    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou


def calculate_metric(p,g):

    if p.sum()==0 and g.sum()!=0:
        return 0,373.128,0,0,0,0

    return (
        metric.binary.dc(p,g),
        metric.binary.hd95(p,g),
        metric.binary.assd(p,g),
        calculate_iou(p, g),
        metric.binary.sensitivity(p,g),
        metric.binary.specificity(p,g)
    )


def summarize(x):

    x=np.asarray(x)

    return {
        "mean":float(x.mean()),
        "std":float(x.std()),
        "median":float(np.median(x)),
        "q1":float(np.percentile(x,25)),
        "q3":float(np.percentile(x,75)),
        "iqr":float(np.percentile(x,75)-np.percentile(x,25))
    }


def logits_adhesion_index(all_fg_logits, all_bg_logits):
    fg = np.asarray(all_fg_logits, dtype=np.float32)
    bg = np.asarray(all_bg_logits, dtype=np.float32)
    kde_fg = gaussian_kde(fg)
    kde_bg = gaussian_kde(bg)
    z_min = min(fg.min(), bg.min())
    z_max = max(fg.max(), bg.max())
    z = np.linspace(z_min, z_max, 500)
    p_fg = kde_fg(z)
    p_bg = kde_bg(z)
    lai = np.trapz(np.minimum(p_fg, p_bg), z)

    return lai

def inter_slice_dice(pred_3d, eps=1e-8):
    """
    Calculate Dice consistency between adjacent slices.

    Args:
        pred_3d:
            numpy array [D,H,W], binary prediction mask

    Returns:
        mean adjacent slice Dice
    """

    D = pred_3d.shape[0]

    dice_list = []

    for i in range(D - 1):

        slice1 = pred_3d[i]
        slice2 = pred_3d[i + 1]

        area1 = slice1.sum()
        area2 = slice2.sum()

        # ignore two empty slices
        if area1 == 0 and area2 == 0:
            continue

        intersection = np.logical_and(
            slice1,
            slice2
        ).sum()

        dice = (2.0 * intersection / (area1 + area2 + eps))
        dice_list.append(dice)

    if len(dice_list) == 0:
        return 1.0

    return np.mean(dice_list)


def compute_LAS(fg_logits, bg_logits):
    """
    LAS = E[(Z_B - Z_F)_+]
    Exact computation, O(N log N)
    """
    fg_tensor = torch.tensor(fg_logits)
    fg = torch.sort(fg_tensor.flatten())[0]

    bg_tensor = torch.tensor(bg_logits)
    bg = torch.sort(bg_tensor.flatten())[0]

    nf, nb = fg.numel(), bg.numel()

    # number and sum of foreground logits <= each background logit
    idx = torch.searchsorted(fg, bg)

    fg_sum = torch.cat([
        fg.new_zeros(1),
        fg.cumsum(0)
    ])[idx]

    violation = idx * bg - fg_sum

    return violation.sum() / (nf * nb)



def to_uint8(x):
    x = np.asarray(x, dtype=np.float32)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    else:
        x = np.zeros_like(x)
    return (x * 255).astype(np.uint8)


def to_bgr(img):
    img = to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def show_cam_on_image(
    img,
    cam,
    pred,
    gt,
    save_root,
    name="case",
    save_img=False,
    save_gt=False,
    save_pred=False,
    save_cam=False,
    save_error=False,
    cam_alpha=0.45,
    mask_alpha=0.50,
):
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # mandatory dirs
    dirs = {
        "cam_overlay": save_root / "cam_overlay",
        "mask_overlay": save_root / "mask_overlay",
    }

    # optional dirs
    optional = {
        "img": save_img,
        "gt": save_gt,
        "pred": save_pred,
        "cam": save_cam,
        "error": save_error,
    }

    for k, flag in optional.items():
        if flag:
            dirs[k] = save_root / k

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    img = to_bgr(img)
    cam = np.squeeze(cam)
    pred = np.squeeze(pred).astype(np.uint8)
    gt = np.squeeze(gt).astype(np.uint8)

    if img.shape[:2] != gt.shape:
        img = cv2.resize(img, (gt.shape[1], gt.shape[0]))

    cam_u8 = to_uint8(cam)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

    # 1. CAM overlay, always save
    cam_overlay = cv2.addWeighted(
        img,
        1 - cam_alpha,
        cam_color,
        cam_alpha,
        0
    )
    cv2.imwrite(
        str(dirs["cam_overlay"] / f"{name}_cam_overlay.png"),
        cam_overlay
    )

    # 2. TP/FP/FN mask overlay, always save
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)

    # OpenCV BGR
    # TP: green
    error[(pred == 1) & (gt == 1)] = [0, 255, 0]
    # FP: yellow
    error[(pred == 1) & (gt == 0)] = [0, 255, 255]
    # FN: red
    error[(pred == 0) & (gt == 1)] = [0, 0, 255]     # FN: blue

    mask_overlay = img.copy()
    region = error.sum(axis=-1) > 0
    mask_overlay[region] = cv2.addWeighted(
        img[region],
        1 - mask_alpha,
        error[region],
        mask_alpha,
        0
    )

    cv2.imwrite(
        str(dirs["mask_overlay"] / f"{name}_mask_overlay.png"),
        mask_overlay
    )

    # optional saves
    if save_img:
        cv2.imwrite(str(dirs["img"] / f"{name}_img.png"), img)

    if save_gt:
        cv2.imwrite(str(dirs["gt"] / f"{name}_gt.png"), gt * 255)

    if save_pred:
        cv2.imwrite(str(dirs["pred"] / f"{name}_pred.png"), pred * 255)

    if save_cam:
        cv2.imwrite(str(dirs["cam"] / f"{name}_cam.png"), cam_color)

    if save_error:
        cv2.imwrite(str(dirs["error"] / f"{name}_error.png"), error)


