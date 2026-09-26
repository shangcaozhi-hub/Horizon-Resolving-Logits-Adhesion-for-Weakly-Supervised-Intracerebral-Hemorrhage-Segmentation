
#####################################################可修改###############################################
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
import utils

import matplotlib.pyplot as plt
bce_loss=torch.nn.BCEWithLogitsLoss()


def _write_gmm_report(model, epoch):
    """Write one snapshot after the epoch's final bank update."""
    report = {"epoch": int(epoch), "round": int(epoch) + 1,
              "phase": "after_final_bank_update_before_optimizer",
              "feature_dim": int(model.feature_dim), "banks": model.gmm_fit_report()}
    with open(model.gmm_report_path, "a", encoding="utf-8") as report_file:
        report_file.write(json.dumps(report, allow_nan=False) + "\n")
    print(f"GMM_REPORT round={epoch + 1}", flush=True)
    for name in model.bank_configs:
        states = report["banks"][name]["classes"]
        print(name, [{"class": s["label"], "weights": s["weights"],
                      "active": s["active_components"], "fitted": s["fitted"],
                      "finite": s["finite"], "positive_definite": s["covariance_positive_definite"]}
                     for s in states], flush=True)


@torch.no_grad()
def _collect_prototype_banks(model, data_loader, optimizer, device, epoch, loss_scaler):
    """Collect banks without gradients, optimizer steps or epoch advancement."""
    rounds = int(getattr(model, "bank_collection_rounds", 0))
    if rounds <= 0:
        return

    original_lrs = [group["lr"] for group in optimizer.param_groups]
    previous_update_state = model.prototype_update_enabled
    batch_norm_state = {
        module: (module.running_mean.clone(), module.running_var.clone(),
                 module.num_batches_tracked.clone())
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    for group in optimizer.param_groups:
        group["lr"] = 0.0

    model.train(True)
    model.prototype_update_enabled = True
    try:
        for stat_round in range(rounds):
            logger = utils.MetricLogger(delimiter="  ")
            header = f"Bank statistics:{stat_round + 1}/{rounds}"
            for samples, targets, skull_mask in logger.log_every(data_loader, 100, header):
                model(samples.to(device, non_blocking=True),
                      label=targets.to(device, non_blocking=True),
                      skull_mask=skull_mask)
            print(f"BANK_STATISTICS completed round {stat_round + 1}/{rounds}", flush=True)

        for group, learning_rate in zip(optimizer.param_groups, original_lrs):
            group["lr"] = learning_rate
        for module, state in batch_norm_state.items():
            module.running_mean.copy_(state[0])
            module.running_var.copy_(state[1])
            module.num_batches_tracked.copy_(state[2])
        checkpoint = {
            "models": model.state_dict(),
            "opt": optimizer.state_dict(),
            "epoch": int(epoch) - 1,
            "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
            "args": getattr(model, "bank_checkpoint_args", None),
            "phase": "bank_statistics_before_training_epoch_10",
            "bank_statistics_rounds": rounds,
        }
        utils.save_on_master(checkpoint, model.bank_checkpoint_path)
        model._bank_collection_done = True
        print(f"BANK_CHECKPOINT saved: {model.bank_checkpoint_path}", flush=True)
    finally:
        model.prototype_update_enabled = previous_update_state
        for module, state in batch_norm_state.items():
            module.running_mean.copy_(state[0])
            module.running_var.copy_(state[1])
            module.num_batches_tracked.copy_(state[2])
        for group, learning_rate in zip(optimizer.param_groups, original_lrs):
            group["lr"] = learning_rate
##########################################################################################

def dice_coeff(input:Tensor,target:Tensor,eps=1e-6):
    inter=(input*target).sum((-1,-2))
    union=input.sum((-1,-2))+target.sum((-1,-2))
    return ((2*inter+eps)/(union+eps)).mean()


def dice_loss(input, target):
    return 1-dice_coeff(input, target)


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, set_training_mode=True):

    model.train(set_training_mode)
    metric_logger=utils.MetricLogger(delimiter="  ")

    collection_epoch = getattr(model, "bank_collection_epoch", None)
    if (collection_epoch is not None and epoch == collection_epoch
            and not getattr(model, "_bank_collection_done", False)):
        _collect_prototype_banks(model, data_loader, optimizer, device, epoch, loss_scaler)

    for samples, targets, skull_mask in metric_logger.log_every(data_loader,100,f"Epoch:{epoch}"):

        img=samples.to(device,non_blocking=True)
        label=targets.to(device,non_blocking=True)

        #####################################################可修改#################################
        model.prototype_update_enabled = getattr(
            model, "prototype_update_during_training", epoch >= 0)
        # Dataset masks precede random rotations/flips; align support with model input.

        scores, _ = model(img, label=label, skull_mask=skull_mask)
        first_fit_path = getattr(model, 'first_fit_checkpoint_path', None)
        if first_fit_path is not None and not getattr(model, '_first_fit_saved', False):
            fit_counts = {name: [int(getattr(model, name).fit_count0),
                                 int(getattr(model, name).fit_count1)]
                          for name in model.bank_configs}
            fit_counts.update({name: getattr(model, name).num_updates.tolist()
                               for name in model.cluster_bank_configs})
            if fit_counts and all(min(counts) > 0 for counts in fit_counts.values()):
                torch.save({'models': model.state_dict(),
                            'args': model.first_fit_checkpoint_args,
                            'epoch': int(epoch),
                            'batch': int(metric_logger.meters['loss'].count) + 1,
                            'fit_counts': fit_counts,
                            'phase': 'all_banks_first_fit_before_optimizer'}, first_fit_path)
                model._first_fit_saved = True
                print(f'FIRST_FIT_CHECKPOINT saved: {first_fit_path}; fits={fit_counts}', flush=True)
        loss = bce_loss(scores.view(-1).float(), label.view(-1))
        # Banks have consumed the final batch; inspect once per epoch before its optimizer step.
        if (metric_logger.meters["loss"].count + 1 == len(data_loader)
                and getattr(model, "gmm_report_path", None) is not None):
            _write_gmm_report(model, epoch)
        ###########################################################################################

        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            10.0)

        loss_scaler.step(optimizer)

        loss_scaler.update()
        torch.cuda.synchronize()

        loss_value = loss.item()
        learning_rate = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value,lr=learning_rate)

        progress_path = getattr(model, "progress_log_path", None)
        if progress_path is not None:
            progress = {
                "epoch": int(epoch),
                "batch": int(metric_logger.meters["loss"].count),
                "num_batches": len(data_loader),
                "loss": float(loss_value),
                "lr": float(learning_rate),
                "grad_norm_after_clip": float(grad_norm.detach().item()),
            }
            if progress["batch"] % 100 == 0 and hasattr(model, "bank_status"):
                progress["banks"] = model.bank_status()
            with open(progress_path, "a", encoding="utf-8") as progress_file:
                progress_file.write(json.dumps(progress) + "\n")

    metric_logger.synchronize_between_processes()
    return {
        k:v.global_avg
        for k,v in metric_logger.meters.items()
    }

@torch.no_grad()
def evaluate_cls(model, loader, device):

    model.eval()
    preds=[]
    labels=[]

    metric_logger=utils.MetricLogger(delimiter="  ")

    for img,label,*_ in metric_logger.log_every(loader, 20, "Cls Val:"):

        img=img.to(device,non_blocking=True)
        label=label.to(device,non_blocking=True)


        ##########################可修改########################
        score = model(img)
        #######################################################

        if isinstance(score,(tuple,list)):
            score=score[0]

        preds.append(torch.sigmoid(score).flatten().cpu())
        labels.append(label.flatten().cpu())

    preds=torch.cat(preds).numpy()
    labels=torch.cat(labels).numpy()

    pred=(preds>0.5).astype(np.float32)

    return {
        "acc":accuracy_score(labels,pred),
        "precision":precision_score(labels,pred,zero_division=0),
        "recall":recall_score(labels,pred,zero_division=0),
        "f1_score":f1_score(labels,pred,zero_division=0)
    }



@torch.no_grad()
def evaluate_dice(model,loader,device,threshold=0.5):

    model.eval()
    dices=[]
    logger=utils.MetricLogger(delimiter="  ")

    for s in logger.log_every(loader,20,"Dice Val:"):

        img=s["image"].to(device,non_blocking=True)
        mask=s["mask"].to(device)

        ###########################可修改###########################

        out = model(img)
        cam = out[1] if isinstance(out, (list, tuple)) else out

        ###########################################################

        cam=torch.sigmoid(cam)
        cam=F.interpolate(
            cam,
            mask.shape[-2:],
            mode="bilinear",
            align_corners=False)

        pred=(cam > threshold).float()

        for p,g in zip(pred,mask):
            inter=(p*g).sum()
            dice=(2*inter+1e-6)/(p.sum()+g.sum()+1e-6)
            dices.append(dice.item())

    dice=np.mean(dices)

    return dice
