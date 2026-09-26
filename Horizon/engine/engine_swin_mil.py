
#####################################################可修改###############################################
import math
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve
from torch import Tensor
import utils

bce_loss=torch.nn.BCEWithLogitsLoss()

def dice_coeff(input:Tensor,target:Tensor,eps=1e-6):
    inter=(input*target).sum((-1,-2))
    union=input.sum((-1,-2))+target.sum((-1,-2))
    return ((2*inter+eps)/(union+eps)).mean()


def dice_loss(input, target):
    return 1-dice_coeff(input, target)


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler, set_training_mode=True, clip_grad=None):

    model.train(set_training_mode)
    metric_logger=utils.MetricLogger(delimiter="  ")

    for samples, targets, mask in metric_logger.log_every(data_loader,100,f"Epoch:{epoch}"):

        img=samples.to(device,non_blocking=True)
        label=targets.to(device,non_blocking=True)

        scores, _ = model(img)
        # Both outputs are raw logits and receive the same image-level label.

        target = label.reshape(-1).float()
        loss_cam = bce_loss(scores.reshape(-1).float(), target)

        loss = loss_cam
        if not math.isfinite(loss.item()):
            raise FloatingPointError('Non-finite SwinMIL training loss')

        optimizer.zero_grad(set_to_none=True)

        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)

        loss_scaler.step(optimizer)

        loss_scaler.update()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item(), loss_cam=loss_cam.item(),
                              lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    return {
        k:v.global_avg
        for k,v in metric_logger.meters.items()
    }
############################################################################################################

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
        scores, _ = model(img)
        score = scores  # Decoder score corresponds to the final CAM.
        ######################################################

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
        _, cam = model(img)
        ######################################################

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
