######################可修改####################
import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import json
import os
from timm.models import create_model

import numpy as np
import tqdm
from network import spol_backbone
from dataloader.fully_datasets import bhsdDataSet, insDataSet
from dataloader.dataset_mri import Brats2021Dataset
from metrics import split_volume_tertiles, calc_auc, calculate_metric, show_cam_on_image, calculate_rve, calculate_icc, logits_adhesion_index, calculate_auprc, d_prime, bhattacharyya_distance

from tqdm import tqdm

import csv

from sklearn.metrics import precision_recall_curve
#########################################################################################################
def save_metrics_csv(csv_path, model_name, result_dict):
    """
    Append evaluation results to csv.

    Args:
        csv_path: csv file path
        model_name: model/seed name
        result_dict: metric dictionary
    """

    # 第一列作为模型标识
    fieldnames = ["Model"] + list(result_dict.keys())

    file_exists = os.path.exists(csv_path)

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        # 文件不存在时写header
        if not file_exists:
            writer.writeheader()

        # 当前结果
        row = {
            "Model": model_name,
            **result_dict
        }

        writer.writerow(row)


@torch.no_grad()
def find_best_threshold(model, valloader, device="cuda"):

    model.eval()
    best_ts = []

    for batch in tqdm(valloader, desc="Finding threshold"):

        img = batch["image"].to(device)
        gt = batch["mask"].cpu().numpy()

        ######################可修改####################
        _, cam, = model(img)

        ###############################################
        prob = torch.relu(cam)
        prob = prob / (F.adaptive_max_pool2d(prob, (1, 1)) + 1e-5)

        prob = F.interpolate(
            prob,
            size=(240, 240),
            mode="bilinear",
            align_corners=True
        ).cpu().numpy()

        if gt.ndim == 3:
            gt = gt[:,None]

        for p, g in zip(prob, gt):

            if g.sum() == 0:
                continue

            best_d, best_t = 0, 0.5

            for t in np.arange(0.05, 0.91, 0.05):
                pred = p > t
                d = 2*np.logical_and(pred,g).sum() / (
                    pred.sum()+g.sum()+1e-8
                )

                if d > best_d:
                    best_d, best_t = d, t
            best_ts.append(best_t)

    if not best_ts:
        raise RuntimeError("No positive slices.")

    best_t = np.median(best_ts)

    print(
        f"threshold mean={np.mean(best_ts):.3f}, "
        f"median={best_t:.3f}, "
        f"std={np.std(best_ts):.3f}"
    )

    return round(float(best_t),3)



if __name__ == '__main__':

    ######################可修改####################
    from network.swin_mil import SwinMIL
    model = SwinMIL(num_classes=1)
    model_path = r'D:\code\Repetition2\checkpoint\swin_cpcl1\hemorrhage\checkpoint-12.pth'
    # D:\code\Repetition3\checkpoint\SwinMIL\seed_1014\hemorrhage\checkpoint-best_dice.pth
    # D:\code\Repetition2\checkpoint\swin_cpcl1\hemorrhage\checkpoint-10.pth
    checkpoint = torch.load(model_path, map_location='cpu')
    ##################################################

    model.load_state_dict(checkpoint['models'], strict=False)
    # 463
    model.eval()
    model.to('cuda')
    output_dir = 'all_result'

    ######################可修改####################
    model_name = 'swin'

    ins = True
    brats = False
    visual = False
    search_threshold = True
    ##############################################
    best_threshold = 0.5

    if brats == False:
        db_val = insDataSet(base_dir="D:/code/Proto_ICH/Datasets/", data_ls="/val_slices.txt")
        valloader = DataLoader(db_val, batch_size=10, shuffle=False, num_workers=2, pin_memory=True)
        best_threshold = find_best_threshold(model, valloader)

    if search_threshold:
        thresholds = np.arange(0.1, 1.0, 0.1).tolist() + [best_threshold]
    else:
        thresholds = [best_threshold]

    vol_thresh = {'small_medium': 0.0, 'medium_large': np.inf}

    if ins == True:
        db_test = insDataSet(base_dir="D:/code/Proto_ICH/Datasets/", data_ls="/test_slices.txt")
        datapath = "/ins/"
        # thresholds = 0.1
        vol_thresh = {'small_medium': 1202.0, 'medium_large': 4962.0}
    else:
        db_test = bhsdDataSet(base_dir="D:/code/Proto_ICH/Datasets/BHSD", data_ls="/data_2D.txt")
        datapath = "/bhsd/"
        vol_thresh = {'small_medium': 1202.0, 'medium_large': 4962.0}
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    if brats == True:
        test_data_path = r"C:\Users\Caozhi\Desktop\prototype_weakly\data\test_2D.txt"
        test_data = []
        f = open(test_data_path, 'r', encoding='utf-8')
        for line in f:
            line = line.strip('\n')
            test_data.append(line)
        datapath = "/brats/"
        db_test = Brats2021Dataset('D:/Datasets/BRATS2021_Training_none_npy/', test_data, [240, 240], stage='val', task='seg')

        testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2,
                                 pin_memory=False, drop_last=True)

    result_metric = {t: {'Dice': [], 'IoU': [], 'HD95': [], 'Sen': [], 'Spe': []} for t in thresholds}
    pre_array = []
    label_array = []
    all_fg_logits = []
    all_bg_logits = []
    auc = []
    auprc = []
    lai = []

    volume = []
    pred_volume = []

    inter_dice = []

    viewed_id = ['001']

    path = output_dir + "/" + model_name + datapath
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(testloader):
            samples_list = []
            img, label, case = sampled_batch['image'], sampled_batch['mask'], \
            sampled_batch['case'][0].replace('.h5', '').split('/')[-1]
            img = img.to('cuda');
            label = label.to('cuda')

            target = (label.sum([1, 2]) > 0).float().view(-1)
            slice_num = int(case.split('_')[2])

            inputsize = 240
            ori_size = (240, 240)

            B, C, H, W = img.size()

            if ins == False:
                img = torch.rot90(img, dims=[2, 3])
                label = torch.rot90(label, dims=[1, 2])

            #################################################可修改######################################################
            scores, cam = model(img)
            scores = scores[0]
            logits = cam
            ############################################################################################################
            # 展平所有像素

            mask = F.interpolate(label.float().unsqueeze(0), size=cam.shape[2:], mode="area")[0]
            mask = mask.unsqueeze(0)

            if label.sum() > 0:
                pos = logits[mask > 0.5]
                neg = logits[mask < 0.1]
                all_fg_logits.extend(pos.detach().cpu().numpy().tolist())
                all_bg_logits.extend(neg.detach().cpu().numpy().tolist())

            norm_cam = torch.relu(cam)
            norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)

            cls_atten_map = F.interpolate(norm_cam.float(), size=ori_size, mode='bilinear')[0]  # 插值到原图大小20x224x224
            cls_atten_map = cls_atten_map.detach().cpu().numpy()
            pre_raw = cls_atten_map.reshape(ori_size).copy()
            pre = pre_raw.copy()

            img = F.interpolate(torch.clamp(img, 0, 1), ori_size, mode='bilinear', align_corners=True)
            img = img.mean(1)
            img = img.squeeze(0).squeeze(0).cpu().numpy()
            label = label.squeeze(0).squeeze(0).cpu().numpy()

            if torch.sigmoid(scores).item() < 0.5:
                pre = np.zeros(ori_size)

            if visual:

                cam = pre * 255
                ispre = False
                if ins == True and brats == False:
                    savepath = output_dir + "/" + model_name + "/ins/" + "vis/"
                elif ins == False and brats == True:
                    savepath = output_dir + "/" + model_name + "/brats/" + "vis/"
                else:
                    savepath = output_dir + "/" + model_name + "/bhsd/" + "vis/"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                vis_pre = (pre_raw >= best_threshold).astype(np.uint8)

                if brats == True:
                    if 0 < i_batch < 5000 and label.mean() > 0.01:
                        show_cam_on_image(img, cam, vis_pre, label, savepath, name=case,
                                          save_img=False, save_gt=False, cam_alpha=0.45, mask_alpha=0.55)
                else:
                    if 0 < i_batch < 6000 and label.sum() > 0:
                        show_cam_on_image(img, cam, vis_pre, label, savepath, name=case,
                                          save_img=False, save_gt=False, cam_alpha=0.45, mask_alpha=0.55)

            patient_id = case.split('_')[0]

            if i_batch == 0:
                viewed_id.append(patient_id)

            # print(pre.sum())
            if slice_num == 0 and i_batch != 0:

                pre_array = np.array(pre_array)
                label_array = np.array(label_array)

                for thresh in thresholds:

                    pred = (pre_array >= thresh).astype(np.uint8)

                    if thresh == best_threshold:
                        pred_vol = pred.sum()
                        pred_volume.append(pred_vol)
                        volume.append(label_array.sum())

                        sample_auc, _ = calc_auc(pre_array, label_array)
                        auc.append(sample_auc)
                        sample_auprc = calculate_auprc(pre_array, label_array)
                        auprc.append(sample_auprc)

                    dice, hd95, asd, iouu, senn, spee = calculate_metric(pred, label_array)

                    result_metric[thresh]['Dice'].append(dice * 100)
                    result_metric[thresh]['IoU'].append(iouu * 100)
                    result_metric[thresh]['HD95'].append(hd95)
                    result_metric[thresh]['Sen'].append(senn * 100)
                    result_metric[thresh]['Spe'].append(spee * 100)
                    print(f"Thresh {thresh} - patient {viewed_id[-1]}: dice: {dice * 100:.2f}, iou: {iouu * 100:.2f}")

                pre_array = []
                label_array = []
                viewed_id.append(patient_id)

            pre_array.append(pre.tolist())
            label_array.append(label.tolist())

        metrics_name = ["Dice", "IoU", "HD95", "Sen", "Spe"]

        # 1. 每个threshold平均指标
        if search_threshold:
            mean_results = []
            for th in thresholds:
                row = {"Threshold": th}
                for m in metrics_name:
                    row[m] = np.nanmean(np.asarray(result_metric[th][m]))
                mean_results.append(row)

            pd.DataFrame(mean_results).to_csv(output_dir + "/" + model_name + datapath + "threshold_mean_metrics.csv", index=False)

        # 2. 外部提供best_threshold
        best_metrics = result_metric[best_threshold]

        stats, box = [], {}

        for m in metrics_name:
            data = np.asarray(best_metrics[m], dtype=np.float32)
            data = data[~np.isnan(data)]

            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1

            stats.append([
                m,
                np.mean(data),
                np.std(data),
                np.median(data),
                q1,
                q3,
                iqr,
                max(q1 - 1.5 * iqr, 0),
                q3 + 1.5 * iqr,
            ])
            box[m] = data

        pd.DataFrame(
            stats,
            columns=["Metric", "Mean", "Std", "Median", "Q1", "Q3", "IQR", "Lower_whisker", "Upper_whisker"]
        ).to_csv(
            output_dir + "/" + model_name + datapath + "best_threshold_statistics.csv",
            index=False)

        pd.DataFrame(box).to_csv(
            output_dir + "/" + model_name + datapath + "best_threshold_boxplot.csv",
            index=False
        )

        import json

        logit_results = {"fore": all_fg_logits, "back": all_bg_logits}

        with open(output_dir + "/" + model_name + datapath + "all_logits.json", "w", encoding="utf-8") as f:
            json.dump(
                logit_results,
                f,
                ensure_ascii=False)

        overall, small, medium, large = calculate_rve(volume, pred_volume, vol_thresh)
        icc = calculate_icc(volume, pred_volume)
        mean_auc = np.mean(auc)
        mean_auprc = np.mean(auprc)
        lai = logits_adhesion_index(all_fg_logits, all_bg_logits)
        d_p = d_prime(all_fg_logits, all_bg_logits)
        bhatt_dis = bhattacharyya_distance(all_fg_logits, all_bg_logits)
        result_dict = {
            "Dice": round(float(stats[0][1]), 2),
            "IoU": round(float(stats[1][1]), 2),
            "HD95": round(float(stats[2][1]), 2),
            "Sen": round(float(stats[3][1]), 2),
            "Spe": round(float(stats[4][1]), 2),
            "AUC": round(float(mean_auc), 2),
            "AUPRC": round(float(mean_auprc), 2),
            "LAI": round(float(lai * 100), 2),
            "d_prime": round(float(d_p), 2),
            "bhatt_dis": round(float(bhatt_dis), 2),
            "ICC": round(float(icc), 2),
            "RVE_overall": round(float(overall), 2),
            "RVE_small": round(float(small), 2),
            "RVE_medium": round(float(medium), 2),
            "RVE_large": round(float(large), 2)}

        csv_path = output_dir + "/" + model_name + datapath + "/all_metrics.csv"

        from pathlib import Path

        model_path = Path(model_path)

        stem = model_path.stem
        if 'best_' in stem:
            metric = stem.split('best_')[-1]  # 'acc'
        else:
            metric = 'unknown'

        parts = model_path.parts
        experiment_name = (f"{parts[-4]}_{parts[-3]}_{metric}")
        print(experiment_name)

        save_metrics_csv(csv_path, experiment_name, result_dict)

        # labels, thresholds = split_volume_tertiles(volume)
        # print(thresholds)