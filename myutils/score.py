from ast import List
import time
import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryAveragePrecision
import pandas as pd
from tqdm import tqdm

from myutils import CLIPSeg, DSS
from myutils.loaders import get_img_by_id, get_img_by_name, get_img, get_imgName, get_imgNames
from myutils.processors import toBinary, toGreyscale

# Load ground truth
PREDS_PATH = "./dataset/preds/"
GT_PATH = "./dataset/masks/gt/"
RESULTS_PATH = "./results/"

# Load img_names
img_names = get_imgNames()

# generates threshold values for global thresholding
n_values = 51
threshold_values = np.linspace(0, 1, n_values)[1:-1]

# Torch metrics
metric_biJ = BinaryJaccardIndex()
metric_biAP = BinaryAveragePrecision()


# TASK
# Save DSS mIOU score, per image, per eigenvector, after found the best mIOU's threshold


def getIOU(pred, gt):
    return metric_biJ(pred, gt).item() * 100


def getBestIOU(pred=None, gt=None, gs_mode=1, model="DSS", pred_id=0, id: int = None, name: str = None):
    """Get best IOU value and its score for one prediction

    Params:
        id: single prediction out from model.
        name: single prediction out from model.

    Return:
        [best threshold, best score]
    """

    ious = []
    img_name = None

    if id is not None and name is None:
        img_name = get_imgName(id)

    elif name is not None and id is None:
        img_name = name

    elif pred is None or gt is None:
        # if img_name is None and (pred is None or gt is None):
        print("Wrong input. At least give pred+gt or id or name")
        return

    # Can get gt and predictions based on img_name, but if gt/pred is provided, prefer them
    if img_name is not None:
        if gt is None:
            # Load from gt's path
            gt = torch.from_numpy(np.array(get_img_by_name(img_name, GT_PATH).convert("1"))).long()
        if pred is None:
            # Load from prediction's path
            pred_path = PREDS_PATH + model + "/" + str(pred_id) + "/" + img_name + ".pt"
            pred = torch.load(pred_path)

    # convert pred to greyscale
    pred = toGreyscale(pred, method=gs_mode)

    # compare all threshold values from the preset
    for idx, t in enumerate(threshold_values):
        # get binary prediction
        bi = toBinary(pred, t)
        # calculate iou score
        iou = metric_biJ(bi, gt).item() * 100
        # iou = round(metric_biJ(pred, gt).item() * 100, 2)
        # print("loop", idx, ", threshold: ", t, ", score: ", iou)
        ious.append(iou)
    # print("Best threshold value: ", threshold_values[max_iou_idx], " with IOU score: ", ious[max_iou_idx])

    return threshold_values[np.argmax(ious)], np.max(ious)


def getBestIOUfromPreds(preds: List = None, gt=None, model="DSS", id: int = None, name: str = None):
    """Get best pred index from list, and IOU value and its score(preds_idx, iou_idx, iou_value)
    from list of predictions.

    Params:
        preds: list of predictions.

    Return:
        best prompt id, [best threshold, best score]
    """
    best_ious_val = []
    best_ious_thr = []

    img_name = None

    if id is not None and name is None:
        img_name = get_imgName(id)

    elif name is not None and id is None:
        img_name = name

    elif preds is None or gt is None:
        # if img_name is None and (pred is None or gt is None):
        print("Wrong input. At least give pred+gt or id or name")
        return

    # Can get gt and predictions based on img_name, but if gt/pred is provided, prefer them
    if img_name is not None:
        if gt is None:
            # Load from gt's path
            gt = torch.from_numpy(np.array(get_img_by_name(img_name, GT_PATH).convert("1"))).long()
        if preds is None:
            preds_tmp = []
            # Load from prediction's path
            for i in range(5):
                pred_path = PREDS_PATH + model + "/" + str(i) + "/" + img_name + ".pt"
                pred = torch.load(pred_path)
                preds_tmp.append(pred)
            preds = preds_tmp

    for idx in range(len(preds)):
        best_iou_thr, best_iou_val = getBestIOU(preds[idx], gt)
        best_ious_val.append(best_iou_val)
        best_ious_thr.append(best_iou_thr)

    max_pred_idx = np.argmax(best_ious_val)

    # print("Best threshold value: ", threshold_values[max_iou_idx], " with IOU score: ", ious[max_iou_idx])

    return max_pred_idx, np.array([best_ious_thr[max_pred_idx], best_ious_val[max_pred_idx]])
    # return np.array([max_pred_idx, best_ious_thr[max_pred_idx], best_ious_val[max_pred_idx]])


def getMeanIOU(model: int = 1, gs_mode=1, bi_mode=1, t=0.5, pred_idx=0, prob_thr=0):
    """Get best mean IOU threshold and its score

    Params:
        model: which model to be scored(1 | 2)
            1: CLIPSEG
            2. DSS
        prob_thr: threshold to classify if the score should be included.

    Return:
        miou score
    """
    if model == 1:
        segment = CLIPSeg.segment
    elif model == 2:
        segment = DSS.segment
    else:
        print("Wrong model number!")
        return None

    ious = []
    start_time = time.time()
    # calculate iou
    for i in tqdm(range(len(img_names)), desc="Calculating Mean IOU...", ncols=100):
        img_name = get_imgName(i + 1)
        source = get_img(id=i + 1)
        gt = torch.from_numpy(np.array(get_img_by_name(img_name, GT_PATH).convert("1"))).long()
        pred = segment(source)[pred_idx]
        pred_gs = toGreyscale(pred, gs_mode)
        if bi_mode == 1:
            pred_bi = toBinary(pred_gs, t=t, mode=bi_mode)
        elif bi_mode == 2:
            pred_bi = toBinary(pred_gs, t=t, mode=bi_mode)[1]
        iou = metric_biJ(pred_bi, gt).item() * 100
        if iou >= prob_thr:
            ious.append(iou)
        else:
            ious.append(0.0)
    end_time = time.time()
    miou = np.mean(ious)
    duration = end_time - start_time
    print(f"MIOU={miou:.2f}%, takes {duration:.2f}s in total, {(duration/len(img_names)):.2f}s per image.")
    return miou, duration


def getBestMeanIOU(model: int = 1, gs_mode=1, bi_mode=1, t=0.5, pred_idx=0, prob_thr=0):
    if model == 1:
        segment = CLIPSeg.segment
    elif model == 2:
        segment = DSS.segment
    else:
        print("Wrong model number!")
        return None

    ious = [[] for _ in range(len(threshold_values))]
    mious = []

    # calculate iou
    for i in tqdm(range(len(img_names)), desc="Calculating Mean IOU...", ncols=100):
        img_name = get_imgName(i + 1)
        source = get_img(id=i + 1)
        gt = torch.from_numpy(np.array(get_img_by_name(img_name, GT_PATH).convert("1"))).long()
        pred = segment(source)[pred_idx]
        pred_gs = toGreyscale(pred, gs_mode)
        # Get Binary Prediction Lists
        if bi_mode == 1:
            preds_bi = [toBinary(pred_gs, t=threshold_values[j], mode=bi_mode) for j in range(len(threshold_values))]
        elif bi_mode == 2:
            preds_bi = [toBinary(pred_gs, t=threshold_values[j], mode=bi_mode)[1] for j in range(len(threshold_values))]
        for j in range(len(threshold_values)):
            iou = metric_biJ(preds_bi[j], gt).item()
            iou = iou * 100 if iou > prob_thr * 100 else 0.0
            ious[j].append(iou)

    mious = [np.mean(ious[i]) for i in range(len(threshold_values))]

    return threshold_values[np.argmax(mious)], np.max(mious)


def saveCSV(t=0.5, bi_mode=1):
    data = {"img_id": [], "threshold": [], "eig_ind": [], "IOU": []}
    df = pd.DataFrame(data)
    for i in tqdm(range(len(img_names)), desc="Saving iou scores to file...", ncols=100):
        preds = DSS.segment(get_img(i + 1))
        gt = torch.from_numpy(np.array(get_img(i + 1, path=GT_PATH).convert("1"))).long()
        for j in range(len(preds)):
            pred_gs = toGreyscale(preds[j])
            pred_bi = toBinary(pred_gs, t, bi_mode)
            iou = metric_biJ(pred_bi, gt).item() * 100
            new_data = [{"img_id": i + 1, "threshold": t, "eig_ind": j, "IOU": iou}]
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv("./results/DSS.csv", index=False)
