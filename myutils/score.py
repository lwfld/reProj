from ast import List
import time
import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryAveragePrecision
import pandas as pd
from tqdm import tqdm

from myutils import CLIPSeg, DSS
from myutils.loaders import get_img, get_imgName, get_imgNames
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

    if gt is not None:
        gt = torch.from_numpy(np.array(gt.convert("1"))).long()

    # Can get gt and predictions based on img_name, but if gt/pred is provided, prefer them
    if img_name is not None:
        if gt is None:
            # Load from gt's path
            gt = torch.from_numpy(np.array(get_img(name=img_name, _gt=True).convert("1"))).long()
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
        preds: Optional, list of predictions.
        gt: ground truth of the input image
        model: model used to make predictions, default=1
            1: CLIPSeg
            2: DSS
        id: Optional, id of the image
        name: Optional, name of the image

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
            gt = torch.from_numpy(np.array(get_img(name=img_name, _gt=True).convert("1"))).long()
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


def getBestPred_(preds, gt, metric_=None):
    """Get best score and its idx from a set of prediction
       The input preds should already be its corresponding format(binary or greyscale)

    Params:
        preds:
        gt:
        metric: which metric to be used(1 | 2)
            1: IOU
            2. AP

    Return:
        score, idx
    """
    assert metric_ is not None, "Please give a metric"

    scores = [metric_(pred, gt).item() * 100 for pred in preds]
    return np.max(scores), np.argmax(scores)


def getIOUPart(segment_=None, gs_mode=1, bi_mode=1, t=0.5, pred_idx=None, range_=None):
    """Get Mean IOU score and its running time

    Params:
        segment_: model's segment function
        gs_mode, default=1: greyscale mode
        bi_mode, default=1: binarilization mode
        t: threshold when bi_mode==1
        pred_idx: choose exactly one pred from predictions

    Return:
        miou score, duration
    """
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    ious = []
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean IOU...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryJaccardIndex()

        # Otsu's method
        if bi_mode == 2:
            preds_bi = [toBinary(pred_gs, mode=2)[1] for pred_gs in preds_gs]
        elif bi_mode == 1:
            preds_bi = [toBinary(pred_gs, t, bi_mode) for pred_gs in preds_gs]

        best_iou, idx = getBestPred_(preds_bi, gt, metric_)
        ious.append(best_iou)

    return ious


def getAPPart(segment_=None, gs_mode=1, pred_idx=None, range_=None, t=None):
    """Get AP score based on prediction and ground truth
    input can either be single or a list

    Params:
        segment_:
        gs_mode:
        pred_idx:
        range_:
        t: confidence threshold

    Return:
        AP score, if input is a list, also returns the index
    """
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    aps = []
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean AP...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryAveragePrecision(thresholds=t)

        best_ap, idx = getBestPred_(preds_gs, gt, metric_)
        aps.append(best_ap)

    return aps


def getMeanIOU(segment_=None, gs_mode=1, bi_mode=1, t=0.5, pred_idx=None, range_=None):
    """Get Mean IOU score and its running time

    Params:
        segment_: model's segment function
        gs_mode, default=1: greyscale mode
        bi_mode, default=1: binarilization mode
        t: threshold when bi_mode==1
        pred_idx: choose exactly one pred from predictions

    Return:
        miou score, duration
    """
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    ious = []
    start_time = time.time()
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean IOU...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryJaccardIndex()

        # Otsu's method
        if bi_mode == 2:
            preds_bi = [toBinary(pred_gs, mode=2)[1] for pred_gs in preds_gs]
        elif bi_mode == 1:
            preds_bi = [toBinary(pred_gs, t, 1) for pred_gs in preds_gs]

        best_iou, idx = getBestPred_(preds_bi, gt, metric_)
        ious.append(best_iou)

    end_time = time.time()
    miou = np.mean(ious)
    duration = end_time - start_time
    print(f"MIOU={miou:.2f}%, takes {duration:.2f}s in total, {(duration/range_):.2f}s per image.")
    return miou, duration


def getBestMeanIOU_ng(segment_=None, gs_mode=1, pred_idx=None, range_=None):
    # getBestMeanIOU(non global threshold version)
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    ious = []

    start_time = time.time()
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean IOU...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryJaccardIndex()

        # Get Binary Prediction Lists
        preds_bi = [[toBinary(pred_gs, t) for pred_gs in preds_gs] for t in threshold_values]

        ious_t = [getBestPred_(pred_bi, gt, metric_)[0] for pred_bi in preds_bi]

        ious.append(np.max(ious_t))

    end_time = time.time()

    miou = np.mean(ious)
    duration = end_time - start_time

    print(f"MIOU={miou:.2f}%, takes {duration:.2f}s in total, {(duration/range_):.2f}s per image.")
    return miou, duration


def getBestMeanIOU(segment_=None, gs_mode=1, pred_idx=None, range_=None):
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    ious = [[] for _ in range(len(threshold_values))]

    start_time = time.time()
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean IOU...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryJaccardIndex()

        # Get Binary Prediction Lists
        preds_bi = [[toBinary(pred_gs, t) for pred_gs in preds_gs] for t in threshold_values]

        for j in range(len(threshold_values)):
            best_iou, idx = getBestPred_(preds_bi[j], gt, metric_)
            ious[j].append(best_iou)

    end_time = time.time()

    mious = [np.mean(ious[i]) for i in range(len(threshold_values))]
    duration = end_time - start_time

    # results
    best_threshold = threshold_values[np.argmax(mious)]
    best_miou = np.max(mious)

    print(
        f"MIOU={best_miou:.2f}% at t={best_threshold}, takes {duration:.2f}s in total, {(duration/range_):.2f}s per image."
    )
    return best_threshold, best_miou, duration


def getMeanAP(segment_=None, gs_mode=1, pred_idx=None, range_=None, t=None):
    """Get AP score based on prediction and ground truth
    input can either be single or a list

    Params:
        segment_:
        gs_mode:
        pred_idx:
        range_:
        t: confidence threshold

    Return:
        AP score, if input is a list, also returns the index
    """
    assert segment_ is not None, "Please give the segment function"

    if range_ is None:
        range_ = len(img_names)

    aps = []
    start_time = time.time()
    # calculate iou
    for i in tqdm(range(range_), desc="Calculating Mean AP...", ncols=100):
        source = get_img(i + 1)
        gt = get_img(i + 1, _gt=True, _asTensor=True)
        # get predictions
        preds = segment_(source)

        # Specifically want one pred
        if pred_idx is not None:
            preds = [preds[pred_idx]]

        # convert to grey scale values
        preds_gs = [toGreyscale(pred, gs_mode) for pred in preds]

        metric_ = BinaryAveragePrecision(thresholds=t)

        best_iou, idx = getBestPred_(preds_gs, gt, metric_)
        aps.append(best_iou)

    end_time = time.time()
    map = np.mean(aps)
    duration = end_time - start_time
    print(f"mAP={map:.2f}%, takes {duration:.2f}s in total, {(duration/range_):.2f}s per image.")
    return map, duration


def getBestMeanIOUfromPreds(model: int = 1, gs_mode=1, bi_mode=1, prob_thr=0, prompts=None, numEigs=5):
    if model == 1:
        preds_length, segment = CLIPSeg.get_segment(prompts)
    elif model == 2:
        preds_length, segment = DSS.get_segment(numEigs)
    else:
        print("Wrong model number!")
        return None

    ious = [[[] for _ in range(len(threshold_values))] for _ in range(preds_length)]
    mious = [[] for _ in range(preds_length)]

    for i in tqdm(range(len(img_names)), desc="Calculating Mean IOU...", ncols=100):
        img_name = get_imgName(i + 1)
        source = get_img(id=i + 1)
        gt = torch.from_numpy(np.array(get_img(name=img_name, _gt=True).convert("1"))).long()
        preds = segment(source)
        pred_gss = [toGreyscale(preds[j], gs_mode) for j in range(preds_length)]
        # Get Binary Prediction Lists
        if bi_mode == 1:
            preds_bi = [
                [toBinary(pred_gss[k], t=threshold_values[j], mode=bi_mode) for j in range(len(threshold_values))]
                for k in range(preds_length)
            ]
        elif bi_mode == 2:
            preds_bi = [
                [toBinary(pred_gss[k], t=threshold_values[j], mode=bi_mode)[1] for j in range(len(threshold_values))]
                for k in range(preds_length)
            ]
        for j in range(len(threshold_values)):
            for k in range(preds_length):
                iou = metric_biJ(preds_bi[k][j], gt).item()
                iou = iou * 100 if iou > prob_thr * 100 else 0.0
                ious[k][j].append(iou)

    mious = [[np.mean(ious[k][i]) for i in range(len(threshold_values))] for k in range(preds_length)]
    best_miou_vals = [np.max(mious[k]) for k in range(preds_length)]
    best_pred_idx = np.argmax(best_miou_vals)
    best_ious = mious[best_pred_idx]

    return best_pred_idx, np.array([threshold_values[np.argmax(best_ious)], np.max(best_ious)])


def saveCSV(file_name, t=0.5, bi_mode=1):
    data = {"img_id": [], "threshold": [], "eig_ind": [], "IOU": []}
    df = pd.DataFrame(data)
    file_path = RESULTS_PATH + file_name + ".csv"
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
    df.to_csv(file_path, index=False)


# def getMeanIOU(model: int = 1, gs_mode=1, bi_mode=1, t=0.5, pred_idx=0, prob_thr=0):
#     """Get best mean IOU threshold and its score

#     Params:
#         model: which model to be scored(1 | 2)
#             1: CLIPSEG
#             2. DSS
#         prob_thr: threshold to classify if the score should be included.

#     Return:
#         miou score
#     """
#     if model == 1:
#         segment = CLIPSeg.segment
#     elif model == 2:
#         segment = DSS.segment
#     else:
#         print("Wrong model number!")
#         return None

#     ious = []
#     start_time = time.time()
#     # calculate iou
#     for i in tqdm(range(len(img_names)), desc="Calculating Mean IOU...", ncols=100):
#         img_name = get_imgName(i + 1)
#         source = get_img(id=i + 1)
#         gt = torch.from_numpy(np.array(get_img(name=img_name, _gt=True).convert("1"))).long()
#         pred = segment(source)[pred_idx]
#         pred_gs = toGreyscale(pred, gs_mode)
#         if bi_mode == 1:
#             pred_bi = toBinary(pred_gs, t=t, mode=bi_mode)
#         elif bi_mode == 2:
#             pred_bi = toBinary(pred_gs, t=t, mode=bi_mode)[1]
#         iou = metric_biJ(pred_bi, gt).item() * 100
#         if iou >= prob_thr:
#             ious.append(iou)
#         else:
#             ious.append(0.0)
#     end_time = time.time()
#     miou = np.mean(ious)
#     duration = end_time - start_time
#     print(f"MIOU={miou:.2f}%, takes {duration:.2f}s in total, {(duration/len(img_names)):.2f}s per image.")
#     return miou, duration
