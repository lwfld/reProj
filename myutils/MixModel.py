from math import comb
from PIL import Image
from myutils.processors import toGreyscale, toBinary
import numpy as np
from myutils import CLIPSeg, DSS
from myutils.score import getBestPred_
from torchmetrics.classification import BinaryAveragePrecision
import torch

prompt = "insect shaped"
_, clipseg_cls = CLIPSeg.get_segment([prompt])
_, _, dss = DSS.get_segment()


def combPreds(preds):
    comb_pred = torch.zeros_like(preds[0])
    for i in range(len(preds)):
        comb_pred = comb_pred + preds[i]
    return comb_pred


def PredstoImage(preds_dss, _toBinary=False):
    comb_pred = preds_dss
    if isinstance(preds_dss, list):
        comb_pred = combPreds(preds_dss)
    pred = toGreyscale(comb_pred, _gs=True)

    if _toBinary:
        pred = toBinary(pred, 0.32) * 255

    image = Image.fromarray(pred.to(torch.uint8).numpy(), mode="L").convert("RGB")
    return image


def getBestIdx(preds_dss, pred_clipseg):
    metric_ = BinaryAveragePrecision()
    _, idx = getBestPred_(preds_dss, toBinary(toGreyscale(pred_clipseg), 0.78), metric_)

    return idx


def getBBoxCords(bi_pred):
    # 获取非零元素的坐标
    nonzero_coords = torch.nonzero(bi_pred)

    # 计算最小矩形的坐标
    bias = 10
    left = max(nonzero_coords[:, 1].min().item() - bias, 0)
    upper = max(nonzero_coords[:, 0].min().item() + bias, 0)
    right = min(nonzero_coords[:, 1].max().item() + bias, bi_pred.shape[1] - 1)
    lower = min(nonzero_coords[:, 0].max().item() - bias, bi_pred.shape[0] - 1)

    return left, upper, right, lower


def DSS2CLIPSeg(input):
    preds_dss = DSS.segment(input)
    pred_clipseg = clipseg_cls(input)
    idx = getBestIdx(preds_dss, pred_clipseg[0])
    return idx, preds_dss[idx]


def DSSPreds2CLIPSeg(input):
    preds_dss = DSS.segment(input)
    pred_comb = PredstoImage(preds_dss)
    pred_clipseg = clipseg_cls(pred_comb)
    return pred_clipseg


def CLIPSeg2DSS(input):
    clip_pred = clipseg_cls(input)[0]
    bi_pred = toBinary(toGreyscale(clip_pred), t=0.78)
    left, upper, right, lower = getBBoxCords(bi_pred)

    cropped_region = input.crop((left, upper, right, lower))

    pred_dss = DSS.segment(cropped_region)[0]
    pred_gs = toGreyscale(pred_dss, _gs=True)

    result_tensor = torch.zeros_like(bi_pred)
    result_tensor[upper:lower, left:right] = pred_gs

    return result_tensor


def getMixModel(type=1):
    if type == 1:

        def DSS2CLIPSeg(input):
            # print(prompt)
            preds_dss = DSS.segment(input)
            pred_clipseg = clipseg_cls(input)
            idx = getBestIdx(preds_dss, pred_clipseg[0])
            return [preds_dss[idx]]

        return DSS2CLIPSeg

    elif type == 2:

        def DSSPreds2CLIPSeg(input):
            preds_dss = DSS.segment(input)
            pred_comb = PredstoImage(preds_dss)
            pred_clipseg = clipseg_cls(pred_comb)
            return pred_clipseg

        return DSSPreds2CLIPSeg

    elif type == 3:

        def CLIPSeg2DSS(input):
            clip_pred = clipseg_cls(input)[0]
            bi_pred = toBinary(toGreyscale(clip_pred), t=0.78)
            left, upper, right, lower = getBBoxCords(bi_pred)

            cropped_region = input.crop((left, upper, right, lower))

            pred_dss = DSS.segment(cropped_region)[0]
            pred_gs = toGreyscale(pred_dss, _gs=True)

            result_tensor = torch.zeros_like(bi_pred)
            result_tensor[upper:lower, left:right] = pred_gs

            return [result_tensor.to(torch.float)]

        return CLIPSeg2DSS
