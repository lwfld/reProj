import numpy as np
import torch
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryJaccardIndex, BinaryAveragePrecision
import pandas as pd

from myutils.img_loader import get_img_by_name, get_imgNames

# Load ground truth
PREDS_PATH = "../dataset/preds/"
GT_PATH = "../dataset/masks/gt/"
CLIP_PATH = PREDS_PATH + "clipseg/"
DSS_PATH = PREDS_PATH + "dss/"
RESULTS_PATH = "../results/"
PREDS_LENGTH = 5

toTensor = transforms.ToTensor()

# Load img_names
img_names = get_imgNames()

# generates threshold values for global thresholding
n_values = 51
threshold_values = np.linspace(0, 1, n_values)[1:-1]

# Torch metrics
metric_biJ = BinaryJaccardIndex()
metric_biAP = BinaryAveragePrecision()

# CLIPSEG
for name in img_names:
    gt = torch.from_numpy(np.array(get_img_by_name(name, GT_PATH).convert('1'))).long()

    for i in range(PREDS_LENGTH):
        path = CLIP_PATH + str(i) + name + ".pt"
        pred = torch.load(path)
        for t in threshold_values:
            iou = metric_biJ(pred, gt.long())
            ap = metric_biAP(pred, gt.long())
