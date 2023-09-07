from PIL import Image
import os
import numpy
import torch

MASK_PATH = "../dataset/masks/"


def norm(pred):
    min = pred.min()
    if min < 0:
        pred += abs(min)
        min = 0
    max = pred.max()
    dst = max - min
    normed_pred = (pred - min).true_divide(dst)
    return normed_pred


def toBinary(pred, t: float, _norm: bool = True):
    """Convert predictions to binary format

    Params:
        pred: prediction out from model.
        t: thredshold, float, from 0 to 1.
        _norm: bool, should normalize, default=True

    Return:
        Tensor consists of 0 and 1.
    """

    # Convert ndarray to tensor
    if type(pred) is numpy.ndarray:
        pred = torch.from_numpy(pred)

    # normalize the pred to [0, 1]
    pred = norm(pred)

    one = torch.ones_like(pred)
    zero = torch.zeros_like(pred)
    pred_bi = torch.where(pred < t, zero, one)
    return pred_bi


def savePic(bi_pred, output_dir: str, file_name: str, _override: bool = False):
    """Convert binary prediction as picture(single channel) in format of png

    Params:
        bi_pred: prediction in binary format
        output_dir: str, output directory
        file_name: str, output file name
        _override: bool, should override if file existed, default=False
    """
    output_path = output_dir + file_name

    if _override or not os.path.isfile(output_path):
        origin = bi_pred
        target = Image.new("1", (origin.shape[1], origin.shape[0]))
        pixels = target.load()
        for i in range(target.size[0]):
            for j in range(target.size[1]):
                pixels[i, j] = int(origin[j][i])

        target.save(output_path)


# def saveMask(t: float, sub_path, preds):
#     path = "./dataset/masks/" + sub_path + str(t) + "/"
#     output_dir = os.path.exists(path)
#     if not output_dir:
#         os.mkdir(path)

#     for pred, i in enumerate(preds):
#         convertPNG(pred, "name", path, t)
#         print("Progress: \t" + str((i + 1)) + "/" + str(len(preds)), end="\r")
#     print("FINISHED!")
