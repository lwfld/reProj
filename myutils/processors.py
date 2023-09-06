from PIL import Image
import os
import numpy
import torch

MASK_PATH = "../dataset/masks/"


def toBinary(pred, t: float):
    # min_vals, _ = torch.min(pred, dim=1, keepdim=True)
    # max_vals, _ = torch.max(pred, dim=1, keepdim=True)

    # pred = (pred - min_vals) / (max_vals - min_vals)
    # one = torch.ones_like(pred)
    # zero = torch.zeros_like(pred)
    # pred_bi = torch.where(pred < t, zero, one)
    # return pred_bi
    if type(pred) is numpy.ndarray:
        pred = torch.from_numpy(pred)
    pred = torch.sigmoid(pred)
    one = torch.ones_like(pred)
    zero = torch.zeros_like(pred)
    pred_bi = torch.where(pred < t, zero, one)
    return pred_bi


def toBinary_dss(pred):
    if type(pred) is numpy.ndarray:
        pred = torch.from_numpy(pred)
    pred = torch.sigmoid(pred)
    one = torch.ones_like(pred)
    zero = torch.zeros_like(pred)
    pred_bi = torch.where(pred is not True, zero, one)
    return pred_bi


def toBinaryPNG(pred, t: float, output_dir: str, file_name: str, override: bool = False):
    output_path = output_dir + file_name

    if override or not os.path.isfile(output_path):
        origin = toBinary(pred, t)
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
