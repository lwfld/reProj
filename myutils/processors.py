from PIL import Image
import os
import numpy as np
import torch
import cv2

MASK_PATH = "../dataset/masks/"


def otsu_thresholding(img_gs):
    # Convert tensor to array
    img_np = img_gs.numpy().astype(np.uint8)

    threshold_val, img_thresh = cv2.threshold(img_np, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold_val, torch.from_numpy(img_thresh).long()


def toGreyscale(pred, method=1):
    """Convert predictions to grey scale format

    Params:
        pred: prediction out from model.
        method: int, should normalize, 1|2
            1: default, minmax normalization
            2: sigmoid

    Return:
        Tensor consists of 0 and 1.
    """
    # Convert ndarray to tensor
    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)

    # use min max values to map predictions to (0, 255)
    if method == 1:
        min = pred.min()
        max = pred.max()
        pred = (pred - min).true_divide(max - min)

    # use sigmoid to map ~
    elif method == 2:
        pred = torch.sigmoid(pred)
    else:
        print("wrong mode.")
        return pred

    # return int(ratio * 255)
    return torch.round(pred * 255).long()


def toBinary(greyscale, t=0.5, mode=1):
    """Convert predictions to binary format

    Params:
        pred: prediction out from model.
        t: thredshold, float, from 0 to 1.
        mode: threhold method, 1 | 2, default: global

    Return:
        Tensor consists of 0 and 1.
    """
    assert isinstance(greyscale, torch.Tensor), "Expected input to be a PyTorch Tensor"

    # Global threshold
    if mode == 1:
        one = torch.ones_like(greyscale)
        zero = torch.zeros_like(greyscale)
        bi = torch.where(greyscale < t * 256, zero, one)
        return bi

    # Otsu's method
    elif mode == 2:
        thr, bi = otsu_thresholding(greyscale)
        return thr, bi

    else:
        return greyscale


def savePic(origin: np.array, output_dir: str, file_name: str, _override: bool = False):
    """Convert binary prediction as picture(single channel) in format of png

    Params:
        origin: prediction in array format
        output_dir: str, output directory
        file_name: str, output file name
        _override: bool, should override if file existed, default=False
    """
    output_path = output_dir + file_name

    if not os.path.isfile(output_path) or _override:
        img = Image.fromarray(origin.astype(np.uint8), mode="L")
        img.save(output_path)


# def savePic(origin: np.array, output_dir: str, file_name: str, _override: bool = False):
#     """Convert binary prediction as picture(single channel) in format of png

#     Params:
#         origin: prediction in array format
#         output_dir: str, output directory
#         file_name: str, output file name
#         _override: bool, should override if file existed, default=False
#     """
#     output_path = output_dir + file_name

#     if not os.path.isfile(output_path) or _override:
#         target = Image.new("L", (origin.shape[1], origin.shape[0]))
#         pixels = target.load()
#         for i in range(target.size[0]):
#             for j in range(target.size[1]):
#                 pixels[i, j] = origin[j][i].item()

#         target.save(output_path)
