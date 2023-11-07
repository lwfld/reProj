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


def toTensor(img, _bi=False):
    if _bi:
        return torch.from_numpy(np.array(img.convert("1"))).long()
    return torch.from_numpy(np.array(img.convert("L"))).long()


def toGreyscale(pred, method=1, _gs=False):
    """Convert predictions to grey scale format

    Params:
        pred: prediction out from model.
        method: int, should normalize, 1|2
            1: default, minmax normalization
            2: sigmoid
            3: direct_pass, no modification
        _gs: bool dafault = False, should the scaled value map to (0-255) greyscale values
             if set to False, remain(0-1)

    Return:
        Tensor consists of 0 and 1.
    """
    assert isinstance(pred, torch.Tensor), "Expected input to be a PyTorch Tensor"

    # use min max values to map predictions to (0, 255)
    if method == 1:
        min = pred.min()
        max = pred.max()
        pred = (pred - min).true_divide(max - min)

    # use sigmoid to map ~
    elif method == 2:
        pred = torch.sigmoid(pred)
    elif method == 3:
        return pred
    else:
        print("wrong mode.")
        return pred

    # return int(ratio * 255)
    if _gs:
        pred = torch.round(pred * 255).long()
    return pred


def toBinary(greyscale, t=0.5, mode=1):
    """Convert predictions to binary format

    Params:
        pred: prediction(normalized) out from model(range[0, 1]), or greyscale format prediction(range[0, 255]).
        t: thredshold, float, from 0 to 1.
        mode: threhold method, 1 | 2.
            1: global thresholding
            2: otsu's method,

    Return: binary mask, if mode==2, also threshold values(unpack first)

    Return:
        Tensor consists of 0 and 1.
    """
    assert isinstance(greyscale, torch.Tensor), "Expected input to be a PyTorch Tensor"

    # Global threshold
    if mode == 1:
        scale_factor = 1
        if greyscale.max() > 1:
            scale_factor = 256
        one = torch.ones_like(greyscale, dtype=torch.uint8)
        zero = torch.zeros_like(greyscale, dtype=torch.uint8)
        bi = torch.where(greyscale < t * scale_factor, zero, one)
        return bi

    # Otsu's method
    elif mode == 2:
        scale_factor = 1
        if greyscale.max() <= 1:
            scale_factor = 256
        thr, bi = otsu_thresholding(greyscale * scale_factor)
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
