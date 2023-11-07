from PIL import Image
import torch
import numpy as np
import os

METAFILE_DIR = "./dataset/metafiles/"
IMAGE_DIR = "./dataset/WildBees/"
GT_DIR = "./dataset/masks/gt/"

# get image id and its name
with open(METAFILE_DIR + "images.txt", "r") as f:
    lines = f.readlines()

img_names = []
for line in lines:
    parts = line.strip().split(" ")
    img_name = parts[1].strip().split(".")
    img_names.append(img_name[0])

f.close()


def get_imgNames():
    return img_names


def get_imgName(id: int):
    assert id >= 1 and id <= len(img_names), f"Wrong input for image id. Range should be within [1, {len(img_names)}]."
    return img_names[id - 1]


def get_imgID(name: str = None):
    try:
        ind = img_names.index(name)
        return ind + 1
    except ValueError:
        print("Image name " + name + " not found.")


def get_img(id: int = None, name: str = None, path=IMAGE_DIR, _gt=False, format="jpg", _asTensor=False):
    """Get images by id or name

    Params:
        id: int, id of the image
        name: str, name of the image
        path: str, path of the file
        _gt: bool, default=False: should load ground truth
        format: str, default="jpg" format of the source image
                ".pt" is also available

    Return:
        Image object of the image
    """
    img_name = name
    if img_name is None and id is not None:
        img_name = get_imgName(id)

    assert img_name is not None, "Please only give one of them: image id / image name"

    if path == IMAGE_DIR and _gt is True:
        path = GT_DIR

    img_path = path + img_name + "." + format

    # check if file exists
    assert os.path.exists(img_path), "File doesn't exist!"

    # load tensors
    if format == "pt" or format == "pth":
        return torch.load(img_path)

    if _gt:
        gt = Image.open(img_path).convert("1")
        if _asTensor:
            return torch.from_numpy(np.array(gt)).long()
        return gt

    return Image.open(img_path)
