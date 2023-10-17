from PIL import Image
import torch

METAFILE_DIR = "./dataset/metafiles/"
IMAGE_DIR = "./dataset/WildBees/"

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
    if id < 1 or id > len(img_names):
        print("Wrong input for image id. Range should be within [1, " + str(len(img_names)), +"].")
        return None
    return img_names[id - 1]


def get_imgID(name: str = None):
    try:
        ind = img_names.index(name)
        return ind + 1
    except ValueError:
        print("Image name " + name + " not found.")


def get_img_by_id(id: int, path=IMAGE_DIR, format="jpg"):
    if id < 1:
        print("Please enter a number greater than 0.")
        return None
    if format == "pt":
        return torch.load(path + img_names[id - 1] + ".pt")

    return Image.open(path + img_names[id - 1] + "." + format)


def get_img_by_name(name: str, path=IMAGE_DIR, format="jpg"):
    return Image.open(path + name + "." + format)


def get_img(id: int = None, name: str = None, path=IMAGE_DIR, format="jpg"):
    img_name = name
    if img_name is None and id is not None:
        img_name = get_imgName(id)
        img_path = path + img_name + "." + format

        if format == "pt":
            return torch.load(img_path)

        return Image.open(img_path)
    return None


# def generatePredPaths():
