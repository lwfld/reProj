from PIL import Image

METAFILE_DIR = "./dataset/metafiles/"
IMAGE_DIR = "./dataset/WildBees/"

# get image id and its name
with open(METAFILE_DIR + "images.txt", "r") as f:
    lines = f.readlines()

img_names = []
for line in lines:
    parts = line.strip().split(" ")
    img_names.append(parts[1])

f.close()


def get_imgNames():
    return img_names


def get_imgName(id: int):
    return img_names[id - 1]


def get_img_by_id(id: int):
    return Image.open(IMAGE_DIR + img_names[id - 1])
