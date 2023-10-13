from PIL import Image

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
    if id < 1:
        print("Please enter a number greater than 0.")
        return None
    return img_names[id - 1]


def get_img_by_id(id: int, path=IMAGE_DIR, format="jpg"):
    if id < 1:
        print("Please enter a number greater than 0.")
        return None
    return Image.open(path + img_names[id - 1] + "." + format)


def get_img_by_name(name: str, path=IMAGE_DIR, format="jpg"):
    return Image.open(path + name + "." + format)
