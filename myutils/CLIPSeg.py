import torch

from clipseg.models.clipseg import CLIPDensePredT
from torchvision import transforms

# load model
model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load("./weights/rd64-uni-refined.pth", map_location=torch.device("cuda")), strict=False)
# model.load_state_dict(torch.load("./clipseg/weights/rd16-uni.pth", map_location=torch.device("cuda")), strict=False)
# model.load_state_dict(torch.load("./clipseg/weights/rd64-uni.pth", map_location=torch.device("cuda")), strict=False)
model.to(torch.device("cuda"))

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ]
)


def get_segment(prompts=None):
    """Get segment function, with setting prompts

    Params:
        prompts_default: Array, prompts to generate predictions

    Return:
        segment Function
    """
    if prompts is None:
        prompts = ["bee", "a bee", "bees", "an image of bee"]

    def _segment(
        input_image,
        prompts=prompts
        # input_image, prompts=["bee", "a photo of bee", "a photo of a bee", "a bright photo of a bee", "an image of bee"]
    ):
        img = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

        preds: torch.Tensor = torch.nn.functional.interpolate(
            # eigenvector, size=(H_pad, W_pad), mode="bilinear"
            preds,
            size=(input_image.size[1], input_image.size[0]),
            mode="bicubic",
        )
        return preds.squeeze(1).cpu()

    return len(prompts), _segment


def segment(
    input_image,
    prompts=["bee", "a bee", "bees", "an image of bee"]
    # input_image, prompts=["bee", "a photo of bee", "a photo of a bee", "a bright photo of a bee", "an image of bee"]
):
    """Get predictions from CLIPSeg model

    Params:
        input_image: image to be segmented.
        prompts: List, list of query words or reference phrases

    Return:
        List of Tensors.
    """
    _, _segment = get_segment(prompts)
    return _segment(input_image)
