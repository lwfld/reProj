import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(torch.device("cuda"))


def seg_clipseg(
    input_image, prompts=["bee", "a photo of bee", "a photo of a bee", "a bright photo of a bee", "an image of bee"]
):
    inputs = processor(
        text=prompts,
        images=[input_image] * len(prompts),
        padding="max_length",
        return_tensors="pt",
    )
    inputs = inputs.to(torch.device("cuda"))

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(input_image.size[1], input_image.size[0]),
        mode="bicubic",
    )

    return preds.squeeze(1).cpu()
