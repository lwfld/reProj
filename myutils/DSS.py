import gc

# from collections import namedtuple

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.sparse.linalg import eigsh
from torch.utils.hooks import RemovableHandle
from torchvision import transforms


def _get_model(name: str):
    if "dino" in name:
        model = torch.hub.load("facebookresearch/dino:main", name)
        model.fc = torch.nn.Identity()
        val_transform = _get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    elif name in ["mocov3_vits16", "mocov3_vitb16"]:
        model = torch.hub.load("facebookresearch/dino:main", name.replace("mocov3", "dino"))
        checkpoint_file, size_char = {
            "mocov3_vits16": ("vit-s-300ep-timm-format.pth", "s"),
            "mocov3_vitb16": ("vit-b-300ep-timm-format.pth", "b"),
        }[name]
        url = f"https://dl.fbaipublicfiles.com/moco-v3/vit-{size_char}-300ep/vit-{size_char}-300ep.pth.tar"
        checkpoint = torch.hub.load_state_dict_from_url(url)
        model.load_state_dict(checkpoint["model"])
        model.fc = torch.nn.Identity()
        val_transform = _get_transform(name)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
    else:
        raise ValueError(f"Unsupported model: {name}")
    model = model.eval()
    return model, val_transform, patch_size, num_heads


def _get_transform(name: str):
    if any(
        x in name
        for x in (
            "dino",
            "mocov3",
            "convnext",
        )
    ):
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose(
            [
                transforms.Resize(size=512, interpolation=TF.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise NotImplementedError()
    return transform


def _get_diagonal(W: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    D = W.dot(np.ones(W.shape[1], W.dtype))
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D


# Cache
torch.cuda.empty_cache()

# Parameters
model_name = "dino_vitb16"  # TODO: Figure out how to make this user-editable

# Load model
model, val_transform, patch_size, num_heads = _get_model(model_name)

# Add hook
which_block = -1
if "dino" in model_name or "mocov3" in model_name:
    feat_out = {}

    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    handle: RemovableHandle = (
        model._modules["blocks"][which_block]
        ._modules["attn"]
        ._modules["qkv"]
        .register_forward_hook(hook_fn_forward_qkv)
    )
else:
    raise ValueError(model_name)


# GPU
if torch.cuda.is_available():
    print("CUDA is available, using GPU.")
    device = torch.device("cuda")
    model.to(device)
else:
    print("CUDA is not available, using CPU.")
    device = torch.device("cpu")


def get_segment(numEig=5, _todense=True):
    """Get segment function, with <input> amount of eigenvectors

    Params:
        numEig_default: int, number of eigenvectors to be generated

    Return:
        segment Function
    """

    @torch.no_grad()
    def segment(inp: Image, numEig=numEig, _todense=_todense):
        # Preprocess image
        images: torch.Tensor = val_transform(inp)
        images = images.unsqueeze(0).to(device)

        # Reshape image
        P = patch_size
        B, C, H, W = images.shape
        H_patch, W_patch = H // P, W // P
        H_pad, W_pad = H_patch * P, W_patch * P
        T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]

        # Crop image to be a multiple of the patch size
        images = images[:, :, :H_pad, :W_pad]

        # Extract features
        if "dino" in model_name or "mocov3" in model_name:
            model.get_intermediate_layers(images)[0].squeeze(0)
            output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            feats = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :].squeeze(0)
        else:
            raise ValueError(model_name)

        # Normalize features
        normalize = True
        if normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        # Compute affinity matrix
        W_feat = feats @ feats.T

        # Feature affinities
        threshold_at_zero = True
        if threshold_at_zero:
            W_feat = W_feat * (W_feat > 0)
        W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
        W_feat = W_feat.cpu().numpy()

        # Diagonal
        W_comb = W_feat

        if _todense:
            D_comb = np.array(_get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check-
        else:
            D_comb = _get_diagonal(W_comb)
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=(numEig + 1), sigma=0, which="LM", M=D_comb)
        except:
            eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=(numEig + 1), which="SM", M=D_comb)
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors.T).float()

        # print(eigenvectors[0].shape)

        # Resolve sign ambiguity
        for k in range(eigenvectors.shape[0]):
            if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
                eigenvectors[k] = 0 - eigenvectors[k]

        preds = []
        # eigenvectors_upscaled = []
        for i in range(1, numEig + 1):
            eigenvector = eigenvectors[i].reshape(1, 1, H_patch, W_patch)  # .reshape(1, 1, H_pad, W_pad)
            eigenvector: torch.Tensor = F.interpolate(
                # eigenvector, size=(H_pad, W_pad), mode="bilinear"
                eigenvector,
                size=(inp.size[1], inp.size[0]),
                mode="bicubic",
            )  # slightly off, but for visualizations this is okay
            preds.append(eigenvector.squeeze())

        # Garbage collection and other memory-related things
        gc.collect()
        # del eigenvector, eigenvector_vis, eigenvectors, W_comb, D_comb
        del eigenvector, eigenvectors, W_comb, D_comb

        # return output_images
        return preds

    return numEig, _todense, segment


@torch.no_grad()
def segment(inp: Image, numEig=5, _todence=True):
    """Get predictions from CLIPSeg model

    Params:
        inp: image to be segmented.
        numEig: int, number of eigenvectors to be generated

    Return:
        List of Tensors.
    """
    _, _, _segment = get_segment(numEig, _todence)
    return _segment(inp)
