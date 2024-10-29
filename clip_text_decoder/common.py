from __future__ import annotations

from enum import Enum
import os
from typing import Callable, Optional, Tuple, Union
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
import clip
import torch
from clip.model import CLIP

from PIL import Image
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo


PreprocessorType = Callable[[Image.Image], Tensor]

V2_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".checkpoints",
    "videobind-v0.2.pth",
)


class LanguageModels(Enum):
    distilgpt2: str = "distilgpt2"
    gpt2: str = "gpt2"
    gpt2_medium: str = "gpt2-medium"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def check_language_model(name: str) -> None:
    allowed = LanguageModels.list()
    if name not in allowed:
        raise ValueError(
            f"Unsupported language model '{name}'. Allowed: {allowed}."
        )


def load_language_model(
    name: str, device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    check_language_model(name)
    config = GPT2Config.from_pretrained(name, add_cross_attention=True)
    model = GPT2LMHeadModel.from_pretrained(name, config=config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model


def load_tokenizer(name: str) -> GPT2Tokenizer:
    check_language_model(name)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class VisionBackbones(Enum):
    blip_base: str = "blip:base"
    clip_rn50: str = "clip:RN50"
    clip_rn101: str = "clip:RN101"
    clip_vit_b32: str = "clip:ViT-B/32"
    clip_vit_b16: str = "clip:ViT-B/16"
    clip_vit_l14: str = "clip:ViT-L/14"
    clip_vit_l14_336px: str = "clip:ViT-L/14@336px"
    imagebind: str = "imagebind"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def check_vision_backbone(backbone: str) -> None:
    allowed = VisionBackbones.list()
    if backbone not in allowed:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Allowed: {allowed}."
        )


def load_vision_backbone(
    backbone: str, device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, PreprocessorType]:
    check_vision_backbone(backbone)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone == VisionBackbones.imagebind.value:
        # we mast return
        #   model: nn.Module
        #   preprocessor: Callable[[Image.Image], Tensor]
        if not os.path.exists(V2_PATH):
            os.makedirs(os.path.dirname(V2_PATH), exist_ok=True)
            torch.hub.download_url_to_file(
                "https://huggingface.co/jondurbin/videobind-v0.2/resolve/main/videobind.pth",
                V2_PATH,
                progress=True,
            )
        imagebind = torch.load(V2_PATH)
        imagebind.eval()
        imagebind.to(device)
        return imagebind, load_and_transform_vision_data

    else:
        # Currently, all other supported backbones are CLIP
        _, name = backbone.split(":")
        return clip.load(name, device=device, jit=False)


def encode_image_tensor(image: Tensor, backbone: nn.Module) -> Tensor:

    if isinstance(backbone, ImageBindModel):
        inputs = {
            ModalityType.VISION: image,
        }
        embeddings = backbone(inputs)

        return embeddings
    else:
        # Currently, all other supported backbones are CLIP
        assert isinstance(backbone, CLIP)
        return backbone.encode_image(image)


def load_and_transform_vision_data(image: Image.Image):

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = image.convert("RGB")
    image = data_transform(image)
    image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)
