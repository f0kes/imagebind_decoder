from __future__ import annotations

from enum import Enum
import os
from typing import Callable, List, Optional, Tuple, Union
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
import numpy as np
import clip
import torch
from clip.model import CLIP

from PIL import Image
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from imagebind.models.multimodal_preprocessors import (
    SimpleTokenizer,
    TextPreprocessor,
)

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
        print(f"Loaded {backbone} model from {V2_PATH}")
        return imagebind, load_and_transform_vision_data

    else:
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
        print(f"Loaded {backbone} model from {V2_PATH}")
        return imagebind, load_and_transform_vision_data


def encode_image_tensor(image: Tensor, backbone: nn.Module) -> Tensor:
    print(f"Encoding image with {backbone.__class__.__name__}...")
    if isinstance(backbone, ImageBindModel):
        inputs = {
            ModalityType.VISION: image,
        }
        embeddings = backbone(inputs)
        print(f"embedding shape: {embeddings[ModalityType.VISION].shape}")

        return embeddings[ModalityType.VISION]
    else:
        # Currently, all other supported backbones are CLIP
        inputs = {
            ModalityType.VISION: image,
        }
        embeddings = backbone(inputs)
        print(f"embedding shape: {embeddings[ModalityType.VISION].shape}")

        return embeddings[ModalityType.VISION]


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


BPE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bpe",
    "bpe_simple_vocab_16e6.txt.gz",
)

PRETRAINED_INFERENCE_MODEL_PATH = (
    "https://drive.google.com/uc?id=1bEAyV2279C4V4iYMaJahREiM58vjy6G1"
    # https://drive.google.com/file/d/1bEAyV2279C4V4iYMaJahREiM58vjy6G1/view?usp=sharing
)

TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH)
LENGTH_TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH, context_length=1024)
TOKEN_CHUNK_SIZE = 74


def split_text_by_token_limit(text, tokenizer, max_tokens=TOKEN_CHUNK_SIZE):
    def fits_in_token_limit(text_segment):
        tokens = tokenizer(text_segment)
        tokens = tokens[tokens != 0][1:-1].tolist()
        return len(tokens) <= max_tokens

    def recursive_split(text, delimiters):
        if fits_in_token_limit(text):
            return [text]
        if not delimiters:
            return split_by_tokens(text)
        delimiter = delimiters[0]
        parts = text.split(delimiter)
        result = []
        current_segment = ""
        for part in parts:
            candidate_segment = (
                current_segment + (delimiter if current_segment else "") + part
            )
            if fits_in_token_limit(candidate_segment):
                current_segment = candidate_segment
            else:
                if current_segment:
                    result.append(current_segment)
                current_segment = part
        if current_segment:
            result.append(current_segment)
        final_result = []
        for segment in result:
            if fits_in_token_limit(segment):
                final_result.append(segment)
            else:
                final_result.extend(recursive_split(segment, delimiters[1:]))
        return final_result

    def split_by_tokens(text):
        tokens = tokenizer(text)
        tokens = tokens[tokens != 0][1:-1].tolist()
        chunks = np.array_split(tokens, int(len(tokens) / max_tokens) or 1)
        return [tokenizer.decode(segment_tokens) for segment_tokens in chunks]

    return recursive_split(text, ["\n", ".", "!", "?", ",", " "])


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokens = [TOKENIZER(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_text_chunks(text, device):
    if not text:
        return []
    all_tokens = LENGTH_TOKENIZER(text)
    all_tokens = all_tokens[all_tokens != 0][1:-1].tolist()

    return [
        load_and_transform_text([segment], device)
        for segment in split_text_by_token_limit(text, LENGTH_TOKENIZER)
    ]


def generate_text_embeddings(
    text: str, imagebind: ImageBindModel, device="cuda"
):
    chunks = load_and_transform_text_chunks(text, device)
    embeddings = [
        imagebind({ModalityType.TEXT: chunk})[ModalityType.TEXT]
        for chunk in chunks
    ]
    return torch.mean(torch.stack(embeddings), dim=0)


def generate_text_embeddings_batch(
    texts: List[str],
    imagebind: ImageBindModel,
    device="cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Generate embeddings for a batch of texts efficiently.
    Args:
        texts: List of text strings to process
        imagebind: ImageBind model instance
        device: Device to process on
        batch_size: Size of internal processing batches
    Returns:
        Tensor of shape (len(texts), embedding_dim) containing mean embeddings
    """
    all_embeddings = []
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Transform all texts in current batch
        batch_chunks = [
            load_and_transform_text_chunks(text, device) for text in batch_texts
        ]
        # Flatten chunks to process them all at once
        flat_chunks = [chunk for chunks in batch_chunks for chunk in chunks]

        if flat_chunks:  # Check if there are any chunks to process
            # Stack chunks into a single tensor
            stacked_chunks = torch.stack(flat_chunks, dim=0)

            with torch.no_grad():  # No need for gradients during embedding generation
                batch_embeddings = imagebind(
                    {ModalityType.TEXT: stacked_chunks}
                )[ModalityType.TEXT]

            # Reshape and mean over chunks for each text
            chunk_idx = 0
            for chunks in batch_chunks:
                num_chunks = len(chunks)
                if num_chunks > 0:
                    text_embeddings = batch_embeddings[
                        chunk_idx : chunk_idx + num_chunks
                    ]
                    mean_embedding = torch.mean(
                        text_embeddings, dim=0, keepdim=True
                    )
                    all_embeddings.append(mean_embedding)
                    chunk_idx += num_chunks
        else:
            # Handle empty chunks case
            embedding_dim = (
                imagebind.embedding_dim
            )  # Adjust this based on your model
            all_embeddings.append(
                torch.zeros((1, embedding_dim), device=device)
            )

    # Stack all embeddings
    return torch.cat(all_embeddings, dim=0)
