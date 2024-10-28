from __future__ import annotations
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from clip_text_decoder.common import load_tokenizer
from functools import lru_cache
import os
import tempfile
from typing import Callable, List, Optional, Tuple, Union
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from imagebind.models.multimodal_preprocessors import (
    SimpleTokenizer,
    TextPreprocessor,
)

import gdown
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
from transformers import GPT2Tokenizer
from imagebind import data

from clip_text_decoder.common import (
    check_language_model,
    check_vision_backbone,
    encode_image_tensor,
    load_language_model,
    load_vision_backbone,
)

BPE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bpe",
    "bpe_simple_vocab_16e6.txt.gz",
)

TOKENIZER = SimpleTokenizer(bpe_path=BPE_PATH)


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokens = [TOKENIZER(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


class DecoderEmbedding(LightningModule):
    def __init__(
        self,
        tokeniser: GPT2Tokenizer,
        vision_backbone: str = "blip:base",
        language_model: str = "distilgpt2",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save_hyperparameters()
        check_vision_backbone(vision_backbone)
        self.vision_backbone = load_vision_backbone(
            vision_backbone, device=device
        )

        check_language_model(language_model)
        self.language_model = load_language_model(language_model, device=device)
        self.tokenizer = tokeniser
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.to(device)

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ):
        target_dim = 768
        batch_size, _, num_features = encoder_hidden_states.shape
        hidden = torch.zeros(
            size=(batch_size, 1, target_dim),
            dtype=encoder_hidden_states.dtype,
            device=encoder_hidden_states.device,
        )
        hidden = encoder_hidden_states[:, :, :target_dim]

        # Generate text
        outputs = self.language_model(
            input_ids=input_ids,
            encoder_hidden_states=hidden,
            attention_mask=attention_mask,
        )

        # Get the most likely tokens
        generated_ids = torch.argmax(outputs.logits, dim=-1)

        # Convert ids to text and get CLIP embeddings
        generated_text = [
            " ".join(
                self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            )
        ]
        if isinstance(self.vision_backbone[0], ImageBindModel):
            inputs = {
                ModalityType.TEXT: load_and_transform_text(
                    generated_text, self.device
                ),
            }
            embeddings = self.vision_backbone[0].embed_text([generated_text])

            return embeddings[ModalityType.TEXT], outputs.logits
        else:
            text_features = self.vision_backbone.encode_text(generated_text)
            return text_features, outputs.logits

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], *_) -> Tensor:
        encoder_hidden_states, input_ids, attention_mask = batch

        # Get result embedding from forward pass
        result_embedding, _ = self.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # Calculate cosine similarity loss
        # encoder_hidden_states is the target embedding
        target = torch.ones(
            result_embedding.size(0), device=self.device
        )  # target=1 means maximize similarity
        loss = self.cosine_loss(result_embedding, encoder_hidden_states, target)

        self.log(
            "training_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], *_
    ) -> Tensor:
        encoder_hidden_states, input_ids, attention_mask = batch

        # Get result embedding from forward pass
        result_embedding, _ = self.forward(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # Calculate cosine similarity loss
        target = torch.ones(result_embedding.size(0), device=self.device)
        loss = self.cosine_loss(result_embedding, encoder_hidden_states, target)

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss


def test_decoder():
    # 1. Create minimal test data

    get_tokenizer = lru_cache()(load_tokenizer)
    batch_size = 2
    seq_length = 10
    hidden_dim = 1024

    # Create dummy data
    encoder_hidden_states = torch.randn(batch_size, 1, hidden_dim)
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    # Create dataset and dataloader
    dataset = TensorDataset(encoder_hidden_states, input_ids, attention_mask)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # 2. Initialize model with minimal settings
    model = DecoderEmbedding(
        vision_backbone="imagebind",  # or use the smallest available model
        language_model="distilgpt2",
        tokeniser=get_tokenizer("distilgpt2"),
    )

    # 3. Create trainer with minimal settings
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,  # Only run 2 training batches
        limit_val_batches=2,  # Only run 2 validation batches
        enable_checkpointing=False,
        logger=False,
    )

    # 4. Run training

    trainer.fit(model, train_loader, val_loader)
    print("Training completed successfully!")

    # 5. Test forward pass

    result_embedding, logits = model(
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
    )
    print("\nForward pass successful!")
    print(f"Result embedding shape: {result_embedding.shape}")
    print(f"Logits shape: {logits.shape}")


if __name__ == "__main__":
    test_decoder()
