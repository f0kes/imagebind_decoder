from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import torch
import os
from clip_text_decoder.model import DecoderInferenceModel
from clip_text_decoder.common import load_tokenizer

app = FastAPI()

decoder = None


# Define a Pydantic model for request validation
class LoadModelRequest(BaseModel):
    checkpoint_path: str | None = None


class PredictRequest(BaseModel):
    embedding: list  # Expecting a list of embedding values for simplicity
    beam_size: int = 1


# Load the model
@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    global decoder
    try:
        path = os.getenv("CHECKPOINT_PATH", "/models/version_1/model.pt")
        if request.checkpoint_path and os.path.exists(request.checkpoint_path):
            path = request.checkpoint_path
        decoder = DecoderInferenceModel.load(path)
        device = torch.device("cuda")
        decoder.to(device=device)
        return {"status": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Predict from embedding
@app.post("/predict")
async def get_text_from_embedding(request: PredictRequest):
    if decoder is None:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Load the model first by calling /load_model.",
        )
    try:
        embedding_tensor = torch.tensor(request.embedding).unsqueeze(0).float()
        prediction = decoder(embedding_tensor, beam_size=request.beam_size)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
