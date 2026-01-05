from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, io, requests
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

MODEL_NAME = "openai/clip-vit-base-patch32"  # 512-dim

device = "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

class EmbedRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

def load_image(req: EmbedRequest) -> Image.Image:
    if req.image_url:
        r = requests.get(req.image_url, timeout=20)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image_url: {r.status_code}")
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    if req.image_base64:
        try:
            data = base64.b64decode(req.image_base64)
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image_base64")

    raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

@app.post("/embed")
def embed(req: EmbedRequest):
    image = load_image(req)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return {"dim": feats.shape[-1], "vector": feats[0].tolist()}
