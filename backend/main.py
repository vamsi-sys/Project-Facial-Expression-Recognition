"""
main.py  —  FastAPI application for Facial Expression Recognition
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import (
    load_resources,
    decode_base64_image,
    decode_uploaded_image,
    analyze_image,
)

app = FastAPI(
    title="Facial Expression Recognition API",
    description="Detects faces and classifies emotions using MobileNetV2 "
                "transfer-learning fine-tuned on FER2013.",
    version="1.0.0",
)

# Allow the frontend (Netlify / any origin) to call this API.
# Tighten to your Netlify URL after first deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model ONCE at startup — not per request
model, labels, face_cascade = load_resources()


class Base64Image(BaseModel):
    image: str  # data URL: "data:image/jpeg;base64,..."


@app.get("/")
def health_check():
    """Health check — also used by Render to verify the service is alive."""
    return {
        "status": "ok",
        "message": "Facial Expression Recognition API is running",
        "emotions": list(labels.values()),
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Upload mode — accepts a multipart image file."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    file_bytes = await file.read()
    image = decode_uploaded_image(file_bytes)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Try a different file.")

    return analyze_image(image, model, labels, face_cascade)


@app.post("/predict/base64")
async def predict_base64(payload: Base64Image):
    """Live camera mode — accepts a base64-encoded JPEG frame every ~1.5 s."""
    image = decode_base64_image(payload.image)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image data")

    return analyze_image(image, model, labels, face_cascade)
