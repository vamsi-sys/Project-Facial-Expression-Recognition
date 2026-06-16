"""
model_utils.py  —  Model loading, face detection, preprocessing & prediction
"""

import os
import json
import base64

import numpy as np
import cv2
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "emotion_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

IMG_SIZE = 96  # Must match training

EMOJI_MAP = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Place emotion_model.keras + labels.json in backend/models/."
        )

    model = load_model(MODEL_PATH)

    with open(LABELS_PATH) as f:
        raw = json.load(f)
    labels = {int(k): v for k, v in raw.items()}

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    return model, labels, face_cascade


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def decode_base64_image(b64: str):
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def decode_uploaded_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Detection & prediction
# ---------------------------------------------------------------------------
def detect_faces(image, cascade):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(48, 48))
    return faces


def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    return np.expand_dims(face_img, axis=0)


def predict_emotion(model, labels, face_img):
    preds    = model.predict(preprocess_face(face_img), verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    top_label = labels[top_idx]
    return {
        "emotion":          top_label,
        "emoji":            EMOJI_MAP.get(top_label, ""),
        "confidence":       float(preds[top_idx]),
        "all_probabilities": {labels[i]: float(preds[i]) for i in range(len(preds))},
    }


def analyze_image(image, model, labels, face_cascade):
    faces = detect_faces(image, face_cascade)
    if len(faces) == 0:
        return {"faces_found": 0, "results": []}

    results = []
    for (x, y, w, h) in faces:
        pred = predict_emotion(model, labels, image[y:y+h, x:x+w])
        pred["box"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        results.append(pred)

    return {"faces_found": len(results), "results": results}
