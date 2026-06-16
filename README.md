# 🎭 Facial Expression Recognition

A full-stack, real-time facial expression recognition web app built with **Transfer Learning**. Point a webcam or upload a photo, and a MobileNetV2-based deep learning model — fine-tuned on the FER2013 dataset — detects faces and classifies the emotion being expressed across 7 categories.

### 🔗 Live Demo

| | |
|---|---|
| **🌐 Frontend (App)** | [projectfacialexpressionrecognition.netlify.app](https://projectfacialexpressionrecognition.netlify.app/) |
| **⚙️ Backend (API)** | [project-facial-expression-recognition.onrender.com](https://project-facial-expression-recognition.onrender.com) |

> ⚠️ **Note:** The backend is hosted on Render's free tier, which spins down after periods of inactivity. The **first request may take 30–60 seconds** while the server wakes up — please be patient on first load.

---

## 📸 What It Does

1. Open the app and choose **Live Camera** or **Upload a Photo**.
2. The image is sent to a FastAPI backend.
3. OpenCV detects all faces in the frame.
4. Each face is cropped, resized, and passed through a MobileNetV2-based deep learning model.
5. The model predicts one of **7 emotions**: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
6. The result is rendered as an animated, polaroid-style card with the annotated photo, an emoji, and a confidence bar chart for every emotion class.

---

## 🧠 How It Works (Architecture)

```
┌──────────────┐        photo / webcam frame        ┌────────────────────┐
│   Frontend    │ ─────────────────────────────────► │      Backend       │
│ (HTML/CSS/JS) │                                     │     (FastAPI)      │
│   Netlify     │ ◄───────────────────────────────── │      Render        │
└──────────────┘   JSON: emotion, confidence,        │  OpenCV face detect │
                    probabilities, bounding box        │  + MobileNetV2 model│
                                                        └────────────────────┘
```

### The Model — Transfer Learning Explained
Instead of training a neural network from scratch (which needs huge datasets and compute), this project uses **transfer learning**:

- **Base**: `MobileNetV2`, pretrained on ImageNet (1.4M images, 1000 classes). It already knows how to extract general visual features — edges, textures, shapes.
- **Custom head**: A small set of new layers (`GlobalAveragePooling2D → Dense(256) → Dropout → Dense(128) → Dropout → Dense(7, softmax)`) added on top, which learns to map those general features specifically to 7 emotion categories.
- **Two-phase training**:
  1. **Frozen backbone** — only the new head trains, while MobileNetV2's weights stay locked. Fast, and prevents the small emotion dataset from destroying the pretrained knowledge.
  2. **Fine-tuning** — the top 30 layers of MobileNetV2 are unfrozen and trained together with the head, using a very small learning rate (`1e-5`) to gently adapt the pretrained features to faces/emotions specifically.
- **Dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — ~35,000 labeled grayscale facial images across 7 emotion classes.

### The Backend — FastAPI + OpenCV
- **Face detection**: OpenCV's Haar Cascade classifier locates face bounding boxes in the incoming image — this runs *before* the deep learning model, so the model only ever sees cropped faces, not whole scenes.
- **Preprocessing**: Each detected face is resized to 96×96, converted to RGB, and normalized to pixel values between 0–1 — matching exactly how images were prepared during training.
- **Inference**: The preprocessed face is passed through the MobileNetV2 model, which outputs a probability for each of the 7 emotions.
- **Two endpoints**:
  - `POST /predict/image` — accepts an uploaded image file (Upload mode)
  - `POST /predict/base64` — accepts a base64-encoded JPEG frame (Live Camera mode, called automatically every ~1.5 seconds while the camera is on)

### The Frontend — Vanilla HTML/CSS/JS
No frameworks, no build step — just static files that:
- Use `getUserMedia()` to access the webcam and `<canvas>` to capture frames
- Use drag-and-drop / file input + `FileReader` for photo uploads
- Render results as animated "developing" polaroid cards with bounding boxes drawn directly onto a canvas, plus a bar chart of all 7 emotion probabilities

---

## 📁 Project Structure

```
fer-final/
├── render.yaml                  ← Render deployment config
├── training/
│   └── train_model.py           ← Transfer learning training script (run on Google Colab)
├── backend/
│   ├── main.py                  ← FastAPI app: routes, CORS, startup model loading
│   ├── model_utils.py           ← Face detection, preprocessing, prediction logic
│   ├── requirements.txt         ← Pinned Python dependencies
│   ├── Dockerfile               ← Container definition (Python 3.11-slim)
│   └── models/
│       ├── emotion_model.keras  ← Trained MobileNetV2 model
│       └── labels.json          ← Maps model output index → emotion name
└── frontend/
    ├── index.html               ← Page structure (mode selection, camera, upload, results)
    ├── style.css                ← "Darkroom / instant photo" visual design system
    └── script.js                ← Mode switching, webcam loop, upload, API calls, rendering
```

---

## 💻 Run It Locally

### Prerequisites
- Python 3.11
- A webcam (for Live Camera mode)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Start the backend
```bash
cd backend
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000` — you should see a JSON response confirming the API and model loaded successfully.

### 3. Start the frontend (in a new terminal)
```bash
cd frontend
python -m http.server 5500
```
Visit `http://localhost:5500` in your browser.

> 💡 By default, `frontend/script.js` points `API_BASE_URL` at the **live Render backend**, so the local frontend will work out of the box even without running the backend locally. To test against your local backend instead, change the line in `script.js`:
> ```js
> API_BASE_URL: "http://localhost:8000",
> ```

---

## 🚀 Deployment Guide

This project is deployed using a two-service architecture: **Render** for the backend API, **Netlify** for the static frontend.

### Backend → Render
1. Push the repo to GitHub.
2. On [render.com](https://render.com) → **New → Web Service** → connect the repo.
3. Set **Root Directory** to `backend`.
4. Render auto-detects the `Dockerfile` → choose **Docker** as the environment.
5. Select the **Free** plan → **Create Web Service**.
6. Wait for the build (~5–8 minutes — TensorFlow is a large dependency). Once live, visiting the root URL returns a JSON health check.

### Frontend → Netlify
1. On [netlify.com](https://app.netlify.com) → **Add new site → Deploy manually** → drag in the `frontend/` folder.
   *(Or: **Import from Git**, set **Publish directory** to `frontend`.)*
2. Netlify provides a live URL automatically.

### Connecting them (CORS)
The backend's `main.py` allows cross-origin requests so the Netlify frontend can call the Render API. For production, `allow_origins` should be locked to the specific Netlify domain rather than `"*"`.

---

## 🔧 Key Technical Fixes Along the Way

Building and deploying this surfaced two real-world version-compatibility issues worth documenting:

| Issue | Cause | Fix |
|---|---|---|
| `TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}` | Model was saved in Colab with a newer Keras 3 version than what was installed on the server; the legacy `.h5` loader couldn't deserialize the extra config key | Converted the model from `.h5` to the native `.keras` format, which uses a more robust loader |
| `ERROR: Could not find a version that satisfies the requirement keras==3.14.1` (build failure) | Keras 3.13+ requires Python ≥3.11, but the Dockerfile was based on `python:3.10-slim` | Updated the Dockerfile's base image to `python:3.11-slim` |

---

## 🛠️ Tech Stack

**Machine Learning**: TensorFlow / Keras · MobileNetV2 (Transfer Learning) · OpenCV (Haar Cascade face detection) · FER2013 dataset

**Backend**: FastAPI · Uvicorn · Python 3.11 · Docker

**Frontend**: HTML5 · CSS3 · Vanilla JavaScript · `getUserMedia` API · Canvas API

**Deployment**: Render (backend, Docker) · Netlify (frontend, static hosting)

---

## 🐞 Troubleshooting

| Symptom | Likely Cause / Fix |
|---|---|
| First request hangs or times out | Render free tier sleeps after inactivity — wait ~30–60s and retry |
| "Model not found" error on backend startup | `backend/models/emotion_model.keras` and `labels.json` must both be committed to the repo |
| CORS error in browser console | Check `allow_origins` in `backend/main.py` includes the frontend's domain |
| Webcam doesn't start | Browsers require HTTPS or `localhost` for camera access — Netlify serves over HTTPS by default |
| No face detected | Ensure good lighting and a clear, front-facing view of the face |

---

## 📜 License

This project is open for educational and personal use.
