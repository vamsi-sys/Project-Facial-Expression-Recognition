# Mood Lab — Facial Expression Recognition

Full-stack facial expression recognition:
- **Model**: MobileNetV2 (ImageNet) → fine-tuned on FER2013 (7 emotions)
- **Backend**: FastAPI + OpenCV + TensorFlow/Keras, served via Docker
- **Frontend**: Vanilla HTML/CSS/JS — live webcam + photo upload modes

```
fer-final/
├── .gitignore
├── render.yaml               ← optional: one-click Render setup
├── training/
│   └── train_model.py        ← run on Colab to retrain (optional)
├── backend/
│   ├── main.py               ← FastAPI routes
│   ├── model_utils.py        ← face detect, preprocess, predict
│   ├── requirements.txt      ← pinned Python dependencies
│   ├── Dockerfile
│   └── models/
│       ├── emotion_model.keras  ← trained model (already included)
│       └── labels.json          ← emotion index map (already included)
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Local testing

```bash
# 1. Backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Visit http://localhost:8000 — should return JSON health check

# 2. Frontend (new terminal)
cd frontend
python -m http.server 5500
# Visit http://localhost:5500
```

---

## Deploy: Backend → Render, Frontend → Netlify

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/fer-final.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy backend on Render
1. render.com → **New → Web Service** → connect your repo
2. Set **Root Directory**: `backend`
3. Environment: **Docker** (auto-detected from Dockerfile)
4. Plan: **Free** → **Create Web Service**
5. Wait ~5 min for the build. Once live, visit your Render URL —
   you'll see: `{"status":"ok","message":"...","emotions":[...]}`

### Step 3 — Update API URL in frontend
Edit `frontend/script.js` line 12:
```js
API_BASE_URL: "https://YOUR-APP-NAME.onrender.com",
```
Then commit and push.

### Step 4 — Deploy frontend on Netlify
- netlify.com → **Add new site → Deploy manually** → drag the `frontend/` folder
- OR connect the GitHub repo, set **Publish directory** to `frontend`

### Step 5 — Lock CORS (after Netlify gives you a URL)
In `backend/main.py`, replace `allow_origins=["*"]` with:
```python
allow_origins=["https://your-site.netlify.app"],
```
Commit + push — Render auto-redeploys.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Render crashes with `quantization_config` error | This build uses the `.keras` format which avoids this — should not happen |
| `Model not found` on startup | Check `backend/models/` has both files committed to git |
| CORS error in browser console | Make sure `allow_origins` in `main.py` matches your Netlify URL |
| Webcam doesn't start | Browser requires `https://` or `localhost` for camera access |
| First request very slow | Free Render services sleep after 15 min idle — first request takes ~30s to wake up |
