/* ==========================================================================
   MOOD LAB — Facial Expression Recognition
   Frontend logic: mode switching, webcam capture, file upload, result cards
   ========================================================================== */

// ─────────────────────────────────────────────────────────────────────────────
// ✏️  STEP 1 — Before deploying the frontend to Netlify, change this URL to
//              your Render backend URL, e.g.:
//              "https://mood-lab-api.onrender.com"
//
//     For local testing leave it as "http://localhost:8000"
// ─────────────────────────────────────────────────────────────────────────────
const CONFIG = {
  API_BASE_URL: "https://project-facial-expression-recognition.onrender.com",
  CAPTURE_INTERVAL_MS: 1500,
};

const EMOTION_ORDER = ["happy", "surprise", "neutral", "fear", "sad", "disgust", "angry"];

// ─── DOM refs ────────────────────────────────────────────────────────────────
const modeSelect    = document.getElementById("mode-select");
const cameraView    = document.getElementById("camera-view");
const uploadView    = document.getElementById("upload-view");
const resultsSection = document.getElementById("results");

const openCameraBtn  = document.getElementById("open-camera-btn");
const openUploadBtn  = document.getElementById("open-upload-btn");
const backButtons    = document.querySelectorAll("[data-back]");

const video              = document.getElementById("video");
const cameraPlaceholder  = document.getElementById("camera-placeholder");
const cameraToggleBtn    = document.getElementById("camera-toggle-btn");
const cameraStatus       = document.getElementById("camera-status");

const dropzone           = document.getElementById("dropzone");
const previewImg         = document.getElementById("preview-img");
const uploadPlaceholder  = document.getElementById("upload-placeholder");
const fileInput          = document.getElementById("file-input");
const analyzeBtn         = document.getElementById("analyze-btn");
const uploadStatus       = document.getElementById("upload-status");

const captureCanvas = document.getElementById("capture-canvas");

// ─── View switching ──────────────────────────────────────────────────────────
function showView(view) {
  modeSelect.classList.toggle("hidden",  view !== "select");
  cameraView.classList.toggle("hidden",  view !== "camera");
  uploadView.classList.toggle("hidden",  view !== "upload");
  resultsSection.classList.add("hidden");
  resultsSection.innerHTML = "";
  if (view !== "camera") stopCamera();
}

openCameraBtn.addEventListener("click", () => showView("camera"));
openUploadBtn.addEventListener("click", () => showView("upload"));
backButtons.forEach(btn => btn.addEventListener("click", () => showView("select")));

// ─── LIVE CAMERA ─────────────────────────────────────────────────────────────
let mediaStream      = null;
let captureIntervalId = null;
let cameraRunning    = false;
let requestInFlight  = false;

cameraToggleBtn.addEventListener("click", () => {
  cameraRunning ? stopCamera() : startCamera();
});

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    video.srcObject = mediaStream;
    cameraPlaceholder.classList.add("hidden");
    cameraRunning = true;
    cameraToggleBtn.textContent = "Stop camera";
    cameraStatus.textContent = "Starting…";
    captureIntervalId = setInterval(captureAndAnalyzeFrame, CONFIG.CAPTURE_INTERVAL_MS);
  } catch (err) {
    cameraStatus.textContent = `Could not access camera: ${err.message}`;
  }
}

function stopCamera() {
  clearInterval(captureIntervalId);
  captureIntervalId = null;
  mediaStream?.getTracks().forEach(t => t.stop());
  mediaStream = null;
  video.srcObject = null;
  cameraRunning = false;
  cameraToggleBtn.textContent = "Start camera";
  cameraPlaceholder.classList.remove("hidden");
  cameraStatus.textContent = "";
}

async function captureAndAnalyzeFrame() {
  if (!video.videoWidth || requestInFlight) return;

  captureCanvas.width  = video.videoWidth;
  captureCanvas.height = video.videoHeight;
  captureCanvas.getContext("2d").drawImage(video, 0, 0);
  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.8);

  requestInFlight = true;
  cameraStatus.textContent = "Analyzing…";

  try {
    const res  = await fetch(`${CONFIG.API_BASE_URL}/predict/base64`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    cameraStatus.textContent = data.faces_found
      ? `${data.faces_found} face${data.faces_found > 1 ? "s" : ""} detected`
      : "No face detected — try moving closer";
    renderResults(data, captureCanvas);
  } catch (err) {
    cameraStatus.textContent = `Error: ${err.message}`;
  } finally {
    requestInFlight = false;
  }
}

// ─── UPLOAD ──────────────────────────────────────────────────────────────────
let selectedFile = null;

fileInput.addEventListener("change", e => { if (e.target.files.length) handleFile(e.target.files[0]); });

["dragover","dragenter"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave","dragend"].forEach(evt =>
  dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"))
);
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) { uploadStatus.textContent = "Please choose an image file."; return; }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    uploadPlaceholder.classList.add("hidden");
    analyzeBtn.disabled = false;
    uploadStatus.textContent = "Ready — click Analyze.";
    resultsSection.classList.add("hidden");
    resultsSection.innerHTML = "";
  };
  reader.readAsDataURL(file);
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  uploadStatus.textContent = "Analyzing…";

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const res  = await fetch(`${CONFIG.API_BASE_URL}/predict/image`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    uploadStatus.textContent = data.faces_found
      ? `${data.faces_found} face${data.faces_found > 1 ? "s" : ""} detected`
      : "No face detected";
    renderResults(data, previewImg);
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ─── RESULTS — polaroid cards ─────────────────────────────────────────────────
function renderResults(data, imageSource) {
  resultsSection.innerHTML = "";
  resultsSection.classList.remove("hidden");

  if (!data.faces_found) {
    resultsSection.innerHTML = `
      <div class="empty-state">
        <p>No face detected.</p>
        <p>Make sure your face is well lit and facing the camera, or try a clearer front-facing photo.</p>
      </div>`;
    return;
  }

  const heading = document.createElement("p");
  heading.className = "results-heading";
  heading.textContent = data.faces_found === 1 ? "Reading" : `Readings — ${data.faces_found} faces`;
  resultsSection.appendChild(heading);

  data.results.forEach(face => resultsSection.appendChild(buildPolaroid(face, imageSource)));
}

function srcDimensions(src) {
  if (src instanceof HTMLVideoElement) return [src.videoWidth,   src.videoHeight];
  if (src instanceof HTMLImageElement) return [src.naturalWidth,  src.naturalHeight];
  return [src.width, src.height];
}

function buildPolaroid(face, imageSource) {
  const [sw, sh] = srcDimensions(imageSource);

  const polaroid = document.createElement("article");
  polaroid.className = "polaroid";

  // Photo frame: redraw source + bounding box
  const frame  = document.createElement("div");
  frame.className = "photo-frame";
  const canvas = document.createElement("canvas");
  canvas.width  = sw;
  canvas.height = sh;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imageSource, 0, 0, sw, sh);

  const { x, y, w, h } = face.box;
  const lw = Math.max(2, sw * 0.006);
  ctx.strokeStyle = "#ff7a59";
  ctx.lineWidth   = lw;
  ctx.strokeRect(x, y, w, h);

  const fs   = Math.max(14, Math.round(sw * 0.03));
  const text = `${face.emotion} ${(face.confidence * 100).toFixed(0)}%`;
  ctx.font      = `600 ${fs}px 'Space Grotesk', sans-serif`;
  const tw      = ctx.measureText(text).width;
  const labelY  = Math.max(0, y - fs - 10);
  ctx.fillStyle = "#ff7a59";
  ctx.fillRect(x, labelY, tw + 16, fs + 10);
  ctx.fillStyle = "#1a1f2b";
  ctx.fillText(text, x + 8, labelY + fs);

  frame.appendChild(canvas);
  polaroid.appendChild(frame);

  // Caption
  const caption  = document.createElement("div");
  caption.className = "caption";

  const readout  = document.createElement("div");
  readout.className = "emotion-readout";

  const emojiEl = document.createElement("span");
  emojiEl.className = "emoji";
  emojiEl.textContent = face.emoji;

  const nameEl = document.createElement("span");
  nameEl.className = "emotion-name";
  nameEl.textContent = face.emotion;

  const confEl = document.createElement("span");
  confEl.className = "confidence";
  confEl.textContent = `${(face.confidence * 100).toFixed(1)}%`;

  readout.append(emojiEl, nameEl, confEl);
  caption.appendChild(readout);

  // Bars
  const bars = document.createElement("div");
  bars.className = "bars";

  EMOTION_ORDER.forEach(emotion => {
    if (!(emotion in face.all_probabilities)) return;
    const val = face.all_probabilities[emotion];

    const row   = document.createElement("div");
    row.className = "bar-row";

    const lbl   = document.createElement("span");
    lbl.className = "bar-label";
    lbl.textContent = emotion;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill  = document.createElement("div");
    fill.className = `bar-fill ${emotion}`;

    const valEl = document.createElement("span");
    valEl.className = "bar-value";
    valEl.textContent = `${(val * 100).toFixed(0)}%`;

    track.appendChild(fill);
    row.append(lbl, track, valEl);
    bars.appendChild(row);

    requestAnimationFrame(() => { fill.style.width = `${val * 100}%`; });
  });

  caption.appendChild(bars);
  polaroid.appendChild(caption);
  return polaroid;
}
