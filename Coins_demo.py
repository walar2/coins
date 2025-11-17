"""
Streamlit app: Live Thai Baht coin detection with YOLOv8 (best.pt or best_coin.pt)
-------------------------------------------------------------------------------
Features
- Live webcam detection in-browser (desktop + Android) via WebRTC
- Optional image/video upload for on-demand inference
- Model hot-swap from sidebar (pick best_coin.pt, best.pt, or upload .pt)
- One-click public HTTPS link using Cloudflare Tunnel (works from Android)
- Auto-install of missing Python deps when run in notebook/Colab

How to run locally
1) pip install -r requirements.txt  (see requirements block below)
2) streamlit run streamlit_yolov8_thai_baht.py
3) Use the Public Link button to expose an HTTPS URL if needed

Requirements (put these lines in a requirements.txt next to this file)
---------------------------------------------------------------------
ultralytics>=8.2.0
streamlit>=1.37
streamlit-webrtc>=0.47.1
opencv-python-headless>=4.9.0.80
av>=11.0.0
numpy>=1.23
torch  # Provided by your environment (CUDA if available)

Notes
- For mobile usage, ensure your deployment provides a HTTPS public URL (e.g., Streamlit Community Cloud or the built-in Cloudflare tunnel button below).
- If you see camera permission prompts on Android, accept them in the browser.
- If GPU is available on the host, YOLO will use it automatically.
"""

# ----------------------------
# Bootstrap: auto-install missing packages (for Colab/Notebook use)
# ----------------------------
import os, sys, subprocess, importlib, tempfile, re, threading, time, stat, urllib.request

def _pip_install_if_missing(pkg_name: str, spec: str):
    try:
        importlib.import_module(pkg_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", spec])

for _pkg, _spec in [
    ("streamlit", "streamlit>=1.37"),
    ("streamlit_webrtc", "streamlit-webrtc>=0.47.1"),
    ("ultralytics", "ultralytics>=8.2.0"),
    ("av", "av>=11.0.0"),
    ("numpy", "numpy>=1.23"),
    ("cv2", "opencv-python-headless>=4.9.0.80"),
]:
    _pip_install_if_missing(_pkg, _spec)

# ----------------------------
# Imports (safe after bootstrap)
# ----------------------------
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ----------------------------
# Public tunnel helper (Cloudflare)
# ----------------------------
import platform

# Platform-aware Cloudflare downloader/runner
# Maps OS/arch -> correct binary URL & filename
_CLOUDFLARE_URLS = {
    ("Windows", "AMD64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    ("Windows", "ARM64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-arm64.exe",
    ("Linux",   "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    ("Linux",   "aarch64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    ("Darwin",  "x86_64"): "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    ("Darwin",  "arm64"):  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64",
}

def _cloudflared_target() -> tuple[str, str]:
    osys = platform.system() or ""
    arch = platform.machine() or ""
    # Normalize some common variants
    arch_norm = {
        "AMD64": "AMD64", "x86_64": "x86_64", "X86_64": "x86_64",
        "aarch64": "aarch64", "arm64": "arm64", "ARM64": "ARM64",
    }.get(arch, arch)
    url = _CLOUDFLARE_URLS.get((osys, arch_norm))
    if not url:
        # Fallbacks
        if osys == "Windows":
            url = _CLOUDFLARE_URLS[("Windows", "AMD64")]
        elif osys == "Darwin":
            url = _CLOUDFLARE_URLS[("Darwin", "arm64")]
        else:
            url = _CLOUDFLARE_URLS[("Linux", "x86_64")]
    # Filename (add .exe for Windows)
    fname = "cloudflared.exe" if osys == "Windows" else "cloudflared"
    return url, fname


def start_cloudflare_tunnel(port: int) -> str:
    """Start a Cloudflare tunnel to http://localhost:port and return the public URL when available.
    Downloads the correct binary for your OS/CPU if needed and launches it.
    """
    url, fname = _cloudflared_target()

    # Search common install locations first
    candidate_paths = [
        os.path.join("/usr/local/bin", fname),
        os.path.join(tempfile.gettempdir(), fname),
        os.path.join(os.getcwd(), fname),
    ]
    bin_path = next((p for p in candidate_paths if os.path.isfile(p)), None)

    if bin_path is None:
        bin_path = candidate_paths[1]
        try:
            urllib.request.urlretrieve(url, bin_path)
            if platform.system() != "Windows":
                os.chmod(bin_path, os.stat(bin_path).st_mode | stat.S_IEXEC)
        except Exception as e:
            raise RuntimeError(f"Failed to download cloudflared: {e}")

    # Build command per-OS
    cmd = [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"]
    popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    if platform.system() == "Windows":
        # On Windows, ensure .exe is used and avoid new console windows
        popen_kwargs.update(shell=False)

    proc = subprocess.Popen(cmd, **popen_kwargs)

    public_url = None
    start = time.time()
    while time.time() - start < 45:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        # Look for trycloudflare URL
        if "trycloudflare.com" in line:
            tokens = [t for t in line.strip().split() if t.startswith("http")]
            if tokens:
                public_url = tokens[0]
                break
    if public_url is None:
        public_url = "(pending...) Check app logs/console for the URL)"

    st.session_state.setdefault("_cf_proc", proc)
    return public_url

# ----------------------------
# Streamlit UI
# ----------------------------
# Prevent 'missing ScriptRunContext' warnings when run as `python script.py` or from IDE scratch
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _IS_STREAMLIT = _get_ctx() is not None
except Exception:
    _IS_STREAMLIT = False

if not _IS_STREAMLIT:
    print("[INFO] This app must be launched with: streamlit run streamlit_yolov8_thai_baht.py")
    print("[TIP] In PyCharm: Run configuration -> Module name: streamlit.web.cli, Parameters: run streamlit_yolov8_thai_baht.py")
    import sys as _sys
    _sys.exit(0)
st.set_page_config(page_title="Thai Baht Coin Detector", page_icon="🪙", layout="wide")
st.title("🪙 Thai Baht Coin Detector — YOLOv8 (Live)")
st.caption("Live object detection on webcam/video. Works from Android via your public link.")

# ===== Sidebar: Public link via Cloudflare Tunnel =====
st.sidebar.header("Public Link")
PORT = int(os.environ.get("PORT", os.environ.get("STREAMLIT_SERVER_PORT", 8501)))
if st.sidebar.button("Start Public Tunnel (HTTPS)"):
    with st.spinner("Starting Cloudflare tunnel..."):
        try:
            url = start_cloudflare_tunnel(PORT)
            st.sidebar.success(f"Tunnel ready: {url}")
            st.sidebar.code(url)
        except Exception as e:
            st.sidebar.error(f"Tunnel error: {e}")
else:
    st.sidebar.write("Click to expose this app to the internet. Use the URL on Android.")

# ===== Embedded model path (no upload) =====
st.sidebar.header("Model Settings")
MODEL_PATH = "C:/Users/minkh/Desktop/Persnl_11/best_coin_thb.pt"
  # <-- embed your model path here
st.sidebar.write(f"Using embedded model: **{os.path.basename(MODEL_PATH)}**")

# Fixed inference size per request
imgsz = 416
st.sidebar.write(f"Inference size (imgsz): **{imgsz}**")

# Confidence/IoU & detections
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
iou_thres = st.sidebar.slider("IoU threshold", 0.1, 0.9, 0.45, 0.01)
max_det = st.sidebar.slider("Max detections", 10, 300, 100, 10)

# ===== Augmentations (inference) =====
st.sidebar.header("Augmentations")
use_tta = st.sidebar.checkbox("Test-time augmentation (YOLO TTA)", value=False, help="Uses flips/scales at inference for robustness (slower)")
use_clahe = st.sidebar.checkbox("CLAHE (contrast boost)", value=False)
use_sharpen = st.sidebar.checkbox("Sharpen", value=False)
use_denoise = st.sidebar.checkbox("Denoise (fast NLM)", value=False)
gamma = st.sidebar.slider("Gamma", 0.5, 1.8, 1.0, 0.05, help="<1.0 brightens, >1.0 darkens")

st.sidebar.markdown("""
**Tip**: On mobile, enable CLAHE + slight sharpen if lighting is flat.
""")

# ----------------------------
# Cache model loading
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    # Loading embedded YOLO model
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")
    m = YOLO(weights_path)
    return m

model_loaded = None
try:
    model_loaded = load_model(MODEL_PATH)
    st.success(f"Loaded model: {os.path.basename(MODEL_PATH)}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------------------
# Helpers
# ----------------------------
import cv2  # safe after bootstrap

# ==== Coin value mapping (name -> baht) ====
# Adjust keys to match your class names if needed.
VALUE_MAP_NAME = {
    "1": 1, "1baht": 1, "1_baht": 1, "1 baht": 1,
    "2": 2, "2baht": 2, "2_baht": 2, "2 baht": 2,
    "5": 5, "5baht": 5, "5_baht": 5, "5 baht": 5,
    "10": 10, "10baht": 10, "10_baht": 10, "10 baht": 10,
}

IMG_SIZE_INFER = 416  # fixed as requested

def _norm_name(name: str) -> str:
    s = name.strip().lower()
    import re as _re
    m = _re.search(r"(\d+)", s)  # pull out the number in the class name, e.g., "coin_10_baht" -> 10
    if m:
        return m.group(1)
    return s

def result_to_counts(res0):
    """
    Convert a YOLO result (results[0]) to:
      - counts: {class_display_name: count}
      - total_value: sum of mapped baht values
    """
    counts = {}
    total_value = 0
    if res0 is None or res0.boxes is None or res0.boxes.cls is None:
        return counts, total_value

    classes = res0.boxes.cls.cpu().numpy().astype(int).tolist()
    names = getattr(model_loaded.model, "names", None) or getattr(model_loaded, "names", None) or {}

    for cid in classes:
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        cname_norm = _norm_name(cname)
        val = VALUE_MAP_NAME.get(cname_norm, 0)
        counts[cname] = counts.get(cname, 0) + 1
        total_value += val

    # Persist last results
    st.session_state["last_counts"] = counts
    st.session_state["last_total_baht"] = total_value
    return counts, total_value



def run_yolo(image_bgr: np.ndarray, return_results: bool = False):
    """Run YOLO inference on a BGR image. Returns annotated frame (and results if requested)."""
    if model_loaded is None:
        return (image_bgr, None) if return_results else image_bgr
    results = model_loaded.predict(
        image_bgr,
        imgsz=IMG_SIZE_INFER,   # fixed 416
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        verbose=False,
        augment=False,          # set True if you want test-time augmentation
    )
    annotated = results[0].plot()  # BGR ndarray
    return (annotated, results[0]) if return_results else annotated


# ----------------------------
# Live webcam via WebRTC
# ----------------------------
# (Live camera removed by request — WebRTC/ICE config deleted)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated = run_yolo(img)
        self.frame_count += 1
        return annotated

# Live camera removed by request.

# ----------------------------
# Image / Video upload (only)

# ----------------------------
st.subheader("Run detection on an image or video")
media_file = st.file_uploader("Upload an image (.jpg/.png) or video (.mp4/.mov)", type=["jpg", "jpeg", "png", "mp4", "mov", "mkv"])

if media_file is not None:
    file_suffix = os.path.splitext(media_file.name)[1].lower()
    tmp_in = os.path.join(tempfile.gettempdir(), f"in{file_suffix}")
    with open(tmp_in, "wb") as f:
        f.write(media_file.read())

    if file_suffix in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(tmp_in)
        if img is None:
            st.error("Could not read the uploaded image.")
        else:
            out, res0 = run_yolo(img, return_results=True)
            counts, total_value = result_to_counts(res0)
            st.image(out[:, :, ::-1], caption="Detections", use_column_width=True)

            # Show a summary + breakdown
            st.subheader(f"Total amount: {total_value} baht")
            if counts:
                st.write("Breakdown by class:")
                st.table({"class": list(counts.keys()), "count": list(counts.values())})
            else:
                st.info("No coins detected.")

    else:
        cap = cv2.VideoCapture(tmp_in)
        if not cap.isOpened():
            st.error("Could not read the uploaded video.")
        else:
            st.info("Processing video... (a short clip will preview)")
            frames = []
            last_res0 = None
            total_preview = 120  # ~4 seconds at 30 fps
            count = 0
            while count < total_preview:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, res0 = run_yolo(frame, return_results=True)
                last_res0 = res0
                frames.append(annotated[:, :, ::-1])  # to RGB for display
                count += 1
            cap.release()

            if frames:
                st.image(frames, caption="Preview (first ~4s)", use_column_width=True)
                # NOTE: Counting on a single frame to avoid double-counting the same coin across frames.
                counts, total_value = result_to_counts(last_res0)
                st.subheader(f"Estimated total amount (last preview frame): {total_value} baht")
                if counts:
                    st.write("Breakdown by class:")
                    st.table({"class": list(counts.keys()), "count": list(counts.values())})
                else:
                    st.info("No coins detected in the sampled frame.")
            else:
                st.warning("No frames found in video.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
**Having trouble?**
- Make sure your weights match your dataset's class order.
- If you deployed on CPU-only, inference will be slower. Consider a GPU host for real-time use.
- On Android, use Chrome and ensure camera permissions are granted.
- Use the **Public Tunnel** button if you need a quick HTTPS link.
""")
