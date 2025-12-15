import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path
import cv2
from PIL import Image

# ================== STREAMLIT CONFIG ==================
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="centered"
)

st.title("Deteksi Keaslian Uang (YOLOv5)")
st.caption("Versi deploy Streamlit Cloud (CPU only)")

# ================== FIX YOLOV5 PATH (WAJIB) ==================
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent
YOLOV5_ROOT = PROJECT_ROOT / "yolov5"

if not YOLOV5_ROOT.exists():
    st.error("❌ Folder 'yolov5' tidak ditemukan. Pastikan ada di repo.")
    st.stop()

sys.path.insert(0, str(YOLOV5_ROOT))

# ================== YOLOV5 IMPORT ==================
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# ================== LOAD MODEL (CPU ONLY) ==================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = DetectMultiBackend("best_windows.pt", device=device)
    model.model.float()
    model.eval()
    return model

model = load_model()

# ================== CONFIDENCE ==================
conf_thres = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

# ================== INPUT MODE ==================
mode = st.radio(
    "Sumber input",
    ("Kamera (Snapshot)", "Upload Gambar")
)

img0 = None

# ================== CAMERA ==================
if mode == "Kamera (Snapshot)":
    cam = st.camera_input("Ambil gambar uang")
    if cam is not None:
        bytes_data = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img0 = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

# ================== UPLOAD IMAGE ==================
else:
    up = st.file_uploader("Upload gambar (.jpg/.png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        img0 = np.array(Image.open(up).convert("RGB"))
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)

# ================== INFERENCE ==================
if img0 is not None:
    img = letterbox(img0, 640)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

    detected = False

    for det in pred:
        if det is not None and len(det):
            detected = True
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                label = f"{model.names[int(cls)]} ({conf:.2f})"
                cv2.rectangle(img0,
                              (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])),
                              (0, 255, 0), 2)
                cv2.putText(img0, label,
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    st.image(img0, channels="BGR", caption="Hasil Deteksi")

    if detected:
        st.success("✅ Uang terdeteksi")
    else:
        st.warning("❌ Tidak ada objek terdeteksi")
