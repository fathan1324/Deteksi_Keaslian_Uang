import streamlit as st
import torch
import numpy as np
import sys
from pathlib import Path
import cv2
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="centered"
)

st.title("Deteksi Keaslian Uang Menggunakan YOLOv5")
st.caption("Versi deploy (snapshot kamera & upload gambar – CPU)")

# ================== YOLOv5 PATH ==================
FILE = Path(__file__).resolve()

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

# ================== LOAD MODEL (CPU ONLY) ==================
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # PAKSA CPU (deploy-safe)
    model = DetectMultiBackend("best_windows1.pt", device=device)
    model.model.float()
    model.model.eval()
    return model

model = load_model()

# ================== CONFIDENCE SLIDER ==================
st.subheader("⚙️ Pengaturan Deteksi")
conf_thres = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# ================== INPUT MODE ==================
st.subheader("📥 Pilih Metode Input")
input_mode = st.radio(
    "Sumber gambar",
    ("Kamera (Snapshot)", "Upload Gambar")
)

img0 = None

# ================== CAMERA INPUT ==================
if input_mode == "Kamera (Snapshot)":
    img_file = st.camera_input("Ambil gambar uang")
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img0 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ================== UPLOAD IMAGE ==================
elif input_mode == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar uang (.jpg / .png)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img0 = np.array(image)
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)

# ================== INFERENCE ==================
if img0 is not None:
    # --- Preprocess ---
    img = letterbox(img0, 640)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)
    img = img.cpu()  # PAKSA CPU

    # --- Predict ---
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres)

    # --- Draw result ---
    detected = False

    for det in pred:
        if det is not None and len(det):
            detected = True
            det[:, :4] = scale_boxes(
                img.shape[2:], det[:, :4], img0.shape
            ).round()

            for *xyxy, conf, cls in det:
                label = f"{model.names[int(cls)]} ({conf:.2f})"
                cv2.rectangle(
                    img0,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    img0,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # --- Output ---
    st.image(img0, channels="BGR", caption="Hasil Deteksi")
    st.write(f"Confidence threshold: {conf_thres}")

    if detected:
        st.success("Objek uang terdeteksi")
    else:
        st.warning("Tidak ada objek terdeteksi")
