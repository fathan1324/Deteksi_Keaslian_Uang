import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# ================== STREAMLIT CONFIG ==================
st.set_page_config(
    page_title="Deteksi Keaslian Uang",
    page_icon="💵",
    layout="centered"
)

st.title("Deteksi Keaslian Uang (YOLOv5)")
st.caption("Streamlit Cloud – CPU only")

# ================== YOLOV5 IMPORT (FINAL & BENAR) ==================
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = DetectMultiBackend("best_windows1.pt", device=device)
    model.model.float()
    model.model.eval()
    return model

model = load_model()

# ================== CONFIDENCE ==================
conf_thres = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

# ================== INPUT ==================
mode = st.radio("Sumber input", ("Kamera (Snapshot)", "Upload Gambar"))
img0 = None

if mode == "Kamera (Snapshot)":
    cam = st.camera_input("Ambil gambar")
    if cam:
        img0 = cv2.imdecode(
            np.frombuffer(cam.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
else:
    up = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if up:
        img0 = cv2.cvtColor(
            np.array(Image.open(up).convert("RGB")),
            cv2.COLOR_RGB2BGR
        )

# ================== INFERENCE ==================
if img0 is not None:
    img = letterbox(img0, 640)[0]
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres)

    detected = False
    for det in pred:
        if det is not None and len(det):
            detected = True
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{model.names[int(cls)]} ({conf:.2f})"
                cv2.rectangle(
                    img0,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    img0, label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

    st.image(img0, channels="BGR")
    st.success("Objek terdeteksi" if detected else "Tidak ada objek terdeteksi")
