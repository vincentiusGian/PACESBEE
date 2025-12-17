import streamlit as st
import os
import gdown
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import time
from collections import Counter

# ===== Class names =====
class_names = ["idle", "gerakan kepala-bawah", "gerakan kepala-leher", "gerakan tangan"]

MODEL_PATH = "models/best.pt"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1wStCza-Ea0AzTR4GZ_qoilQtDH6sdpyC",
        MODEL_PATH,
        quiet=False
    )

model = YOLO(MODEL_PATH)

# ===== Boundary Evolution Equation (BEE) =====
def boundary_evolution_equation(mask, iterations=25, alpha=0.2, beta=0.05):
    mask = mask.astype(np.float32) / 255.0
    phi = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    phi = (phi - phi.max()/2)

    for _ in range(iterations):
        gx, gy = np.gradient(phi)
        grad_norm = np.sqrt(gx**2 + gy**2) + 1e-8
        curvature = cv2.Laplacian(phi, cv2.CV_32F)
        dphi = alpha * curvature - beta * grad_norm
        phi += dphi
        phi = np.clip(phi, -5, 5)

    refined = (phi > 0).astype(np.uint8) * 255
    return refined

# ===== Streamlit UI =====
st.title("🎥 PACES - Paper-based Anti Cheating Examination System")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    behavior_counts = Counter()

    # ===== Font settings =====
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    font_thickness = 3


    frame_skip = 5
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # ===== Filter ringan =====
                if conf < 0.3:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < 100:
                    continue

                behavior_counts[class_names[cls_id]] += 1

                # Crop ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # ===== Buat mask dari grayscale =====
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                if np.mean(gray[mask > 0]) < 127:
                    mask = cv2.bitwise_not(mask)

                # Apply BEE refinement
                refined_mask = boundary_evolution_equation(mask, iterations=25, alpha=0.2, beta=0.05)

                # ===== Inverse mask =====
                refined_mask = cv2.bitwise_not(refined_mask)

                # Hitung persentase area BEE (inverse)
                bee_area = np.sum(refined_mask > 0)
                roi_area = refined_mask.size
                bee_percentage = (bee_area / roi_area) * 100

                # Find refined contours
                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame[y1:y2, x1:x2], contours, -1, (0, 0, 255), 2)

                # ===== Labels terpisah =====
                # YOLO prediction (atas)
                y_label_top = max(y1 - 20, 10)
                yolo_text = f"{class_names[cls_id]} {conf*100:.1f}%"
                (text_w, text_h), _ = cv2.getTextSize(yolo_text, font, font_scale, font_thickness)
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x1, y_label_top - text_h - 5),
                    (x1 + text_w + 10, y_label_top),
                    (0, 255, 0),
                    -1
                )
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                cv2.putText(frame, yolo_text, (x1 + 5, y_label_top - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

                # BEE prediction (bawah)
                y_label_bottom = y2 + 25
                bee_text = f"BEE {bee_percentage:.1f}%"
                (text_w2, text_h2), _ = cv2.getTextSize(bee_text, font, font_scale, font_thickness)
                overlay = frame.copy()
                cv2.rectangle(
                    overlay,
                    (x1, y_label_bottom - text_h2 - 5),
                    (x1 + text_w2 + 10, y_label_bottom),
                    (0, 255, 0),
                    -1
                )
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                cv2.putText(frame, bee_text, (x1 + 5, y_label_bottom - 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Show frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        current_frame += 1
        progress = current_frame / frame_count
        progress_bar.progress(progress)
        progress_text.text(f"Processing frame {current_frame}/{frame_count}")

        # time.sleep(0.02)

    cap.release()
    progress_bar.empty()
    progress_text.text("Video has been processed")

    # ===== Behavioral Event Evaluation (BEE) =====
    st.subheader("📊 Behavioral Event Evaluation (BEE) Results")
    total = sum(behavior_counts.values()) or 1
    bee_data = {cls: f"{(behavior_counts[cls]/total)*100:.1f}%" for cls in class_names}
    st.write(bee_data)

    # ===== Interpretasi sederhana =====
    st.subheader("🧠 BEE Analysis")
    most_common = behavior_counts.most_common(1)[0][0] if behavior_counts else "idle"

    if most_common == "idle":
        st.info("✅ Siswa tampak fokus dan tidak menunjukkan perilaku mencurigakan.")
    elif most_common == "gerakan kepala-bawah":
        st.warning("⚠️ Terdeteksi sering menunduk — kemungkinan sedang melihat ke bawah (mungkin mencontek?).")
    elif most_common == "gerakan kepala-leher":
        st.warning("⚠️ Sering menggerakkan kepala/leher — perlu diawasi, bisa jadi sedang melihat sekitar.")
    elif most_common == "gerakan tangan":
        st.error("🚨 Gerakan tangan sering terdeteksi — kemungkinan sedang memegang atau memindahkan sesuatu.")
    else:
        st.info("Tidak ada perilaku signifikan yang terdeteksi.")
