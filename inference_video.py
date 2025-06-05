import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

# Config
VIDEO_PATH = "input/test_demo1.mp4"
OUTPUT_PATH = "output/result.mp4"
HF_MODEL = "adonaivera/siglip-person-search-openset"
MODEL_FILENAME = "best_model_epoch_22_loss_0.0273.pt"
TEXT_PROMPT = "A woman wearing blue jeans and a casual top, carrying a white shoulder bag."
SIM_THRESHOLD = 0.15
INTERVAL_FRAMES = 30

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load SigLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(HF_MODEL)
model = AutoModel.from_pretrained(HF_MODEL).to(device)
model.eval()

# Encode target text
with torch.no_grad():
    text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt", padding=True).to(device)
    text_feat = model.get_text_features(**text_inputs)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

# Load YOLOv8 with tracker
detector = YOLO("yolov8n.pt")

# Prepare video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

track_last_eval_frame = {}
track_last_sim = {}
frame_idx = 0

# Frame loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.track(frame, persist=True)
    boxes = results[0].boxes

    if boxes is None:
        out.write(frame)
        frame_idx += 1
        continue

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].item())
        track_id = int(box.id[0].item()) if box.id is not None else -1

        if cls_id != 0:
            continue

        # Default bounding box for all people
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Evaluate similarity only every INTERVAL_FRAMES
        last_eval = track_last_eval_frame.get(track_id, -INTERVAL_FRAMES)
        if frame_idx - last_eval >= INTERVAL_FRAMES:
            crop = frame[y1:y2, x1:x2]
            image_input = processor(images=crop, return_tensors="pt").to(device)

            with torch.no_grad():
                image_feat = model.get_image_features(**image_input)
                image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
                sim = torch.matmul(image_feat, text_feat.T).item()

            track_last_sim[track_id] = sim
            track_last_eval_frame[track_id] = frame_idx

        else:
            sim = track_last_sim.get(track_id, 0.0)

        if sim > SIM_THRESHOLD:
            # Highlight match in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{sim:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Done. Output saved to {OUTPUT_PATH}")
