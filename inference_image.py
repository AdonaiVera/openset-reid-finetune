import os
import torch
import cv2
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel
from PIL import Image

# Config
IMAGE_PATH = "assets/test_image1.jpg"
OUTPUT_PATH = "output/processed_image.jpg"
HF_MODEL = "adonaivera/siglip-person-search-openset"
TEXT_PROMPT = "A man wearing a white t-shirt and carrying a brown shoulder bag."
SIM_THRESHOLD = 0.15

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model from Hugging Face
processor = AutoProcessor.from_pretrained(HF_MODEL)
model = AutoModel.from_pretrained(HF_MODEL).to(device)
model.eval()

# Encode prompt text
with torch.no_grad():
    text_inputs = processor(text=TEXT_PROMPT, return_tensors="pt", padding=True).to(device)
    text_feat = model.get_text_features(**text_inputs)
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)

# Load image
image = cv2.imread(IMAGE_PATH)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect persons with YOLO
detector = YOLO("yolov8n.pt")
results = detector(rgb_image)[0]
boxes = results.boxes

# Process detections
if boxes is not None:
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].item())

        if cls_id != 0:
            continue  # Only process persons

        # Draw default box (green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop and encode with SigLIP
        crop = rgb_image[y1:y2, x1:x2]
        image_input = processor(images=crop, return_tensors="pt").to(device)

        with torch.no_grad():
            image_feat = model.get_image_features(**image_input)
            image_feat = torch.nn.functional.normalize(image_feat, dim=-1)
            sim = torch.matmul(image_feat, text_feat.T).item()

        # Label the similarity
        cv2.putText(image, f"{sim:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Highlight matched person
        if sim > SIM_THRESHOLD:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "match_found", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Save output
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Processed image saved to {OUTPUT_PATH}")
