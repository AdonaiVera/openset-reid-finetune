import os
import shutil
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoProcessor, AutoModel

REPO_NAME = "adonaivera/siglip-person-search-openset"
LOCAL_DIR = "siglip-person-search-openset"
MODEL_PATH = "models/best_model_epoch_22_loss_0.0273.pt"
BASE_MODEL = "google/siglip-base-patch16-224"

# Step 1: Create local dir structure
os.makedirs(LOCAL_DIR, exist_ok=True)

# Step 2: Copy model weights
shutil.copy(MODEL_PATH, os.path.join(LOCAL_DIR, "pytorch_model.bin"))

# Step 3: Save config and processor from base model
model = AutoModel.from_pretrained(BASE_MODEL)
model.config.save_pretrained(LOCAL_DIR)

processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(LOCAL_DIR)

# Step 4: Add README.md
readme_text = f"""\
# SigLIP Person Search (Open-Set)

Fine-tuned version of [`{BASE_MODEL}`] for open-set person search using multimodal similarity.

**Usage:** Given a text prompt (e.g., `"man wearing red t-shirt and jeans"`), this model can rank person detections by similarity.

**Base model:** `{BASE_MODEL}`  
**Author:** @adonaivera  
"""
with open(os.path.join(LOCAL_DIR, "README.md"), "w") as f:
    f.write(readme_text)

# Step 5: Create and push to HF Hub
create_repo(REPO_NAME, exist_ok=True)
upload_folder(repo_id=REPO_NAME, folder_path=LOCAL_DIR, commit_message="Initial upload of fine-tuned SigLIP model")
print(f"✅ Uploaded to https://huggingface.co/{REPO_NAME}")
