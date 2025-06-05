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

# Step 2: Load base model and load your fine-tuned weights
model = AutoModel.from_pretrained(BASE_MODEL)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

# Step 3: Save model and processor in HF format
model.save_pretrained(LOCAL_DIR)

processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.save_pretrained(LOCAL_DIR)


# Step 4: Add README.md
readme_text = f"""\
---
license: apache-2.0
tags:
  - image-feature-extraction
  - image-text-retrieval
  - multimodal
  - siglip
  - person-search
datasets:
  - custom
language:
  - en
pipeline_tag: image-feature-extraction
---

# 🔍 SigLIP Person Search - Open Set

This model is a fine-tuned version of **`google/siglip-base-patch16-224`** for open-set **person retrieval** based on **natural language descriptions**. It's built to support **image-text similarity** in real-world retail and surveillance scenarios.

## 🧠 Use Case

This model allows you to search for people in crowded environments (like malls or stores) using only a **text prompt**, for example:

> "A man wearing a white t-shirt and carrying a brown shoulder bag"

The model will return person crops that match the description.

## 💾 Training

* Base: `google/siglip-base-patch16-224`
* Loss: Cosine InfoNCE
* Data: ReID dataset with multimodal attributes (generated via Gemini)
* Epochs: 10
* Usage: Retrieval-style search (not classification)

## 📈 Intended Use

* Smart surveillance
* Anonymous retail behavior tracking
* Human-in-the-loop retrieval
* Visual search & retrieval systems

## 🔧 How to Use

```python
from transformers import AutoProcessor, AutoModel
import torch

processor = AutoProcessor.from_pretrained("adonaivera/siglip-person-search-openset")
model = AutoModel.from_pretrained("adonaivera/siglip-person-search-openset")

text = "A man wearing a white t-shirt and carrying a brown shoulder bag"
inputs = processor(text=text, return_tensors="pt")
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
```

## 📌 Notes

* This model is optimized for **feature extraction** and **cosine similarity matching**
* It's not meant for classification or image generation
* Similarity threshold tuning is required depending on your application
"""
with open(os.path.join(LOCAL_DIR, "README.md"), "w") as f:
    f.write(readme_text)

# Step 5: Create and push to HF Hub
create_repo(REPO_NAME, exist_ok=True)
upload_folder(repo_id=REPO_NAME, folder_path=LOCAL_DIR, commit_message="Initial upload of fine-tuned SigLIP model")
print(f"✅ Uploaded to https://huggingface.co/{REPO_NAME}")
