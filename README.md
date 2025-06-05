# 🧠 SigLIP Person Finder + 📦 Multi-view ReID Dataset

[![Gradio Demo](https://huggingface.co/datasets/huggingface/badges/raw/main/gradio.svg)](https://huggingface.co/spaces/adonaivera/siglip-person-finder)

An open-set person search system using natural language descriptions and a curated ReID dataset enriched with rich semantic attributes.


## 🎯 Use Case

*"Can we find the woman in the white t-shirt with a brown shoulder bag?"*

That’s the goal. Instead of closed-set ID matching, this system enables **text-based retrieval** for both **images** and **videos** using a fine-tuned [SigLIP](https://arxiv.org/abs/2303.15343) model.


## 🎥 Try the Demo

Upload an image or video and enter a description:

## 📚 Example Prompt

> “A woman wearing blue jeans and a casual top, carrying a handbag over her shoulder.”

Result:
![example](assets/processed_image1.jpg)


> “A man is walking next to a bicycle, wearing blue denim bermuda shorts and a plain white t-shirt.”

Result:


## 🧩 Key Features

| Step                     | Component / Description                                                                          |
| ------------------------ |------------------------------------------------------------------------------------------------- |
| **1. Dataset Creation**  | Used **FiftyOne + Gemini Multimodal** to validate rich natural descriptions.                  |
| **2. YOLOv8**            | Detects people in each frame (image or video).                                                |
| **3. SigLIP**            | Embeds both person crops and text prompts into a shared embedding space.                      |
| **4. Cosine Similarity** | Measures how well a detected person matches the text description.                             |
| **5. Tracking**          | Tracks people over frames and reuses embeddings to reduce computation.                        |
| **6. Thresholding**      | Only draw bounding boxes in **red** if similarity exceeds your chosen threshold.              |

## 🗃️ Dataset: Multiview ReID + Visual Attributes

https://github.com/user-attachments/assets/aaa672df-0810-4fc4-8be9-fa44e60b9ee2

A curated, attribute-rich person re-identification dataset based on **Market-1501**, enhanced with:

* ✅ Multi-view images per person
* ✅ Detailed physical and clothing attributes
* ✅ Natural language descriptions
* ✅ Global attribute consolidation

## 📊 Dataset Statistics

| Subset     | Samples   |
| ---------- | --------- |
| Train      | 3,181     |
| Evaluation | 1,726     |
| Test       | 1,548     |
| **Total**  | **6,455** |

👉 [HF Dataset with Fiftyone](https://huggingface.co/datasets/adonaivera/fiftyone-multiview-reid-attributes)


## 🛠️ Dataset Creation Process

1. **Base Dataset**:
   * Used **Market-1501** as the foundation, which provides multi-camera views per identity.

2. **Duplicate Removal**:
   * Applied **DINOv2** embeddings to identify and remove near-duplicate samples.

3. **Attribute Generation**:
   * Used **Google Gemini Vision** to automatically generate:
     * Physical appearance details
     * Clothing descriptions
     * Natural language summaries

4. **Multi-view Merging**:
   * Attributes were consolidated across views for consistent representation.

## 🧱 Dataset Structure

Each sample includes:

* `filepath`: Path to image
* `person_id`: Person identity
* `camera_id`: Camera source
* `tags`: One of `["train", "query", "gallery"]`
* `attributes`:
  ```json
  {
    "gender": "Male",
    "age": "Adult",
    "ethnicity": "Unknown",
    "appearance": {...},
    "clothing": {...},
    "accessories": {...},
    "posture": {...},
    "actions": {...}
  }
  ```
* `description`: A clean natural language summary per person


## 📥 Installation

```bash
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add fiftyone huggingface-hub python-dotenv google-generativeai torch numpy torchvision pillow datasets transformers wandb sentencepiece ultralytics gradio spaces

Then run the app:

```bash
python app.py
```


## 🔑 Environment Setup

Add .env with the following variables:
```bash
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## 🧠 Why This Dataset?

This dataset is designed to enhance re-identification tasks with rich semantic cues.

📌 **Use cases include**:

* Person re-identification benchmarking
* Multi-view attribute consistency studies
* Natural language-based person search
* Attribute-conditioned retrieval systems

## ❗ Limitations & Ethical Considerations

* ⚠️ The base Market-1501 dataset may contain inherent demographic or collection biases.
* ⚠️ All attribute descriptions are **AI-generated** — may contain occasional hallucinations or uncertain estimations.
* ⚠️ Not suitable for deployment in **real-world surveillance** or **law enforcement** contexts without further validation.

## 🧠 Why SigLIP?

SigLIP replaces CLIP’s softmax loss with **pairwise sigmoid loss**, allowing independent image-text pair evaluations. This improves:

* Open-set scalability
* Cross-view generalization
* Natural language matching

## ⚡ Real-Time Constraint & Tracking Optimization

One of the key design goals was **near real-time performance** in video-based person search. Running inference for every frame is computationally expensive, especially with large transformer-based models like SigLIP.

To address this:

- ✅ We applied a **tracking mechanism** to follow detected individuals across frames.
- ✅ We only recompute SigLIP similarity every **30 frames (~1 second for 30 FPS)**, assuming identity consistency across tracked frames.
- ✅ This strategy drastically reduces redundant computations while maintaining retrieval accuracy.

This makes the system more suitable for real-world retail and surveillance contexts where speed is a priority.

## 🧪 Training Overview

We trained our open-set text-to-image retrieval model with the following setup:

| Item              | Description                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **Model**         | Fine-tuned [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) |
| **Samples**       | 3,181 multiview samples with natural language + attribute labels                                     |
| **Epochs**        | 10 (rapid prototyping with early stopping)                                                           |
| **Loss Function** | Symmetric cross-entropy over cosine similarities (InfoNCE-style)                                     |
| **Batch Size**    | 16                                                                                                   |
| **Optimizer**     | AdamW with cosine scheduler                                                                          |
| **Metrics**       | ✅ Recall\@1 and ✅ Recall\@5 on the query split (Test)                                              |
| **Thresholding**  | Cosine similarity threshold (0.1 – 0.3) used for open-set retrieval                                  |
| **Tracking**      | Applied to avoid recomputation over repeated detections in videos                                    |
| **Tools**         | [Weights & Biases](https://wandb.ai/) for loss monitoring & logging                                  |

We adopted a cosine-based InfoNCE loss approach for independent image-text pair comparisons, which enables scalable, open-set retrieval. During evaluation, **Recall\@1** and **Recall\@5** were logged to monitor retrieval effectiveness.

🧠 *Early results show improvement from near-random cosine scores (\~0.01) to meaningful matches (\~0.2–0.3), even with a small dataset.*

## ⚠️ Limitations

* Small dataset (\~3k training samples)
* Needs better lighting/angles for retail deployment
* SigLIP is slower for realtime solutions; uses re-evaluation every 30 frames for speed

## 🔭 Next Steps

* 🧪 Evaluate larger SigLIP variants
* 📏 Increase the dataset size (Working on a **larger curated multi-view dataset** with [Carlos](https://www.linkedin.com/in/phdcarloshinojosa/) and [Karen](https://www.linkedin.com/in/karenyanethsanchez/))
* 🧰 ONNX + TensorRT acceleration
* 🧠 Crowdsource richer descriptions

## 🙏 Thanks

* 📸 [Market-1501](https://github.com/zhunzhong07/IDE-baseline-Market-1501) authors
* 🔍 [FiftyOne](https://voxel51.com/fiftyone/)
* 🧠 [Gemini Vision](https://deepmind.google/discover/blog/gemini-1-5/)
* 📦 [SigLIP](https://github.com/google-research/vision_transformer)

## 📬 Contact

For questions, improvements, or bug reports:
➡️ Open an issue in the [GitHub repository](https://github.com/AdonaiVera/openset-reid-finetune)

## 📚 References

@inproceedings{zheng2015scalable,
  title={Scalable Person Re-identification: A Benchmark},
  author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages={1116--1124},
  year={2015}
}

