# 📦 FiftyOne-Compatible Multiview Person ReID with Visual Attributes

A curated, attribute-rich person re-identification dataset based on **Market-1501**, enhanced with:

* ✅ Multi-view images per person
* ✅ Detailed physical and clothing attributes
* ✅ Natural language descriptions
* ✅ Global attribute consolidation

## 📊 Dataset Statistics

| Subset    | Samples   |
| --------- | --------- |
| Train     | 3,181     |
| Query     | 1,726     |
| Gallery   | 1,548     |
| **Total** | **6,455** |

## 📥 Installation

```bash
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add fiftyone huggingface-hub python-dotenv google-generativeai torch numpy torchvision pillow datasets transformers wandb
```

## 🔑 Environment Setup

Add .env with the following variables:
```bash
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## 🚀 Usage

```python
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset("adonaivera/fiftyone-multiview-reid-attributes")
session = fo.launch_app(dataset)
```

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

## 📜 License

**CC-BY-4.0**
Please cite and credit if using in academic or applied research.

## 🙏 Acknowledgments

* Market-1501 dataset creators
* Google Gemini Vision model
* Voxel51 & FiftyOne team

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
