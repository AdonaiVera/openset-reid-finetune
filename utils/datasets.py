from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from fiftyone.utils.huggingface import load_from_hub

class TextImageDataset(Dataset):
    def __init__(self, dataset, split='train', processor=None, augment=False):
        self.fo_dataset = dataset.match_tags(split)
        self.samples = list(self.fo_dataset.iter_samples())
        self.processor = processor
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=384, scale=(0.9, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        ]) if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.filepath
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        attributes = []
        if sample.attributes:
            for category, attrs in sample.attributes.items():
                if isinstance(attrs, dict):
                    for attr, value in attrs.items():
                        if value and value != "Unknown":
                            attributes.append(f"{attr}: {value}")
                elif attrs and attrs != "Unknown":
                    attributes.append(f"{category}: {attrs}")

        text = sample.description
        if attributes:
            text = f"{text}. This person has {', '.join(attributes)}"

        return {
            "image": image,
            "text": text
        }
    
def load_dataset():
    dataset = load_from_hub(
        repo_id="adonaivera/fiftyone-multiview-reid-attributes",
        dataset_name="fiftyone-multiview-reid2",
        overwrite=True
    )
    return dataset