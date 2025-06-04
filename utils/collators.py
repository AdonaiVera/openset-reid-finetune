import torch

class TextImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        inputs = self.processor(images=images, text=texts, padding=True, truncation=True, return_tensors="pt")

        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        return inputs