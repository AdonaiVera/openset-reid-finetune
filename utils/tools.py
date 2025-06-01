import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

class ImageDuplicateRemover:
    def __init__(self, device=None):
        """
        Initialize the ImageDuplicateRemover with DINOv2 model.
        
        Args:
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DINOv2 model
        print("Loading DINOv2 model...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformation
        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def remove_repeated_images(self, image_paths: List[str], similarity_threshold: float = 0.95, batch_size: int = 32) -> List[str]:
        """
        Remove duplicate images using DINOv2 embeddings similarity comparison with batch processing.
        
        Args:
            image_paths (List[str]): List of paths to images
            similarity_threshold (float): Threshold for considering images as duplicates (0.0 to 1.0)
            batch_size (int): Number of images to process at once
            
        Returns:
            List[str]: List of unique image paths
        """
        # Process images in batches
        all_embeddings = []
        valid_paths = []
        
        print("Computing embeddings in batches...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            for image_path in batch_paths:
                try:
                    img = Image.open(image_path).convert('RGB')
                    tensor = self.transform(img).unsqueeze(0)
                    batch_tensors.append(tensor)
                    valid_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack batch and compute embeddings
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                embeddings = self.model(batch)
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings
        if not all_embeddings:
            return []
        
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        
        print("Finding unique images...")
        # Compute similarity matrix using matrix multiplication
        similarity_matrix = np.dot(all_embeddings, all_embeddings.T)
        
        # Find unique images
        unique_indices = []
        used_indices = set()
        
        for i in range(len(similarity_matrix)):
            if i in used_indices:
                continue
            
            unique_indices.append(i)
            # Find similar images
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            # Add all similar images to used set
            used_indices.update(similar_indices[similar_indices > i])
        
        final_images = [valid_paths[i] for i in unique_indices]
        
        print(f"Removed {len(valid_paths) - len(final_images)} duplicate images")
        return final_images 