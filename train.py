import argparse
import multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os

from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.datasets import TextImageDataset, load_dataset
from utils.collators import TextImageCollator

class SigLIPTrainer:
    def __init__(self, model, dataloader, optimizer, scheduler, device, temperature=0.07):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.temperature = temperature

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(self.dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

            image_features = self.model.get_image_features(pixel_values=batch['pixel_values'])
            text_features = self.model.get_text_features(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
            labels = torch.arange(len(image_features), device=self.device)

            loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_image.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            wandb.log({"loss": loss.item()})
            total_loss += loss.item()
            tqdm.write(f"Step {step} | Loss: {loss.item():.4f}")

        return total_loss / len(self.dataloader)

    @torch.no_grad()
    def validate(self, val_dataloader):
        self.model.eval()
        sims = []
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            image_feats = self.model.get_image_features(batch['pixel_values'])
            text_feats = self.model.get_text_features(batch['input_ids'], batch['attention_mask'])
            image_feats = F.normalize(image_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)
            cosine_sim = torch.sum(image_feats * text_feats, dim=-1)
            sims.extend(cosine_sim.cpu().tolist())
        return np.mean(sims)

def main():
    parser = argparse.ArgumentParser(description="Train SigLIP for open-set text-image similarity.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    # Init W&B
    wandb.init(project="text-image-siglip", config=vars(args))

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset and model
    print("Loading dataset and model...")
    dataset = load_dataset()
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare dataloaders
    train_ds = TextImageDataset(dataset, split="train", processor=processor, augment=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TextImageCollator(processor),
        num_workers=mp.cpu_count() - 1,
    )

    val_ds = TextImageDataset(dataset, split="val", processor=processor, augment=False)
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TextImageCollator(processor),
        num_workers=mp.cpu_count() - 1,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = args.epochs * len(train_dl)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    # Trainer
    trainer = SigLIPTrainer(model, train_dl, optimizer, scheduler, device)

    # Training loop with early stopping
    best_val_score = -float("inf")
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        avg_loss = trainer.train_epoch()
        val_score = trainer.validate(val_dl)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val Cosine Sim: {val_score:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": avg_loss,
            "val_score": val_score
        })

        if val_score > best_val_score:
            best_val_score = val_score
            no_improve_epochs = 0
            best_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch + 1}_val_{val_score:.4f}.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved to {best_path}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement. Patience counter: {no_improve_epochs}/{args.patience}")

        if no_improve_epochs >= args.patience:
            print("🛑 Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
