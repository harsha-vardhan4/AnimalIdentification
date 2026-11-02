from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import AnimalCNN


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
CHECKPOINT = MODEL_DIR / "animal_cnn.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 4
SEED = 42
DEVICE = torch.device("cpu")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    if not (DATA / "train").is_dir():
        raise FileNotFoundError(
            "Dataset missing. Run python src/prepare_data.py first."
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    train_ds = datasets.ImageFolder(
        DATA / "train", transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        DATA / "val", transform=eval_transform
    )

    if train_ds.classes != val_ds.classes:
        raise ValueError("Training and validation classes differ.")

    if len(train_ds.classes) != 10:
        raise ValueError("Expected exactly 10 classes.")

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    print("Classes:", train_ds.classes)
    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print("Training device:", DEVICE)

    model = AnimalCNN(num_classes=len(train_ds.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

        train_loss = total_loss / count
        train_acc = correct / count

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (
                    logits.argmax(1) == labels
                ).sum().item()
                val_count += labels.size(0)

        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": best_state,
                "class_names": train_ds.classes,
                "image_size": IMAGE_SIZE,
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.5, 0.5, 0.5],
            }, CHECKPOINT)

            print("  Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training did not save a model.")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved model: {CHECKPOINT}")


if __name__ == "__main__":
    main()