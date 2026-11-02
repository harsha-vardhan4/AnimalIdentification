from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from model import AnimalCNN


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "models" / "animal_cnn.pth"
DEVICE = torch.device("cpu")


def main():
    torch.set_num_threads(2)

    if not CHECKPOINT.exists():
        raise FileNotFoundError("Train the model first.")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    classes = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ])

    test_ds = datasets.ImageFolder(DATA / "test", transform=transform)

    if test_ds.classes != classes:
        raise ValueError("Test classes do not match checkpoint classes.")

    loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=0
    )

    model = AnimalCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            predictions = logits.argmax(dim=1).cpu()

            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())

    print(classification_report(
        actual, predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(classes)))
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=classes
    )
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.tight_layout()

    output = ROOT / "models" / "confusion_matrix.png"
    plt.savefig(output, dpi=150)
    print(f"Confusion matrix saved to: {output}")


if __name__ == "__main__":
    main()