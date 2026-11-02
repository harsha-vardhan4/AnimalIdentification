from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

from model import AnimalCNN


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "animal_cnn.pth"


def predict_image(image_path):
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Train the CNN before making predictions.")

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    classes = checkpoint["class_names"]
    size = checkpoint["image_size"]

    model = AnimalCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ])

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    top_k = min(3, len(classes))
    scores, indices = probabilities.topk(top_k)

    print(f"\nImage: {image_path}")
    print("Predictions:")

    for score, index in zip(scores.tolist(), indices.tolist()):
        print(f"  {classes[index]}: {score:.2%}")

    return classes[indices[0]], scores[0].item()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/animal.jpg")
        sys.exit(1)

    label, confidence = predict_image(sys.argv[1])
    print(f"\nPredicted animal: {label}")
    print(f"Model confidence: {confidence:.2%}")