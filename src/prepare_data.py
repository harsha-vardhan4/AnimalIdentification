from pathlib import Path
import random
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CLASSES = [
    "bear", "cat", "cow", "deer", "dog",
    "elephant", "Giraffe", "horse", "lion", "tiger",
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    rng = random.Random(42)

    if not RAW.exists():
        raise FileNotFoundError(f"Create the dataset folder: {RAW}")

    if OUT.exists():
        shutil.rmtree(OUT)

    for animal in CLASSES:
        folder = RAW / animal

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        images = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
            and valid_image(p)
        ]

        rng.shuffle(images)

        if len(images) < 20:
            raise ValueError(
                f"{animal}: only {len(images)} valid images. "
                "Add at least 20; more images are strongly recommended."
            )

        n = len(images)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, paths in splits.items():
            destination = OUT / split / animal
            destination.mkdir(parents=True, exist_ok=True)

            for index, source in enumerate(paths):
                # Unique filenames prevent collisions.
                target = destination / f"{index:06d}_{source.name}"
                shutil.copy2(source, target)

        print(
            f"{animal:10s}: "
            + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        )

    print(f"\nDataset prepared at: {OUT}")


if __name__ == "__main__":
    main()