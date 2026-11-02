from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import AnimalCNN


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "animal_cnn.pth"

st.set_page_config(
    page_title="Animal Identification",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 Animal Identification")
st.write("Identify one of 10 animal classes using a CNN trained from scratch.")


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    classes = checkpoint["class_names"]
    model = AnimalCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((checkpoint["image_size"],
                           checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ])

    return model, classes, transform


if not CHECKPOINT.exists():
    st.warning("The trained model is not available yet.")
    st.code("python src/train.py", language="powershell")
else:
    uploaded = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width="stretch")

        if st.button("Identify Animal", type="primary"):
            with st.spinner("Analyzing image..."):
                model, classes, transform = load_model()
                tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(tensor), dim=1
                    )[0]

                scores, indices = probabilities.topk(
                    min(3, len(classes))
                )

            best_index = indices[0].item()
            best_score = scores[0].item()

            st.subheader(f"Prediction: {classes[best_index].title()}")
            st.metric("Model confidence", f"{best_score:.1%}")
            st.caption(
                "Confidence is a model score, not a guarantee of correctness."
            )

            st.write("Top predictions")
            for score, index in zip(
                scores.tolist(), indices.tolist()
            ):
                st.write(f"**{classes[index].title()}** — {score:.1%}")
                st.progress(score)