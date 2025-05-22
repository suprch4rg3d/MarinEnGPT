import torch
import clip
from PIL import Image

def load_clip_model():
    # Load CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def analyze_image_with_clip(image_path: str) -> str:
    """
    Analyze an image using CLIP and generate a descriptive text.
    """
    # Load the CLIP model at the beginning to avoid reloading it every time
    model, preprocess, device = load_clip_model()
    
    # Open the image and preprocess it for CLIP
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Generate a prompt to describe the image using CLIP and GPT
    text_prompts = [
        "a detailed schematic of a ship engine",
        "an electrical circuit diagram",
        "a general layout of a marine vessel",
        "a safety protocol diagram",
        "a mechanical part diagram",
        "a close-up view of ship machinery"
    ]
    text_inputs = clip.tokenize(text_prompts).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Calculate similarity and find the best description
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    best_match_index = similarities.argmax().item()
    description = text_prompts[best_match_index]

    return description
