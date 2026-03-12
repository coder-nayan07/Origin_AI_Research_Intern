import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def test_setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        model.to(device)
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    test_setup()