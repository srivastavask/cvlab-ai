import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Initialize ONCE (global variables persist across runs)
processor, model = None, None

def load_model():
    global processor, model
    if model is None:  # Only load if not already loaded
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")

def describe_image(image_path: str) -> str:
    load_model()  # Safe to call multiple times (checks if already loaded)
    image = Image.open(image_path).convert("RGB")
    
    # Improved prompt
    inputs = processor(
        images=image,
        text="Describe this clothing item for an e-commerce search: a",
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

if __name__ == "__main__":
    desc = describe_image("images (1).jpeg")  # Replace with your image
    print(f"Description: {desc}")