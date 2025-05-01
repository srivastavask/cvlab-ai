# First install required packages
from tensorflow.keras.utils import pad_sequences

import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Missing import
import pickle

def load_models():
    try:
        # Load with proper error handling
        caption_model = load_model("model.keras")
        feature_extractor = load_model("feature_extractor.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return caption_model, feature_extractor, tokenizer
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

caption_model, feature_extractor, tokenizer = load_models()

def generate_caption(image):
    if image is None:  # Handle empty input
        return "Please upload an image"
    
    try:
        # Convert and validate image
        img = Image.fromarray(image.astype('uint8')).resize((224, 224))
        img_array = np.array(img) / 255.0
        if img_array.shape != (224, 224, 3):
            raise ValueError("Invalid image dimensions")
            
        # Feature extraction
        img_input = np.expand_dims(img_array, axis=0)
        image_features = feature_extractor.predict(img_input, verbose=0)
        
        # Caption generation
        caption = "startseq"
        for _ in range(34):  # Use max_length from training
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=34)  # Match training max_length
            yhat = caption_model.predict([image_features, seq], verbose=0)
            predicted_word = tokenizer.index_word.get(np.argmax(yhat), "")
            
            if not predicted_word or predicted_word == "endseq":
                break
            caption += " " + predicted_word

        return caption.replace("startseq", "").replace("endseq", "").strip()
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create interface with improved UI
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="📷 Upload Image"),
    outputs=gr.Textbox(label="📝 Generated Caption"),
    title="AI Image Caption Generator",
    examples=["/kaggle/input/flickr8k/Images/667626_18933d713e.jpg"],  # Add example images
    allow_flagging="never"
)

demo.launch(share=True)  # Enable sharing option
