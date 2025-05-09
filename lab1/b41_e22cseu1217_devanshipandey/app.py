import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("car_plate_detector.h5")

# Function to detect car plate and draw bounding box
def detect_car_plate(image):
    orig_height, orig_width = image.shape[:2]  # Get original image size

    # Resize for model input (224x224)
    img_resized = cv2.resize(image, (224, 224))
    img_normalized = img_resized / 255.0  # Normalize

    # Prepare for prediction
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict bounding box
    pred = model.predict(img_input)[0]

    # Scale bounding box back to original image size
    xmin = int(pred[0] * orig_width)
    ymin = int(pred[1] * orig_height)
    xmax = int(pred[2] * orig_width)
    ymax = int(pred[3] * orig_height)

    # Draw rectangle on the original image
    img_with_box = image.copy()
    cv2.rectangle(img_with_box, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)  # Green bounding box

    return img_with_box

# Streamlit UI
st.title("Car Plate Detection App 🚗📸")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Detect car plate
    processed_image = detect_car_plate(image)

    # Convert processed image to PIL format for display
    processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    # Display uploaded image and processed image
    st.image([image, processed_image_pil], caption=["Original Image", "Detected Car Plate"], width=300)

