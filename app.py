import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io

# Load the pre-trained model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: TB, Normal
model.load_state_dict(torch.load("models/tuberculosis_model.pth"))
model.eval()

# Define transformation for the uploaded image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Helper function to make predictions
def predict_image(image_bytes):
    try:
        # Try opening the image using PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure the image is in RGB format (in case it's grayscale or in another mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformation (resize, normalize, etc.)
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict using the model
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            return predicted.item(), confidence

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Streamlit UI layout
st.title("Tuberculosis Detection in Chest X-rays")
st.write(
    "Upload a chest X-ray image to detect whether it shows signs of Tuberculosis (TB)."
)

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded image file
        image_bytes = uploaded_file.read()  # Read the file in bytes
        image = Image.open(io.BytesIO(image_bytes))  # Try to open it as an image

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions
        st.write("Classifying image...")

        # Get prediction and confidence
        label, confidence = predict_image(image_bytes)

        if label is not None:
            # Map prediction to class labels
            if label == 0:
                result = "Normal"
            else:
                result = "Tuberculosis (TB)"

            # Display results
            st.subheader(f"Prediction: {result}")
            st.write(f"Confidence: {confidence[label]:.2f}%")
        else:
            st.error("Could not classify the image.")

    except Exception as e:
        st.error(f"Error opening image: {e}")
