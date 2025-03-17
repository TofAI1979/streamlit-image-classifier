import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import io

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🎨 UI Improvements
st.set_page_config(page_title="AI Image Classifier", page_icon="📷", layout="wide")

st.title("📷 AI Image Classifier")
st.markdown("#### Upload an image or paste a link to classify it!")
st.write("Supported formats: **JPG, PNG, JPEG**")

# Upload Section
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# URL Input Section
image_url = st.text_input("Or enter an image URL")

# Store images to process
images_to_process = []

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images_to_process.append(image)

# Handle image from URL
if image_url:
    try:
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            images_to_process.append(image)
    except Exception as e:
        st.error(f"⚠️ Unable to load image from URL: {e}")

# Classify and display images
if images_to_process:
    cols = st.columns(len(images_to_process))  # Create dynamic columns for images
    for i, image in enumerate(images_to_process):
        try:
            # Show a processing message
            with st.spinner("🔍 Classifying..."):
                # Apply transformations
                input_tensor = transform(image).unsqueeze(0)

                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)

                # Get predicted class
                predicted_class = torch.argmax(output[0]).item()
                predicted_label = class_labels[predicted_class]

            # Display the image
            with cols[i]:  
                st.image(image, use_container_width=True)
                st.success(f"✅ {predicted_label}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
