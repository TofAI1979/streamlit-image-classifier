import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# Load the pre-trained ResNet-18 model
# Previous: model = models.resnet18(pretrained=True)
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
st.markdown("#### Upload images of cats and dogs, and let AI classify them!")
st.write("Supported formats: **JPG, PNG, JPEG**")

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Upload Section
uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# Store files in session state if uploaded
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Remove All Button (Fixed)
if st.session_state.uploaded_files:
    if st.button("🗑️ Remove All", key="remove_all_button"):
        st.session_state.uploaded_files = []  # Clear the session state
        st.experimental_rerun()  # Refresh the app

# Display uploaded images if they exist
if st.session_state.uploaded_files:
    cols = st.columns(len(st.session_state.uploaded_files))  # Create dynamic columns for images
    for i, uploaded_file in enumerate(st.session_state.uploaded_files):
        try:
            # Open image
            image = Image.open(uploaded_file).convert("RGB")

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

            # Display the image in a column **without a caption**
            with cols[i]:  
                st.image(image, use_container_width=True)  # Removed caption
                st.success(f"✅ {predicted_label}")  # Still show the prediction
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
