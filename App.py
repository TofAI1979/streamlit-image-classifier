import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
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
st.markdown("#### Upload images and classify them!")
st.write("Supported formats: **JPG, PNG, JPEG**")

# Upload Section
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

# Store images and results
images_to_process = []
classifications = []

# Handle uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images_to_process.append((uploaded_file.name, image))

# Classify and display images
if images_to_process:
    cols = st.columns(len(images_to_process))  # Create dynamic columns for images
    for i, (image_name, image) in enumerate(images_to_process):
        try:
            # Resize image for uniform preview (~3x3 cm)
            preview_image = image.resize((90, 90))

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

                # Store classification result
                classifications.append(f"{image_name}: {predicted_label}")

            # Display the resized image
            with cols[i]:  
                st.image(preview_image, use_container_width=False)
                st.success(f"✅ {predicted_label}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Add "Download Predictions" button if classifications exist
if classifications:
    st.markdown("---")  # Separator for better layout
    st.markdown("### 📥 Download Classification Results")

    # Convert classifications to a downloadable text file
    classification_text = "\n".join(classifications)
    file_buffer = io.BytesIO()
    file_buffer.write(classification_text.encode())
    file_buffer.seek(0)

    # Display the download button
    st.download_button(
        label="📄 Download Results",
        data=file_buffer,
        file_name="classification_results.txt",
        mime="text/plain",
    )
