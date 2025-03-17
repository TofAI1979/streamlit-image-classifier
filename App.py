import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Load ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
class_labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit App
st.title("Dog and Cat Image Classifier 🐶🐱")

# File uploader for multiple images
uploaded_files = st.file_uploader("Upload images of dogs and cats", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Get predicted class
        predicted_class = torch.argmax(output[0]).item()
        predicted_label = class_labels[predicted_class]

        # Display results
        st.image(image, caption=f"Predicted: {predicted_label}", use_column_width=True)
        st.write(f"### Predicted Label: {predicted_label}")

