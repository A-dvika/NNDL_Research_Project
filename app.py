import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from torchvision import models, transforms
from PIL import Image

# Disable Streamlit's file watcher to prevent errors
os.environ["TORCH_HOME"] = "./"

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2(weights=None)  # No pretrained weights since we load our own
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 7)

# Load trained model weights
model_path = "C://Users//HP//Desktop//NNDL_Project//mobilenet_v2//mobilenet_v2_emotion_epoch_15.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Emotion labels (update as per FER-2013 dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Streamlit UI
st.title("Real-Time Emotion Detection")

# Start webcam button
start_webcam = st.button("Start Webcam")

if start_webcam:
    cap = cv2.VideoCapture(0)  # Open webcam

    # Check if the webcam is opened
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        st.success("Webcam started! Close the app to stop.")

        # Display webcam feed with real-time prediction
        stframe = st.empty()  # Create a placeholder for the video frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            # Convert BGR (OpenCV) to RGB (PIL format)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)

            # Preprocess image
            image_tensor = transform(image_pil).unsqueeze(0).to(device)

            # Perform inference
            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                emotion = emotion_labels[predicted_class]

            # Display predicted emotion on frame
            cv2.putText(image, f"Emotion: {emotion}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the frame in Streamlit
            stframe.image(image, channels="RGB", use_container_width=True)

        cap.release()  # Release the webcam
        cv2.destroyAllWindows()  # Close OpenCV windows
