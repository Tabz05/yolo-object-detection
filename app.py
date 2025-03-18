import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

# Title of the web app
st.title("YOLO Object Detection App")

# Load your fine-tuned YOLO model
model = YOLO("runs\\detect\\train\\weights\\best.pt")  # Replace with your model path

# File uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Display the original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    results = model(image)

    # Extract bounding boxes, labels, and confidence scores
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        labels = result.boxes.cls  # Class labels
        confidences = result.boxes.conf  # Confidence scores

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for box, label, confidence in zip(boxes, labels, confidences):
            x1, y1, x2, y2 = box.tolist()
            label_name = model.names[int(label)]  # Get class name from label ID
            confidence_score = float(confidence)

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Add label and confidence score
            text = f"{label_name} {confidence_score:.2f}"
            draw.text((x1, y1 - 10), text, fill="red")

    # Display the image with bounding boxes
    st.image(image, caption="Detected Objects", use_column_width=True)
else:
    st.write("Please upload an image file.")
