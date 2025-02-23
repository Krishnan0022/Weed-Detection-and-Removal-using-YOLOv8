# Weed Detection and Removal using YOLOv8

## Overview
This project aims to develop an AI-powered system for detecting and removing weeds in agricultural fields using the YOLOv8 object detection model. By distinguishing between crops and weeds, the system facilitates automated weed removal strategies, enhancing crop yield and reducing manual labor.

## Features
- **Automated Weed Detection**: Uses YOLOv8 for precise identification of weeds.
- **Real-time Processing**: Implements OpenCV and DeepSORT for tracking.
- **Model Training**: Trains YOLOv8 on labeled datasets for accurate classification.
- **Edge Deployment**: Runs on Raspberry Pi or Jetson Nano for field use.
- **Web Interface**: Deploys using Streamlit for easy access.

## Dataset
We use the [Crop and Weed Detection Dataset](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes), which contains labeled images of crops and weeds with bounding boxes formatted for YOLO.

### Downloading the Dataset
```bash
kaggle datasets download -d ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes
unzip crop-and-weed-detection-data-with-bounding-boxes.zip -d data/
```

## Installation
### Clone the Repository
```bash
git clone https://github.com/yourusername/weed-detection-yolov8.git
cd weed-detection-yolov8
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
The `requirements.txt` includes:
```text
ultralytics
opencv-python
numpy
torch
torchvision
streamlit
supervision
```

## Data Preparation
1. Organize the dataset:
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
2. Ensure images and labels are in YOLO format.

## Training the Model
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='data/dataset.yaml', epochs=50, imgsz=640)
```
Define `dataset.yaml` with paths and class names.

## Real-Time Detection
```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/train/exp/weights/best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls[0])]
            confidence = box.conf[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Weed Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Deployment using Streamlit
Create `app.py`:
```python
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("Weed Detection using YOLOv8")
model = YOLO('runs/train/exp/weights/best.pt')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    results = model(np.array(image))
    st.image(results.render()[0], caption="Detection Results", use_column_width=True)
```
Run the app:
```bash
streamlit run app.py
```

## Results
- **Precision & Recall Analysis**: Evaluate detection accuracy.
- **Speed & Latency**: Measure real-time performance on edge devices.
- **Deployment Feasibility**: Assess practical field application.

## Contributing
Contributions are welcome! Fork the repository and submit a pull request.


