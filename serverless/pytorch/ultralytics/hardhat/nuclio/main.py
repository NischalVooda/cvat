import cv2
import json, base64, io, os
from PIL import Image
from ultralytics import YOLO
import numpy as np

def init_context(context):
    context.logger.info("Initializing context... 0%")

    # Download the model using wget
    os.system("wget -O /opt/nuclio/best.pt https://github.com/NischalVooda/Data-Annotation/raw/main/best.pt")
    
    # Load the custom YOLO model
    model = YOLO('/opt/nuclio/best.pt')

    use_gpu = os.getenv("RUN_ON_GPU", 'False').lower() in ('true', '1')
    print(f"CUDA-STATUS: {use_gpu}")
    if use_gpu:
        model.to('cuda')

    context.user_data.model = model
    context.user_data.class_names = [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "No-Vest",
        "Worker", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ]

    context.logger.info("Context initialized...100%")

def handler(context, event):
    context.logger.info("Running custom YOLO model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    
    # Read the image using OpenCV from the decoded bytes
    np_image = np.frombuffer(buf.getvalue(), np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        return context.Response(
            body=json.dumps({"error": "Failed to load image"}),
            headers={}, content_type='application/json', status_code=400
        )

    model = context.user_data.model
    class_names = context.user_data.class_names

    results = model(image)
    detected_classes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0].item()
            if confidence < threshold:
                continue

            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = class_names[class_id]

            detected_classes.append({
                'confidence': confidence,
                'label': class_name,
                'points': [x1, y1, x2, y2],
                'type': 'rectangle'
            })

    return context.Response(
        body=json.dumps(detected_classes),
        headers={}, content_type='application/json', status_code=200
    )
