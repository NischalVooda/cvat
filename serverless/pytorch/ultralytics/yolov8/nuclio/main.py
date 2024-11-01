from ultralytics import YOLO
import json, base64, io, os
from PIL import Image

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read/Install the DL model
    model = YOLO('yolov8n.pt')

    use_gpu = os.getenv("RUN_ON_GPU", 'False').lower() in ('true', '1') # Import the GPU env variable and covert to a boolean value
    print(f"CUDA-STATUS: {use_gpu}")
    if use_gpu:
        model.to('cuda')

    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.6))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    yolo_results = context.user_data.model(image, conf=threshold)
    results = yolo_results[0]

    encoded_results = [] # JSON format
    for idx, class_idx in enumerate(results.boxes.cls):
        confidence = results.boxes.conf[idx].item()
        label = results.names[int(class_idx.item())]
        points = results.boxes.xyxy[idx].tolist()
        encoded_results.append({
            'confidence': confidence,
            'label': label,
            'points': points,
            'type': 'rectangle'
        })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)