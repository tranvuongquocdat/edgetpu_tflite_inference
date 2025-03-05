from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import asyncio
import cv2
import numpy as np
import base64
import time
import os
from edgetpumodel import EdgeTPUModel
from utils import get_image_tensor
import picamera2
import argparse

app = FastAPI()

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="models/yolo12n_320.tflite",
                    help='Path to the model file')
parser.add_argument('--names_path', type=str, default="models/label_yolo12.yaml",
                    help='Path to the labels file')
args = parser.parse_args()

# Initialize camera
camera = picamera2.Picamera2()
camera_config = camera.create_preview_configuration(main={"format": "RGB888","size": (640, 480)})
camera.configure(camera_config)

# Initialize model
model = EdgeTPUModel(args.model_path, args.names_path, conf_thresh=0.7, iou_thresh=0.25)
input_shape = model.get_image_size()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera.start()
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture and process image
            full_image = camera.capture_array()
            full_image, net_image, pad = get_image_tensor(full_image, input_shape[0])
            
            # Predict
            pred = model.forward(net_image)
            det = model.process_predictions(pred[0], full_image, pad)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 20 == 0:
                end_time = time.time()
                fps = 10 / (end_time - start_time)
                start_time = time.time()
            else:
                fps = None
                
            # Encode image to send
            _, buffer = cv2.imencode('.jpg', full_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Send data
            await websocket.send_json({
                "image": img_str,
                "detections": det,
                "fps": fps
            })
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        camera.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
