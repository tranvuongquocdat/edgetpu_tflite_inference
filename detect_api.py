import websockets
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
import json

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="models/normal_model.tflite",
                    help='Path to the model file')
parser.add_argument('--names_path', type=str, default="models/normal_label.yaml",
                    help='Path to the labels file')
args = parser.parse_args()

# Initialize camera
camera = picamera2.Picamera2()
camera_config = camera.create_preview_configuration(main={"format": "RGB888","size": (640, 480)})
camera.configure(camera_config)

# Initialize model
model = EdgeTPUModel(args.model_path, args.names_path, conf_thresh=0.5, iou_thresh=0.25)
input_shape = model.get_image_size()

async def process_client(websocket):
    camera.start()
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture and process image
            full_image = camera.capture_array()
            # Rotate image 90 degrees counterclockwise
            full_image = cv2.rotate(full_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            full_image, net_image, pad = get_image_tensor(full_image, input_shape[0])
            
            # Predict
            pred = model.forward(net_image)
            det = model.process_predictions(pred[0], full_image, pad)
            
            # Calculate FPS and CPU temp every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                cpu_temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
                start_time = time.time()
            else:
                fps = None
                cpu_temp = None
                
            # Encode image to send
            _, buffer = cv2.imencode('.jpg', full_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Send data
            await websocket.send(json.dumps({
                "image": img_str,
                "detections": det.tolist(),  # Convert numpy array to list
                "fps": fps,
                "cpu_temp": cpu_temp
            }))
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        camera.stop()

async def main():
    server = await websockets.serve(
        process_client,
        "0.0.0.0",
        8000,
        ping_interval=20,
        ping_timeout=20
    )
    print("Server started on ws://0.0.0.0:8000")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down")
