from edgetpumodel import EdgeTPUModel
from utils import get_image_tensor
import picamera2
import numpy as np
import time
import os
import argparse

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="models/yolo12n_320.tflite",
                    help='Path to the model file')
parser.add_argument('--names_path', type=str, default="models/label_yolo12.yaml",
                    help='Path to the labels file')
args = parser.parse_args()

# Initialize the camera
camera = picamera2.Picamera2()
camera_config = camera.create_preview_configuration(main={"format": "RGB888","size": (640, 480)})
camera.configure(camera_config)
camera.start()

# Use command line arguments for model paths
model = EdgeTPUModel(args.model_path, args.names_path, conf_thresh=0.7, iou_thresh=0.25)
input_shape = model.get_image_size()

# Variables to calculate FPS
frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture image
        full_image = camera.capture_array()
        # Resize and preprocess image
        full_image, net_image, pad = get_image_tensor(full_image, input_shape[0])
        # Predict
        pred = model.forward(net_image)
                
        det = model.process_predictions(pred[0], full_image, pad)        
        # Print results
        print(det)
        
        # Count frames and calculate FPS
        frame_count += 1
        if frame_count % 20 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            cpu_temp = os.popen("vcgencmd measure_temp").readline()
            print(f"FPS: {fps:.2f}, CPU Temp: {cpu_temp}")
            start_time = time.time()

except KeyboardInterrupt:
    print("Stopping...")
finally:
    camera.stop()

