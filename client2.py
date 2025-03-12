import websockets
import asyncio
import cv2
import numpy as np
import json
import base64
from websockets.exceptions import ConnectionClosed

SERVER_IP = "192.168.137.246"  # Thay đổi IP này thành IP của Raspberry Pi
SERVER_PORT = "8000"

async def connect_to_server():
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}"  # Loại bỏ /ws vì apicam.py không sử dụng path
    last_fps = None
    last_cpu_temp = None
    last_inference_latency = None
    
    while True:  # Add reconnection loop
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print(f"Connected to server at {uri}")
                
                while True:
                    try:
                        # Receive data from server with timeout
                        data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(data)
                        
                        # Update last values if new data is available
                        if 'fps' in data and data['fps'] is not None:
                            last_fps = data['fps']
                        if 'cpu_temp' in data and data['cpu_temp'] is not None:
                            last_cpu_temp = data['cpu_temp']
                        if 'inference_latency' in data and data['inference_latency'] is not None:
                            last_inference_latency = data['inference_latency']
                        
                        # Decode image
                        img_bytes = base64.b64decode(data['image'])
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("Error: Could not decode image")
                            continue
                            
                        # Draw detections - apicam.py already draws on the image, but we can enhance it
                        detections = data['detections']
                        for det in detections:
                            # Format in apicam.py: [x1, y1, x2, y2, confidence, class_id, class_name]
                            x1, y1, x2, y2 = map(int, det[:4])
                            conf = det[4]
                            cls_id = int(det[5])
                            cls_name = det[6]
                            
                            # We don't need to redraw since the server already drew them
                            # But we could add additional visualization if needed
                        
                        # Always show the last known values
                        y_offset = 30
                        if last_fps is not None:
                            fps_text = f"FPS: {last_fps:.2f}"
                            cv2.putText(frame, fps_text,
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, (0, 255, 0), 2)
                            y_offset += 30
                            
                        if last_inference_latency is not None:
                            latency_text = f"Inference: {last_inference_latency:.2f} ms"
                            cv2.putText(frame, latency_text,
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, (0, 255, 0), 2)
                            y_offset += 30
                            
                        if last_cpu_temp is not None:
                            temp_text = f"CPU Temp: {last_cpu_temp}°C"
                            cv2.putText(frame, temp_text,
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.7, (0, 255, 0), 2)
                        
                        # Display frame
                        cv2.imshow('AI Camera Detection', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return  # Clean exit
                            
                    except asyncio.TimeoutError:
                        print("Timeout waiting for server response")
                        break
                        
        except ConnectionClosed:
            print("Connection closed by server, attempting to reconnect...")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)  # Wait before reconnecting
        except Exception as e:
            print(f"Connection error: {e}")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)  # Wait before reconnecting

if __name__ == "__main__":
    while True:  # Add infinite loop to keep program running
        try:
            asyncio.get_event_loop().run_until_complete(connect_to_server())
        except KeyboardInterrupt:
            print("\nExiting program")
            break 