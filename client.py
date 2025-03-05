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
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}/ws"
    
    while True:  # Add reconnection loop
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print(f"Connected to server at {uri}")
                
                while True:
                    try:
                        # Receive data from server with timeout
                        data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(data)
                        
                        # Decode image
                        img_bytes = base64.b64decode(data['image'])
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("Error: Could not decode image")
                            continue
                            
                        # Draw detections
                        detections = data['detections']
                        for det in detections:
                            x1, y1, x2, y2 = map(int, det[:4])
                            conf = det[4]
                            cls = int(det[5])
                            
                            # Add label with class name and confidence
                            label = f"Class {cls}: {conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Show FPS and CPU temp if available
                        if data['fps']:
                            fps_text = f"FPS: {data['fps']:.2f}"
                            cv2.putText(frame, fps_text,
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (0, 255, 0), 2)
                            
                        if data['cpu_temp']:
                            temp_text = f"CPU Temp: {data['cpu_temp']}°C"
                            cv2.putText(frame, temp_text,
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (0, 255, 0), 2)
                        
                        # Display frame
                        cv2.imshow('Detection Results', frame)
                        
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