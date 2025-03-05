import websockets
import asyncio
import cv2
import numpy as np
import json
import base64

SERVER_IP = "192.168.1.100"  # Thay đổi IP này thành IP của Raspberry Pi
SERVER_PORT = "8000"

async def connect_to_server():
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                # Receive data from server
                data = await websocket.recv()
                data = json.loads(data)
                
                # Decode image
                img_bytes = base64.b64decode(data['image'])
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                # Draw detections
                detections = data['detections']
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    conf = det[4]
                    cls = int(det[5])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Class: {cls} Conf: {conf:.2f}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
                
                # Show FPS if available
                if data['fps']:
                    cv2.putText(frame, f"FPS: {data['fps']:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Detection Results', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(connect_to_server()) 