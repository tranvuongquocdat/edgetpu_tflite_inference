# apicam.py (Server)
import asyncio
import websockets
import cv2
import numpy as np
import json
import base64
import time

async def handle_client(websocket, path):
    cap = cv2.VideoCapture(0)  # Mở camera (0 là camera mặc định)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera")
                break

            # Resize frame để xử lý nhanh hơn
            frame = cv2.resize(frame, (640, 480))
            
            # Giả lập detection (có thể thay bằng model AI thực tế)
            detections = [
                [100, 100, 200, 200, 0.95, 0, "object"]  # [x1, y1, x2, y2, conf, class_id, class_name]
            ]
            
            # Vẽ detections lên frame
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det[6]} {det[4]:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)

            # Tính FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Mã hóa frame thành base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Chuẩn bị dữ liệu gửi đi
            data = {
                'image': img_base64,
                'detections': detections,
                'fps': fps,
                'cpu_temp': 50.5,  # Giả lập nhiệt độ CPU
                'inference_latency': 25.3  # Giả lập độ trễ
            }

            # Gửi dữ liệu
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.033)  # ~30 FPS

    except websockets.ConnectionClosed:
        print("Kết nối bị đóng")
    finally:
        cap.release()

async def main():
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",  # Lắng nghe trên tất cả interfaces
        8000
    )
    print("Server đang chạy tại 0.0.0.0:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())