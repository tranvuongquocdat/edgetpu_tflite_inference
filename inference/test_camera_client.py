# client.py
import websockets
import asyncio
import cv2
import numpy as np
import json
import base64
from websockets.exceptions import ConnectionClosed

SERVER_IP = "192.168.137.162"  # Thay bằng IP thực của Raspberry Pi
SERVER_PORT = "8000"

async def connect_to_server():
    uri = f"ws://{SERVER_IP}:{SERVER_PORT}"
    last_metrics = {'fps': None, 'cpu_temp': None, 'inference_latency': None}
    
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print(f"Đã kết nối tới server tại {uri}")
                
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(data)
                        
                        # Cập nhật metrics
                        for key in last_metrics:
                            if key in data and data[key] is not None:
                                last_metrics[key] = data[key]
                        
                        # Giải mã hình ảnh
                        img_bytes = base64.b64decode(data['image'])
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("Lỗi: Không thể giải mã hình ảnh")
                            continue
                        
                        # Hiển thị thông tin metrics
                        y_offset = 30
                        for key, value in last_metrics.items():
                            if value is not None:
                                text = f"{key.upper()}: {value:.2f}{'°C' if key == 'cpu_temp' else 'ms' if key == 'inference_latency' else ''}"
                                cv2.putText(frame, text, (10, y_offset), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                y_offset += 30
                        
                        # Hiển thị frame
                        cv2.imshow('AI Camera Detection', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            return
                            
                    except asyncio.TimeoutError:
                        print("Hết thời gian chờ phản hồi từ server")
                        break
                        
        except ConnectionClosed:
            print("Kết nối bị đóng, đang thử kết nối lại...")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            cv2.destroyAllWindows()
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(connect_to_server())
    except KeyboardInterrupt:
        print("\nĐang thoát chương trình")
        cv2.destroyAllWindows()