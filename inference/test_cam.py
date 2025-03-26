from picamera2 import Picamera2
import time

# Khởi tạo đối tượng Picamera2
picam2 = Picamera2()

# Tạo cấu hình mặc định
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

# Bật camera
print("Đang khởi động camera...")
picam2.start()

# Chờ 5 giây
time.sleep(5)

# Chụp ảnh và lưu
print("Đang chụp ảnh...")
picam2.capture_file("test_image.jpg")
print("Đã chụp ảnh, kiểm tra file 'test_image.jpg'.")

# Dừng camera
picam2.stop()