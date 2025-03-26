from picamera2 import Picamera2, Preview
import time

# Khởi tạo đối tượng Picamera2
picam2 = Picamera2()

# Tạo cấu hình xem trước mặc định
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

# Bật chế độ xem trước (preview)
picam2.start_preview(Preview.QTGL)  # Sử dụng QTGL để hiển thị cửa sổ xem trước
picam2.start()

# Chờ 5 giây để xem trước
print("Đang hiển thị preview, kiểm tra màn hình...")
time.sleep(5)

# Chụp một bức ảnh và lưu dưới tên "test_image.jpg"
picam2.capture_file("test_image.jpg")
print("Đã chụp ảnh, kiểm tra file 'test_image.jpg' trong thư mục hiện tại.")

# Dừng camera
picam2.stop_preview()
picam2.stop()