#pragma once
// ĐỪNG commit file này lên git thật (thêm "secrets.h" vào .gitignore).
// Đây chỉ là nơi TÁCH RIÊNG thông tin nhạy cảm khỏi esp32_client.ino,
// để không lỡ tay đẩy WiFi password/IP nội bộ lên kho công khai.

#define WIFI_SSID       "TEN_WIFI_CUA_BAN"
#define WIFI_PASSWORD   "MAT_KHAU_WIFI"

// IP máy tính chạy data_collection_server.py / inference_server.py
#define SERVER_HOST     "192.168.1.100"
#define SERVER_PORT     5000

// ID DUY NHẤT cho mỗi ESP32. Server (features.py/inference_server.py) tách
// lịch sử & trạng thái báo động theo device_id -> BẮT BUỘC phải khác nhau
// nếu bạn triển khai nhiều thiết bị, nếu không dữ liệu sẽ trộn lẫn.
#define DEVICE_ID       "esp32-phong-khach"
