import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_dummy_csv(filename="sensor_data.csv"):
    data = []
    start_time = datetime.now()
    seq = 0
    device_id = "esp32-test-01"

    print("Đang tạo dữ liệu giả lập...")

    # ==========================================
    # SESSION 1: Phòng bình thường (Label = 0)
    # ==========================================
    for i in range(60):  # 60 giây (60 mẫu)
        ts = start_time + timedelta(seconds=i)
        data.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"), device_id, "session_normal_1", seq,
            30.0 + np.random.normal(0, 0.2),  # temp: Ổn định quanh 30 độ
            65.0 + np.random.normal(0, 0.5),  # hum: Ổn định quanh 65%
            2000 + np.random.normal(0, 50),  # light: Sáng phòng bình thường
            0,  # ir: Không có lửa
            400 + np.random.normal(0, 10),  # gas: Không khí sạch
            1, 0  # sensor_ok = 1, label = 0
        ])
        seq += 1

    start_time += timedelta(seconds=120)  # Khoảng nghỉ giữa các phiên

    # ==========================================
    # SESSION 2: Đốt giấy thật (Label = 1)
    # ==========================================
    for i in range(60):
        ts = start_time + timedelta(seconds=i)
        data.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"), device_id, "session_fire_1", seq,
            30.0 + i * 0.4 + np.random.normal(0, 0.2),  # temp: Tăng dần lên
            65.0 - i * 0.3 + np.random.normal(0, 0.5),  # hum: Giảm dần xuống (khô)
            3500 + np.random.normal(0, 50),  # light: Ánh lửa hắt sáng LDR
            1,  # ir: Phát hiện tia hồng ngoại
            400 + i * 25 + np.random.normal(0, 10),  # gas: Tăng vọt (khói)
            1, 1  # sensor_ok = 1, label = 1
        ])
        seq += 1

    start_time += timedelta(seconds=120)

    # ==========================================
    # SESSION 3: Phòng có bật điều hòa (Label = 0)
    # ==========================================
    for i in range(60):
        ts = start_time + timedelta(seconds=i)
        data.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"), device_id, "session_normal_2", seq,
            25.0 + np.random.normal(0, 0.2),
            55.0 + np.random.normal(0, 0.5),
            1800 + np.random.normal(0, 50),
            0,
            420 + np.random.normal(0, 10),
            1, 0
        ])
        seq += 1

    start_time += timedelta(seconds=120)

    # ==========================================
    # SESSION 4: Cháy vật liệu nhựa (Label = 1)
    # ==========================================
    for i in range(60):
        ts = start_time + timedelta(seconds=i)
        data.append([
            ts.strftime("%Y-%m-%d %H:%M:%S"), device_id, "session_fire_2", seq,
            25.0 + i * 0.5 + np.random.normal(0, 0.2),  # Nhiệt độ tăng dốc hơn
            55.0 - i * 0.2 + np.random.normal(0, 0.5),  # Ẩm tụt
            500 + np.random.normal(0, 50),  # Ánh sáng tối đi (do khói đen đặc che)
            1,
            420 + i * 30 + np.random.normal(0, 10),  # Nồng độ khói gas cực cao
            1, 1
        ])
        seq += 1

    # Tạo DataFrame và xuất ra CSV
    columns = [
        "timestamp", "device_id", "session", "seq",
        "temp", "hum", "light", "ir", "gas",
        "sensor_ok", "label"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f" Đã tạo thành công file '{filename}'!")
    print(f"   - Tổng số dòng: {len(df)}")
    print(f"   - Số session: 4")
    print(f"   - Số mẫu an toàn (Label 0): {len(df[df['label'] == 0])}")
    print(f"   - Số mẫu cháy (Label 1): {len(df[df['label'] == 1])}")
    print("\n Bạn có thể chạy file train_model.py ngay bây giờ.")


if __name__ == "__main__":
    create_dummy_csv()