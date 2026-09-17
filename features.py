"""
BƯỚC 1 - Định nghĩa CHUNG về dữ liệu & đặc trưng (feature engineering)
------------------------------------------------------------------------
File này KHÔNG chạy trực tiếp. Nó được import bởi:
    - data_collection_server.py  (biết cần ghi những cột nào)
    - train_model.py             (tính đặc trưng khi huấn luyện)
    - inference_server.py        (tính đặc trưng khi dự đoán real-time)

Vì sao phải tách ra? Trước đây train_model.py và inference_server.py tự tính
đặc trưng theo 2 đoạn code khác nhau. Chỉ cần sửa 1 bên là mô hình sẽ nhận
đầu vào lệch so với lúc train (lỗi "train/serve skew") -> dự đoán sai mà
không báo lỗi gì cả. Giờ chỉ còn MỘT nơi duy nhất tính đặc trưng.

File này cố tình KHÔNG phụ thuộc pandas/numpy để cả server nhẹ cũng dùng được.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence

# ====== SCHEMA CỦA FILE CSV ======
SENSOR_FIELDS: List[str] = ["temp", "hum", "light", "ir", "gas"]
META_FIELDS: List[str] = ["timestamp", "device_id", "session", "seq"]
CSV_HEADERS: List[str] = META_FIELDS + SENSOR_FIELDS + ["sensor_ok", "label"]

# ====== NHÃN (label) ======
# Cố tình dùng NHIỀU nhãn thay vì chỉ 0/1: khi train ta gộp lại thành
# 0 = an toàn, 1 = cháy. Nhưng nếu lúc thu thập chỉ ghi 0/1 thì thông tin
# "đây là khói bếp" hay "đây là hơi nước" sẽ mất vĩnh viễn.
LABEL_NAMES: Dict[int, str] = {
    0: "binh thuong",
    1: "chay / khoi that",
    2: "hoi nuoc / nau an",
    3: "nong nhung khong khoi (may say, nang)",
    4: "thay doi anh sang manh",
    5: "khoi thuoc la / nhieu khac",
}
FIRE_LABEL = 1
VALID_LABELS = set(LABEL_NAMES)

# ====== GIỚI HẠN HỢP LỆ CỦA CẢM BIẾN (dùng để loại dữ liệu rác) ======
SENSOR_RANGES: Dict[str, tuple] = {
    "temp": (-40.0, 85.0),
    "hum": (0.0, 100.0),
    "light": (0.0, 4095.0),
    "ir": (0.0, 1.0),
    "gas": (0.0, 4095.0),
}

# ====== THAM SỐ THỜI GIAN ======
SAMPLE_PERIOD_S = 1.0          # ESP32 gửi 1 mẫu / giây
WINDOWS_S: Sequence[float] = (5.0, 30.0)  # 2 cửa sổ: phản ứng nhanh & xu thế
GAP_RESET_S = 4.0              # mất mẫu quá lâu -> xoá lịch sử, không tính diff
BUFFER_S = 45.0                # giữ tối đa 45 giây lịch sử trong RAM

DIFF_FIELDS: List[str] = ["temp", "hum", "light", "gas"]
STAT_FIELDS: List[str] = ["temp", "gas"]


def _feature_names() -> List[str]:
    """Danh sách cột đặc trưng, ĐÚNG THỨ TỰ mà mô hình mong đợi."""
    names: List[str] = list(SENSOR_FIELDS)
    for window in WINDOWS_S:
        w = int(window)
        for field in DIFF_FIELDS:
            names.append(f"{field}_diff_{w}s")
        for field in STAT_FIELDS:
            names.append(f"{field}_mean_{w}s")
            names.append(f"{field}_std_{w}s")
    names += [
        "temp_rate_s",        # độ C mỗi giây
        "gas_rate_s",         # ADC mỗi giây
        "gas_above_min",      # gas hiện tại cao hơn mức nền bao nhiêu
        "temp_up_hum_down",   # nhiệt tăng ĐỒNG THỜI ẩm giảm = dấu hiệu lửa
        "abs_hum",            # độ ẩm tuyệt đối (g/m3), ổn định hơn RH
        "history_ok",         # 1 = đã đủ lịch sử, 0 = vừa khởi động/mất mẫu
    ]
    return names


FEATURE_COLS: List[str] = _feature_names()


def absolute_humidity(temp_c: float, rh_percent: float) -> float:
    """Độ ẩm tuyệt đối (g/m3). RH phụ thuộc nhiệt độ, còn AH thì không:
    khi lửa làm nóng không khí, RH tụt nhưng AH gần như không đổi."""
    try:
        saturation = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        return (saturation * rh_percent * 2.1674) / (273.15 + temp_c)
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0.0


def validate_reading(data: Dict[str, Any]) -> Dict[str, float]:
    """Kiểm tra & chuyển kiểu 1 bản ghi cảm biến.

    Ném ValueError nếu thiếu trường hoặc giá trị nằm ngoài dải vật lý.
    Nhờ vậy server trả về HTTP 400 rõ ràng thay vì HTTP 500 (trước đây
    float(None) làm cả request chết và ESP32 nhận -1 rồi... không làm gì).
    """
    cleaned: Dict[str, float] = {}
    for field in SENSOR_FIELDS:
        if field not in data or data[field] is None:
            raise ValueError(f"thieu truong '{field}'")
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            raise ValueError(f"truong '{field}' khong phai so: {data[field]!r}")
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"truong '{field}' khong hop le: {value}")
        low, high = SENSOR_RANGES[field]
        if not (low <= value <= high):
            raise ValueError(f"truong '{field}'={value} ngoai dai cho phep [{low}, {high}]")
        cleaned[field] = value
    return cleaned


class FeatureExtractor:
    """Tính đặc trưng từ chuỗi mẫu theo THỜI GIAN THỰC (streaming).

    Điểm khác biệt quan trọng so với bản cũ: cửa sổ được tính theo GIÂY,
    không theo SỐ MẪU. Nếu ESP32 reboot hoặc mất gói, "5 mẫu trước" có thể
    là 40 giây trước - lúc đó đặc trưng diff_5s là con số vô nghĩa mà mô
    hình vẫn tin là thật.
    """

    def __init__(
        self,
        windows: Sequence[float] = WINDOWS_S,
        gap_reset_s: float = GAP_RESET_S,
        buffer_s: float = BUFFER_S,
    ) -> None:
        self.windows = sorted(float(w) for w in windows)
        self.gap_reset_s = gap_reset_s
        self.buffer_s = max(buffer_s, self.windows[-1] + SAMPLE_PERIOD_S)
        self._tol = SAMPLE_PERIOD_S * 0.5
        self._buf: Deque[Dict[str, float]] = deque()
        self.dropped_gaps = 0

    def reset(self) -> None:
        self._buf.clear()

    @property
    def size(self) -> int:
        return len(self._buf)

    def push(self, ts: float, reading: Dict[str, float]) -> Dict[str, float]:
        """Thêm 1 mẫu (ts = giây, epoch hoặc tương đối) và trả về đặc trưng."""
        ts = float(ts)
        if self._buf:
            gap = ts - self._buf[-1]["ts"]
            if gap <= 0 or gap > self.gap_reset_s:
                self.dropped_gaps += 1
                self.reset()

        sample: Dict[str, float] = {"ts": ts}
        for field in SENSOR_FIELDS:
            sample[field] = float(reading[field])
        self._buf.append(sample)

        while len(self._buf) > 1 and ts - self._buf[0]["ts"] > self.buffer_s:
            self._buf.popleft()

        return self._compute()

    def _compute(self) -> Dict[str, float]:
        current = self._buf[-1]
        now = current["ts"]
        feats: Dict[str, float] = {field: current[field] for field in SENSOR_FIELDS}

        short_diff: Dict[str, float] = {field: 0.0 for field in DIFF_FIELDS}
        short_age = 0.0
        gas_min = current["gas"]

        for window in self.windows:
            w = int(window)
            win = [s for s in self._buf if now - s["ts"] <= window + self._tol]
            oldest = win[0]
            age = now - oldest["ts"]
            # Chỉ tin đặc trưng khi cửa sổ thật sự phủ >= 60% khoảng thời gian
            valid = age >= window * 0.6

            for field in DIFF_FIELDS:
                diff = (current[field] - oldest[field]) if valid else 0.0
                feats[f"{field}_diff_{w}s"] = diff
                if window == self.windows[0]:
                    short_diff[field] = diff

            for field in STAT_FIELDS:
                values = [s[field] for s in win]
                feats[f"{field}_mean_{w}s"] = statistics.fmean(values)
                feats[f"{field}_std_{w}s"] = statistics.pstdev(values) if len(values) > 1 else 0.0

            if window == self.windows[0]:
                short_age = age if valid else 0.0
            if window == self.windows[-1]:
                gas_min = min(s["gas"] for s in win)

        denominator = short_age if short_age > 0 else 1.0
        feats["temp_rate_s"] = short_diff["temp"] / denominator
        feats["gas_rate_s"] = short_diff["gas"] / denominator
        feats["gas_above_min"] = current["gas"] - gas_min
        feats["temp_up_hum_down"] = max(0.0, short_diff["temp"]) * max(0.0, -short_diff["hum"])
        feats["abs_hum"] = absolute_humidity(current["temp"], current["hum"])
        span = now - self._buf[0]["ts"]
        feats["history_ok"] = 1.0 if span >= self.windows[-1] * 0.6 else 0.0

        return {name: float(feats[name]) for name in FEATURE_COLS}


def features_to_vector(feats: Dict[str, float]) -> List[float]:
    """Đổi dict đặc trưng thành list đúng thứ tự FEATURE_COLS."""
    return [float(feats[name]) for name in FEATURE_COLS]


def iter_session_features(
    rows: Iterable[Dict[str, Any]],
    ts_getter=lambda row: row["ts"],
) -> Iterator[Dict[str, float]]:
    """Tính đặc trưng cho MỘT session (đã sắp xếp theo thời gian).

    train_model.py dùng hàm này để tạo dữ liệu huấn luyện, nên dữ liệu train
    và dữ liệu lúc chạy thật được sinh ra bởi CÙNG một đoạn code.
    """
    extractor = FeatureExtractor()
    for row in rows:
        yield extractor.push(ts_getter(row), row)
