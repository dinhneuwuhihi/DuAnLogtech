"""
BƯỚC 4 - Server suy luận thời gian thực (Real-time Inference)
------------------------------------------------------------------
Sau khi có fire_model.pkl (từ train_model.py), chạy file NÀY thay cho
data_collection_server.py. ESP32 không cần đổi code: endpoint vẫn là /predict.

    pip install -r requirements.txt
    python inference_server.py

NHỮNG THỨ ĐÃ SỬA SO VỚI BẢN CŨ:
1. Lịch sử cảm biến tách RIÊNG theo device_id (trước đây dùng deque toàn cục:
   cắm 2 con ESP32 là dữ liệu trộn vào nhau và mọi đặc trưng diff thành rác),
   và có Lock vì Flask chạy nhiều luồng.
2. Cửa sổ tính theo GIÂY chứ không theo SỐ MẪU (xem features.py): mất gói
   hay ESP32 reboot thì lịch sử được xoá thay vì tính "5 giây" từ 40 giây.
3. Kiểm tra dữ liệu đầu vào -> trả HTTP 400 rõ ràng. Bản cũ float(None) làm
   request chết với HTTP 500, ESP32 nhận -1 rồi KHÔNG làm gì cả.
4. LUẬT NGƯỠNG CỨNG ghi đè mô hình (fail-safe): gas quá cao / quá nóng /
   nhiệt tăng quá nhanh là báo động ngay, bất kể AI nói gì.
5. Chống rung (debounce) k-trong-n + trễ nhả (hysteresis): 1 mẫu nhiễu không
   được phép kích bơm/relay; và đã báo động thì phải "yên" một lúc mới tắt.
6. Trả về p_fire = XÁC SUẤT CHÁY. Bản cũ trả proba[prediction] nên
   "confidence 0.95" có thể là 95% an toàn hoặc 95% cháy - ESP32 không biết.
7. Ghi log mọi lần suy luận ra inference_log.csv: vừa để truy vết, vừa là
   tập dữ liệu thực địa cho lần train sau.
"""

from __future__ import annotations
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import joblib
from flask import Flask, jsonify, request

from features import (
    FEATURE_COLS,
    FeatureExtractor,
    features_to_vector,
    validate_reading,
)

MODEL_FILE = "fire_model.pkl"
LOG_FILE = "inference_log.csv"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# ---- Luật ngưỡng cứng: sống còn, không phụ thuộc AI ----
GAS_HARD_LIMIT = 2500      # ADC MQ-2: khói rất đậm đặc
TEMP_HARD_LIMIT = 60.0     # độ C
TEMP_RISE_HARD_LIMIT = 5.0 # độ C tăng trong 5 giây

# ---- Chống rung quyết định ----
DEBOUNCE_K, DEBOUNCE_N = 3, 5   # cần 3/5 mẫu dương mới báo động
CLEAR_MARGIN = 0.7              # ngưỡng nhả = 0.7 * ngưỡng báo (hysteresis)
CLEAR_SAMPLES = 5               # cần 5 mẫu liên tiếp "sạch" mới tắt báo động
DEVICE_TIMEOUT_S = 15.0         # quá lâu không thấy mẫu -> coi là offline

# ---- CẤU HÌNH EMAIL CẢNH BÁO ----
EMAIL_SENDER = "toladinhne@gmail.com"  # Sửa thành email gửi
EMAIL_PASSWORD = "lzfc glif ijse ydzh"  # Sửa thành Mật khẩu ứng dụng (16 ký tự)
EMAIL_RECEIVER = "toladinhne@gmail.com"  # Sửa thành email nhận cảnh báo


def send_alert_email(device_id, temp, gas, reason):
    """Hàm gửi email chạy trong luồng nền (background thread)"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"🚨 CẢNH BÁO CHÁY/KHÓI TỪ {device_id} 🚨"

    body = f"""
    HỆ THỐNG PHÁT HIỆN NGUY CƠ CHÁY/KHÓI!

    - Thiết bị: {device_id}
    - Lý do kích hoạt: {reason}
    - Nhiệt độ hiện tại: {temp:.1f} °C
    - Nồng độ Gas hiện tại: {gas}
    - Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Vui lòng kiểm tra hiện trường ngay lập tức!
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"📧 Đã gửi email cảnh báo thành công tới {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"📧 Lỗi khi gửi email: {e}")


def load_bundle(path: str) -> Dict[str, Any]:
    file = Path(path)
    if not file.exists():
        sys.exit(f"Khong tim thay {file.resolve()} - hay chay train_model.py truoc.")
    try:
        bundle = joblib.load(file)
    except Exception as exc:  # pickle sai phien ban sklearn, file hong...
        sys.exit(f"Khong doc duoc {file}: {exc}\n"
                 f"Thuong do sklearn khac phien ban luc train -> train lai mo hinh.")

    cols = bundle.get("feature_cols")
    if cols != FEATURE_COLS:
        sys.exit("Danh sach dac trung trong mo hinh KHAC features.py hien tai.\n"
                 "Ban da sua features.py sau khi train -> phai chay lai train_model.py.")
    bundle.setdefault("threshold", 0.5)
    return bundle


BUNDLE = load_bundle(MODEL_FILE)
MODEL = BUNDLE["model"]
THRESHOLD = float(BUNDLE["threshold"])
CLEAR_THRESHOLD = THRESHOLD * CLEAR_MARGIN


class DeviceState:
    """Lịch sử & trạng thái báo động của MỘT thiết bị."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id
        self.extractor = FeatureExtractor()
        self.window: Deque[int] = deque(maxlen=DEBOUNCE_N)
        self.clean_streak = 0
        self.alarm = False
        self.samples = 0
        self.last_seen: float = 0.0
        self.last_p_fire: float = 0.0
        self.last_reason: str = "-"
        self.email_sent = False

    def decide(self, p_fire: float, hard_alarm: bool) -> Dict[str, Any]:
        """Kết hợp mô hình + luật cứng + chống rung."""
        self.window.append(1 if p_fire >= THRESHOLD else 0)
        model_alarm = sum(self.window) >= DEBOUNCE_K

        if hard_alarm:
            self.alarm = True
            self.clean_streak = 0
            reason = "rule"
        elif model_alarm:
            self.alarm = True
            self.clean_streak = 0
            reason = "model"
        else:
            # Đã báo động thì phải "sạch" liên tục mới được nhả (hysteresis)
            if self.alarm:
                if p_fire < CLEAR_THRESHOLD:
                    self.clean_streak += 1
                else:
                    self.clean_streak = 0
                if self.clean_streak >= CLEAR_SAMPLES:
                    self.alarm = False
            reason = "alarm_latched" if self.alarm else "safe"

        self.last_reason = reason
        return {"alarm": self.alarm, "reason": reason}


class InferenceEngine:
    def __init__(self) -> None:
        self._devices: Dict[str, DeviceState] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.total_alarms = 0
        self._log_path = Path(LOG_FILE)
        self._init_log()

    def _init_log(self) -> None:
        if not self._log_path.exists():
            header = ["timestamp", "device_id"] + FEATURE_COLS + ["p_fire", "alarm", "reason"]
            with self._log_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def _state(self, device_id: str) -> DeviceState:
        state = self._devices.get(device_id)
        if state is None:
            state = DeviceState(device_id)
            self._devices[device_id] = state
            print(f"+ Thiet bi moi: {device_id}")
        return state

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        reading = validate_reading(payload)              # ném ValueError nếu rác
        device_id = str(payload.get("device_id") or "esp32-unknown")[:32]
        now = time.time()

        with self._lock:
            state = self._state(device_id)
            # Mất liên lạc quá lâu -> lịch sử cũ không còn ý nghĩa
            if state.last_seen and now - state.last_seen > DEVICE_TIMEOUT_S:
                state.extractor.reset()
                state.window.clear()
                print(f"! {device_id}: mat lien lac {now - state.last_seen:.0f}s -> xoa lich su")
            state.last_seen = now
            state.samples += 1

            feats = state.extractor.push(now, reading)
            vector = features_to_vector(feats)
            p_fire = float(MODEL.predict_proba([vector])[0][1])

            hard_alarm = (
                reading["gas"] >= GAS_HARD_LIMIT
                or reading["temp"] >= TEMP_HARD_LIMIT
                or feats["temp_diff_5s"] >= TEMP_RISE_HARD_LIMIT
            )
            decision = state.decide(p_fire, hard_alarm)
            state.last_p_fire = p_fire
            self.total_requests += 1
            if decision["alarm"]:
                self.total_alarms += 1

            # Nếu có báo động VÀ chưa gửi email cho sự cố này
            if decision["alarm"] and not state.email_sent:
                print(f"Bắt đầu gửi email cảnh báo...")
                # Tạo Thread riêng để gửi email (tránh làm kẹt server chờ 2-3s)
                threading.Thread(
                    target=send_alert_email,
                    args=(device_id, reading["temp"], reading["gas"], decision["reason"])
                ).start()
                state.email_sent = True  # Đánh dấu đã gửi để không spam mail mỗi giây

            # Nếu hệ thống đã an toàn trở lại (được reset), mở khóa cờ email
            elif not decision["alarm"]:
                state.email_sent = False

                self._append_log(device_id, feats, p_fire, decision)

            self._append_log(device_id, feats, p_fire, decision)

        prediction = 1 if decision["alarm"] else 0
        status = "NGUY HIEM (chay/khoi)" if prediction else "An toan"
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {device_id} "
              f"temp={reading['temp']:g} hum={reading['hum']:g} gas={reading['gas']:g} "
              f"d_temp5s={feats['temp_diff_5s']:+.1f} -> {status} "
              f"p_fire={p_fire:.3f} ({decision['reason']})")

        return {
            "prediction": prediction,           # giữ nguyên để ESP32 cũ vẫn chạy
            "p_fire": round(p_fire, 3),
            "confidence": round(p_fire if prediction else 1.0 - p_fire, 3),
            "alarm": bool(decision["alarm"]),
            "reason": decision["reason"],
            "threshold": round(THRESHOLD, 3),
            "warming_up": feats["history_ok"] < 1.0,
        }

    def _append_log(self, device_id: str, feats: Dict[str, float],
                    p_fire: float, decision: Dict[str, Any]) -> None:
        row = [datetime.now().isoformat(timespec="seconds"), device_id]
        row += [round(feats[name], 4) for name in FEATURE_COLS]
        row += [round(p_fire, 4), int(decision["alarm"]), decision["reason"]]
        try:
            with self._log_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except OSError as exc:
            print(f"! Khong ghi duoc log: {exc}")

    def health(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            devices = {
                dev.device_id: {
                    "so_mau": dev.samples,
                    "giay_tu_lan_cuoi": round(now - dev.last_seen, 1) if dev.last_seen else None,
                    "online": bool(dev.last_seen and now - dev.last_seen <= DEVICE_TIMEOUT_S),
                    "p_fire": round(dev.last_p_fire, 3),
                    "dang_bao_dong": dev.alarm,
                    "ly_do": dev.last_reason,
                }
                for dev in self._devices.values()
            }
            return {
                "status": "ok",
                "mode": "inference",
                "model_version": BUNDLE.get("version"),
                "trained_at": BUNDLE.get("trained_at"),
                "sklearn_luc_train": BUNDLE.get("sklearn_version"),
                "threshold": round(THRESHOLD, 3),
                "threshold_nha": round(CLEAR_THRESHOLD, 3),
                "uptime_giay": round(now - self.started_at, 1),
                "tong_request": self.total_requests,
                "tong_lan_bao_dong": self.total_alarms,
                "thiet_bi": devices,
            }

    def reset(self, device_id: str) -> bool:
        with self._lock:
            state = self._devices.get(device_id)
            if state is None:
                return False
            state.extractor.reset()
            state.window.clear()
            state.alarm = False
            state.clean_streak = 0
            return True


app = Flask(__name__)
engine = InferenceEngine()


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON khong hop le hoac thieu body"}), 400
    try:
        return jsonify(engine.predict(payload))
    except ValueError as exc:
        print(f"!! Du lieu khong hop le: {exc}")
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # không để 1 request làm sập đầu báo cháy
        print(f"!! Loi suy luan: {exc}")
        return jsonify({"error": "loi noi bo", "detail": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(engine.health())


@app.route("/reset/<device_id>", methods=["GET", "POST"])
def reset(device_id: str):
    """Xoá trạng thái báo động của 1 thiết bị (sau khi đã xử lý sự cố)."""
    if not engine.reset(device_id):
        return jsonify({"error": f"chua tung thay thiet bi '{device_id}'"}), 404
    return jsonify({"message": f"Da reset {device_id}"})


def run() -> None:
    print("=" * 70)
    print(f"Model : {MODEL_FILE} (version {BUNDLE.get('version')}, "
          f"train luc {BUNDLE.get('trained_at')})")
    print(f"Nguong bao dong: {THRESHOLD:.3f} | nguong nha: {CLEAR_THRESHOLD:.3f} "
          f"| debounce {DEBOUNCE_K}/{DEBOUNCE_N}")
    print(f"Luat cung: gas>={GAS_HARD_LIMIT} | temp>={TEMP_HARD_LIMIT} "
          f"| temp tang>={TEMP_RISE_HARD_LIMIT}C/5s")
    print(f"Log suy luan: {Path(LOG_FILE).resolve()}")
    print("=" * 70)
    try:
        # Flask dev server không dành cho chạy lâu dài; waitress ổn định trên Windows.
        from waitress import serve
        print(f"Chay bang waitress tai http://{SERVER_HOST}:{SERVER_PORT}")
        serve(app, host=SERVER_HOST, port=SERVER_PORT, threads=8)
    except ImportError:
        print("! Chua co waitress (pip install waitress) -> dung Flask dev server.")
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    run()
