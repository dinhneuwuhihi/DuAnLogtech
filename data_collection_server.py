"""
BƯỚC 2 - Server thu thập dữ liệu (Data Collection)
----------------------------------------------------
Chạy file này trên MÁY TÍNH của bạn (đóng vai trò Server).
ESP32 gửi dữ liệu cảm biến đến đây, server ghi vào sensor_data.csv kèm nhãn.

Cách dùng:
1. Cài thư viện:  pip install -r requirements.txt
2. Chạy:          python data_collection_server.py
3. TRƯỚC MỖI LẦN thu thập, mở 1 "session" (phiên) mới:
       http://<IP_MAY_TINH>:5000/start_session/phong_khach_binh_thuong
   Sau đó đặt nhãn cho phiên:
       http://<IP_MAY_TINH>:5000/set_label/0   -> bình thường
       http://<IP_MAY_TINH>:5000/set_label/1   -> cháy/khói thật
       http://<IP_MAY_TINH>:5000/set_label/2   -> hơi nước / nấu ăn
       http://<IP_MAY_TINH>:5000/set_label/3   -> nóng nhưng không khói (máy sấy)
       http://<IP_MAY_TINH>:5000/set_label/4   -> thay đổi ánh sáng mạnh
       http://<IP_MAY_TINH>:5000/set_label/5   -> khói thuốc / nhiễu khác
   Xem danh sách nhãn: http://<IP_MAY_TINH>:5000/labels
4. Theo dõi tiến độ: http://<IP_MAY_TINH>:5000/stats

VÌ SAO CẦN "SESSION"?
  - Dữ liệu là chuỗi thời gian 1Hz: 2 dòng liền nhau gần như giống nhau.
    Nếu chia train/test ngẫu nhiên, mô hình sẽ "học thuộc" và cho accuracy
    ảo ~0.99. train_model.py chia theo session để tránh rò rỉ dữ liệu.
  - Đặc trưng tốc độ thay đổi (diff) không được tính vắt qua 2 phiên khác nhau.

VÌ SAO CẦN CÁC NHÃN 2..5 (hard negatives)?
  Hơi nước nồi lẩu, khói thuốc, hơi nóng máy sấy... chính là thứ gây báo
  động giả. Thu thập và gắn nhãn riêng chúng thì mô hình mới học được
  "nóng + khói thật" khác "chỉ hơi nước" ở đâu. Khi train, mọi nhãn khác 1
  đều được gộp về 0.
"""

import csv
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from features import CSV_HEADERS, FIRE_LABEL, LABEL_NAMES, VALID_LABELS, validate_reading

DEFAULT_CSV_FILE = "sensor_data.csv"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
DEFAULT_DEVICE_ID = "esp32-unknown"


class SensorDataManager:
    """Quản lý trạng thái thu thập + ghi CSV (an toàn với nhiều luồng)."""

    def __init__(self, file_path: str = DEFAULT_CSV_FILE) -> None:
        self.file_path = Path(file_path)
        self.headers = list(CSV_HEADERS)
        self.current_label: int = 0
        self.session: str = self._new_session_name("mac_dinh")
        self.label_changed_at: Optional[str] = None
        self.written_rows = 0
        self.rejected_rows = 0
        # Flask xử lý nhiều request song song -> hai writerow lồng nhau có thể
        # làm hỏng dòng CSV. Lock đảm bảo mỗi lần chỉ 1 luồng ghi file.
        self._lock = threading.Lock()
        self._seq: Dict[str, int] = defaultdict(int)
        self._init_storage()

    @staticmethod
    def _new_session_name(name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "session"
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"

    def _init_storage(self) -> None:
        """Tạo file CSV kèm tiêu đề nếu chưa có.

        Gọi ngay trong constructor (chứ không chỉ trong `__main__`) để khi
        chạy bằng waitress/gunicorn file vẫn có header đúng.
        """
        if not self.file_path.exists():
            with self.file_path.open(mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.headers)

    # ---------- trạng thái ----------
    def start_session(self, name: str) -> str:
        with self._lock:
            self.session = self._new_session_name(name)
            self._seq.clear()
            return self.session

    def set_label(self, label: int) -> bool:
        if label not in VALID_LABELS:
            return False
        with self._lock:
            self.current_label = label
            self.label_changed_at = datetime.now().isoformat(timespec="seconds")
        print(f"*** LABEL -> {label} ({LABEL_NAMES[label]}) | session={self.session} ***")
        return True

    # ---------- ghi dữ liệu ----------
    def append_reading(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Kiểm tra payload, ghi 1 dòng vào CSV và trả về bản ghi đã lưu."""
        reading = validate_reading(payload)  # ném ValueError nếu dữ liệu rác
        device_id = str(payload.get("device_id") or DEFAULT_DEVICE_ID)[:32]
        sensor_ok = int(bool(payload.get("sensor_ok", 1)))
        timestamp = datetime.now().isoformat()

        with self._lock:
            self._seq[device_id] += 1
            row = {
                "timestamp": timestamp,
                "device_id": device_id,
                "session": self.session,
                "seq": self._seq[device_id],
                "sensor_ok": sensor_ok,
                "label": self.current_label,
                **reading,
            }
            with self.file_path.open(mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([row[h] for h in self.headers])
            self.written_rows += 1
            total = self.written_rows

        values = " ".join(f"{k}={reading[k]:g}" for k in reading)
        print(f"[{timestamp}] #{total} {device_id} {values} label={self.current_label}")
        return row

    # ---------- thống kê ----------
    def get_statistics(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "session_hien_tai": self.session,
            "label_hien_tai": self.current_label,
            "ten_label_hien_tai": LABEL_NAMES.get(self.current_label, "?"),
            "doi_label_luc": self.label_changed_at,
            "so_dong_bi_tu_choi": self.rejected_rows,
        }
        if not self.file_path.exists():
            return {**base, "total": 0, "theo_label": {}, "theo_session": {}}

        by_label: Counter = Counter()
        by_session: Counter = Counter()
        fire_rows = 0
        with self.file_path.open(mode="r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = row.get("label")
                by_label[label] += 1
                by_session[row.get("session") or "?"] += 1
                if label == str(FIRE_LABEL):
                    fire_rows += 1

        total = sum(by_label.values())
        return {
            **base,
            "total": total,
            "so_mau_chay": fire_rows,
            "theo_label": {
                f"{k} ({LABEL_NAMES.get(int(k), '?') if str(k).isdigit() else '?'})": v
                for k, v in sorted(by_label.items())
            },
            "theo_session": dict(by_session),
        }


app = Flask(__name__)
data_manager = SensorDataManager()


@app.route("/predict", methods=["POST"])
def collect_data():
    """Nhận dữ liệu cảm biến trong giai đoạn THU THẬP.

    Endpoint vẫn tên /predict để ESP32 không phải đổi code; ở bước này ta
    chỉ LƯU dữ liệu, chưa chạy AI.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON khong hop le hoac thieu body"}), 400

    try:
        data_manager.append_reading(payload)
    except ValueError as exc:
        data_manager.rejected_rows += 1
        print(f"!! Tu choi du lieu: {exc}")
        return jsonify({"error": str(exc)}), 400

    # Phản hồi để ESP32 biết server còn sống. Chỉ báo nguy hiểm khi ta đang
    # chủ động gắn nhãn cháy, các nhãn nhiễu (2..5) vẫn coi là an toàn.
    prediction = 1 if data_manager.current_label == FIRE_LABEL else 0
    return jsonify({
        "prediction": prediction,
        "p_fire": float(prediction),
        "alarm": bool(prediction),
        "mode": "data_collection",
        "label": data_manager.current_label,
        "session": data_manager.session,
    })


@app.route("/set_label/<int:label>", methods=["GET", "POST"])
def set_label(label: int):
    """Đổi nhãn đang thu thập. Có cả POST cho script, GET cho trình duyệt."""
    if not data_manager.set_label(label):
        return jsonify({
            "error": "label khong hop le",
            "cac_label_hop_le": {str(k): v for k, v in LABEL_NAMES.items()},
        }), 400
    return jsonify({
        "message": f"Da chuyen sang label = {label} ({LABEL_NAMES[label]})",
        "luu_y": "Hay bo qua vai giay dau tien sau khi doi nhan - train_model.py tu cat cua so chuyen tiep.",
    })


@app.route("/start_session/<name>", methods=["GET", "POST"])
def start_session(name: str):
    """Mở phiên thu thập mới (bắt buộc cho việc chia train/test không rò rỉ)."""
    session = data_manager.start_session(name)
    return jsonify({"message": "Da mo session moi", "session": session})


@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({str(k): v for k, v in LABEL_NAMES.items()})


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(data_manager.get_statistics())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mode": "data_collection",
        "file": str(data_manager.file_path.resolve()),
        "session": data_manager.session,
        "label": data_manager.current_label,
    })


if __name__ == "__main__":
    print(f"Ghi du lieu vao: {data_manager.file_path.resolve()}")
    print(f"Session: {data_manager.session} | label = {data_manager.current_label}")
    # host="0.0.0.0" để ESP32 trong cùng LAN gọi tới được.
    # debug=False: debug=True làm server tự reload -> mất trạng thái session/label.
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
