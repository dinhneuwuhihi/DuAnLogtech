"""
BƯỚC 3 - Huấn luyện mô hình AI trên máy tính
------------------------------------------------
Chạy SAU KHI đã thu thập đủ dữ liệu bằng data_collection_server.py.

    pip install -r requirements.txt
    python train_model.py

Kết quả: fire_model.pkl (gồm mô hình + danh sách đặc trưng + NGƯỠNG quyết định)
dùng cho inference_server.py.

NHỮNG THỨ ĐÃ SỬA SO VỚI BẢN CŨ (quan trọng, đọc kỹ):
1. KHÔNG dùng train_test_split ngẫu nhiên nữa.
   Dữ liệu là chuỗi 1Hz, dòng N và N+1 gần như y hệt nhau. Chia ngẫu nhiên =
   "bạn của mỗi dòng test đều nằm trong tập train" -> mô hình học thuộc,
   accuracy báo 0.99 nhưng ra thực tế báo động giả liên tục. Giờ chia theo
   SESSION: cả một phiên ghi hình nằm hẳn ở train hoặc hẳn ở test.
2. Đặc trưng được tính bằng features.py - đúng đoạn code mà inference_server
   dùng lúc chạy thật (hết lỗi train/serve skew).
3. Không dùng accuracy làm thước đo chính. Với đầu báo cháy, cái đáng quan
   tâm là RECALL của lớp 1 (bỏ sót cháy = thảm hoạ), SỐ BÁO ĐỘNG GIẢ MỖI GIỜ
   (báo giả nhiều = người ta rút phích) và TRỄ PHÁT HIỆN (bao nhiêu giây).
4. Tự chọn & lưu NGƯỠNG xác suất (không dùng mặc định 0.5), có hiệu chỉnh
   xác suất (calibration) vì predict_proba của RandomForest không phải xác
   suất thật, trong khi ta hiển thị nó như "độ tin cậy".
5. So sánh với BASELINE bằng luật ngưỡng đơn giản. Nếu AI không thắng rõ
   luật ngưỡng thì... dùng luật ngưỡng, đơn giản và dễ giải thích hơn.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from features import (
    FEATURE_COLS,
    FIRE_LABEL,
    LABEL_NAMES,
    SENSOR_FIELDS,
    SENSOR_RANGES,
    iter_session_features,
)

CSV_FILE = "sensor_data.csv"
MODEL_FILE = "fire_model.pkl"
REPORT_FILE = "training_report.json"

TEST_SIZE = 0.3          # tỉ lệ SESSION dành cho test
RANDOM_STATE = 42
TRANSITION_S = 10.0      # bỏ 10s đầu sau khi đổi nhãn (khói chưa tới cảm biến)
SESSION_GAP_S = 60.0     # dữ liệu cũ không có cột session -> cắt theo khoảng trống
MAX_FALSE_ALARM_RATE = 0.01   # tối đa 1% mẫu an toàn được phép báo động
MIN_RECALL = 0.80             # nếu không đạt thì cảnh báo
DEBOUNCE_K, DEBOUNCE_N = 3, 5  # giống luật k-trong-n của inference_server


# ============================ 1. ĐỌC & LÀM SẠCH ============================
def load_dataset(csv_file: str) -> pd.DataFrame:
    path = Path(csv_file)
    if not path.exists():
        sys.exit(f"Khong tim thay {path.resolve()} - hay chay data_collection_server.py truoc.")

    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts", "label"] + SENSOR_FIELDS)
    df["label"] = df["label"].astype(int)

    # Tương thích với file CSV cũ (chưa có device_id/session/sensor_ok)
    if "device_id" not in df.columns:
        df["device_id"] = "legacy"
    if "sensor_ok" not in df.columns:
        df["sensor_ok"] = 1
    if "session" not in df.columns or df["session"].isna().all():
        print("! CSV khong co cot 'session' -> tu suy ra tu cac khoang trong thoi gian.")
        df = df.sort_values("ts")
        gap = df["ts"].diff().dt.total_seconds().fillna(1e9)
        df["session"] = (gap > SESSION_GAP_S).cumsum().map(lambda i: f"auto_{i:03d}")

    df["session"] = df["session"].astype(str)
    df = df.sort_values(["session", "ts"]).reset_index(drop=True)

    before = len(df)
    df = df[df["sensor_ok"].astype(int) == 1]
    for field, (low, high) in SENSOR_RANGES.items():
        df = df[(df[field] >= low) & (df[field] <= high)]
    df = df.drop_duplicates(subset=["session", "ts"])
    print(f"Loai bo {before - len(df)} dong rac/trung lap, con {len(df)} dong.")
    return df.reset_index(drop=True)


def drop_transition_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ các dòng ngay sau khi đổi nhãn.

    Khi bạn gọi /set_label/1 rồi mới bật bật lửa, vài giây đầu KHÔNG hề có
    khói nhưng đã bị gắn nhãn "cháy". Những dòng đó dạy mô hình điều sai.
    """
    keep = np.ones(len(df), dtype=bool)
    for _, group in df.groupby("session", sort=False):
        labels = group["label"].to_numpy()
        times = group["ts"].to_numpy()
        change_at = times[0]
        for i in range(len(group)):
            if i > 0 and labels[i] != labels[i - 1]:
                change_at = times[i]
            elapsed = (times[i] - change_at) / np.timedelta64(1, "s")
            if elapsed < TRANSITION_S:
                keep[group.index[i]] = False
    print(f"Bo {int((~keep).sum())} dong trong cua so chuyen tiep ({TRANSITION_S:.0f}s sau moi lan doi nhan).")
    return df[keep].reset_index(drop=True)


# ============================ 2. ĐẶC TRƯNG ============================
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tính đặc trưng theo TỪNG session bằng đúng code của inference_server."""
    frames: List[pd.DataFrame] = []
    for session, group in df.groupby("session", sort=False):
        group = group.sort_values("ts")
        rows = group[SENSOR_FIELDS].to_dict("records")
        epochs = (group["ts"].astype("int64") // 10**9).to_numpy()
        feats = list(iter_session_features(
            ({**row, "ts": float(ts)} for row, ts in zip(rows, epochs)),
        ))
        block = pd.DataFrame(feats, index=group.index)
        block["session"] = session
        block["ts"] = group["ts"].to_numpy()
        block["label"] = group["label"].to_numpy()
        frames.append(block)
    return pd.concat(frames).sort_values(["session", "ts"]).reset_index(drop=True)


# ============================ 3. CHIA TẬP ============================
def split_by_session(feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sessions = feat["session"].unique()
    fire_sessions = feat.loc[feat["y"] == 1, "session"].unique()

    if len(sessions) < 2:
        print("! Chi co 1 session -> chia theo THOI GIAN (70% dau train / 30% cuoi test).")
        print("! Hay thu thap nhieu phien rieng biet de danh gia dang tin cay hon.")
        cut = int(len(feat) * (1 - TEST_SIZE))
        return feat.iloc[:cut].copy(), feat.iloc[cut:].copy()

    if len(fire_sessions) < 2:
        print("! Chi co 1 phien co nhan chay -> khong the giu nguyen ca su kien chay o test.")
        print("! Ket qua danh gia se lac quan hon thuc te. Hay ghi them nhieu lan chay khac nhau.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(feat, feat["y"], groups=feat["session"]))
    return feat.iloc[train_idx].copy(), feat.iloc[test_idx].copy()


# ============================ 4. HUẤN LUYỆN ============================
def fit_model(train: pd.DataFrame):
    """RandomForest + hiệu chỉnh xác suất (nếu đủ session để tách tập hiệu chỉnh)."""
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    X = train[FEATURE_COLS].to_numpy()
    y = train["y"].to_numpy()

    sessions = train["session"].nunique()
    if sessions >= 4 and y.sum() >= 50 and (y == 0).sum() >= 50:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        fit_idx, cal_idx = next(splitter.split(X, y, groups=train["session"]))
        if len(np.unique(y[cal_idx])) == 2:
            base.fit(X[fit_idx], y[fit_idx])
            calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
            calibrated.fit(X[cal_idx], y[cal_idx])
            print("Da hieu chinh xac suat (Platt scaling) tren tap calibration rieng.")
            return calibrated, base

    base.fit(X, y)
    print("! Khong du du lieu/session de hieu chinh xac suat -> dung predict_proba tho.")
    return base, base


# ============================ 5. ĐÁNH GIÁ ============================
def rates_at(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    recall = tp / positives
    precision = tp / max(tp + fp, 1)
    return {
        "threshold": float(thr),
        "recall": recall,
        "precision": precision,
        "false_alarm_rate": fp / negatives,
        "f2": (5 * precision * recall / (4 * precision + recall)) if (precision + recall) else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Chọn ngưỡng nhỏ nhất (nhạy nhất) mà tỉ lệ báo giả còn chấp nhận được."""
    candidates = np.unique(np.round(np.concatenate([p, [0.5]]), 3))
    feasible = [rates_at(y_true, p, t) for t in candidates]
    ok = [m for m in feasible if m["false_alarm_rate"] <= MAX_FALSE_ALARM_RATE]
    if ok:
        best = max(ok, key=lambda m: (m["recall"], -m["threshold"]))
    else:
        print(f"! Khong nguong nao dat ti le bao gia <= {MAX_FALSE_ALARM_RATE:.1%} -> chon theo F2.")
        best = max(feasible, key=lambda m: m["f2"])
    return best


def debounced_alarms(test: pd.DataFrame, p: np.ndarray, thr: float) -> Tuple[int, float]:
    """Mô phỏng luật k-trong-n của server: đếm SỐ LẦN báo động giả, không
    phải số DÒNG báo giả (một lần báo động kéo dài vẫn chỉ là 1 lần)."""
    df = test.assign(p=p).sort_values(["session", "ts"])
    false_episodes = 0
    safe_seconds = 0.0
    for _, group in df.groupby("session", sort=False):
        window: List[int] = []
        alarm = False
        previous_ts = None
        for _, row in group.iterrows():
            window.append(1 if row["p"] >= thr else 0)
            window = window[-DEBOUNCE_N:]
            now_alarm = sum(window) >= DEBOUNCE_K
            if now_alarm and not alarm and row["y"] == 0:
                false_episodes += 1
            alarm = now_alarm
            if row["y"] == 0:
                if previous_ts is not None:
                    delta = (row["ts"] - previous_ts).total_seconds()
                    safe_seconds += delta if 0 < delta <= 5 else 1.0
                else:
                    safe_seconds += 1.0
            previous_ts = row["ts"]
    hours = max(safe_seconds / 3600.0, 1e-6)
    return false_episodes, false_episodes / hours


def detection_latency(test: pd.DataFrame, p: np.ndarray, thr: float) -> List[float]:
    """Với mỗi "sự kiện cháy" trong tập test: bao nhiêu giây sau khi bắt đầu
    thì mô hình mới báo? Bỏ sót hoàn toàn = inf."""
    df = test.assign(p=p).sort_values(["session", "ts"])
    latencies: List[float] = []
    for _, group in df.groupby("session", sort=False):
        fire = group[group["y"] == 1]
        if fire.empty:
            continue
        gap = fire["ts"].diff().dt.total_seconds().fillna(1e9)
        event_id = (gap > SESSION_GAP_S).cumsum()
        for _, event in fire.groupby(event_id):
            hit = event[event["p"] >= thr]
            if hit.empty:
                latencies.append(float("inf"))
            else:
                latencies.append((hit["ts"].iloc[0] - event["ts"].iloc[0]).total_seconds())
    return latencies


def baseline_scores(test: pd.DataFrame) -> np.ndarray:
    """Luật ngưỡng "ngu" để so sánh: khói tăng vọt HOẶC nhiệt tăng nhanh."""
    gas_jump = test["gas_above_min"].to_numpy() > 300
    temp_jump = test["temp_diff_5s"].to_numpy() > 2.0
    hot = test["temp"].to_numpy() > 55
    return (gas_jump | temp_jump | hot).astype(float)


# ============================ MAIN ============================
def main() -> None:
    print("=" * 70)
    df = load_dataset(CSV_FILE)
    print("\nSo mau theo nhan:")
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  {label} ({LABEL_NAMES.get(int(label), '?')}): {count}")
    print("\nSo mau theo session:")
    print(df.groupby("session")["label"].value_counts().unstack(fill_value=0).to_string())

    df = drop_transition_rows(df)

    feat = build_feature_frame(df)
    # Gộp mọi nhãn khác 1 về 0: 2..5 là "hard negatives" (hơi nước, máy sấy...)
    feat["y"] = (feat["label"] == FIRE_LABEL).astype(int)
    if feat["y"].nunique() < 2:
        sys.exit("Du lieu chi co mot lop - can ca mau binh thuong (0) va mau chay (1).")

    train, test = split_by_session(feat)
    print(f"\nTrain: {len(train)} dong / {train['session'].nunique()} session "
          f"({int(train['y'].sum())} mau chay)")
    print(f"Test : {len(test)} dong / {test['session'].nunique()} session "
          f"({int(test['y'].sum())} mau chay)")
    if test["y"].nunique() < 2:
        sys.exit("Tap test khong co du 2 lop - hay thu thap them cac phien khac nhau.")

    model, forest = fit_model(train)

    X_test = test[FEATURE_COLS].to_numpy()
    y_test = test["y"].to_numpy()
    p_fire = model.predict_proba(X_test)[:, 1]

    best = choose_threshold(y_test, p_fire)
    threshold = best["threshold"]

    print("\n" + "=" * 70)
    print(f"NGUONG DA CHON: {threshold:.3f}  (khong dung mac dinh 0.5)")
    print(f"  Recall lop chay     : {best['recall']:.3f}  <- quan trong nhat")
    print(f"  Precision           : {best['precision']:.3f}")
    print(f"  Ti le bao gia (dong): {best['false_alarm_rate']:.4f}")
    print(f"  ROC-AUC             : {roc_auc_score(y_test, p_fire):.3f}")
    print(f"  PR-AUC (AP)         : {average_precision_score(y_test, p_fire):.3f}")

    print("\nConfusion matrix (hang = thuc te, cot = du doan):")
    print(confusion_matrix(y_test, (p_fire >= threshold).astype(int)))
    print("\nBao cao chi tiet:")
    print(classification_report(y_test, (p_fire >= threshold).astype(int),
                                target_names=["An toan (0)", "Chay/Khoi (1)"],
                                digits=3, zero_division=0))

    episodes, per_hour = debounced_alarms(test, p_fire, threshold)
    print(f"Bao dong gia (sau luat {DEBOUNCE_K}/{DEBOUNCE_N}): {episodes} lan "
          f"= {per_hour:.2f} lan/gio")

    latencies = detection_latency(test, p_fire, threshold)
    if latencies:
        finite = [x for x in latencies if np.isfinite(x)]
        missed = len(latencies) - len(finite)
        if finite:
            print(f"Tre phat hien: trung vi {np.median(finite):.1f}s, "
                  f"xau nhat {max(finite):.1f}s trong {len(finite)} su kien chay")
        print(f"So su kien chay BI BO SOT hoan toan: {missed}/{len(latencies)}")

    baseline = baseline_scores(test)
    base_metrics = rates_at(y_test, baseline, 0.5)
    print("\nSO SANH VOI LUAT NGUONG DON GIAN (khong dung AI):")
    print(f"  recall={base_metrics['recall']:.3f} precision={base_metrics['precision']:.3f} "
          f"bao gia={base_metrics['false_alarm_rate']:.4f}")
    if base_metrics["f2"] >= best["f2"]:
        print("  ! Luat nguong khong kem hon mo hinh -> can them du lieu / them dac trung,")
        print("    hoac dung luon luat nguong cho on dinh va de giai thich.")

    print("\nMuc do quan trong cua dac trung:")
    for name, importance in sorted(zip(FEATURE_COLS, forest.feature_importances_),
                                   key=lambda item: -item[1])[:12]:
        print(f"  {name:20s} {importance:.3f}")

    csv_hash = hashlib.sha256(Path(CSV_FILE).read_bytes()).hexdigest()[:16]
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "threshold": threshold,
        "version": "2.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(feat)),
        "n_sessions": int(feat["session"].nunique()),
        "csv_sha256_16": csv_hash,
        "test_metrics": {k: v for k, v in best.items()},
        "false_alarms_per_hour": per_hour,
    }
    joblib.dump(bundle, MODEL_FILE)
    Path(REPORT_FILE).write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDa luu mo hinh: {MODEL_FILE}  (kem nguong {threshold:.3f})")
    print(f"Da luu bao cao : {REPORT_FILE}")
    if best["recall"] < MIN_RECALL:
        print(f"\n! CANH BAO: recall {best['recall']:.2f} < {MIN_RECALL} - mo hinh con bo sot "
              f"nhieu truong hop chay. Hay thu thap them du lieu chay that.")
    print("=" * 70)


if __name__ == "__main__":
    main()
