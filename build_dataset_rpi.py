"""
build_dataset_rpi.py — сбор датасета на Raspberry Pi (mediapipe 0.10.x с mp.solutions)

Берёт 8 видео из папки videos/, для каждого кадра:
  1. MediaPipe mp.solutions.hands → 21 точка
  2. Нормализация (центр на запястье, scale по max_dist)
  3. One-hot метка из gesture_labels.json

Результат: CSV (42 координаты + 8 one-hot) → перенести на ПК для train_student.py

Запуск на RPi: python3 build_dataset_rpi.py
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import csv
import math
import json
import sys
from pathlib import Path

import mediapipe as mp

# ── КОНФИГ ──
VIDEOS_DIR = Path("videos")
OUTPUT_CSV = Path("distillation_dataset.csv")
LABEL_MAP_FILE = Path("gesture_labels.json")
FRAME_STEP = 3  # каждый 3-й кадр (60fps → 20fps)

# ── ЗАГРУЗКА МЭППИНГА ──
if LABEL_MAP_FILE.exists():
    with open(LABEL_MAP_FILE) as f:
        mapping = json.load(f)
    GESTURE_VIDEOS = [(item["file"], item["label"]) for item in mapping]
    CLASSES = [item["gesture"] for item in mapping]
else:
    print(f"Файл {LABEL_MAP_FILE} не найден. Создай его с маппингом видео → классы.")
    sys.exit(1)

print(f"Классы ({len(CLASSES)}): {CLASSES}")

# ── MP.SOLUTIONS.HANDS ──
print("Loading MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ── НОРМАЛИЗАЦИЯ ──
def normalize(landmarks):
    base_x, base_y = landmarks[0].x, landmarks[0].y
    points = []
    max_dist = 0.0
    for lm in landmarks:
        nx = lm.x - base_x
        ny = lm.y - base_y
        points.append((nx, ny))
        dist = math.hypot(nx, ny)
        if dist > max_dist:
            max_dist = dist
    result = []
    if max_dist > 0:
        for nx, ny in points:
            result.append(nx / max_dist)
            result.append(ny / max_dist)
    else:
        for _ in points:
            result.extend([0.0, 0.0])
    return result

# ── СБОР ДАТАСЕТА ──
all_rows = []
total = 0
skipped = 0

headers = [f"coord_{i}" for i in range(42)] + [f"prob_{c}" for c in CLASSES]

for filename, label_idx in GESTURE_VIDEOS:
    video_path = VIDEOS_DIR / f"{filename}.mp4"
    if not video_path.exists():
        print(f"  ⚠️  Пропущен: {video_path}")
        continue

    print(f"  {filename}.mp4 → класс {label_idx} ({CLASSES[label_idx]})")
    cap = cv2.VideoCapture(str(video_path))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok = 0

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % FRAME_STEP != 0:
            frame_i += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            coords = normalize(lm)
            one_hot = [0.0] * len(CLASSES)
            one_hot[label_idx] = 1.0
            all_rows.append(coords + one_hot)
            ok += 1
        else:
            skipped += 1

        total += 1
        frame_i += 1

    cap.release()
    print(f"    → {ok} записей из {fc} кадров")

hands.close()

# ── СОХРАНЕНИЕ ──
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(all_rows)

print(f"\nГотово!")
print(f"  Записей: {len(all_rows)}")
print(f"  Пропущено (нет руки): {skipped}")
print(f"  Файл: {OUTPUT_CSV}")