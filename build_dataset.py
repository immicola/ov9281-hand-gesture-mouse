"""
build_dataset.py — сбор датасета для дистилляции из 8 видео жестов

Берёт 8 видео (по одному на жест), для каждого кадра:
  1. MediaPipe HandLandmarker → 21 точка
  2. Нормализация (центр на запястье, scale по max_dist)
  3. One-hot метка по номеру видео

Результат: CSV (42 координаты + 8 вероятностей) для train_student.py

Запуск: python3 build_dataset.py
"""

import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import cv2
import csv
import math
import json
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── КОНФИГ ────────────────────────────────────────────────────────────────────
GESTURES_DIR = Path("videos")
OUTPUT_CSV = Path("distillation_dataset.csv")
HAND_MODEL = Path("hand_landmarker.task")
FRAME_STEP = 3  # каждый 3-й кадр (60fps → 20fps)

# Список видеофайлов: (имя_файла_без_расширения, метка_класса)
GESTURE_VIDEOS = [
    ("none",         0),      # пустой жест / без руки (опционально)
    ("closed_fist",  1),
    ("open_palm",    2),
    ("point_up",     3),
    ("thumb_down",   4),
    ("thumb_up",     5),
    ("victory",      6),
    ("iloveyou",     7),
]

# Альтернатива — загрузить из JSON
LABEL_MAP_FILE = Path("gesture_labels.json")

# ── ЗАГРУЗКА HAND LANDMARKER ─────────────────────────────────────────────────
print("Loading HandLandmarker (CPU)...")
base_options = python.BaseOptions(
    model_asset_path=str(HAND_MODEL),
    delegate=python.BaseOptions.Delegate.CPU,
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ── НОРМАЛИЗАЦИЯ (как в статье) ──────────────────────────────────────────────
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
    return result  # 42 float

# ── ЗАГРУЗКА МЭППИНГА ────────────────────────────────────────────────────────
if LABEL_MAP_FILE.exists():
    with open(LABEL_MAP_FILE) as f:
        mapping = json.load(f)
    GESTURE_VIDEOS = [(item["file"], item["label"]) for item in mapping]
    CLASSES = [item["gesture"] for item in mapping]
else:
    CLASSES = [v[0] for v in GESTURE_VIDEOS]

print(f"Классы ({len(CLASSES)}): {CLASSES}")
print(f"Папка с видео: {GESTURES_DIR.resolve()}")

# ── СБОР ДАТАСЕТА ─────────────────────────────────────────────────────────────
all_rows = []
total_frames = 0
skipped = 0

# Заголовки CSV
headers = [f"coord_{i}" for i in range(42)] + [f"prob_{c}" for c in CLASSES]

for filename, label_idx in GESTURE_VIDEOS:
    video_path = GESTURES_DIR / f"{filename}.mp4"
    if not video_path.exists():
        print(f"  ⚠️  Пропущен: {video_path}")
        continue

    print(f"  Обработка: {filename}.mp4 (класс {label_idx}: {CLASSES[label_idx]})")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_from_video = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            coords = normalize(hand)

            # One-hot: 1.0 на позиции label_idx, 0.0 на остальных
            one_hot = [0.0] * len(CLASSES)
            one_hot[label_idx] = 1.0

            all_rows.append(coords + one_hot)
            frames_from_video += 1
        else:
            skipped += 1

        frame_idx += 1
        total_frames += 1

    cap.release()
    print(f"    → {frames_from_video} записей из {frame_count} кадров")

landmarker.close()

# ── СОХРАНЕНИЕ ────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(all_rows)

print(f"\nГотово!")
print(f"  Всего кадров обработано : {total_frames}")
print(f"  Записей в датасете      : {len(all_rows)}")
print(f"  Пропущено (нет руки)    : {skipped}")
print(f"  Результат               : {OUTPUT_CSV}")