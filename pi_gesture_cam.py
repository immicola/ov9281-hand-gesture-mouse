#!/usr/bin/env python3
"""
Pi Gesture Cam — TFLite gesture recognition on Raspberry Pi 4 + OV9281
Захват с rpicam-vid → MediaPipe Hands (21 landmarks) → TFLite (8 классов) → вывод на экран

Модель: student_model_7kb.tflite (обучена через knowledge distillation)
Классы: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import queue
import subprocess
from collections import deque
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_model_7kb.tflite")

CAPTURE_W = 640
CAPTURE_H = 480
FRAMERATE = 30

MIN_DETECTION = 0.65
MIN_TRACKING = 0.45

SMOOTH_FRAMES = 3           # скользящее окно для стабилизации жеста

CLASSES = [
    "None", "Closed_Fist", "Open_Palm", "Pointing_Up",
    "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"
]

GESTURE_COLORS = {
    "None":       (128, 128, 128),
    "Closed_Fist":(200,   0, 255),
    "Open_Palm":  (  0, 220, 255),
    "Pointing_Up":(  0, 180, 255),
    "Thumb_Down": (  0,   0, 200),
    "Thumb_Up":   (  0, 200,   0),
    "Victory":    (  0, 255, 128),
    "ILoveYou":   (200, 100, 200),
}

# ─── TFLite MODEL ─────────────────────────────────────────────────────────────
print(f"Loading TFLite model: {MODEL_PATH}")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_details  = interpreter.get_input_details()
out_details = interpreter.get_output_details()
print(f"  Input  → {in_details[0]['shape']}  ({in_details[0]['dtype']})")
print(f"  Output → {out_details[0]['shape']} ({out_details[0]['dtype']})")


def normalize_landmarks(landmarks):
    """21 MediaPipe landmark → 42 нормированных координат (как в teacher_labeling.py)"""
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
    return result  # список из 42 float


def classify_gesture(landmarks):
    """Нормируем landmarks → TFLite inference → возвращаем (label, prob, all_probs)"""
    coords = normalize_landmarks(landmarks)
    inp = np.array(coords, dtype=np.float32).reshape(1, 42)
    interpreter.set_tensor(in_details[0]["index"], inp)
    interpreter.invoke()
    probs = interpreter.get_tensor(out_details[0]["index"])[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs[idx], probs


# ─── CAMERA (rpicam-vid YUV420) ──────────────────────────────────────────────
FRAME_SIZE = (CAPTURE_W * CAPTURE_H * 3) // 2  # YUV420

print(f"Starting rpicam-vid ({CAPTURE_W}x{CAPTURE_H} @ {FRAMERATE}fps)...")
cam_proc = subprocess.Popen([
    "rpicam-vid",
    "--width",     str(CAPTURE_W),
    "--height",    str(CAPTURE_H),
    "--framerate", str(FRAMERATE),
    "--nopreview",
    "--codec",     "yuv420",
    "-t",          "0",
    "--output",    "-",
], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_queue = queue.Queue(maxsize=2)


def _camera_reader():
    while cam_proc.poll() is None:
        try:
            buf = b""
            while len(buf) < FRAME_SIZE:
                chunk = cam_proc.stdout.read(FRAME_SIZE - len(buf))
                if not chunk:
                    return
                buf += chunk
            arr = np.frombuffer(buf, dtype=np.uint8)
            gray = arr[:CAPTURE_W * CAPTURE_H].reshape(CAPTURE_H, CAPTURE_W)
            try:
                frame_queue.put_nowait(gray)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(gray)
        except Exception as e:
            print(f"Camera reader error: {e}")
            break
    try:
        cam_proc.stdout.close()
    except Exception:
        pass


threading.Thread(target=_camera_reader, daemon=True).start()
print("Waiting for first frame...")

# ─── MEDIAPIPE HANDS ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=MIN_DETECTION,
    min_tracking_confidence=MIN_TRACKING,
)
mp_draw = mp.solutions.drawing_utils

# ─── SMOOTHING ────────────────────────────────────────────────────────────────
gesture_history = deque(maxlen=SMOOTH_FRAMES)
stable_gesture = "None"

# ─── FPS COUNTER ──────────────────────────────────────────────────────────────
frame_count = 0
fps_timer = time.monotonic()
current_fps = 0

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
print("\nReady! Press 'q' to quit.\n")

while True:
    # ── Get frame ─────────────────────────────────────────────────────────
    try:
        gray = frame_queue.get(timeout=5)
        frame_count += 1
        if frame_count == 1:
            print("First frame received!")
    except queue.Empty:
        print("Frame timeout — restarting camera?")
        continue

    # ── FPS ───────────────────────────────────────────────────────────────
    if frame_count % 30 == 0:
        now = time.monotonic()
        current_fps = int(30 / (now - fps_timer))
        fps_timer = now

    # ── Convert YUV → BGR ─────────────────────────────────────────────────
    # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame = cv2.flip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1)

    # ── MediaPipe Hands ───────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands_detector.process(rgb)
    rgb.flags.writeable = True

    label = "No Hand"
    confidence = 0.0
    all_probs = None
    color = (80, 80, 80)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        lm = hand_landmarks.landmark

        # ── TFLite inference ──────────────────────────────────────────────
        label, confidence, all_probs = classify_gesture(lm)
        color = GESTURE_COLORS.get(label, (180, 180, 180))

        # ── Smooth ────────────────────────────────────────────────────────
        gesture_history.append(label)
        if len(gesture_history) >= SMOOTH_FRAMES and all(g == label for g in gesture_history):
            stable_gesture = label
        else:
            label = stable_gesture
    else:
        gesture_history.clear()
        stable_gesture = "None"

    # ─── HUD ──────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    # Gesture name (large, top-left)
    cv2.putText(frame, f"Gesture: {label}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"Conf: {confidence:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # FPS (top-right)
    fps_text = f"FPS: {current_fps}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, fps_text, (w - tw - 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    # Probability bars (bottom-left)
    if all_probs is not None:
        bar_x = 10
        bar_y = h - 170
        bar_w = 120
        bar_h = 12
        gap = 2

        for i, (cls, prob) in enumerate(zip(CLASSES, all_probs)):
            y = bar_y + i * (bar_h + gap)
            fill = int(prob * bar_w)
            clr = GESTURE_COLORS.get(cls, (100, 100, 100))

            # background
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h),
                          (40, 40, 40), -1)
            # filled bar
            if fill > 0:
                cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + bar_h),
                              clr, -1)
            # border
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h),
                          (60, 60, 60), 1)
            # label
            cv2.putText(frame, f"{cls[:11]}", (bar_x + bar_w + 6, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # ── Show ──────────────────────────────────────────────────────────────
    cv2.imshow("Pi Gesture Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
cam_proc.terminate()
cv2.destroyAllWindows()
print("\nDone.")