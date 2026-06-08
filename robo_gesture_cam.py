#!/usr/bin/env python3
"""
Robo Gesture Cam — Raspberry Pi 4 + OV9281 + MediaPipe → UART → Pico → USB → PC (ROS2)

Захват с rpicam-vid → MediaPipe Hands → rule-based finger counting (5 жестов)
→ UART команды на Pico (GEST:name + X:value) → USB на PC → thor_bridge.py → ROS2

Жесты:
  FIST    (0 пальцев) → GRIPPER_CLOSE
  ONE     (1 палец)   → IDLE / point
  TWO     (2 пальца)  → J2 shoulder (джойстик по X)
  THREE   (3 пальца)  → J1 rotation (джойстик по X)
  FOUR    (4 пальца)  → J3 elbow    (джойстик по X)
  FIVE    (5 пальцев) → GRIPPER_OPEN

Запуск:
  python3 robo_gesture_cam.py           # с экраном
  python3 robo_gesture_cam.py --headless # без экрана

Зависимости:
  pip install opencv-contrib-python==4.10.0.84 numpy==1.26.4 mediapipe==0.10.18 pyserial
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
import threading
import queue
import serial
import time
import sys
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FRAME_W = 640
FRAME_H = 480
FRAMERATE = 15
FRAME_SIZE = (FRAME_W * FRAME_H * 3) // 2

MIN_DETECT = 0.65
MIN_TRACK = 0.45
GESTURE_FRAMES = 3

SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 115200
CMD_THROTTLE = 0.15
X_THROTTLE = 0.03
DEAD_ZONE = 0.04

JOINT_GESTURES = ("THREE", "TWO", "FOUR")

HEADLESS = "--headless" in sys.argv

# ─── COLORS ───────────────────────────────────────────────────────────────────
GESTURE_COLORS = {
    "FIST":    (200,   0, 255),
    "ONE":     (  0, 255, 255),
    "TWO":     (  0, 255, 128),
    "THREE":   (  0, 200, 200),
    "FOUR":    (  0, 180, 255),
    "FIVE":    (  0, 255,  80),
    "UNKNOWN": (180, 180, 180),
    "NO HAND": ( 80,  80,  80),
}

JOINT_LABELS = {
    "THREE": "J1 rotation",
    "TWO":   "J2 shoulder",
    "FOUR":  "J3 elbow",
}

JOINT_COLORS = {
    "THREE": (  0, 200, 200),
    "TWO":   (  0, 255, 128),
    "FOUR":  (  0, 180, 255),
}

# ─── UART ──────────────────────────────────────────────────────────────────────
pico = None
try:
    pico = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
    print(f"UART: connected to Pico on {SERIAL_PORT}")
except Exception as e:
    print(f"UART: {e} — running in preview mode")


def uart_send(cmd):
    if pico:
        try:
            pico.write(cmd.encode())
            pico.flush()
            return True
        except Exception as e:
            print(f"UART error: {e}")
            return False
    return False


def send_gesture(name):
    return uart_send(f"GEST:{name}\n")


def send_x(value):
    return uart_send(f"X:{value:.3f}\n")


# ─── CAMERA (rpicam-vid YUV420) ──────────────────────────────────────────────
print(f"Starting rpicam-vid ({FRAME_W}x{FRAME_H} @ {FRAMERATE}fps)...")
cam_proc = subprocess.Popen([
    "rpicam-vid",
    "--width",     str(FRAME_W),
    "--height",    str(FRAME_H),
    "--framerate", str(FRAMERATE),
    "--nopreview",
    "--codec",     "yuv420",
    "-t",          "0",
    "--output",    "-",
], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_queue = queue.Queue(maxsize=2)


def _reader():
    while cam_proc.poll() is None:
        try:
            buf = b""
            while len(buf) < FRAME_SIZE:
                chunk = cam_proc.stdout.read(FRAME_SIZE - len(buf))
                if not chunk:
                    return
                buf += chunk
            arr = np.frombuffer(buf, dtype=np.uint8)
            gray = arr[:FRAME_W * FRAME_H].reshape(FRAME_H, FRAME_W)
            try:
                frame_queue.put_nowait(gray)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait(gray)
        except Exception as e:
            print(f"Reader error: {e}")
            break
    try:
        cam_proc.stdout.close()
    except Exception:
        pass


threading.Thread(target=_reader, daemon=True).start()
print("Waiting for first frame...")


# ─── FINGER LOGIC ─────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_center(lm):
    idx = (0, 1, 5, 9, 13, 17)
    return (sum(lm[i].x for i in idx) / 6,
            sum(lm[i].y for i in idx) / 6)


def finger_extended(lm, tip_idx, pip_idx):
    tip = lm[tip_idx]
    pip = lm[pip_idx]
    mcp = lm[tip_idx - 3]
    wrist = lm[0]

    d_tip_wrist = dist(tip, wrist)
    d_pip_wrist = dist(pip, wrist)
    if d_tip_wrist < d_pip_wrist * 1.08:
        return False

    d_tip_mcp = dist(tip, mcp)
    d_pip_mcp = dist(pip, mcp)
    if d_tip_mcp < d_pip_mcp * 1.05:
        return False

    return True


def thumb_extended(lm):
    t, ip = lm[4], lm[3]
    wrist = lm[0]
    d_t_w = dist(t, wrist)
    d_ip_w = dist(ip, wrist)

    if d_t_w < d_ip_w * 1.08:
        return False
    return d_t_w > d_ip_w * 1.2


def count_fingers(lm):
    ext = {
        'T': thumb_extended(lm),
        'I': finger_extended(lm, 8, 6),
        'M': finger_extended(lm, 12, 10),
        'R': finger_extended(lm, 16, 14),
        'P': finger_extended(lm, 20, 18),
    }
    return sum(ext.values()), ext


def classify_gesture(count, ext):
    if count == 0:
        return "FIST"
    if count == 1:
        return "ONE" if ext.get('I') else "UNKNOWN"
    if count == 2 and ext.get('I') and ext.get('M'):
        return "TWO"
    if count == 3 and ext.get('I') and ext.get('M') and ext.get('R'):
        return "THREE"
    if count == 4 and not ext.get('T'):
        return "FOUR"
    if count == 5:
        return "FIVE"
    return "UNKNOWN"


# ─── DRAW HELPERS ─────────────────────────────────────────────────────────────
def draw_joystick(frame, label, px):
    h, w = frame.shape[:2]
    cx = w - 90
    cy = h // 2
    dz = 18
    mz = 60

    cv2.putText(frame, label, (cx - 40, cy - mz - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), mz, (80, 80, 80), 1)
    cv2.circle(frame, (cx, cy), dz, (120, 120, 120), 1)
    cv2.line(frame, (cx - mz, cy), (cx + mz, cy), (60, 60, 60), 1)
    cv2.putText(frame, "-", (cx - mz - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    cv2.putText(frame, "+", (cx + mz + 4, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

    off = (px - 0.5) * 2
    hx = int(cx + off * mz)
    hx = max(cx - mz, min(cx + mz, hx))
    color = JOINT_COLORS.get(label, (0, 255, 255))

    cv2.line(frame, (cx, cy), (hx, cy), color, 2)
    cv2.circle(frame, (hx, cy), 10, color, -1)
    cv2.circle(frame, (hx, cy), 10, (255, 255, 255), 2)

    arrow = ">" if off > DEAD_ZONE else ("<" if off < -DEAD_ZONE else ".")
    cv2.putText(frame, f"{arrow}  {off:+.2f}", (cx - 18, cy + mz + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_gripper(frame, state):
    h, w = frame.shape[:2]
    x0 = w - 130
    y0 = h - 50
    x1 = w - 20
    color = (0, 200, 0) if state == "OPEN" else (200, 100, 0)
    filled = (x1 - x0) if state == "OPEN" else 0

    cv2.rectangle(frame, (x0, y0), (x1, y0 + 12), (80, 80, 80), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + 12), color, -1)
    cv2.putText(frame, f"GRIP {state}", (x0 - 10, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK,
    )

    gest_hist = deque(maxlen=GESTURE_FRAMES)
    stable_gest = "NO HAND"
    last_sent = ""
    last_gest_t = 0.0
    last_x_t = 0.0
    smooth_px = 0.5
    first_frame = True

    print(f"\nReady! {'Headless' if HEADLESS else 'GUI'} mode. Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                gray = frame_queue.get(timeout=5)
            except queue.Empty:
                print("Frame timeout")
                continue

            if first_frame:
                print("First frame received!")
                first_frame = False

            # ── YUV → BGR ─────────────────────────────────────────────────
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame = cv2.flip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1)

            # ── MediaPipe ─────────────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            lbl = "NO HAND"
            color = GESTURE_COLORS["NO HAND"]
            now = time.monotonic()

            if result.multi_hand_landmarks:
                hl = result.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS
                )
                lm = hl.landmark

                n, ext = count_fingers(lm)
                raw = classify_gesture(n, ext)

                gest_hist.append(raw)
                if len(gest_hist) >= GESTURE_FRAMES:
                    if all(g == raw for g in gest_hist):
                        stable_gest = raw

                lbl = stable_gest
                color = GESTURE_COLORS.get(lbl, (180, 180, 180))

                # Palm X for joystick
                px, _ = palm_center(lm)
                smooth_px = smooth_px * 0.6 + px * 0.4

                # UART: gesture
                if lbl not in ("NO HAND", "UNKNOWN") and lbl != last_sent and now - last_gest_t > CMD_THROTTLE:
                    send_gesture(lbl)
                    print(f">>> {lbl}")
                    last_sent = lbl
                    last_gest_t = now

                # UART: joystick X
                if lbl in JOINT_GESTURES and now - last_x_t > X_THROTTLE:
                    send_x(smooth_px)
                    last_x_t = now

                # ── HUD ──────────────────────────────────────────────────
                if not HEADLESS:
                    h, w = frame.shape[:2]

                    cv2.circle(frame, (int(lm[0].x * w), int(lm[0].y * h)),
                               8, (0, 255, 0), 2)

                    fl = [k for k, v in ext.items() if v]
                    cv2.putText(frame, f"Fingers: {n} [{','.join(fl)}]",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 255), 1)

                    if lbl in JOINT_LABELS:
                        draw_joystick(frame, JOINT_LABELS[lbl], smooth_px)
                    elif lbl == "FIST":
                        draw_gripper(frame, "CLOSE")
                    elif lbl == "FIVE":
                        draw_gripper(frame, "OPEN")

                    uart_color = (0, 200, 0) if pico else (0, 100, 200)
                    cv2.putText(frame, f"UART: {'OK' if pico else 'NO'}  Sent: {last_sent}",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, uart_color, 1)

                    cv2.putText(frame, f"Gesture: {lbl}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            else:
                gest_hist.clear()
                stable_gest = "NO HAND"

            # ── Show ─────────────────────────────────────────────────────
            if not HEADLESS:
                cv2.imshow("Robo Gesture Cam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cam_proc.terminate()
        if pico:
            pico.close()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()