#!/usr/bin/env python3
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
from collections import deque, Counter

FRAME_W = 640
FRAME_H = 480
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

print("Запуск rpicam-vid (YUV420)...")
proc = subprocess.Popen([
    "rpicam-vid",
    "--width", str(FRAME_W),
    "--height", str(FRAME_H),
    "--framerate", "15",
    "--nopreview",
    "--codec", "yuv420",
    "-t", "0",
    "--output", "-"
], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
print(f"rpicam-vid запущен ({FRAME_W}x{FRAME_H})")

fq = queue.Queue(maxsize=2)

def _reader():
    while proc.poll() is None:
        try:
            buf = b""
            while len(buf) < FRAME_SIZE:
                chunk = proc.stdout.read(FRAME_SIZE - len(buf))
                if not chunk:
                    return
                buf += chunk
            arr = np.frombuffer(buf, dtype=np.uint8)
            gray = arr[:FRAME_W * FRAME_H].reshape(FRAME_H, FRAME_W)
            try:
                fq.put_nowait(gray)
            except queue.Full:
                try: fq.get_nowait()
                except: pass
                fq.put_nowait(gray)
        except Exception as e:
            print(f"Reader error: {e}")
            break
    try: proc.stdout.close()
    except: pass

threading.Thread(target=_reader, daemon=True).start()
print("Ждём первый кадр...")

pico = None
try:
    pico = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.01)
    print(f"UART: connected to Pico on {SERIAL_PORT}")
except Exception as e:
    print(f"UART: failed ({e}) — running in preview mode")

def send_gesture(name):
    if pico:
        try:
            pico.write(f"GEST:{name}\n".encode())
            pico.flush()
            return True
        except Exception as e:
            print(f"UART error: {e}")
            return False
    return False

def send_x(value):
    if pico:
        try:
            pico.write(f"X:{value:.3f}\n".encode())
            pico.flush()
            return True
        except Exception as e:
            print(f"UART error: {e}")
            return False
    return False

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

def thumb_extended(lm, hand_label):
    t, ip = lm[4], lm[3]
    wrist = lm[0]
    d_t_w = dist(t, wrist)
    d_ip_w = dist(ip, wrist)
    if d_t_w < d_ip_w * 1.08:
        return False
    return d_t_w > d_ip_w * 1.2

def count_fingers(lm, hand_label):
    ext = {'T': False, 'I': False, 'M': False, 'R': False, 'P': False}
    ext['I'] = finger_extended(lm, 8, 6)
    ext['M'] = finger_extended(lm, 12, 10)
    ext['R'] = finger_extended(lm, 16, 14)
    ext['P'] = finger_extended(lm, 20, 18)
    ext['T'] = thumb_extended(lm, hand_label)
    return sum(ext.values()), ext

def classify_gesture(count, ext):
    if count == 0:
        return "FIST"
    if count == 1:
        if ext.get('I'): return "ONE"
        return "UNKNOWN"
    if count == 2 and ext.get('I') and ext.get('M'):
        return "TWO"
    if count == 3 and ext.get('I') and ext.get('M') and ext.get('R') and not ext.get('P') and not ext.get('T'):
        return "THREE"
    if count == 4 and not ext.get('T'):
        return "FOUR"
    if count == 5:
        return "FIVE"
    return "UNKNOWN"

_GESTURE_COLORS = {
    "FIST":    (200, 0, 255),
    "ONE":     (0, 255, 255),
    "TWO":     (0, 255, 128),
    "THREE":   (0, 200, 200),
    "FOUR":    (0, 180, 255),
    "FIVE":    (0, 255, 80),
    "UNKNOWN": (180, 180, 180),
    "NO HAND": (80, 80, 80),
}

JOINT_LABELS = {"THREE": "J1 rotation", "TWO": "J2 shoulder", "FOUR": "J3 elbow"}
JOINT_COLORS = {"THREE": (0, 200, 200), "TWO": (0, 255, 128), "FOUR": (0, 180, 255)}

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

def draw_gripper_indicator(frame, state):
    h, w = frame.shape[:2]
    x0 = w - 130
    y0 = h - 50
    x1 = w - 20
    color = (0, 200, 0) if state == "OPEN" else (200, 100, 0)
    fw = x1 - x0
    filled = fw if state == "OPEN" else 0
    cv2.rectangle(frame, (x0, y0), (x1, y0 + 12), (80, 80, 80), 1)
    cv2.rectangle(frame, (x0, y0), (x0 + filled, y0 + 12), color, -1)
    cv2.putText(frame, f"GRIP {state}", (x0 - 10, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=MIN_DETECT,
        min_tracking_confidence=MIN_TRACK,
    )
    draw_utils = mp.solutions.drawing_utils

    gest_hist = deque(maxlen=GESTURE_FRAMES)
    stable_gest = "NO HAND"
    last_sent = ""
    last_gest_t = 0.0
    last_x_t = 0.0
    smooth_px = 0.5
    last_res = None
    first_frame = True

    print("Ready! Gestures active. Press 'q' to quit.\n")

    while True:
        try:
            gray = fq.get(timeout=5)
        except queue.Empty:
            print("Frame timeout")
            continue

        if first_frame:
            print("Первый кадр получен!")
            first_frame = False

        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame = cv2.flip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        last_res = hands.process(rgb)
        rgb.flags.writeable = True

        res = last_res
        lbl = "NO HAND"
        color = _GESTURE_COLORS["NO HAND"]
        now = time.monotonic()

        if res and res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            draw_utils.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            lm = hl.landmark
            hd = res.multi_handedness[0].classification[0].label if res.multi_handedness else 'Right'

            n, ext = count_fingers(lm, hd)
            raw = classify_gesture(n, ext)

            gest_hist.append(raw)
            if len(gest_hist) >= GESTURE_FRAMES:
                if all(g == raw for g in gest_hist):
                    stable_gest = raw

            lbl = stable_gest
            color = _GESTURE_COLORS.get(lbl, (180, 180, 180))

            px, _ = palm_center(lm)
            smooth_px = smooth_px * 0.6 + px * 0.4

            if lbl != last_sent and lbl not in ("NO HAND", "UNKNOWN") and now - last_gest_t > CMD_THROTTLE:
                ok = send_gesture(lbl)
                if ok:
                    print(f">>> {lbl}")
                    last_sent = lbl
                    last_gest_t = now

            if lbl in JOINT_GESTURES and now - last_x_t > X_THROTTLE:
                send_x(smooth_px)
                last_x_t = now

            h, w = frame.shape[:2]
            cx = int(lm[0].x * w)
            cy = int(lm[0].y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)

            fl = [k for k, v in ext.items() if v]
            cv2.putText(frame, f"Fingers: {n} [{','.join(fl)}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Stable: {lbl}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            if lbl in JOINT_LABELS:
                draw_joystick(frame, JOINT_LABELS[lbl], smooth_px)
            elif lbl == "FIST":
                draw_gripper_indicator(frame, "CLOSE")
            elif lbl == "FIVE":
                draw_gripper_indicator(frame, "OPEN")

        else:
            gest_hist.clear()
            stable_gest = "NO HAND"

        uart_status = "UART OK" if pico else "PREVIEW"
        cv2.putText(frame, f"{uart_status}  Sent: {last_sent}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 200, 0) if pico else (0, 100, 200), 1)
        cv2.putText(frame, f"Gesture: {lbl}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Robo Arm Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    proc.terminate()
    if pico:
        pico.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
