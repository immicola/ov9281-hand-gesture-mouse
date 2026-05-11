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
from collections import deque

FRAME_W = 640
FRAME_H = 480
FRAME_SIZE = (FRAME_W * FRAME_H * 3) // 2

MIN_DETECT = 0.7
MIN_TRACK = 0.5
GESTURE_FRAMES = 3
PROCESS_EVERY_N = 2

SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 115200
CMD_THROTTLE = 0.5

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
            return True
        except Exception as e:
            print(f"UART write error: {e}")
            return False
    return False

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def finger_extended(lm, tip_idx, pip_idx):
    tip = lm[tip_idx]
    pip = lm[pip_idx]
    mcp = lm[tip_idx - 3]
    wrist = lm[0]
    if tip.y >= pip.y:
        return False
    if dist(tip, wrist) < dist(pip, wrist) * 1.08:
        return False
    if dist(tip, mcp) < dist(pip, mcp) * 1.05:
        return False
    return True

def thumb_extended(lm, hand_label):
    t, ip, mcp = lm[4], lm[3], lm[2]
    wrist = lm[0]
    if dist(t, wrist) < dist(ip, wrist) * 1.08:
        return False
    pcx = (lm[0].x + lm[5].x + lm[9].x) / 3
    if abs(t.x - pcx) > 0.08 and dist(t, lm[5]) > dist(ip, lm[5]) * 1.15:
        return True
    if (t.y < ip.y - 0.03 and
        dist(t, wrist) > dist(ip, wrist) * 1.2 and
        dist(t, wrist) > dist(mcp, wrist) * 1.15):
        return True
    if hand_label == 'Right':
        if t.x + 0.01 < ip.x and dist(t, lm[5]) > dist(ip, lm[5]) * 1.1:
            return True
    else:
        if t.x > ip.x + 0.01 and dist(t, lm[5]) > dist(ip, lm[5]) * 1.1:
            return True
    return False

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
        if ext.get('T'): return "LIKE"
        if ext.get('I'): return "ONE"
        return "UNKNOWN"
    if count == 2 and ext.get('I') and ext.get('M'):
        return "TWO"
    if count == 3:
        return "THREE"
    if count == 4 and not ext.get('T'):
        return "FOUR"
    if count == 5:
        return "FIVE"
    return "UNKNOWN"

_GESTURE_COLORS = {
    "FIST":    (200, 0, 255),
    "LIKE":    (0, 200, 255),
    "ONE":     (0, 255, 255),
    "TWO":     (0, 255, 128),
    "THREE":   (0, 200, 200),
    "FOUR":    (0, 180, 255),
    "FIVE":    (0, 255, 80),
    "UNKNOWN": (180, 180, 180),
    "NO HAND": (80, 80, 80),
}

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
    last_send_t = 0.0
    send_ok = False
    frame_idx = 0
    last_res = None

    print("Robo Arm Gesture → Thor Controller. Press 'q' to quit.\n")

    while True:
        try:
            gray = fq.get(timeout=5)
        except queue.Empty:
            print("Frame timeout")
            continue

        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame = cv2.flip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1)

        frame_idx += 1
        if frame_idx % PROCESS_EVERY_N == 0:
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
            else:
                stable_gest = raw

            lbl = stable_gest
            color = _GESTURE_COLORS.get(lbl, (180, 180, 180))

            if lbl != last_sent and lbl not in ("NO HAND", "UNKNOWN") and now - last_send_t > CMD_THROTTLE:
                send_ok = send_gesture(lbl)
                if send_ok:
                    print(f">>> {lbl}")
                    last_sent = lbl
                    last_send_t = now
                else:
                    print(f"FAIL: {lbl}")

            h, w = frame.shape[:2]
            cx = int(lm[0].x * w)
            cy = int(lm[0].y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)

            fl = [k for k, v in ext.items() if v]
            cv2.putText(frame, f"Fingers: {n} [{','.join(fl)}]",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Stable: {stable_gest}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        else:
            if frame_idx % PROCESS_EVERY_N == 0:
                gest_hist.clear()
                stable_gest = "NO HAND"

        # HUD: status line
        if pico:
            status = f"UART OK  Last: {last_sent}"
            sc = (0, 200, 0)
        else:
            status = "UART OFF (preview mode)"
            sc = (0, 100, 200)
        cv2.putText(frame, status, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

        cv2.putText(frame, f"Gesture: {lbl}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Robo Arm → Thor via Pico", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    proc.terminate()
    if pico:
        pico.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
