#!/usr/bin/env python3
"""
Hand Gesture Mouse Controller - OV9281 + Picamera2
Raspberry Pi 4 · Debian Trixie · labwc (Wayland)

Uses /dev/uinput (kernel virtual input) — works on Wayland, no X11 needed.

Combines:
- camera_opencv.py: OV9281 capture via Picamera2
- hand_gesture_mouse_v2.py: Gesture recognition + mouse control

Setup (one-time):
    sudo apt install python3-evdev libopencv-dev
    source gesture_env/bin/activate
    pip install picamera2 opencv-python mediapipe evdev numpy
    sudo chmod 666 /dev/video0
    # Allow your user to write to uinput without sudo:
    echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
    sudo udevadm control --reload && sudo udevadm trigger
    sudo usermod -aG input $USER
    # Log out and back in, then run without sudo:
    source gesture_env/bin/activate
    python3 hand_gesture_ov9281.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import queue
import math
from collections import deque
import select

PICAMERA2_AVAILABLE = False
picam = None

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("Picamera2 not available, using OpenCV VideoCapture")

try:
    import evdev
    from evdev import UInput, AbsInfo, ecodes as e

    SCREEN_W = 2560
    SCREEN_H = 1440

    cap_evdev = UInput(
        name="gesture-mouse",
        events={
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=0, min=0, max=SCREEN_W - 1,
                          fuzz=0, flat=0, resolution=0)),
                (e.ABS_Y, AbsInfo(value=0, min=0, max=SCREEN_H - 1,
                          fuzz=0, flat=0, resolution=0)),
            ],
            e.EV_SYN: [],
        },
        input_props=[e.INPUT_PROP_POINTER],
    )
    USE_UINPUT = True
    print(f"uinput virtual mouse created ({SCREEN_W}x{SCREEN_H})")
    print("  -> If cursor doesn't move, check: ls -la /dev/uinput")
    print("     and run: sudo usermod -aG input $USER  then re-login")

except Exception as ex:
    print(f"evdev/uinput unavailable ({ex})")
    print("Falling back to pyautogui (won't work on Wayland)")
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    USE_UINPUT = False
    SCREEN_W, SCREEN_H = pyautogui.size()


def mouse_move(x, y):
    if USE_UINPUT:
        cap_evdev.write(e.EV_ABS, e.ABS_X, int(np.clip(x, 0, SCREEN_W - 1)))
        cap_evdev.write(e.EV_ABS, e.ABS_Y, int(np.clip(y, 0, SCREEN_H - 1)))
        cap_evdev.syn()
    else:
        pyautogui.moveTo(x, y)

def mouse_click():
    if USE_UINPUT:
        cap_evdev.write(e.EV_KEY, e.BTN_LEFT, 1); cap_evdev.syn()
        time.sleep(0.02)
        cap_evdev.write(e.EV_KEY, e.BTN_LEFT, 0); cap_evdev.syn()
    else:
        pyautogui.click()

def mouse_down():
    if USE_UINPUT:
        cap_evdev.write(e.EV_KEY, e.BTN_LEFT, 1); cap_evdev.syn()
    else:
        pyautogui.mouseDown()

def mouse_up():
    if USE_UINPUT:
        cap_evdev.write(e.EV_KEY, e.BTN_LEFT, 0); cap_evdev.syn()
    else:
        pyautogui.mouseUp()


mouse_queue = queue.Queue(maxsize=4)

def mouse_worker():
    while True:
        cmd = mouse_queue.get()
        if cmd is None:
            break
        action = cmd[0]
        if action == 'move':   mouse_move(cmd[1], cmd[2])
        elif action == 'click': mouse_click()
        elif action == 'down':  mouse_down()
        elif action == 'up':    mouse_up()

mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
mouse_thread.start()

def enqueue(cmd):
    try:
        mouse_queue.put_nowait(cmd)
    except queue.Full:
        if cmd[0] == 'move':
            pass
        else:
            try: mouse_queue.get_nowait()
            except: pass
            mouse_queue.put_nowait(cmd)


CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 400

CLICK_COOLDOWN     = 0.4
PINCH_THRESHOLD    = 0.045
PINCH_RELEASE      = 0.08
GESTURE_DEBOUNCE   = 2
FIST_MAX_FINGERS   = 0
OPEN_MIN_FINGERS    = 2
MIN_DETECT_CONF    = 0.65
MIN_TRACK_CONF     = 0.45

DEAD_ZONE   = 0.045
MAX_ZONE    = 0.30
MAX_SPEED   = 120
STICK_SMOOTH = 0.98


import select

def init_capture():
    import subprocess, signal
    import sys

    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    proc = subprocess.Popen([
        "rpicam-vid", "--width", "640", "--height", "400",
        "--framerate", "30", "--nopreview",
        "-t", "0", "--output", "-"
    ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print("rpicam-vid started (640x400)")
    return proc

def frame_reader_thread(proc, frame_queue):
    import io
    buffer = b""
    while proc.poll() is None:
        try:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buffer += chunk
            
            while True:
                start = buffer.find(b'\xff\xd8')
                if start < 0:
                    if len(buffer) > 131072:
                        buffer = buffer[-4096:]
                    break
                buffer = buffer[start:]
                end = buffer.find(b'\xff\xd9', 2)
                if end < 0:
                    if len(buffer) > 131072:
                        buffer = buffer[-4096:]
                    break
                
                jpeg = buffer[:end+2]
                buffer = buffer[end+2:]
                
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if frame is not None and frame.shape[0] > 100:
                    try:
                        frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                        except:
                            pass
                        frame_queue.put_nowait(frame)
        except Exception:
            break
    
    try:
        proc.stdout.close()
    except:
        pass

if PICAMERA2_AVAILABLE:
    print("Initializing Picamera2...")
    try:
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "YUV420"},
        )
        config["raw"] = None
        picam.configure(config)
        picam.start()
        time.sleep(1)
        print(f"Picamera2 running ({CAPTURE_WIDTH}x{CAPTURE_HEIGHT})")
    except Exception as picam_err:
        print(f"Picamera2 init failed ({picam_err}), falling back to OpenCV")
        PICAMERA2_AVAILABLE = False
        picam = None
        cap = init_capture()
else:
    print("Initializing OpenCV VideoCapture...")
    cap = init_capture()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=MIN_DETECT_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)
mp_draw = mp.solutions.drawing_utils


cur_x = SCREEN_W / 2
cur_y = SCREEN_H / 2
smooth_x = 0.5
smooth_y = 0.5
centre_x = 0.5
centre_y = 0.5

is_dragging      = False
last_click_time  = 0.0
pinch_active    = False
pinch_released  = True

gesture_history = deque(maxlen=GESTURE_DEBOUNCE)
current_stable_gesture = "NONE"


def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def count_extended_fingers(lm):
    wrist = lm[0]
    fingers = {'thumb': False, 'index': False, 'middle': False,
              'ring': False, 'pinky': False}

    if lm[8].y < lm[6].y and dist(lm[8], wrist) > dist(lm[6], wrist) * 1.1:
        fingers['index'] = True
    if lm[12].y < lm[10].y and dist(lm[12], wrist) > dist(lm[10], wrist) * 1.1:
        fingers['middle'] = True
    if lm[16].y < lm[14].y and dist(lm[16], wrist) > dist(lm[14], wrist) * 1.1:
        fingers['ring'] = True
    if lm[20].y < lm[18].y and dist(lm[20], wrist) > dist(lm[18], wrist) * 1.1:
        fingers['pinky'] = True

    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    palm_center_x = (lm[0].x + lm[5].x + lm[9].x) / 3
    thumb_dist = abs(thumb_tip.x - palm_center_x)
    if thumb_dist > 0.08 and dist(thumb_tip, wrist) > dist(thumb_mcp, wrist) * 1.1:
        fingers['thumb'] = True

    return sum(fingers.values()), fingers

def get_palm_center(lm):
    PALM_IDX = (0, 1, 5, 9, 13, 17)
    x = sum(lm[i].x for i in PALM_IDX) / len(PALM_IDX)
    y = sum(lm[i].y for i in PALM_IDX) / len(PALM_IDX)
    return x, y

def stick_velocity(sx, sy, cx, cy):
    dx = sx - cx
    dy = sy - cy
    r = math.sqrt(dx*dx + dy*dy)
    if r < DEAD_ZONE:
        return 0.0, 0.0
    t = min((r - DEAD_ZONE) / (MAX_ZONE - DEAD_ZONE), 1.0)
    speed = t * t * MAX_SPEED
    return (dx / r) * speed, (dy / r) * speed

def detect_gesture(lm):
    fingers_ext, _ = count_extended_fingers(lm)
    pinch_dist = dist(lm[4], lm[8])
    if pinch_dist < PINCH_THRESHOLD:
        return "PINCH"
    if fingers_ext <= FIST_MAX_FINGERS:
        return "FIST"
    if fingers_ext >= OPEN_MIN_FINGERS:
        return "OPEN"
    return "PARTIAL"

def get_stable_gesture(cur):
    global gesture_history, current_stable_gesture
    gesture_history.append(cur)
    if len(gesture_history) < GESTURE_DEBOUNCE:
        return current_stable_gesture
    if all(g == cur for g in gesture_history):
        current_stable_gesture = cur
    return current_stable_gesture


def read_frame_jpeg(pipe):
    import cv2
    import os

    data = b""
    while True:
        chunk = os.read(pipe, 65536)
        if not chunk:
            return None
        data += chunk

        start = data.find(b'\xff\xd8')
        if start >= 0:
            end = data.find(b'\xff\xd9', start)
            if end >= 0:
                jpeg = data[start:end+2]
                data = data[end+2:]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if frame is not None and frame.shape == (400, 640):
                    return frame
    return None

print("\nHand Gesture Mouse - OV9281 + Picamera2")
print("  OPEN PALM (2+ fingers) : Move cursor")
print("  FIST (0 fingers)       : Drag")
print("  PINCH (thumb+index)    : Click")
print("  PARTIAL                 : Transition (ignored)")
print("Press 'q' to quit\n")

rpicam_proc = init_capture()
frame_queue = queue.Queue(maxsize=2)

reader = threading.Thread(target=frame_reader_thread, args=(rpicam_proc, frame_queue), daemon=True)
reader.start()

print("Frame reader started, waiting for frames...")

try:
    while True:
        try:
            gray = frame_queue.get(timeout=3)
        except queue.Empty:
            continue
        
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = hands.process(img_rgb)
        img_rgb.flags.writeable = True

        gesture_text = "NO HAND"
        color = (80, 80, 80)

        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
            lm = hl.landmark

            raw = detect_gesture(lm)
            fingers_ext, fingers = count_extended_fingers(lm)
            stable = get_stable_gesture(raw)

            palm_x, palm_y = get_palm_center(lm)
            smooth_x = smooth_x + STICK_SMOOTH * (palm_x - smooth_x)
            smooth_y = smooth_y + STICK_SMOOTH * (palm_y - smooth_y)
            vx, vy = stick_velocity(smooth_x, smooth_y, centre_x, centre_y)
            cur_x = np.clip(cur_x + vx, 0, SCREEN_W - 1)
            cur_y = np.clip(cur_y + vy, 0, SCREEN_H - 1)

            if stable == "PINCH":
                if pinch_released:
                    now = time.monotonic()
                    if now - last_click_time > CLICK_COOLDOWN:
                        enqueue(('click',))
                        last_click_time = now
                        print(f"CLICK at ({int(cur_x)}, {int(cur_y)})")
                    pinch_active = True
                    pinch_released = False
                if is_dragging:
                    enqueue(('up',))
                    is_dragging = False
                gesture_text = "CLICK"
                color = (0, 255, 80)

            elif stable == "FIST":
                if not is_dragging:
                    is_dragging = True
                    enqueue(('down',))
                    print(f"DRAG START at ({int(cur_x)}, {int(cur_y)})")
                enqueue(('move', int(cur_x), int(cur_y)))
                if pinch_active:
                    pinch_released = True
                    pinch_active = False
                gesture_text = "DRAG"
                color = (255, 0, 200)

            elif stable == "OPEN":
                if is_dragging:
                    enqueue(('up',))
                    is_dragging = False
                enqueue(('move', int(cur_x), int(cur_y)))
                if pinch_active and dist(lm[4], lm[8]) > PINCH_RELEASE:
                    pinch_released = True
                    pinch_active = False
                gesture_text = "MOVE"
                color = (0, 220, 255)

            else:
                if is_dragging:
                    enqueue(('move', int(cur_x), int(cur_y)))
                    gesture_text = "DRAG (HOLD)"
                    color = (255, 100, 150)
                else:
                    gesture_text = "TRANSITION"
                    color = (180, 180, 180)

            h, w = img.shape[:2]
            cx_px = int(centre_x * w)
            cy_px = int(centre_y * h)
            hx_px = int(smooth_x * w)
            hy_px = int(smooth_y * h)
            cv2.circle(img, (cx_px, cy_px), int(DEAD_ZONE * w), (80, 80, 80), 1)
            cv2.circle(img, (cx_px, cy_px), int(MAX_ZONE * w), (120, 120, 120), 1)
            cv2.line(img, (cx_px, cy_px), (hx_px, hy_px), (0, 220, 255), 2)
            cv2.circle(img, (hx_px, hy_px), 8, color, -1)

            t_px = (int(lm[4].x * w), int(lm[4].y * h))
            i_px = (int(lm[8].x * w), int(lm[8].y * h))
            mid_px = ((t_px[0]+i_px[0])//2, (t_px[1]+i_px[1])//2)
            pinch_dist = dist(lm[4], lm[8])
            is_pinching = pinch_dist < PINCH_THRESHOLD
            dot_col = (0, 255, 80) if is_pinching else (200, 200, 200)
            cv2.circle(img, t_px, 10, dot_col, -1)
            cv2.circle(img, i_px, 10, dot_col, -1)
            cv2.line(img, t_px, i_px, dot_col, 2)
            cv2.circle(img, mid_px, 5, dot_col, -1)

            cv2.putText(img, f"Pinch: {pinch_dist:.3f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            finger_list = [k[0].upper() for k, v in fingers.items() if v]
            cv2.putText(img, f"Fingers: {fingers_ext} [{','.join(finger_list)}]",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(img, f"Raw: {raw} -> Stable: {stable}",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        else:
            if is_dragging:
                enqueue(('up',))
                is_dragging = False
            pinch_active = False
            pinch_released = True
            gesture_history.clear()
            current_stable_gesture = "NONE"

        cv2.putText(img, f"Gesture: {gesture_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Cursor: {int(cur_x)},{int(cur_y)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if is_dragging:
            cv2.putText(img, "DRAGGING",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 200), 2)

        cv2.imshow("Hand Gesture Controller [rpicam]", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if is_dragging:
        mouse_up()
    mouse_queue.put(None)
    mouse_thread.join(timeout=1)
    if USE_UINPUT:
        cap_evdev.close()

    rpicam_proc.terminate()
    rpicam_proc.wait(timeout=2)

    cv2.destroyAllWindows()
    print("Exited cleanly.")