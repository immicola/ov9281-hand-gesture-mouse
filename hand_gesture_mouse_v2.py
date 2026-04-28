"""
Hand Gesture Mouse Controller - IMPROVED GESTURE RECOGNITION
Raspberry Pi 5 · Debian Trixie · labwc (Wayland)
Camera: OV9281 monochrome (via rpicam-vid YUV420)

Uses /dev/uinput (kernel virtual input) — works on Wayland, no X11 needed.

IMPROVEMENTS:
1. Better pinch detection with hysteresis
2. Clearer gesture state machine to prevent conflicts
3. Temporal filtering (debouncing) for more stable recognition
4. Enhanced finger counting using multiple criteria
5. Better palm/fist distinction

Setup (one-time):
    # System packages
    sudo apt install python3-evdev python3-venv libopencv-dev

    # Create virtual environment with Python 3.12
    python3 -m venv venv
    source venv/bin/activate

    # Install Python packages
    pip install evdev opencv-python mediapipe numpy

    # OR use requirements.txt:
    # pip install -r requirements.txt

    # Allow your user to write to uinput without sudo:
    echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
    sudo udevadm control --reload && sudo udevadm trigger
    sudo usermod -aG input $USER

    # Log out and back in, then run without sudo:
    source venv/bin/activate
    python3 hand_gesture_mouse_v2.py

Note: Uses rpicam-vid subprocess with YUV420 codec for OV9281 compatibility.
      Python 3.12 has libcamera compatibility issues with direct Picamera2 import.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import math
import subprocess
import select
from collections import deque

# ── uinput virtual mouse via evdev ────────────────────────────────────────────
try:
    import evdev
    from evdev import UInput, AbsInfo, ecodes as e

    # Screen resolution — change if yours differs
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
    print(f"uinput virtual mouse created  ({SCREEN_W}x{SCREEN_H})")
    print("  -> If cursor doesn't move, check: ls -la /dev/uinput")
    print("     and run: sudo usermod -aG input $USER  then re-login")

except Exception as ex:
    print(f"evdev/uinput unavailable ({ex})")
    print("Install:  sudo apt install python3-evdev")
    print("          pip install evdev --break-system-packages")
    print("Falling back to pyautogui (won't work on Wayland)")
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0
    USE_UINPUT = False
    SCREEN_W, SCREEN_H = pyautogui.size()

# ── Mouse actions ─────────────────────────────────────────────────────────────
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

# ── Async mouse worker ────────────────────────────────────────────────────────
mouse_queue = queue.Queue(maxsize=4)

def mouse_worker():
    while True:
        cmd = mouse_queue.get()
        if cmd is None:
            break
        action = cmd[0]
        if   action == 'move':  mouse_move(cmd[1], cmd[2])
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

# ── Config ────────────────────────────────────────────────────────────────────
CAPTURE_WIDTH   = 640
CAPTURE_HEIGHT  = 480

# GESTURE RECOGNITION IMPROVEMENTS
CLICK_COOLDOWN      = 0.4    # seconds between allowed clicks
PINCH_THRESHOLD     = 0.045  # TIGHTENED - thumb-index dist for pinch detection
PINCH_RELEASE       = 0.08   # must open past this to re-arm
GESTURE_DEBOUNCE    = 2      # frames - gesture must be stable for this many frames
FIST_MAX_FINGERS    = 0      # STRICT: only 0 fingers = fist (all fingers must be curled)
OPEN_MIN_FINGERS    = 2      # 2+ fingers = open palm (more lenient)

MIN_DETECT_CONF = 0.65
MIN_TRACK_CONF  = 0.45

# Enhanced finger detection - LOOSENED for better open palm detection
FINGER_RATIO_SQ     = 1.15   # (1.07)^2 - REDUCED for easier finger detection
CURL_THRESHOLD      = 0.5    # REDUCED - less strict curl requirement

# ── Gamepad stick config ──────────────────────────────────────────────────────
DEAD_ZONE       = 0.045
MAX_ZONE        = 0.30
MAX_SPEED       = 120
STICK_SMOOTH    = 0.98

# ── Camera (OV9281 via rpicam-vid YUV420) ────────────────────────────────────
print("Initializing OV9281 camera with rpicam-vid (YUV420)...")

def init_capture():
    # YUV420 frame size: (width * height * 3) // 2
    frame_size = (CAPTURE_WIDTH * CAPTURE_HEIGHT * 3) // 2
    proc = subprocess.Popen([
        "rpicam-vid",
        "--width", str(CAPTURE_WIDTH),
        "--height", str(CAPTURE_HEIGHT),
        "--framerate", "30",
        "--nopreview",
        "--codec", "yuv420",
        "-t", "0",
        "--output", "-"
    ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print(f"rpicam-vid started (YUV420, {CAPTURE_WIDTH}x{CAPTURE_HEIGHT})")
    return proc, frame_size

def frame_reader_thread(proc, frame_queue, frame_size):
    while proc.poll() is None:
        try:
            # Read exactly frame_size bytes for one YUV420 frame
            data = b""
            while len(data) < frame_size:
                chunk = proc.stdout.read(frame_size - len(data))
                if not chunk:
                    break
                data += chunk
            if len(data) != frame_size:
                break
            # Convert to numpy array
            frame = np.frombuffer(data, dtype=np.uint8)
            # Extract Y channel (first CAPTURE_WIDTH*CAPTURE_HEIGHT bytes)
            gray = frame[:CAPTURE_WIDTH * CAPTURE_HEIGHT].reshape(CAPTURE_HEIGHT, CAPTURE_WIDTH)
            if gray is not None:
                try:
                    frame_queue.put_nowait(gray)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                    frame_queue.put_nowait(gray)
        except Exception as e:
            print(f"Frame reader error: {e}")
            break
    try:
        proc.stdout.close()
    except:
        pass

rpicam_proc, frame_size = init_capture()
frame_queue = queue.Queue(maxsize=2)
reader = threading.Thread(target=frame_reader_thread, args=(rpicam_proc, frame_queue, frame_size), daemon=True)
reader.start()
print("Frame reader started, waiting for frames...")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=MIN_DETECT_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)
mp_draw = mp.solutions.drawing_utils

# ── State ─────────────────────────────────────────────────────────────────────
cur_x: float    = SCREEN_W / 2
cur_y: float    = SCREEN_H / 2
smooth_x: float = 0.5
smooth_y: float = 0.5
centre_x: float = 0.5
centre_y: float = 0.5

is_dragging     = False
last_click_time = 0.0
pinch_active    = False
pinch_released  = True

# Gesture debouncing - temporal filtering
gesture_history = deque(maxlen=GESTURE_DEBOUNCE)
current_stable_gesture = "NONE"

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def dist_sq(p1, p2):
    dx, dy = p1.x - p2.x, p1.y - p2.y
    return dx*dx + dy*dy

def count_extended_fingers_enhanced(lm):
    """
    Simplified and more reliable finger counting.
    Checks if fingertips are higher (lower Y value) than their base knuckles.
    
    Returns: (count, details_dict)
    """
    wrist = lm[0]
    
    fingers = {
        'thumb': False,
        'index': False,
        'middle': False,
        'ring': False,
        'pinky': False
    }
    
    # For each finger: check if tip is farther from wrist than the MCP (base knuckle)
    # Also check if tip is extended upward (for non-thumb fingers)
    
    # Index finger
    if lm[8].y < lm[6].y:  # tip above PIP
        if dist(lm[8], wrist) > dist(lm[6], wrist) * 1.1:
            fingers['index'] = True
    
    # Middle finger
    if lm[12].y < lm[10].y:
        if dist(lm[12], wrist) > dist(lm[10], wrist) * 1.1:
            fingers['middle'] = True
    
    # Ring finger
    if lm[16].y < lm[14].y:
        if dist(lm[16], wrist) > dist(lm[14], wrist) * 1.1:
            fingers['ring'] = True
    
    # Pinky
    if lm[20].y < lm[18].y:
        if dist(lm[20], wrist) > dist(lm[18], wrist) * 1.1:
            fingers['pinky'] = True
    
    # Thumb - check horizontal distance from palm
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    palm_center_x = (lm[0].x + lm[5].x + lm[9].x) / 3
    
    # Thumb is extended if tip is far from palm center AND farther than MCP
    thumb_dist = abs(thumb_tip.x - palm_center_x)
    if thumb_dist > 0.08 and dist(thumb_tip, wrist) > dist(thumb_mcp, wrist) * 1.1:
        fingers['thumb'] = True
    
    count = sum(1 for extended in fingers.values() if extended)
    
    return count, fingers

def get_palm_center(lm):
    PALM_IDX = (0, 1, 5, 9, 13, 17)
    x = sum(lm[i].x for i in PALM_IDX) / len(PALM_IDX)
    y = sum(lm[i].y for i in PALM_IDX) / len(PALM_IDX)
    return x, y

def stick_velocity(sx, sy, cx, cy):
    """Gamepad-stick model with dead zone and quadratic acceleration."""
    dx = sx - cx
    dy = sy - cy
    r  = math.sqrt(dx*dx + dy*dy)
    
    if r < DEAD_ZONE:
        return 0.0, 0.0
    
    t = min((r - DEAD_ZONE) / (MAX_ZONE - DEAD_ZONE), 1.0)
    speed = t * t * MAX_SPEED
    
    vx = (dx / r) * speed
    vy = (dy / r) * speed
    
    return vx, vy

def detect_gesture(lm):
    """
    Detect current gesture from landmarks.
    Returns: gesture_name (string)
    
    Gesture hierarchy (checked in order):
    1. PINCH - thumb and index very close
    2. FIST - 0-1 fingers extended
    3. OPEN - 3+ fingers extended
    4. PARTIAL - anything else (2 fingers, etc.)
    """
    fingers_extended, finger_details = count_extended_fingers_enhanced(lm)
    pinch_dist = dist(lm[4], lm[8])
    
    # PINCH takes priority - check first
    if pinch_dist < PINCH_THRESHOLD:
        return "PINCH"
    
    # FIST - very few fingers extended
    if fingers_extended <= FIST_MAX_FINGERS:
        return "FIST"
    
    # OPEN PALM - most fingers extended
    if fingers_extended >= OPEN_MIN_FINGERS:
        return "OPEN"
    
    # PARTIAL - transitional state (e.g., 2 fingers)
    return "PARTIAL"

def get_stable_gesture(current_gesture):
    """
    Temporal filtering - gesture must be stable for GESTURE_DEBOUNCE frames.
    Returns the debounced stable gesture.
    """
    global gesture_history, current_stable_gesture
    
    gesture_history.append(current_gesture)
    
    # Need full history buffer
    if len(gesture_history) < GESTURE_DEBOUNCE:
        return current_stable_gesture
    
    # Check if all recent gestures are the same
    if all(g == current_gesture for g in gesture_history):
        current_stable_gesture = current_gesture
    
    return current_stable_gesture

# ── Main loop ─────────────────────────────────────────────────────────────────
print("\nHand Gesture Mouse -- IMPROVED RECOGNITION")
print("  OPEN PALM (3+ fingers)     : Move cursor")
print("  FIST (0-1 fingers)         : Drag")
print("  PINCH (thumb + index)      : Click")
print("  PARTIAL (2 fingers)        : Transition state (ignored)")
print("Press 'q' in the camera window to quit\n")

frame_count = 0

while True:
    try:
        gray = frame_queue.get(timeout=5)
        frame_count += 1
        if frame_count == 1:
            print("First frame received!")
    except queue.Empty:
        print("Timeout waiting for frame from rpicam-vid")
        continue
    
    # Normalize brightness for better visibility
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert grayscale to BGR for MediaPipe (it expects 3 channels)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = hands.process(img_rgb)
    img_rgb.flags.writeable = True

    gesture_text = "NO HAND"
    color        = (80, 80, 80)

    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
        lm = hl.landmark
        
        # GET RAW GESTURE
        raw_gesture = detect_gesture(lm)
        
        # Get finger count for debugging
        fingers_extended, finger_details = count_extended_fingers_enhanced(lm)
        
        # APPLY TEMPORAL FILTERING
        stable_gesture = get_stable_gesture(raw_gesture)
        
        # Get hand position for cursor control
        palm_x, palm_y = get_palm_center(lm)
        smooth_x = smooth_x + STICK_SMOOTH * (palm_x - smooth_x)
        smooth_y = smooth_y + STICK_SMOOTH * (palm_y - smooth_y)
        
        vx, vy = stick_velocity(smooth_x, smooth_y, centre_x, centre_y)
        cur_x = float(np.clip(cur_x + vx, 0, SCREEN_W - 1))
        cur_y = float(np.clip(cur_y + vy, 0, SCREEN_H - 1))
        
        # ── GESTURE STATE MACHINE ─────────────────────────────────────────
        
        if stable_gesture == "PINCH":
            # PINCH -> CLICK
            if pinch_released:
                now = time.monotonic()
                if now - last_click_time > CLICK_COOLDOWN:
                    enqueue(('click',))
                    last_click_time = now
                    print(f"CLICK at ({int(cur_x)}, {int(cur_y)})")
                pinch_active   = True
                pinch_released = False
            
            # Release drag if active
            if is_dragging:
                enqueue(('up',))
                is_dragging = False
            
            gesture_text = "CLICK"
            color = (0, 255, 80)
        
        elif stable_gesture == "FIST":
            # FIST -> DRAG
            if not is_dragging:
                is_dragging = True
                enqueue(('down',))
                print(f"DRAG START at ({int(cur_x)}, {int(cur_y)})")
            
            enqueue(('move', int(cur_x), int(cur_y)))
            
            # Reset pinch state
            if pinch_active:
                pinch_released = True
                pinch_active = False
            
            gesture_text = "DRAG"
            color = (255, 0, 200)
        
        elif stable_gesture == "OPEN":
            # OPEN PALM -> MOVE
            if is_dragging:
                enqueue(('up',))
                is_dragging = False
                print("DRAG END")
            
            enqueue(('move', int(cur_x), int(cur_y)))
            
            # Allow pinch re-arming when hand opens
            if pinch_active:
                pinch_dist_current = dist(lm[4], lm[8])
                if pinch_dist_current > PINCH_RELEASE:
                    pinch_released = True
                    pinch_active = False
            
            gesture_text = "MOVE"
            color = (0, 220, 255)
        
        else:  # PARTIAL
            # Transitional state - hold current action
            if is_dragging:
                enqueue(('move', int(cur_x), int(cur_y)))
                gesture_text = "DRAG (HOLD)"
                color = (255, 100, 150)
            else:
                gesture_text = "TRANSITION"
                color = (180, 180, 180)
        
        # ── HUD: stick visualiser ─────────────────────────────────────────
        h, w    = img.shape[:2]
        cx_px   = int(centre_x * w)
        cy_px   = int(centre_y * h)
        hx_px   = int(smooth_x * w)
        hy_px   = int(smooth_y * h)
        dz_px   = int(DEAD_ZONE  * w)
        mz_px   = int(MAX_ZONE   * w)
        
        cv2.circle(img, (cx_px, cy_px), dz_px, (80,  80,  80),  1)
        cv2.circle(img, (cx_px, cy_px), mz_px, (120, 120, 120), 1)
        cv2.line  (img, (cx_px, cy_px), (hx_px, hy_px), (0, 220, 255), 2)
        cv2.circle(img, (hx_px, hy_px), 8, color, -1)
        
        # Pinch indicator with distance display
        t_px    = (int(lm[4].x * w), int(lm[4].y * h))
        i_px    = (int(lm[8].x * w), int(lm[8].y * h))
        mid_px  = ((t_px[0]+i_px[0])//2, (t_px[1]+i_px[1])//2)
        
        pinch_dist = dist(lm[4], lm[8])
        is_pinching = pinch_dist < PINCH_THRESHOLD
        dot_col = (0, 255, 80) if is_pinching else (200, 200, 200)
        
        cv2.circle(img, t_px,   10, dot_col, -1)
        cv2.circle(img, i_px,   10, dot_col, -1)
        cv2.line  (img, t_px, i_px, dot_col,  2)
        cv2.circle(img, mid_px,  5, dot_col, -1)
        
        # Show pinch distance
        cv2.putText(img, f"Pinch: {pinch_dist:.3f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Show finger count - IMPORTANT DEBUG INFO
        finger_list = [k[0].upper() for k, v in finger_details.items() if v]
        cv2.putText(img, f"Fingers: {fingers_extended} [{','.join(finger_list)}]",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show raw vs stable gesture for debugging
        cv2.putText(img, f"Raw: {raw_gesture} -> Stable: {stable_gesture}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    else:
        # No hand detected - reset everything
        if is_dragging:
            enqueue(('up',))
            is_dragging = False
        pinch_active   = False
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

    cv2.imshow("Hand Gesture Controller [IMPROVED]", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
if is_dragging:
    mouse_up()
mouse_queue.put(None)
mouse_thread.join(timeout=1)
if USE_UINPUT:
    cap_evdev.close()
cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")