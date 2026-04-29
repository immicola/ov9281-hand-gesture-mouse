"""
Hand Gesture Mouse Controller
Raspberry Pi 4 · OV9281 (rpicam-vid YUV420) → Raspberry Pi Pico (USB HID Mouse)

Жесты:
  OPEN PALM (2+ пальца)  → движение курсора
  FIST (0 пальцев)       → drag
  PINCH (большой+указат) → клик (курсор ЗАМОРОЖЕН во время pinch)
  PARTIAL                → игнорируется

Параметры для подстройки:
  SENSITIVITY_X/Y  — скорость курсора (больше = быстрее)
  SMOOTH           — сглаживание (больше = плавнее, но медленнее отклик)
  MIN_MOVE_PX      — минимальный сдвиг в пикселях (порог дрожи)
  PINCH_FREEZE_MS  — сколько мс держать курсор замороженным после pinch
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import math
import subprocess
import serial
from collections import deque

# ── UART -> Pico ───────────────────────────────────────────────────────────────
try:
    pico_serial = serial.Serial('/dev/serial0', 115200, timeout=0.01)
    print("UART: связь с Pico установлена!")
except Exception as ex:
    print(f"Ошибка UART: {ex}")
    pico_serial = None

# ── Mouse actions ─────────────────────────────────────────────────────────────
def _send(data: bytes):
    if pico_serial:
        try:
            pico_serial.write(data)
        except Exception:
            pass

def mouse_move_rel(dx: int, dy: int):
    if dx == 0 and dy == 0:
        return
    while abs(dx) > 0 or abs(dy) > 0:
        sx = max(-127, min(127, dx))
        sy = max(-127, min(127, dy))
        _send(f"M,{sx},{sy}\n".encode())
        dx -= sx
        dy -= sy

def mouse_click(): _send(b"CLICK\n")
def mouse_down():  _send(b"DOWN\n")
def mouse_up():    _send(b"UP\n")

# ── Async worker ──────────────────────────────────────────────────────────────
_mq = queue.Queue(maxsize=6)

def _mouse_worker():
    while True:
        cmd = _mq.get()
        if cmd is None:
            break
        t = cmd[0]
        if   t == 'rel':   mouse_move_rel(cmd[1], cmd[2])
        elif t == 'click': mouse_click()
        elif t == 'down':  mouse_down()
        elif t == 'up':    mouse_up()

threading.Thread(target=_mouse_worker, daemon=True).start()

def enqueue(cmd):
    try:
        _mq.put_nowait(cmd)
    except queue.Full:
        if cmd[0] == 'rel':
            pass
        else:
            try: _mq.get_nowait()
            except: pass
            _mq.put_nowait(cmd)

# ── Конфиг ────────────────────────────────────────────────────────────────────
CAPTURE_W = 640
CAPTURE_H = 480

SENSITIVITY_X  = 2000   # скорость по X (увеличь если медленно)
SENSITIVITY_Y  = 2000   # скорость по Y
SMOOTH         = 0.7    # сглаживание позиции ладони (0..1, больше = плавнее)
MIN_MOVE_PX    = 2      # минимальный сдвиг в пикселях — всё меньше = дрожь, игнорируем

# Заморозка курсора во время pinch (мс) — курсор не двигается пока идёт клик
PINCH_FREEZE_MS = 300

PINCH_THRESH   = 0.045
PINCH_RELEASE  = 0.08
CLICK_COOLDOWN = 0.4
GESTURE_FRAMES = 3      # увеличено с 2 до 3 — жест должен держаться 3 кадра
FIST_MAX       = 0
OPEN_MIN       = 2

MIN_DETECT = 0.65
MIN_TRACK  = 0.45

# ── Камера ────────────────────────────────────────────────────────────────────
print("Запуск rpicam-vid (YUV420)...")
_frame_size = (CAPTURE_W * CAPTURE_H * 3) // 2
_proc = subprocess.Popen([
    "rpicam-vid",
    "--width",     str(CAPTURE_W),
    "--height",    str(CAPTURE_H),
    "--framerate", "30",
    "--nopreview",
    "--codec",     "yuv420",
    "-t",          "0",
    "--output",    "-"
], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
print(f"rpicam-vid запущен ({CAPTURE_W}x{CAPTURE_H})")

_fq = queue.Queue(maxsize=2)

def _reader():
    while _proc.poll() is None:
        try:
            buf = b""
            while len(buf) < _frame_size:
                chunk = _proc.stdout.read(_frame_size - len(buf))
                if not chunk:
                    return
                buf += chunk
            arr  = np.frombuffer(buf, dtype=np.uint8)
            gray = arr[:CAPTURE_W * CAPTURE_H].reshape(CAPTURE_H, CAPTURE_W)
            try:
                _fq.put_nowait(gray)
            except queue.Full:
                try: _fq.get_nowait()
                except: pass
                _fq.put_nowait(gray)
        except Exception as e:
            print(f"Frame reader error: {e}")
            break
    try: _proc.stdout.close()
    except: pass

threading.Thread(target=_reader, daemon=True).start()
print("Ждём первый кадр...")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
_mp   = mp.solutions.hands
hands = _mp.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=MIN_DETECT,
    min_tracking_confidence=MIN_TRACK,
)
_draw = mp.solutions.drawing_utils

# ── Состояние ─────────────────────────────────────────────────────────────────
smooth_px: float = 0.5
smooth_py: float = 0.5
prev_px = None
prev_py = None

is_dragging    = False
last_click_t   = 0.0
pinch_active   = False
pinch_released = True
frozen_until   = 0.0   # время до которого курсор заморожен (монотонное время)

gest_hist   = deque(maxlen=GESTURE_FRAMES)
stable_gest = "NONE"

# ── Вспомогательные функции ───────────────────────────────────────────────────
def _dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def count_fingers(lm):
    w   = lm[0]
    ext = {'T': False, 'I': False, 'M': False, 'R': False, 'P': False}
    for tip, pip, letter in [(8,6,'I'),(12,10,'M'),(16,14,'R'),(20,18,'P')]:
        if lm[tip].y < lm[pip].y and _dist(lm[tip], w) > _dist(lm[pip], w) * 1.1:
            ext[letter] = True
    pcx = (lm[0].x + lm[5].x + lm[9].x) / 3
    if abs(lm[4].x - pcx) > 0.08 and _dist(lm[4], w) > _dist(lm[2], w) * 1.1:
        ext['T'] = True
    return sum(ext.values()), ext

def palm_center(lm):
    idx = (0, 1, 5, 9, 13, 17)
    return (sum(lm[i].x for i in idx) / 6,
            sum(lm[i].y for i in idx) / 6)

def detect_gesture(lm):
    n, _ = count_fingers(lm)
    pd   = _dist(lm[4], lm[8])
    if pd < PINCH_THRESH: return "PINCH"
    if n  <= FIST_MAX:    return "FIST"
    if n  >= OPEN_MIN:    return "OPEN"
    return "PARTIAL"

def get_stable(raw):
    global gest_hist, stable_gest
    gest_hist.append(raw)
    if len(gest_hist) >= GESTURE_FRAMES and all(g == raw for g in gest_hist):
        stable_gest = raw
    return stable_gest

# ── Главный цикл ──────────────────────────────────────────────────────────────
print("\nГотово! Жесты активны. 'q' — выход.\n")
fc = 0

while True:
    try:
        gray = _fq.get(timeout=5)
        fc  += 1
        if fc == 1:
            print("Первый кадр получен!")
    except queue.Empty:
        print("Таймаут кадра от rpicam-vid")
        continue

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img  = cv2.flip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res  = hands.process(rgb)
    rgb.flags.writeable = True

    lbl   = "NO HAND"
    color = (80, 80, 80)
    now   = time.monotonic()

    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        _draw.draw_landmarks(img, hl, _mp.HAND_CONNECTIONS)
        lm = hl.landmark

        raw = detect_gesture(lm)
        sg  = get_stable(raw)
        n, ext_map = count_fingers(lm)

        # ── Позиция ладони → относительный сдвиг ─────────────────────────
        px, py = palm_center(lm)
        smooth_px = smooth_px * SMOOTH + px * (1 - SMOOTH)
        smooth_py = smooth_py * SMOOTH + py * (1 - SMOOTH)

        if prev_px is not None:
            dx = int((smooth_px - prev_px) * SENSITIVITY_X)
            dy = int((smooth_py - prev_py) * SENSITIVITY_Y)
        else:
            dx, dy = 0, 0

        prev_px, prev_py = smooth_px, smooth_py

        # Порог минимального движения — убирает дрожь
        if abs(dx) < MIN_MOVE_PX: dx = 0
        if abs(dy) < MIN_MOVE_PX: dy = 0

        # Курсор заморожен? (после pinch)
        cursor_frozen = now < frozen_until

        # ── Жестовый автомат ─────────────────────────────────────────────
        if sg == "PINCH":
            if pinch_released:
                if now - last_click_t > CLICK_COOLDOWN:
                    enqueue(('click',))
                    last_click_t  = now
                    frozen_until  = now + PINCH_FREEZE_MS / 1000.0  # замораживаем курсор
                    print("CLICK")
                pinch_active   = True
                pinch_released = False
            if is_dragging:
                enqueue(('up',))
                is_dragging = False
            lbl   = "CLICK"
            color = (0, 255, 80)

        elif sg == "FIST":
            if not is_dragging:
                enqueue(('down',))
                is_dragging = True
                print("DRAG START")
            if not cursor_frozen and (dx or dy):
                enqueue(('rel', dx, dy))
            if pinch_active:
                pinch_released = True
                pinch_active   = False
            lbl   = "DRAG"
            color = (255, 0, 200)

        elif sg == "OPEN":
            if is_dragging:
                enqueue(('up',))
                is_dragging = False
                print("DRAG END")
            if not cursor_frozen and (dx or dy):
                enqueue(('rel', dx, dy))
            if pinch_active and _dist(lm[4], lm[8]) > PINCH_RELEASE:
                pinch_released = True
                pinch_active   = False
            lbl   = "MOVE"
            color = (0, 220, 255)

        else:  # PARTIAL
            if is_dragging and not cursor_frozen and (dx or dy):
                enqueue(('rel', dx, dy))
            lbl   = "DRAG(HOLD)" if is_dragging else "PARTIAL"
            color = (255, 100, 150) if is_dragging else (180, 180, 180)

        # ── HUD ───────────────────────────────────────────────────────────
        h, w  = img.shape[:2]
        sp_px = (int(smooth_px * w), int(smooth_py * h))
        cv2.circle(img, sp_px, 12, color, -1)

        # Индикатор заморозки
        if cursor_frozen:
            cv2.putText(img, "FROZEN", (int(smooth_px*w)+15, int(smooth_py*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255), 2)

        t_px = (int(lm[4].x*w), int(lm[4].y*h))
        i_px = (int(lm[8].x*w), int(lm[8].y*h))
        pd   = _dist(lm[4], lm[8])
        dc   = (0,255,80) if pd < PINCH_THRESH else (200,200,200)
        cv2.circle(img, t_px, 10, dc, -1)
        cv2.circle(img, i_px, 10, dc, -1)
        cv2.line  (img, t_px, i_px, dc, 2)
        cv2.putText(img, f"Pinch:{pd:.3f}",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        fl = [k for k,v in ext_map.items() if v]
        cv2.putText(img, f"Fingers:{n} [{','.join(fl)}]",
                    (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(img, f"Raw:{raw} Stable:{sg}  dx={dx:+d} dy={dy:+d}",
                    (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)

    else:
        if is_dragging:
            enqueue(('up',))
            is_dragging = False
        pinch_active   = False
        pinch_released = True
        gest_hist.clear()
        stable_gest = "NONE"
        prev_px = prev_py = None

    cv2.putText(img, f"Gesture: {lbl}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if is_dragging:
        cv2.putText(img, "DRAGGING", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,200), 2)

    cv2.imshow("Hand Gesture -> Pico HID", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
if is_dragging:
    mouse_up()
_mq.put(None)
_proc.terminate()
if pico_serial:
    pico_serial.close()
cv2.destroyAllWindows()
print("Выход.")