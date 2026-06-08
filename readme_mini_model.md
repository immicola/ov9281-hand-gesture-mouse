# Pi Gesture Cam — Mini Model

Распознавание жестов рук в реальном времени на Raspberry Pi 4/5 + OV9281.

Два подхода:
1. **TFLite (distilled)** — `pi_gesture_cam.py` — MediaPipe Hands → маленькая TFLite модель (7 KB), обученная через knowledge distillation
2. **Rule-based (finger counting)** — `robo_gesture_cam.py` / `robo_arm_v2.py` — MediaPipe Hands → геометрический подсчёт пальцев → UART

---

## 1. TFLite Distilled Model (`pi_gesture_cam.py`)

Маленькая нейронка (7 KB, student_model_7kb.tflite), обучена через knowledge distillation из MediaPipe GestureRecognizer.

### Жесты (8 классов)

| Класс | Описание |
|---|---|
| `thumbs_up` | Большой палец вверх |
| `pistol` | Пистолетик (большой + указательный) |
| `pointy` | Указательный палец вверх |
| `peace` | Мир (✌️) |
| `3fingers` | Три пальца |
| `4fingers` | Четыре пальца |
| `open` | Раскрытая ладонь |
| `close` | Кулак |

### Обучение

| Скрипт | Назначение |
|---|---|
| `teacher_labeling.py` | Сбор soft labels через MediaPipe GestureRecognizer |
| `build_dataset.py` / `build_dataset_rpi.py` | Сбор датасета (21 landmark → 42 координаты) |
| `distillation_dataset.csv` | Датасет: 42 координаты + 8 вероятностей от teacher |
| `train_student.py` | Обучение студента (KL Divergence) |
| `student_model_7kb.tflite` | Готовая TFLite модель float32 (8 классов, 7 KB) |

---

## 2. Robo Gesture Cam (`robo_gesture_cam.py`)

Rule-based finger counting для управления роботом через UART на Pico.  
MediaPipe Hands → геометрический подсчёт пальцев → `GEST:<name>\n` + `X:<value>\n` → UART → Pico bridge → USB → PC (ROS2).

### Жесты (6 классов + NO HAND / UNKNOWN)

| Жест | Пальцы | UART команда | Действие |
|---|---|---|---|
| `FIST` | 0 | `GEST:FIST` | GRIPPER_CLOSE |
| `ONE` | 1 (указательный) | `GEST:ONE` | IDLE / point |
| `TWO` | 2 (указ. + сред.) | `GEST:TWO` | J2 shoulder (джойстик по X) |
| `THREE` | 3 (указ. + сред. + безым.) | `GEST:THREE` | J1 rotation (джойстик по X) |
| `FOUR` | 4 (без большого) | `GEST:FOUR` | J3 elbow (джойстик по X) |
| `FIVE` | 5 | `GEST:FIVE` | GRIPPER_OPEN |

### Robo Arm v2 (`robo_arm_v2.py`)

Старая версия того же подхода. Те же жесты (FIST — FIVE), тот же UART-протокол.

---

## Системные требования

- Raspberry Pi 4 или 5 (aarch64)
- Debian Bookworm / Trixie
- Камера OV9281 (или любая через rpicam-vid)
- rpicam-vid (libcamera-apps)

## Установка

### 1. Подготовка системы

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-evdev python3-venv libopencv-dev
```

### 2. Виртуальное окружение (Python 3.11)

Проверенная конфигурация (всё работает вместе):

```bash
python3 --version
# Python 3.11.15

source ~/arducam/11venv/bin/activate
pip install --upgrade pip
```

### 3. Установка зависимостей

**Совместимая связка:**

```bash
pip install opencv-contrib-python==4.10.0.84 numpy==1.26.4 mediapipe==0.10.18 tflite-runtime pyserial evdev
```

**Установленные версии:**

```
Python:  3.11.15
mediapipe             0.10.18
numpy                 1.26.4
opencv-contrib-python 4.10.0.84
pyserial              3.5
tflite-runtime        2.14.0
```

**Почему так:**
- mediapipe 0.10.18 **требует** `numpy<2` и `opencv-contrib-python`
- opencv-contrib-python 4.10 тянет numpy 1.26 — идеально
- opencv-python 4.13 тянет numpy 2 — **несовместим** с mediapipe

---

## Файлы проекта

| Файл | Назначение |
|---|---|
| `pi_gesture_cam.py` | TFLite инференс (distilled student_model_7kb) |
| `robo_gesture_cam.py` | Rule-based finger counting → UART → Pico → ROS2 |
| `robo_arm_v2.py` | Rule-based → UART (старая версия) |
| `hand_gesture_mouse_v2.py` | Жесты → виртуальная мышь (uinput) |
| `student_model_7kb.tflite` | TFLite модель float32 (7 KB, 8 классов) |
| `gesture_labels.json` | Метки классов для датасета |
| `build_dataset.py` | Сбор датасета (ключевые точки) |
| `build_dataset_rpi.py` | Сбор датасета на RPi |
| `distillation_dataset.csv` | Датасет: 42 координаты + 8 вероятностей |
| `train_student.py` | Обучение студента (KL Divergence) |
| `teacher_labeling.py` | Сбор soft labels через GestureRecognizer |
| `mediapipe_keypoints.py` | Извлечение 42 keypoints из видео |
| `dataset.json` | Ключевые точки MediaPipe (21 × 2) |
| `run_locate.py` | Локализация / отладка |

---

## Запуск

```bash
source ~/arducam/11venv/bin/activate
cd ~/arducam

# TFLite Distilled Model
python pi_gesture_cam.py

# Robo Gesture Cam (с экраном)
python robo_gesture_cam.py

# Robo Gesture Cam (без экрана, только UART)
python robo_gesture_cam.py --headless

# Robo Arm v2 (старая версия)
python robo_arm_v2.py

# Mouse HID
python hand_gesture_mouse_v2.py
```

`q` — выход.

---

## UART-протокол

Текстовый, newline-delimited, 115200 бод:

| Формат | Пример | Описание |
|---|---|---|
| `GEST:<name>\n` | `GEST:THREE\n` | Выбрать сустав или управлять схватом |
| `X:<value>\n` | `X:0.723\n` | Горизонтальная позиция ладони (0.0–1.0) |

Pico работает как dumb bridge: UART (GPIO 0/1) ↔ USB CDC, без логики.

---

## Проверки

```bash
# Камера
rpicam-hello -t 3000

# Устройства видео
ls -la /dev/video*

# Группа input (для uinput)
groups $USER | grep input

# Нет ли процессов, занявших камеру
ps aux | grep -E "(rpicam|python|libcamera)" | grep -v grep
```

## Проблемы

| Симптом | Решение |
|---|---|
| `ImportError: No module named 'mediapipe'` | Активировать venv: `source 11venv/bin/activate` |
| `Device or resource busy` | `pkill -f rpicam; pkill -f python` |
| Камера не определяется | `rpicam-hello -t 3000`, проверить шлейф OV9281 |
| numpy 2 конфликтует | `pip install numpy==1.26.4` |
| `mediapipe requires opencv-contrib-python` | `pip install opencv-contrib-python==4.10.0.84` |

## Лицензия

MIT License
