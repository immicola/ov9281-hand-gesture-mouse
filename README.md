# Pi Gesture Cam

Распознавание жестов рук в реальном времени на Raspberry Pi 4/5 + OV9281.

## Возможности (3 режима)

| Режим | Скрипт | Назначение |
|---|---|---|
| **TFLite Cam** | `pi_gesture_cam.py` | MediaPipe Hands → TFLite модель → вывод на экран |
| **Mouse HID** | `hand_gesture_mouse_v2.py` | Жесты → виртуальная мышь через `/dev/uinput` |
| **Robo Arm** | `robo_arm_v2.py` | Жесты → UART (/dev/serial0) → Raspberry Pi Pico |

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

### 2. Права для виртуальной мыши (только для Mouse HID)

```bash
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload && sudo udevadm trigger
sudo usermod -aG input $USER
```

**Важно:** перелогиниться (`logout → login` или `reboot`), чтобы группа `input` применилась.

### 3. Виртуальное окружение (Python 3.11)

MediaPipe не поддерживает Python 3.13. Используй **3.11 или 3.12**.

```bash
python3 -m venv ~/arducam/11venv
source ~/arducam/11venv/bin/activate
pip install --upgrade pip
```

### 4. Установка зависимостей

**Совместимая связка** (opencv-contrib + numpy 1.26 + mediapipe):

```bash
pip install opencv-contrib-python==4.10.0.84 numpy==1.26.4 mediapipe==0.10.18 tflite-runtime pyserial evdev
```

**Проверенные версии (11venv):**

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

## Жесты

### TFLite / Teacher MediaPipe (8 классов)

| Класс | Описание |
|---|---|
| `None` | Неопределён / шум |
| `Closed_Fist` | Кулак |
| `Open_Palm` | Раскрытая ладонь |
| `Pointing_Up` | Указательный палец вверх |
| `Thumb_Down` | Большой палец вниз |
| `Thumb_Up` | Большой палец вверх |
| `Victory` | Мир (✌️) |
| `ILoveYou` | Рок-рожа (🤘) |

### Mouse HID (3 основных жеста)

| Жест | Действие |
|---|---|
| OPEN PALM (3+ пальца) | Движение курсора (джойстик от центра ладони) |
| FIST (0-1 палец) | Drag (перетаскивание) |
| PINCH (большой + указательный сведены) | Левый клик |

### Robo Arm UART (6 жестов)

`FIST`, `ONE`, `TWO`, `THREE`, `FOUR`, `FIVE` — отсылаются через UART на Pico.

## Файлы проекта

| Файл | Назначение |
|---|---|
| `pi_gesture_cam.py` | TFLite инференс + вывод на экран |
| `hand_gesture_mouse_v2.py` | Жесты → виртуальная мышь (uinput) |
| `robo_arm_v2.py` | Жесты → UART → Pico |
| `student_model_7kb.tflite` | TFLite модель (8 классов) |
| `dataset.json` | Ключевые точки MediaPipe (21 × 2) |
| `distillation_dataset.csv` | Датасет для дистилляции (42 координаты + 8 вероятностей) |
| `teacher_labeling.py` | Сбор soft labels через GestureRecognizer |
| `train_student.py` | Обучение студента (KL Divergence) |
| `mediapipe_keypoints.py` | Извлечение 42 keypoints из видео |

## Запуск

```bash
source ~/arducam/11venv/bin/activate
cd ~/arducam

# TFLite Cam
python pi_gesture_cam.py

# Mouse HID
python hand_gesture_mouse_v2.py

# Robo Arm
python robo_arm_v2.py
```

`q` — выход.

## Проверки

```bash
# Камера
rpicam-hello -t 3000

# Устройства видео
ls -la /dev/video*

# Права uinput
ls -la /dev/uinput
# Должно быть: crw-rw---- 1 root input ...

# Группа input
groups $USER | grep input

# Нет ли процессов, занявших камеру
ps aux | grep -E "(rpicam|python|libcamera)" | grep -v grep
```

## Проблемы

| Симптом | Решение |
|---|---|
| `cv2.normalize not found` | Обновить opencv или заменить на `cv2.convertScaleAbs` |
| `ImportError: No module named 'mediapipe'` | Активировать venv: `source 11venv/bin/activate` |
| `Device or resource busy` | `pkill -f rpicam; pkill -f python` |
| Курсор не двигается | Проверить права `/dev/uinput`, перелогиниться |
| Камера не определяется | `rpicam-hello -t 3000`, проверить шлейф OV9281 |
| numpy 2 конфликтует | `pip install numpy==1.26.4` |
| `mediapipe requires opencv-contrib-python` | `pip install opencv-contrib-python==4.10.0.84` |

## Лицензия

MIT License
