# Hand Gesture Mouse Controller для OV9281

Управление мышью через жесты руки с использованием монохромной камеры OV9281 на Raspberry Pi.

## Особенности

- Распознавание жестов через MediaPipe
- Виртуальная мышь через `/dev/uinput` (работает на Wayland)
- Использование камеры OV9281 через `rpicam-vid` (YUV420)
- Улучшенная фильтрация жестов и предотвращение дребезга

## Жесты

| Жест | Действие |
|------|----------|
| OPEN PALM (3+ пальца) | Движение курсора |
| FIST (0-1 палец) | Drag (перетаскивание) |
| PINCH (большой + указательный) | Клик |
| PARTIAL (2 пальца) | Переходное состояние (игнорируется) |

## Системные требования

- **Raspberry Pi 4/5** с Debian Trixie/Bookworm
- **Python 3.12**
- **Камера OV9281** (монохромная)
- **Wayland** (labwc) — не требует X11

## Установка на новом Raspberry Pi

### 1. Подготовка системы

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка системных пакетов
sudo apt install -y python3-evdev python3-venv libopencv-dev

# Проверка что камера OV9281 определяется
rpicam-hello -t 3000
```

### 2. Создание виртуального окружения

```bash
# Создание директории проекта
mkdir -p ~/arducam
cd ~/arducam

# Создание виртуального окружения с Python 3.12
python3 -m venv venv

# Активация окружения
source venv/bin/activate
```

### 3. Установка Python-пакетов

```bash
# Установка зависимостей
pip install evdev opencv-python mediapipe numpy

# Или через requirements.txt:
# pip install -r requirements.txt
```

### 4. Настройка прав доступа (uinput)

Этот шаг необходим для работы виртуальной мыши без sudo:

```bash
# Создание правила udev для uinput
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules

# Перезагрузка прав udev
sudo udevadm control --reload && sudo udevadm trigger

# Добавление пользователя в группу input
sudo usermod -aG input $USER

# ВАЖНО: Выйти и снова зайти в систему (или перезагрузиться)
# Без этого права группы input не применятся
```

### 5. Копирование скрипта

```bash
# Скопировать hand_gesture_mouse_v2.py в ~/arducam/
# Например через scp:
# scp hand_gesture_mouse_v2.py user@raspberry-pi:~/arducam/

# Или создать файл напрямую:
nano ~/arducam/hand_gesture_mouse_v2.py
# (вставить содержимое файла)
```

### 6. Запуск

```bash
cd ~/arducam
source venv/bin/activate
python3 hand_gesture_mouse_v2.py
```

Для выхода нажмите **'q'** в окне камеры.

## Проверка работоспособности

```bash
# 1. Проверка что камера OV9281 определяется
rpicam-hello -t 3000

# 2. Проверка устройств видео
ls -la /dev/video*

# 3. Проверка прав uinput
ls -la /dev/uinput
# Должно быть: crw-rw---- 1 root input ...

# 4. Проверка что пользователь в группе input
groups $USER | grep input

# 5. Проверка что процессы не блокируют камеру
ps aux | grep -E "(rpicam|python|libcamera)" | grep -v grep
# Если есть процессы — завершить их: pkill -f имя_процесса
```

## Возможные проблемы

| Проблема | Решение |
|----------|---------|
| `ImportError: No module named 'mediapipe'` | Активируйте venv: `source venv/bin/activate` |
| `Device or resource busy` | Завершите другие процессы: `pkill -f rpicam; pkill -f python` |
| Курсор не двигается | Проверьте права uinput, перелогиньтесь |
| `Undefined symbol: PyThreadState_GetUnchecked` | Нормально — скрипт использует rpicam-vid, а не прямой импорт Picamera2 |
| Камера не определяется | Проверьте подключение OV9281, запустите `rpicam-hello` |

## Технические детали

Скрипт использует `rpicam-vid` с кодеком YUV420 вместо прямого импорта Picamera2 из-за проблем совместимости с Python 3.12.

**Пайплайн захвата:**
1. `rpicam-vid --codec yuv420` запускается как subprocess
2. Кадры YUV420 читаются из stdout
3. Извлекается Y-канал (grayscale)
4. Нормализация яркости
5. Конвертация в BGR для MediaPipe

## Файлы проекта

- `hand_gesture_mouse_v2.py` — основной скрипт
- `requirements.txt` — зависимости Python
- `camera_opencv.py` — пример работы с камерой через Picamera2
- `hand_gesture_ov9281.py` — альтернативная реализация

## Лицензия

MIT License
