# Архитектура системы управления Thor Robot Arm через жесты

## 1. Обзор

Система позволяет управлять роботом-манипулятором Thor (6-DOF + схват) с помощью жестов руки, распознаваемых через камеру. Компьютерное зрение работает на Raspberry Pi 4, данные передаются по цепочке через UART, USB, WebSocket и ROS 2 до физического робота.

---

## 2. Аппаратная схема

| Компонент | Роль | Соединение |
|-----------|------|------------|
| **Arducam OV9281** | Монохромная глобальный затвор камера | CSI шлейф → Raspberry Pi 4 |
| **Raspberry Pi 4** | Захват кадра, MediaPipe, отправка UART | — |
| **Raspberry Pi Pico (RP2040)** | UART→USB CDC мост (без логики) | GPIO 0/1 → Pi GPIO 14/15; USB → Windows |
| **Windows 11 + WSL2** | thor_bridge.py, rosbridge_server, ROS 2 Humble | USB от Pico; usbipd к роботу |
| **Thor Robot Arm** | 6-DOF рука + схват | USB → Windows (usbipd → WSL) |

---

## 3. Data Flow (полный путь сигнала)

```
OV9281 (камера)
  │ CSI
  ▼
rpicam-vid (YUV420, 640×480, 15fps)
  │ stdout
  ▼
robo_arm_v2.py (MediaPipe Hands)
  │ распознаёт жест, позицию ладони
  │ UART (GPIO 14/15, 115200)
  ▼
Pico code.py (UART→USB CDC ретранслятор)
  │ USB (CDC ACM)
  ▼
thor_bridge.py (WSL, /dev/ttyACM0)
  │ парсит GEST:/X:, мапит на суставы
  │ WebSocket (ws://localhost:9090)
  ▼
rosbridge_server (ROS 2 Humble)
  │ топик /joint_group_position_controller/command
  ▼
ros2_control → hardware interface (C++ libserial)
  │ usbipd (USB over IP)
  ▼
Thor Robot Arm (сервомоторы)
```

---

## 4. Raspberry Pi 4 — `robo_arm_v2.py`

### 4.1 Захват кадра

- `rpicam-vid` запускается как subprocess с `--codec yuv420 --framerate 15`
- Фоновый тред `_reader()` читает ровно `FRAME_SIZE = (640*480*3)//2 = 460800` байт на кадр
- Извлекается Y-канал (grayscale), кадр кладётся в `queue.Queue(maxsize=2)`
- Если очередь полна — старый кадр отбрасывается (real-time, без backpressure)

### 4.2 Детекция руки

- **Библиотека**: MediaPipe Hands (`model_complexity=0` — lite)
- **Пороги**: `min_detection_confidence=0.65`, `min_tracking_confidence=0.45`
- **max_num_hands=1** — только одна рука

### 4.3 Подсчёт пальцев

- Для каждого пальца (кроме большого): сравнение расстояний `tip→wrist` vs `pip→wrist` и `tip→mcp` vs `pip→mcp`
- Большой палец: отдельная логика через `thumb_extended()` (абдукция от центра ладони + отношения расстояний)

### 4.4 Таблица жестов

| Жест | Пальцы | Действие |
|------|--------|----------|
| `FIST` | 0 | Закрыть схват |
| `ONE` | 1 (указательный) | Игнорируется (промежуточный) |
| `TWO` | 2 (указ. + сред.) | Выбрать сустав J2 (shoulder) |
| `THREE` | 3 (указ. + сред. + безым., без мизинца и большого) | Выбрать сустав J1 (base rotation) |
| `FOUR` | 4 (без большого) | Выбрать сустав J3 (elbow) |
| `FIVE` | 5 | Открыть схват |

### 4.5 Стабилизация и троттлинг

- **Стабилизация**: `deque(maxlen=3)` — жест должен совпасть 3 кадра подряд
- **Троттлинг команд**: `CMD_THROTTLE = 0.15с` между отправками жестов
- **Троттлинг X**: `X_THROTTLE = 0.03с` между отправками позиции ладони

### 4.6 Управление суставом

- Любой из трёх жестов (`THREE`, `TWO`, `FOUR`) активирует соответствующий сустав
- Позиция ладони по X (0.0–1.0) непрерывно отправляется как `X:<value>\n`
- На стороне thor_bridge `value` преобразуется в смещение сустава

---

## 5. UART-протокол

Асинхронный текст, newline-delimited, 115200 бод:

| Формат | Пример | Описание |
|--------|--------|----------|
| `GEST:<name>\n` | `GEST:THREE\n` | Выбрать сустав или управлять схватом |
| `X:<value>\n` | `X:0.723\n` | Горизонтальная позиция ладони (0.0–1.0) |

---

## 6. Raspberry Pi Pico — `code.py`

**13 строк кода** на CircuitPython. Никакой логики — чистый UART→USB CDC мост:

```
loop:
  data = uart.read(64)
  if data:
    usb_cdc.console.write(data)
```

Критично: Pico **не парсит** и **не буферизирует** данные. Он просто ретранслирует байты с UART (GPIO 0/1) на USB CDC. Нет `time.sleep()` — минимальная задержка.

---

## 7. WSL2 — `thor_bridge.py`

Запускается на WSL (Ubuntu 22.04), читает serial порт Pico (`/dev/ttyACM0`).

### 7.1 Потоки

- **`serial_reader()`**: читает serial блоками по 64 байта, парсит строки `GEST:` и `X:`
- **`ws_worker()`**: держит WebSocket к rosbridge_server, публикует при изменениях

### 7.2 Обработка X-позиции

```
offset = value - 0.5              // центрирование
if |offset| < DEAD_ZONE(0.04): return  // мёртвая зона
t = clamp((|offset| - 0.04) / (0.25 - 0.04), 0, 1)
speed = t² × JOYSTICK_GAIN(0.5)  // квадратичная кривая
joint = clamp(joint + delta, min, max)
```

### 7.3 Маппинг жестов

```python
GESTURE_JOINT_MAP = {
    "THREE": 0,  # J1 — base rotation
    "TWO":   1,  # J2 — shoulder
    "FOUR":  2,  # J3 — elbow
}
```

### 7.4 WebSocket (самописный)

- Реализация RFC 6455 без внешних библиотек
- `threading.Event()` — публикация только при изменениях
- `PUBLISH_THROTTLE = 0.01с`
- Топик: `/joint_group_position_controller/command`
- Тип: `std_msgs/Float64MultiArray`

Формат сообщения:
```json
{"op": "publish", "topic": "/joint_group_position_controller/command",
 "msg": {"data": [j1, j2, j3, j4, j5, j6, gripper]}}
```

---

## 8. ROS 2 / Thor-ROS

- **ROS 2**: Humble
- **Контроллер**: `joint_group_position_controller` (ros2_control)
- **Hardware interface**: C++ через `libserial`
- **Соединение с роботом**: `usbipd` — USB-устройство робота пробрасывается из Windows в WSL по сети
- **Альтернативный UI**: Asgard (веб-интерфейс через rosbridge)

---

## 9. End-to-End: пошагово

1. Камера OV9281 захватывает кадр 640×480
2. `rpicam-vid` выводит YUV420 в stdout
3. `robo_arm_v2.py` извлекает grayscale, нормализует, подаёт в MediaPipe
4. MediaPipe находит 21 точку кисти, определяется жест
5. Стабилизация: жест должен подтвердиться 3 кадра подряд
6. Отправка `GEST:THREE\n` по UART
7. Pico ретранслирует байты на USB → Windows → WSL
8. `thor_bridge.py` парсит `THREE` → выбирает J1 (base rotation)
9. Циклическая отправка `X:0.723\n` управляет поворотом J1
10. `thor_bridge.py` публикует положение всех суставов в ROS 2 топик
11. `ros2_control` → hardware interface → usbipd → физический Thor Arm

---

## 10. Архитектурные заметки

- **Pico — dumb bridge**: не содержит логики управления, только ретрансляция UART↔USB. Это упрощает firmware, но создаёт дополнительное звено задержки.
- **5 последовательных serial/network hop-ов**: Pi→Pico→USB→WSL→WS→ROS→usbipd→robot. Каждый hop добавляет латентность.
- **Текстовый протокол**: команды вида `GEST:THREE\n` — человекочитаемо, но избыточно. Бинарный протокол мог бы быть эффективнее.
- **Самописный WebSocket**: `thor_bridge.py` содержит собственную реализацию WebSocket (RFC 6455) вместо зависимости от внешних библиотек.
- **Wayland-совместимость**: `os.environ.setdefault("QT_QPA_PLATFORM", "xcb")` в начале скриптов — обход падения OpenCV под labwc/Wayland.
- **Три поколения**: система эволюционировала от жеста LIKE (большой палец) → THREE (три пальца), от 15fps → 30fps, от Counter-стабилизации → all-match.
- **Управление velocity-режимом**: X-позиция ладони работает как джойстик — скорость вращения сустава пропорциональна отклонению от центра с квадратичной кривой и мёртвой зоной.
