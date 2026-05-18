#!/usr/bin/env python3
import sys
import json
import struct
import hashlib
import base64
import socket
import os
import time
import threading
import math

ROSBRIDGE_HOST = "localhost"
ROSBRIDGE_PORT = 9090
RECONNECT_DELAY = 3.0

HOME = [0.0, 0.8, -1.5, 0.0, 0.3, 0.0]
GRIP_OPEN = 0.0
GRIP_CLOSE = -1.2

JOINT_RANGES = [
    (-2.967, 2.967),
    (-1.57, 1.57),
    (-1.57, 1.57),
]

GESTURE_JOINT_MAP = {
    "THREE": 0,
    "TWO": 1,
    "FOUR": 2,
}

DEAD_ZONE = 0.04
MAX_DIST = 0.25
JOYSTICK_GAIN = 0.5
PUBLISH_THROTTLE = 0.01

TOPIC = "/joint_group_position_controller/command"

current_joints = list(HOME)
current_gripper = GRIP_OPEN
active_joint = -1
lock = threading.Lock()
changed_event = threading.Event()

_TIMEOUT = object()

class WSClient:
    def __init__(self):
        self.sock = None
        self.buf = b""

    def connect(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect((host, port))
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Handshake failed")
            resp += chunk
        if b"101" not in resp.split(b"\r\n")[0]:
            raise ConnectionError(f"Bad handshake: {resp[:200]}")
        self.sock.settimeout(0.01)
        self.buf = b""
        print(f"  WS connected to {host}:{port}")

    def send_text(self, text):
        data = text.encode("utf-8")
        frame = bytearray()
        frame.append(0x81)
        mask = os.urandom(4)
        length = len(data)
        if length < 126:
            frame.append(0x80 | length)
        elif length < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack(">Q", length))
        frame.extend(mask)
        frame.extend(bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        self.sock.sendall(bytes(frame))

    def recv(self):
        while True:
            if len(self.buf) < 2:
                try:
                    self.buf += self.sock.recv(4096)
                except socket.timeout:
                    return _TIMEOUT
                if not self.buf:
                    return None
            opcode = self.buf[0] & 0x0F
            masked = self.buf[1] & 0x80
            length = self.buf[1] & 0x7F
            offset = 2
            if length == 126:
                while len(self.buf) < offset + 2:
                    self.buf += self.sock.recv(1)
                length = struct.unpack(">H", self.buf[offset:offset+2])[0]
                offset += 2
            elif length == 127:
                while len(self.buf) < offset + 8:
                    self.buf += self.sock.recv(1)
                length = struct.unpack(">Q", self.buf[offset:offset+8])[0]
                offset += 8
            if masked:
                while len(self.buf) < offset + 4:
                    self.buf += self.sock.recv(1)
                mask = self.buf[offset:offset+4]
                offset += 4
            while len(self.buf) < offset + length:
                self.buf += self.sock.recv(1)
            payload = self.buf[offset:offset+length]
            self.buf = self.buf[offset+length:]
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                try:
                    self.sock.sendall(bytes([0x8A, 0x00]))
                except:
                    pass
                continue
            if opcode == 0xA:
                continue
            return payload.decode("utf-8")

    def close(self):
        if self.sock:
            try:
                self.sock.sendall(bytes([0x88, 0x00]))
            except:
                pass
            self.sock.close()
            self.sock = None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def publish_joints(ws):
    with lock:
        data = list(current_joints) + [current_gripper]
    msg = {"op": "publish", "topic": TOPIC, "msg": {"data": data}}
    ws.send_text(json.dumps(msg))


def handle_gesture(name):
    global active_joint, current_gripper
    if name in GESTURE_JOINT_MAP:
        active_joint = GESTURE_JOINT_MAP[name]
        changed_event.set()
        print(f"  Joint selected: {name} → J{active_joint + 1}")
    elif name == "FIST":
        with lock:
            current_gripper = GRIP_CLOSE
        changed_event.set()
        print(f"  GRIP CLOSE")
    elif name == "FIVE":
        with lock:
            current_gripper = GRIP_OPEN
        changed_event.set()
        print(f"  GRIP OPEN")
    else:
        print(f"  Ignored gesture: {name}")


def handle_x(value):
    offset = value - 0.5
    dist = abs(offset)
    if dist < DEAD_ZONE:
        return
    j = active_joint
    if j < 0:
        return
    t = min((dist - DEAD_ZONE) / (MAX_DIST - DEAD_ZONE), 1.0)
    speed = t * t * JOYSTICK_GAIN
    delta = speed if offset > 0 else -speed
    mn, mx = JOINT_RANGES[j]
    with lock:
        current_joints[j] = clamp(current_joints[j] + delta, mn, mx)
    changed_event.set()


def ws_worker(ws, host, port):
    while True:
        try:
            ws.connect(host, port)
            ws.send_text(json.dumps({
                "op": "advertise",
                "topic": TOPIC,
                "type": "std_msgs/Float64MultiArray"
            }))
            print(f"  Advertised {TOPIC}")
            last_pub = 0
            while True:
                changed_event.wait(timeout=0.02)
                changed_event.clear()
                while True:
                    msg = ws.recv()
                    if msg is _TIMEOUT:
                        break
                    if msg is None:
                        raise ConnectionError("WS closed")
                now = time.monotonic()
                if now - last_pub > PUBLISH_THROTTLE:
                    publish_joints(ws)
                    last_pub = now
        except Exception as e:
            print(f"  WS error: {e}")
        print(f"  Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)


def serial_reader(port):
    import serial
    while True:
        try:
            ser = serial.Serial(port, 115200, timeout=0.01)
            print(f"Serial: opened {port}")
            buf = ""
            while True:
                data = ser.read(64)
                if data:
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("GEST:"):
                            handle_gesture(line[5:])
                        elif line.startswith("X:"):
                            try:
                                handle_x(float(line[2:]))
                            except:
                                pass
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        except Exception as e:
            print(f"Reader error: {e}")
        time.sleep(2)


def main():
    if len(sys.argv) < 2:
        print("Usage: thor_bridge.py <serial_port> [--host HOST] [--port PORT]")
        print("  Example: thor_bridge.py /dev/ttyACM0")
        sys.exit(1)

    port = sys.argv[1]
    host = ROSBRIDGE_HOST
    rport = ROSBRIDGE_PORT
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--port" and i + 1 < len(sys.argv):
            rport = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    print("Thor Bridge (Joystick Mode)")
    print(f"  Serial port: {port}")
    print(f"  Rosbridge:   {host}:{rport}")
    print(f"  Topic:       {TOPIC}")
    print(f"  THREE → J1 (base),     palm ← → joint ← −")
    print(f"  TWO   → J2 (shoulder), palm ← → joint ← −")
    print(f"  FOUR  → J3 (elbow),    palm ← → joint ← −")
    print(f"  FIST → gripper close")
    print(f"  FIVE → gripper open")
    print()

    ws = WSClient()
    threading.Thread(target=ws_worker, args=(ws, host, rport), daemon=True).start()
    threading.Thread(target=serial_reader, args=(port,), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        ws.close()


if __name__ == "__main__":
    main()
