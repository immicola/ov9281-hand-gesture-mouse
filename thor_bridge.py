#!/usr/bin/env python3
"""
thor_bridge.py — Pico USB serial → Thor-ROS via rosbridge WebSocket

Usage:
  python3 thor_bridge.py /dev/ttyACM0        # default port
  python3 thor_bridge.py /dev/ttyACM0 --host localhost --port 9090

Runs in WSL2 alongside Thor-ROS.
Needs Pico USB serial forwarded via usbipd.
"""

import sys
import json
import struct
import hashlib
import base64
import socket
import os
import time
import threading

ROSBRIDGE_HOST = "localhost"
ROSBRIDGE_PORT = 9090
RECONNECT_DELAY = 3.0

# Gesture → [joint1..joint6 (rad), gripper (rad)]
# Joint limits: j1/j4/j6 ±2.967, j2/j3/j5 ±1.57, gripper -1.57..0
GESTURE_POSES = {
    "LIKE":  ([0.0, 0.8, -1.5, 0.0, 0.3, 0.0],  0.0),
    "FIVE":  ([0.0, 0.5, -1.2, 0.0, -0.3, 0.0], 0.0),
    "FIST":  ([0.0, 0.5, -1.2, 0.0, -0.3, 0.0], -1.2),
    "ONE":   ([0.0, 0.3, -0.3, 0.0, -1.2, 0.0], 0.0),
    "TWO":   ([1.2, 0.5, -1.0, 0.5, -0.5, 0.0], 0.0),
}

TOPIC = "/joint_group_position_controller/command"
ADVERTISE_DONE = threading.Event()
ADVERTISE_DONE.set()


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

        self.sock.settimeout(None)
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
                self.buf += self._recv_n(2 - len(self.buf))
            opcode = self.buf[0] & 0x0F
            masked = self.buf[1] & 0x80
            length = self.buf[1] & 0x7F
            offset = 2

            if length == 126:
                while len(self.buf) < offset + 2:
                    self.buf += self._recv_n(1)
                length = struct.unpack(">H", self.buf[offset:offset+2])[0]
                offset += 2
            elif length == 127:
                while len(self.buf) < offset + 8:
                    self.buf += self._recv_n(1)
                length = struct.unpack(">Q", self.buf[offset:offset+8])[0]
                offset += 8

            if masked:
                while len(self.buf) < offset + 4:
                    self.buf += self._recv_n(1)
                mask = self.buf[offset:offset+4]
                offset += 4

            while len(self.buf) < offset + length:
                self.buf += self._recv_n(1)

            payload = self.buf[offset:offset+length]
            self.buf = self.buf[offset+length:]

            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self.sock.sendall(bytes([0x8A, 0x00]))
                continue
            if opcode == 0xA:
                continue
            return payload.decode("utf-8")

    def _recv_n(self, n):
        while True:
            chunk = self.sock.recv(max(n, 4096))
            if not chunk:
                raise ConnectionError("Connection closed")
            return chunk

    def close(self):
        if self.sock:
            try:
                self.sock.sendall(bytes([0x88, 0x00]))
            except:
                pass
            self.sock.close()
            self.sock = None


def advertise_topic(ws):
    ws.send_text(json.dumps({
        "op": "advertise",
        "topic": TOPIC,
        "type": "std_msgs/Float64MultiArray"
    }))
    ADVERTISE_DONE.set()
    print(f"  Advertised {TOPIC}")


def send_pose(ws, gesture_name):
    pose = GESTURE_POSES.get(gesture_name)
    if not pose:
        print(f"  Unknown gesture: {gesture_name}")
        return False

    joints, gripper = pose
    msg = {
        "op": "publish",
        "topic": TOPIC,
        "msg": {
            "data": list(joints) + [gripper]
        }
    }
    ws.send_text(json.dumps(msg))
    deg = [f"{math.degrees(j):.0f}" for j in joints] + [f"{math.degrees(gripper):.0f}"]
    print(f"  >>> {gesture_name}  →  [{', '.join(deg)}] deg")
    return True


def serial_reader(port, ws):
    import serial
    while True:
        try:
            ser = serial.Serial(port, 115200, timeout=0.1)
            print(f"Serial: opened {port}")
            ADVERTISE_DONE.wait()
            buf = ""
            while True:
                data = ser.read(64)
                if data:
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("GEST:"):
                            gesture = line[5:]
                            send_pose(ws, gesture)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        except Exception as e:
            print(f"Reader error: {e}")
        time.sleep(2)


def ws_reconnector(ws, host, port):
    while True:
        try:
            ws.connect(host, port)
            advertise_topic(ws)
            while True:
                msg = ws.recv()
                if msg is None:
                    break
        except Exception as e:
            print(f"WS error: {e}")
        print(f"  Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)


def main():
    import math as _m
    global math
    math = _m

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

    print(f"Thor Bridge")
    print(f"  Serial port: {port}")
    print(f"  Rosbridge:   {host}:{rport}")
    print(f"  Topic:       {TOPIC}")
    print(f"  Gestures:    {', '.join(GESTURE_POSES.keys())}")
    print()

    ws = WSClient()
    threading.Thread(target=ws_reconnector, args=(ws, host, rport), daemon=True).start()
    threading.Thread(target=serial_reader, args=(port, ws), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        ws.close()


if __name__ == "__main__":
    main()
