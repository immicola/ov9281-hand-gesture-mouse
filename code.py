import board
import busio
import usb_hid
from adafruit_hid.mouse import Mouse

uart = busio.UART(board.GP0, board.GP1, baudrate=115200, timeout=0.01)
mouse = Mouse(usb_hid.devices)

buffer = b""

while True:
    if uart.in_waiting:
        buffer += uart.read(uart.in_waiting)
        if b"\n" in buffer:
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                try:
                    cmd_str = line.decode('utf-8').strip()
                    if not cmd_str: continue
                    parts = cmd_str.split(',')
                    cmd_type = parts[0]
                    
                    if cmd_type == 'M':
                        mouse.move(x=int(parts[1]), y=int(parts[2]))
                    
                    elif cmd_type == 'C':
                        target_x = int(parts[1])
                        target_y = int(parts[2])
                        
                        # 1. Жестко бьемся в левый верхний угол (0, 0)
                        # 30 шагов по -127 = -3810 пикселей (хватит, чтобы упереться в край даже на 4K)
                        for _ in range(30):
                            mouse.move(x=-127, y=-127)
                        
                        # 2. Мгновенно едем в заданную точку от угла (в центр)
                        cx, cy = 0, 0
                        while cx < target_x or cy < target_y:
                            step_x = min(127, target_x - cx)
                            step_y = min(127, target_y - cy)
                            mouse.move(x=step_x, y=step_y)
                            cx += step_x
                            cy += step_y
                            
                    elif cmd_type == 'CLICK':
                        mouse.click(Mouse.LEFT_BUTTON)
                    elif cmd_type == 'DOWN':
                        mouse.press(Mouse.LEFT_BUTTON)
                    elif cmd_type == 'UP':
                        mouse.release(Mouse.LEFT_BUTTON)
                except Exception:
                    pass