import board
import busio
import usb_cdc
import time

uart = busio.UART(board.GP0, board.GP1, baudrate=115200)
serial = usb_cdc.console

while True:
    if uart.in_waiting > 0:
        data = uart.read(64)
        if data:
            serial.write(data)
    time.sleep(0.005)
