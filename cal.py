# cal.py — запусти на Pi, двигает курсор в угол потом на X,Y шагов
import serial, time

s = serial.Serial('/dev/serial0', 115200)
time.sleep(0.5)

# Сначала в угол
s.write(b"C,450,250\n")  # 0,0 = только угол, никуда не едем