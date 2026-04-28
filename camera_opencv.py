#!/usr/bin/env python3
"""
Просмотр камеры OV9281 через OpenCV
Не использует picamera2 preview — только захват кадров в numpy
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

def main():
    cam = Picamera2()

    # Явно создаём конфиг БЕЗ raw стрима
    # create_video_configuration не добавляет raw по умолчанию
    config = cam.create_video_configuration(
        main={"size": (640, 400), "format": "YUV420"},
    )

    # Убираем raw если всё равно добавился
    config["raw"] = None

    print("Конфигурация до применения:")
    print("  main:", config.get("main"))
    print("  raw: ", config.get("raw"))

    cam.configure(config)

    print("Конфигурация после configure:")
    final = cam.camera_configuration()
    print("  main:", final.get("main"))
    print("  raw: ", final.get("raw"))

    # Запуск БЕЗ preview — только захват в память
    cam.start()
    time.sleep(1)  # дать автоэкспозиции стабилизироваться

    print("Стрим запущен. Нажмите 'q' для выхода, 's' для снимка.")
    shot = 0

    while True:
        # Захват YUV420 кадра
        frame = cam.capture_array("main")

        # YUV420 → Grayscale (просто берём Y-канал, первые 400 строк)
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 1:
            gray = frame[:, :, 0]
        else:
            # YUV420 упакован как (600, 640) — Y в верхних 400 строках
            if frame.shape[0] > 400:
                gray = frame[:400, :]
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_I420)

        # Нормализовать яркость для лучшей видимости
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Подсказка на экране
        cv2.putText(gray, "OV9281 | 'q' quit | 's' save",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

        cv2.imshow("OV9281 Camera", gray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot += 1
            fname = f"snap_{shot:03d}.png"
            cv2.imwrite(fname, gray)
            print(f"Сохранено: {fname}")

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()