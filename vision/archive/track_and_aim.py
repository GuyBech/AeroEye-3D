r"""
AeroEye-3D — Phase 6e: closed-loop tracking.

Camera detects the bright target (left eye) -> calib.json affine map turns the
pixel into (pan, tilt) -> the turret aims there. This is the whole system
working together.

    ..\.venv\Scripts\python.exe track_and_aim.py [COM3] [cam]

Close the window to quit. Laser fires only while a target is locked.
Safety: keep the working area clear; the laser follows the target.
"""
import sys
import time
import json

import cv2
import numpy as np
import serial

import stereo
import detector

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM3")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN  = "AeroEye-3D tracking"

calib = json.load(open("calib.json"))
PAN_C  = calib["pan"]      # pan  = a*x + b*y + c
TILT_C = calib["tilt"]     # tilt = d*x + e*y + f


def pixel_to_angle(x, y):
    pan  = PAN_C[0] * x + PAN_C[1] * y + PAN_C[2]
    tilt = TILT_C[0] * x + TILT_C[1] * y + TILT_C[2]
    return int(round(pan)), int(round(tilt))


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)

    last_sent = (None, None, None)
    last_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray, _ = stereo.split_lr(frame)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        target = detector.detect(gray)

        pan = tilt = None
        laser = 0
        if target:
            pan, tilt = pixel_to_angle(target.x, target.y)
            laser = 1
            cv2.circle(disp, (target.x, target.y), max(target.radius, 6), (0, 255, 0), 2)
            cv2.drawMarker(disp, (target.x, target.y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(disp, f"target=({target.x},{target.y}) -> pan={pan} tilt={tilt}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(disp, "no target", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # send at most ~8 Hz, and only when something meaningfully changed
        now = time.time()
        if now - last_t > 0.12:
            if target:
                cmd = (pan, tilt, 1)
            else:
                cmd = (last_sent[0], last_sent[1], 0) if last_sent[0] else (90, 20, 0)
            if cmd != last_sent:
                ser.write(f"{cmd[0]},{cmd[1]},{cmd[2]}\n".encode())
                ser.readline()
                last_sent = cmd
            last_t = now

        cv2.imshow(WIN, disp)
        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    ser.write(b"90,20,0\n")   # park, laser off
    cap.release(); cv2.destroyAllWindows(); ser.close()


if __name__ == "__main__":
    main()
