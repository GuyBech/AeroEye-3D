r"""Diagnostic / building block: detect the laser dot by ON-minus-OFF frame
differencing. Robust to ambient light and a dim monochrome view — whatever
changed between laser-off and laser-on is the dot.
    ..\.venv\Scripts\python.exe laser_diff.py [COM3] [cam]
"""
import sys
import time
import cv2
import numpy as np
import serial
import stereo

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM3")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)


def grab_left(cap, n=5):
    frame = None
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.03)
    left, _ = stereo.split_lr(frame)
    return left


def set_laser(ser, on):
    ser.write(f"90,20,{1 if on else 0}\n".encode())
    time.sleep(0.4)
    ser.readline()


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)

    set_laser(ser, False); time.sleep(0.3)
    off = grab_left(cap).astype(np.int16)
    set_laser(ser, True);  time.sleep(0.3)
    on  = grab_left(cap).astype(np.int16)

    diff = np.clip(on - off, 0, 255).astype(np.uint8)
    diff_blur = cv2.GaussianBlur(diff, (9, 9), 0)
    _, maxv, _, maxloc = cv2.minMaxLoc(diff_blur)
    print(f"diff brightest={maxv:.0f} at {maxloc}")

    disp = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(disp, maxloc, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.putText(disp, f"diff max={maxv:.0f} @ {maxloc}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("laser_diff.png", disp)
    print("Saved laser_diff.png")

    cap.release(); ser.close()


if __name__ == "__main__":
    main()
