r"""
AeroEye-3D — Phase 6e(3D): depth-aware closed-loop tracking.

Detects the target in BOTH eyes, computes disparity (depth), and uses the
affine3d calib3d.json map  angle = f(x, y, disparity)  to aim correctly at
any distance. Stereo sanity checks reject mismatches (the two eyes must see
the target on a similar row, with a plausible positive disparity); when the
match is invalid the laser is held OFF.

    ..\.venv\Scripts\python.exe track_and_aim_3d.py [COM3] [cam]
Close the window to quit.
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
WIN  = "AeroEye-3D tracking 3D"

calib = json.load(open("calib3d.json"))
PAN_C, TILT_C = calib["pan"], calib["tilt"]

DISP_MIN, DISP_MAX = 8, 200      # plausible disparity range
ROW_TOL = 40                     # max |y_left - y_right| for a valid stereo match


def aim(x, y, disp):
    pan  = PAN_C[0]*x + PAN_C[1]*y + PAN_C[2]*disp + PAN_C[3]
    tilt = TILT_C[0]*x + TILT_C[1]*y + TILT_C[2]*disp + TILT_C[3]
    return int(round(pan)), int(round(tilt))


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)

    last_sent, last_t = (None, None, None), 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tl, tr = detector.detect(left), detector.detect(right)
        vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
        w = left.shape[1]

        valid = False
        pan = tilt = disp = dy = None
        if tl and tr:
            disp = tl.x - tr.x
            dy = abs(tl.y - tr.y)
            if DISP_MIN <= disp <= DISP_MAX and dy <= ROW_TOL:
                valid = True
                pan, tilt = aim(tl.x, tl.y, disp)
        if tl: cv2.drawMarker(vis, (tl.x, tl.y), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if tr: cv2.drawMarker(vis, (w + tr.x, tr.y), (0, 128, 255), cv2.MARKER_CROSS, 18, 2)

        if valid:
            cv2.putText(vis, f"LOCK disp={disp} dy={dy} -> pan={pan} tilt={tilt}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            why = "no target"
            if tl and tr:
                why = f"disp={disp} dy={dy} REJECT (need {DISP_MIN}<=disp<={DISP_MAX}, dy<={ROW_TOL})"
            cv2.putText(vis, why, (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        now = time.time()
        if now - last_t > 0.12:
            if valid:
                cmd = (pan, tilt, 1)
            else:
                cmd = (last_sent[0], last_sent[1], 0) if last_sent[0] else (90, 20, 0)
            if cmd != last_sent:
                ser.write(f"{cmd[0]},{cmd[1]},{cmd[2]}\n".encode()); ser.readline()
                last_sent = cmd
            last_t = now

        cv2.imshow(WIN, cv2.resize(vis, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    ser.write(b"90,20,0\n")
    cap.release(); cv2.destroyAllWindows(); ser.close()


if __name__ == "__main__":
    main()
