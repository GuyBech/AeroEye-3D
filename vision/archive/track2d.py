r"""
AeroEye-3D — 2D closed-loop tracking (laser co-located with camera).

With the laser next to the camera lens there is no parallax, so aiming needs
only the LEFT eye pixel and the quadratic calib2d.json map:
    angle = f(x, y, x*y, x^2, y^2)
No stereo matching, no disparity — simpler and more robust.

Detector选:  --flash  brightest spot (flashlight)
             --shape  round moving object (tennis ball)
             (default) any moving object

    ..\.venv\Scripts\python.exe track2d.py [COM3] [cam] [--flash|--shape]
Close the window to quit. Laser only fires while a target is present.
"""
import sys
import time
import json

import cv2
import numpy as np
import serial

import stereo

if "--shape" in sys.argv:
    from detector_shape import BallDetector as Det
elif "--flash" in sys.argv:
    import detector as _bright
    class Det:
        def detect(self, gray):
            return _bright.detect(gray)
else:
    from detector_motion import MotionDetector as Det

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM3")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN  = "AeroEye-3D track 2D"

calib = json.load(open("calib2d.json"))
PAN_C, TILT_C = calib["pan"], calib["tilt"]
QUAD = calib.get("model") == "quad2d"

WARMUP  = 25
COAST_S = 0.8
SEND_DT = 0.04
ALPHA   = 0.5


def aim(x, y):
    if QUAD:
        feats = [x, y, x * y, x * x, y * y, 1.0]
    else:
        feats = [x, y, 1.0]
    pan  = sum(c * f for c, f in zip(PAN_C, feats))
    tilt = sum(c * f for c, f in zip(TILT_C, feats))
    return int(round(pan)), int(round(tilt))


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)
    det = Det()

    frame_i = 0
    sm = None
    last_lock, last_lock_t = None, 0.0
    last_sent, last_t = (None, None, None), 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_i += 1
        left, _ = stereo.split_lr(frame)
        t = det.detect(left)
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)

        now = time.time()
        locked = False
        if frame_i > WARMUP and t:
            raw = aim(t.x, t.y)
            if sm is None:
                sm = [float(raw[0]), float(raw[1])]
            else:
                sm[0] = ALPHA * raw[0] + (1 - ALPHA) * sm[0]
                sm[1] = ALPHA * raw[1] + (1 - ALPHA) * sm[1]
            last_lock = (int(round(sm[0])), int(round(sm[1])))
            last_lock_t = now
            locked = True
            cv2.circle(vis, (t.x, t.y), max(t.radius, 6), (0, 255, 0), 2)
            cv2.putText(vis, f"LOCK ({t.x},{t.y}) -> pan={last_lock[0]} tilt={last_lock[1]}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        coasting = (not locked) and last_lock and (now - last_lock_t < COAST_S)
        if not locked:
            msg = "COAST" if coasting else ("warming up" if frame_i <= WARMUP else "no target (laser off)")
            cv2.putText(vis, msg, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        if now - last_t > SEND_DT:
            if locked or coasting:
                cmd = (last_lock[0], last_lock[1], 1)
            else:
                cmd = (last_sent[0], last_sent[1], 0) if last_sent[0] else (72, 25, 0)
            if cmd != last_sent:
                ser.write(f"{cmd[0]},{cmd[1]},{cmd[2]}\n".encode()); ser.readline()
                last_sent = cmd
            last_t = now

        cv2.imshow(WIN, cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    ser.write(b"72,25,0\n")
    cap.release(); cv2.destroyAllWindows(); ser.close()


if __name__ == "__main__":
    main()
