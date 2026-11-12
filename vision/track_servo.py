r"""
AeroEye-3D — CLOSED-LOOP tracking (visual servoing) for a bright target.

The camera sees BOTH the flashlight target and the laser dot. Each frame:
  1. detect the two bright blobs in the left eye:
       - the LARGER blob  = the flashlight
       - the SMALLER blob = the laser dot
  2. aim open-loop from calib3d.json (pixel + disparity -> angle), then
  3. add a feedback correction that nudges pan/tilt until the laser dot pixel
     sits ON the flashlight pixel. The correction integrates over frames, so
     any residual calibration bias is driven to zero automatically.

    ..\.venv\Scripts\python.exe track_servo.py [COM3] [cam]
Close the window to quit.
"""
import sys
import time
import json

import cv2
import numpy as np
import serial

import stereo

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN  = "AeroEye-3D servo tracking"

calib = json.load(open("calib3d.json"))
PAN_C, TILT_C = calib["pan"], calib["tilt"]

ABS_FLOOR  = 90
PEAK_DELTA = 45
MIN_SEP    = 12       # px — blobs closer than this are the same light
GAIN       = 0.35     # feedback strength (fraction of the pixel error per frame)
CORR_MAX   = 20.0     # deg — clamp the integrated correction
COAST_S    = 0.8
SEND_DT    = 0.03


def aim(x, y, disp):
    f = [x, y, disp, x * y, x * x, y * y, 1.0]
    return (sum(c * v for c, v in zip(PAN_C, f)),
            sum(c * v for c, v in zip(TILT_C, f)))


DOT_FLOOR = 150       # laser dot must be at least this bright (board is ~130)


def find_target(gray):
    """Brightest blob = the flashlight."""
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    _, maxv, _, _ = cv2.minMaxLoc(b)
    if maxv < ABS_FLOOR:
        return None
    thr = max(ABS_FLOOR, int(maxv) - PEAK_DELTA)
    mask = (b >= thr).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return (int(x), int(y), max(int(r), 2), cv2.contourArea(c))


def find_laser(gray, tgt):
    """The laser dot: brightest point OUTSIDE the target's own glow. Two-stage,
    with its own (lower) threshold — the saturated flashlight must not hide it."""
    b = cv2.GaussianBlur(gray, (3, 3), 0)
    if tgt is not None:
        b = b.copy()
        cv2.circle(b, (tgt[0], tgt[1]), max(int(tgt[2] * 2.5), MIN_SEP), 0, -1)
    _, mv, _, loc = cv2.minMaxLoc(b)
    if mv < DOT_FLOOR:
        return None
    return (loc[0], loc[1], 3, 1.0)


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)

    corr = [0.0, 0.0]                # integrated feedback correction (deg)
    last_cmd, last_cmd_t = None, 0.0
    last_seen_t = 0.0
    last_sent = (None, None, None)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tgt = find_target(left)
        laser_dot = find_laser(left, tgt) if tgt else None
        tgt_r = find_target(right)
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        now = time.time()

        if tgt and tgt_r:
            disp = tgt[0] - tgt_r[0]
            disp = max(2, min(60, disp))
            base_pan, base_tilt = aim(tgt[0], tgt[1], disp)

            # visual feedback: nudge so the laser dot lands on the target
            if laser_dot:
                ex = tgt[0] - laser_dot[0]
                ey = tgt[1] - laser_dot[1]
                corr[0] += GAIN * PAN_C[0] * ex          # d pan  ~ PAN_C[0] per px x
                corr[1] += GAIN * TILT_C[1] * ey         # d tilt ~ TILT_C[1] per px y
                corr[0] = max(-CORR_MAX, min(CORR_MAX, corr[0]))
                corr[1] = max(-CORR_MAX, min(CORR_MAX, corr[1]))
                cv2.circle(vis, (laser_dot[0], laser_dot[1]), max(laser_dot[2], 4), (0, 0, 255), 2)
                cv2.line(vis, (laser_dot[0], laser_dot[1]), (tgt[0], tgt[1]), (255, 255, 0), 1)

            pan = int(round(base_pan + corr[0]))
            tilt = int(round(base_tilt + corr[1]))
            last_cmd, last_cmd_t = (pan, tilt), now
            last_seen_t = now
            cv2.circle(vis, (tgt[0], tgt[1]), max(tgt[2], 6), (0, 255, 0), 2)
            cv2.putText(vis, f"pan={pan} tilt={tilt} corr=({corr[0]:+.1f},{corr[1]:+.1f})"
                        + ("" if laser_dot else "  [dot not seen]"),
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            coasting = last_cmd and (now - last_seen_t < COAST_S)
            cv2.putText(vis, "COAST" if coasting else "no target (laser off)",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            if not coasting:
                last_cmd = None

        if last_cmd:
            cmd = (last_cmd[0], last_cmd[1], 1)
        else:
            cmd = (last_sent[0], last_sent[1], 0) if last_sent[0] else (72, 25, 0)
        if cmd != last_sent and now - getattr(main, "_t", 0) > SEND_DT:
            ser.write(f"{cmd[0]},{cmd[1]},{cmd[2]}\n".encode()); ser.readline()
            last_sent = cmd
            main._t = now

        cv2.imshow(WIN, cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(20)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    ser.write(b"72,25,0\n")
    cap.release(); cv2.destroyAllWindows(); ser.close()


if __name__ == "__main__":
    main()
