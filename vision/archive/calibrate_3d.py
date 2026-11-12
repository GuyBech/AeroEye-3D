r"""
AeroEye-3D — Phase 6d(3D): depth-aware calibration.

Adds stereo disparity to the map so aiming is correct at ANY distance:
    pan  = a*x + b*y + c*disparity + d
    tilt = e*x + f*y + g*disparity + h

Procedure (mouse only):
  1. Put the flashlight somewhere in view (both eyes must see it). The window
     shows both eyes with the detected target and the live disparity.
  2. Drag PAN / TILT until the laser dot lands ON the flashlight.
  3. Drag RECORD to 1 -> the tool briefly blanks the laser, detects the target
     in both eyes, and stores (x, y, disparity) <-> (pan, tilt).
  4. Repeat ~12-15 times, IMPORTANT: at several DISTANCES (near/mid/far) and
     spread across the view.
  5. Drag SAVE to 1 -> fits and writes calib3d.json.

Run:  ..\.venv\Scripts\python.exe calibrate_3d.py [COM3] [cam]
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
OUT  = "calib3d.json"
WIN  = "AeroEye-3D calibrate 3D"

points = []            # (x, y, disparity, pan, tilt)
_ser = None

DISP_MIN, DISP_MAX = 8, 200    # a valid stereo match has a plausible disparity
ROW_TOL = 40                   # and the two eyes see it on a similar row


def stereo_ok(tl, tr):
    if not (tl and tr):
        return False, None, None
    disp = tl.x - tr.x
    dy = abs(tl.y - tr.y)
    ok = (DISP_MIN <= disp <= DISP_MAX) and (dy <= ROW_TOL)
    return ok, disp, dy


def send(pan, tilt, laser):
    _ser.write(f"{pan},{tilt},{int(laser)}\n".encode())
    _ser.readline()


def read_frame(cap, n=6):
    frame, got = None, 0
    for _ in range(40):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f; got += 1
            if got >= n: break
    return frame


def detect_both(cap):
    """Return (x_left, y_left, disparity) for a VALID stereo match, else None."""
    frame = read_frame(cap)
    if frame is None:
        return None
    left, right = stereo.split_lr(frame)
    tl, tr = detector.detect(left), detector.detect(right)
    ok, disp, _ = stereo_ok(tl, tr)
    if ok:
        return tl.x, tl.y, disp
    return None


def fit_and_save():
    if len(points) < 6:
        print(f"need >=6 points, have {len(points)}"); return
    P = np.array(points, float)
    A = np.column_stack([P[:, 0], P[:, 1], P[:, 2], np.ones(len(P))])   # x,y,disp,1
    pan_c,  *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
    tilt_c, *_ = np.linalg.lstsq(A, P[:, 4], rcond=None)
    ep = np.mean(np.abs(A @ pan_c  - P[:, 3]))
    et = np.mean(np.abs(A @ tilt_c - P[:, 4]))
    json.dump({"model": "affine3d", "pan": pan_c.tolist(), "tilt": tilt_c.tolist(),
               "points": points, "n": len(points)}, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT} ({len(points)} pts)  fit error pan~{ep:.1f} tilt~{et:.1f} deg")


def main():
    global _ser
    _ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); _ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)
    cv2.createTrackbar("pan",    WIN, 90, 180, lambda v: None); cv2.setTrackbarMin("pan", WIN, 0)
    cv2.createTrackbar("tilt",   WIN, 20, 150, lambda v: None); cv2.setTrackbarMin("tilt", WIN, 15)
    cv2.createTrackbar("laser",  WIN, 1, 1, lambda v: None)
    cv2.createTrackbar("RECORD", WIN, 0, 1, lambda v: None)
    cv2.createTrackbar("SAVE",   WIN, 0, 1, lambda v: None)
    print(__doc__)

    last = (None, None, None)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tl, tr = detector.detect(left), detector.detect(right)
        ok_match, disp_val, dy_val = stereo_ok(tl, tr)

        p = cv2.getTrackbarPos("pan", WIN)
        t = cv2.getTrackbarPos("tilt", WIN)
        l = cv2.getTrackbarPos("laser", WIN)
        if (p, t, l) != last:
            send(p, t, l); last = (p, t, l)

        if cv2.getTrackbarPos("RECORD", WIN) == 1:
            send(p, t, 0); time.sleep(0.3)            # blank laser for a clean read
            res = detect_both(cap)
            send(p, t, l)                             # laser back to slider state
            if res:
                x, y, d = res
                points.append([x, y, d, p, t])
                print(f"pt {len(points)}: pixel=({x},{y}) disp={d} -> pan{p} tilt{t}")
            else:
                print("REJECTED — bad stereo match (reposition so both eyes see the target)")
            cv2.setTrackbarPos("RECORD", WIN, 0)

        if cv2.getTrackbarPos("SAVE", WIN) == 1:
            fit_and_save(); cv2.setTrackbarPos("SAVE", WIN, 0)

        vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
        if tl: cv2.drawMarker(vis, (tl.x, tl.y), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        if tr: cv2.drawMarker(vis, (left.shape[1] + tr.x, tr.y), (0, 128, 255), cv2.MARKER_CROSS, 18, 2)
        match_col = (0, 255, 0) if ok_match else (0, 0, 255)
        match_txt = "MATCH OK" if ok_match else "bad match"
        cv2.putText(vis, f"pan={p} tilt={t} disp={disp_val} dy={dy_val} [{match_txt}] pts={len(points)}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_col, 1)
        cv2.putText(vis, "record only when MATCH OK | vary DISTANCE | SAVE=1",
                    (8, left.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.imshow(WIN, vis)
        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    send(90, 20, 0)
    cap.release(); cv2.destroyAllWindows(); _ser.close()


if __name__ == "__main__":
    main()
