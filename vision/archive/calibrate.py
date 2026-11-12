r"""
AeroEye-3D — MANUAL pixel -> turret-angle calibration (mouse-only).

Everything is driven by the mouse, so no keyboard focus is needed:
  * Drag the PAN / TILT sliders to move the turret. Watch the red laser dot
    move inside the camera view.
  * LEFT-CLICK exactly on the laser dot to record a point
    -> stores (clicked pixel) <-> (current pan, tilt).
  * RIGHT-CLICK anywhere to undo the last recorded point.
  * Drag the LASER slider (0/1) to toggle the laser.
  * Drag the SAVE slider to 1 to fit the map and write calib.json.
  * Close the window (X) to quit.

Collect 5-6 points spread across the view (corners + middle), then SAVE.

Run:  ..\.venv\Scripts\python.exe calibrate.py [COM3] [cam]
"""
import sys
import time
import json

import cv2
import numpy as np
import serial

import stereo

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM3")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
OUT  = "calib.json"
WIN  = "AeroEye-3D calibrate"

points = []            # (px, py, pan, tilt)
_ser = None
pan = tilt = laser = None


def send(pan, tilt, laser):
    _ser.write(f"{pan},{tilt},{int(laser)}\n".encode())
    _ser.readline()


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        p = cv2.getTrackbarPos("pan", WIN)
        t = cv2.getTrackbarPos("tilt", WIN)
        points.append([x, y, p, t])
        print(f"recorded pt {len(points)}: pixel=({x},{y}) angle=(pan{p},tilt{t})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            print("undo", points.pop())


def fit_and_save():
    if len(points) < 4:
        print(f"need >=4 points, have {len(points)}"); return
    P = np.array(points, float)
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    pan_c,  *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    tilt_c, *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
    ep = np.mean(np.abs(A @ pan_c  - P[:, 2]))
    et = np.mean(np.abs(A @ tilt_c - P[:, 3]))
    json.dump({"model": "affine", "pan": pan_c.tolist(), "tilt": tilt_c.tolist(),
               "points": points, "n": len(points)}, open(OUT, "w"), indent=2)
    print(f"\nSaved {OUT} ({len(points)} pts)  fit error pan~{ep:.1f} tilt~{et:.1f} deg")


def main():
    global _ser
    _ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); _ser.reset_input_buffer()

    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)
    cv2.createTrackbar("pan",   WIN, 90, 165, lambda v: None)
    cv2.setTrackbarMin("pan",   WIN, 15)
    cv2.createTrackbar("tilt",  WIN, 20, 150, lambda v: None)
    cv2.setTrackbarMin("tilt",  WIN, 3)
    cv2.createTrackbar("laser", WIN, 1, 1, lambda v: None)
    cv2.createTrackbar("SAVE",  WIN, 0, 1, lambda v: None)
    cv2.setMouseCallback(WIN, on_mouse)
    print(__doc__)

    last = (None, None, None)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray, _ = stereo.split_lr(frame)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        p = cv2.getTrackbarPos("pan", WIN)
        t = cv2.getTrackbarPos("tilt", WIN)
        l = cv2.getTrackbarPos("laser", WIN)
        if (p, t, l) != last:
            send(p, t, l); last = (p, t, l)

        if cv2.getTrackbarPos("SAVE", WIN) == 1:
            fit_and_save()
            cv2.setTrackbarPos("SAVE", WIN, 0)

        for (px, py, *_a) in points:
            cv2.drawMarker(disp, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(disp, f"pan={p} tilt={t} laser={l}  pts={len(points)}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(disp, "L-click=record  R-click=undo  SAVE slider=fit",
                    (8, gray.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.imshow(WIN, disp)

        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release(); cv2.destroyAllWindows(); _ser.close()


if __name__ == "__main__":
    main()
