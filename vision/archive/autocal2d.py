r"""
AeroEye-3D — automatic 2D calibration (laser co-located with camera).

Now that the laser/turret sits right next to the camera lens, parallax is
negligible and the pixel->angle map no longer depends on distance. So we fit a
simple 2D map from ONE automatic sweep:
    pan  = a*x + b*y + c
    tilt = d*x + e*y + f

The sweep commands a grid of (pan, tilt); at each pose it grabs laser-OFF and
laser-ON frames of the LEFT eye, subtracts, and takes the brightest difference
as the laser dot. No human aiming error. Saves calib2d.json.

    ..\.venv\Scripts\python.exe autocal2d.py [COM3] [cam]
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
OUT  = "calib2d.json"
WIN  = "AeroEye-3D autocal 2D (live)"

PANS  = list(range(45, 100, 3))    # dense sweep over the window the camera sees
TILTS = list(range(3, 49, 3))
SETTLE_MOVE = 0.45
SETTLE_LASER = 0.15
DIFF_THRESH = 12


def grab_left(cap, n=4):
    acc, got = None, 0
    for _ in range(40):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        l, _ = stereo.split_lr(f)
        l = l.astype(np.float32)
        acc = l if acc is None else acc + l
        got += 1
        if got >= n:
            break
    return acc / got


def cmd(ser, p, t, l):
    ser.write(f"{p},{t},{l}\n".encode()); time.sleep(0.03); ser.readline()


def detect_pose(cap, ser, p, t):
    cmd(ser, p, t, 0); time.sleep(SETTLE_MOVE)
    off = grab_left(cap)
    cmd(ser, p, t, 1); time.sleep(SETTLE_LASER)
    on = grab_left(cap)
    d = cv2.GaussianBlur(np.clip(on - off, 0, 255).astype(np.uint8), (5, 5), 0)
    _, maxv, _, loc = cv2.minMaxLoc(d)
    valid = maxv >= DIFF_THRESH

    vis = cv2.cvtColor(on.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    col = (0, 255, 0) if valid else (0, 0, 255)
    if valid:
        cv2.drawMarker(vis, loc, col, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(vis, f"pan={p} tilt={t} diff={maxv:.0f} {'HIT' if valid else 'skip'}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.imshow(WIN, cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(1)
    return [loc[0], loc[1], p, t] if valid else None


def fit(points):
    P = np.array(points, float)
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    pc, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    tc, *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
    ep = np.abs(A @ pc - P[:, 2]); et = np.abs(A @ tc - P[:, 3])
    return pc, tc, ep, et


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)

    points = []
    print(f"Sweeping {len(PANS)}x{len(TILTS)} poses...")
    for t in TILTS:
        hits = 0
        for p in PANS:
            r = detect_pose(cap, ser, p, t)
            if r:
                points.append(r); hits += 1
        print(f"  tilt={t:3d}: {hits} hits")
    cmd(ser, 90, 20, 0)
    cap.release(); ser.close(); cv2.destroyAllWindows()

    # Reject the "stuck blob" (laser-module glow): a real wall dot lands on a
    # different pixel for each pose, so any pixel cell hit many times is glow.
    from collections import Counter
    cells = Counter((p[0] // 8, p[1] // 8) for p in points)
    points = [p for p in points if cells[(p[0] // 8, p[1] // 8)] <= 3]

    print(f"\ntotal points after glow-filter: {len(points)}")
    if len(points) < 6:
        print("not enough points — make sure the laser lands where the camera looks")
        return

    # iterative outlier cleaning
    P = list(points)
    for _ in range(40):
        pc, tc, ep, et = fit(P)
        if max(ep.max(), et.max()) <= 2.5 or len(P) <= 15:
            break
        P.pop(int(np.argmax(ep + et)))
    pc, tc, ep, et = fit(P)
    json.dump({"model": "affine2d", "pan": pc.tolist(), "tilt": tc.tolist(),
               "points": P, "n": len(P)}, open(OUT, "w"), indent=2)
    print(f"kept {len(P)}/{len(points)} | PAN mean={ep.mean():.2f} max={ep.max():.1f} | "
          f"TILT mean={et.mean():.2f} max={et.max():.1f} deg")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
