r"""
AeroEye-3D — automatic pixel<->angle calibration via laser-dot differencing.

Sweeps the turret over a grid of (pan, tilt). At each pose it captures a
laser-OFF and a laser-ON frame of the stereo LEFT eye, subtracts them, and
takes the brightest spot of the difference as the laser dot. Poses where the
dot is clearly visible (diff >= THRESH) become calibration points. With >=4
of them it fits an affine map pixel->angle and saves calib.json.

First run it as a SCAN to see where the dot is visible and how strong it is:
    ..\.venv\Scripts\python.exe autocal.py [COM3] [cam]
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

PANS  = list(range(40, 141, 20))     # 40,60,80,100,120,140
TILTS = [10, 25, 40, 55]
THRESH = 18                          # min diff brightness to count the dot as visible


def grab_left(cap, n=6):
    frame = None
    got = 0
    # keep reading until we have a few good frames (or give up after ~2s)
    for _ in range(60):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            got += 1
            if got >= n:
                break
        time.sleep(0.03)
    if frame is None:
        raise RuntimeError("camera returned no frames — check the USB connection")
    left, _ = stereo.split_lr(frame)
    return left.astype(np.int16)


def command(ser, pan, tilt, laser):
    ser.write(f"{pan},{tilt},{laser}\n".encode())
    time.sleep(0.05)
    ser.readline()


def detect_dot(cap, ser, pan, tilt):
    command(ser, pan, tilt, 0); time.sleep(0.25)
    off = grab_left(cap)
    command(ser, pan, tilt, 1); time.sleep(0.25)
    on = grab_left(cap)
    diff = cv2.GaussianBlur(np.clip(on - off, 0, 255).astype(np.uint8), (9, 9), 0)
    _, maxv, _, maxloc = cv2.minMaxLoc(diff)
    return maxv, maxloc


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)

    print(f"Scanning {len(PANS)}x{len(TILTS)} poses (thresh={THRESH})...\n")
    pts = []
    for tilt in TILTS:
        row = []
        for pan in PANS:
            maxv, loc = detect_dot(cap, ser, pan, tilt)
            hit = maxv >= THRESH
            row.append(f"p{pan}:{maxv:3.0f}{'*' if hit else ' '}")
            if hit:
                pts.append([loc[0], loc[1], pan, tilt])
        print(f"tilt={tilt:3d} | " + "  ".join(row))

    command(ser, 90, 20, 0)   # park, laser off
    print(f"\nVisible points (diff>={THRESH}): {len(pts)}")
    for p in pts:
        print(f"  pixel=({p[0]},{p[1]})  angle=(pan{p[2]},tilt{p[3]})")

    if len(pts) >= 4:
        P = np.array(pts, float)
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        pan_c,  *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        tilt_c, *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
        err_p = np.mean(np.abs(A @ pan_c  - P[:, 2]))
        err_t = np.mean(np.abs(A @ tilt_c - P[:, 3]))
        json.dump({"model": "affine", "pan": pan_c.tolist(), "tilt": tilt_c.tolist(),
                   "points": pts, "n": len(pts)}, open(OUT, "w"), indent=2)
        print(f"\nSaved {OUT}  fit error pan~{err_p:.1f} tilt~{err_t:.1f} deg")
    else:
        print("\nNot enough visible points to fit. Aim the camera so it sees where the laser lands.")

    cap.release(); ser.close()


if __name__ == "__main__":
    main()
