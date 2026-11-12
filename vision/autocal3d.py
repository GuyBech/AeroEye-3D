r"""
AeroEye-3D — automatic depth-aware calibration, v2 (dense + quadratic).

Sweeps a dense grid of (pan, tilt); at each pose diffs laser-OFF/ON in BOTH
eyes to find the dot -> pixel + disparity with no human error. Accumulates
points across runs (one run per board distance), then fits a quadratic
depth-aware map:
    angle = f(x, y, disp, x*y, x^2, y^2)
which handles residual parallax (the disp term) AND lens distortion (the
quadratic terms). Saves calib3d.json.

Run once PER board distance:
    ..\.venv\Scripts\python.exe autocal3d.py [COM4] [cam]
    ..\.venv\Scripts\python.exe autocal3d.py --reset      # start fresh

--reset snapshots the whole active calibration into calib_backups/ first (see
calib_snapshot.py), then clears the accumulated points AND sets the ball
correction aside: aim_correction.json is a patch fitted relative to the OLD
board model, and on top of a new one it would be silently wrong. Re-run the
ball calibration after the sweep; until then the tracker uses the board model
alone.

The sweep range covers the whole frame. The July sweep used pan 45-97 and
tilt 3-47, which never reached the right side of the image (pan < 45) or its
top, so every target there was aimed by extrapolation — and the live runs
missed consistently to the right. Poses whose dot lands outside the frame or
off the board are simply skipped.
"""
import sys
import os
import time
import json

import cv2
import numpy as np
import serial

import stereo
import calib_snapshot

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
PTS  = "autocal_points.json"
OUT  = "calib3d.json"
CORR = "aim_correction.json"
WIN  = "AeroEye-3D autocal3D (live)"

PANS  = list(range(25, 106, 4))      # was 45..97: the right side was never swept
TILTS = list(range(3, 56, 4))        # was 3..47
SETTLE_MOVE = 0.45
SETTLE_LASER = 0.15
DIFF_THRESH = 12
DISP_MIN, DISP_MAX = 2, 220
ROW_TOL = 40


def grab(cap, n=4):
    accl = accr = None; got = 0
    for _ in range(40):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        l, r = stereo.split_lr(f)
        l = l.astype(np.float32); r = r.astype(np.float32)
        accl = l if accl is None else accl + l
        accr = r if accr is None else accr + r
        got += 1
        if got >= n:
            break
    return accl / got, accr / got


def cmd(ser, p, t, l):
    ser.write(f"{p},{t},{l}\n".encode()); time.sleep(0.03); ser.readline()


def dot(diff, on):
    """Find the laser dot: the diff локates the change region, then we snap to
    the brightest ON pixel nearby - the PRIMARY dot, not its optical ghost
    (cheap laser lenses cast a dim ghost ~20px away that can win the diff)."""
    d = cv2.GaussianBlur(np.clip(diff, 0, 255).astype(np.uint8), (5, 5), 0)
    _, mv, _, loc = cv2.minMaxLoc(d)
    x0, y0 = loc
    w = 25
    sub = on[max(0, y0 - w):y0 + w, max(0, x0 - w):x0 + w]
    _, _, _, sloc = cv2.minMaxLoc(cv2.GaussianBlur(sub.astype(np.uint8), (3, 3), 0))
    return mv, (max(0, x0 - w) + sloc[0], max(0, y0 - w) + sloc[1])


def detect_pose(cap, ser, p, t):
    cmd(ser, p, t, 0); time.sleep(SETTLE_MOVE)
    off_l, off_r = grab(cap)
    cmd(ser, p, t, 1); time.sleep(SETTLE_LASER)
    on_l, on_r = grab(cap)
    vl, ll = dot(on_l - off_l, on_l)
    vr, lr = dot(on_r - off_r, on_r)
    disp = ll[0] - lr[0]
    dy = abs(ll[1] - lr[1])
    valid = (vl >= DIFF_THRESH and vr >= DIFF_THRESH
             and DISP_MIN <= disp <= DISP_MAX and dy <= ROW_TOL)

    vis = cv2.cvtColor(cv2.hconcat([on_l.astype(np.uint8), on_r.astype(np.uint8)]),
                       cv2.COLOR_GRAY2BGR)
    col = (0, 255, 0) if valid else (0, 0, 255)
    cv2.drawMarker(vis, ll, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.drawMarker(vis, (on_l.shape[1] + lr[0], lr[1]), (0, 128, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(vis, f"pan={p} tilt={t} disp={disp} {'HIT' if valid else 'skip'}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.imshow(WIN, cv2.resize(vis, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(1)
    return [ll[0], ll[1], disp, p, t] if valid else None


def feats(P):
    x, y, d = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack([x, y, d, x * y, x * x, y * y, np.ones(len(P))])


def fit(points):
    P = np.array(points, float)
    A = feats(P)
    pc, *_ = np.linalg.lstsq(A, P[:, 3], rcond=None)
    tc, *_ = np.linalg.lstsq(A, P[:, 4], rcond=None)
    ep = np.abs(A @ pc - P[:, 3]); et = np.abs(A @ tc - P[:, 4])
    return pc, tc, ep, et


def main():
    if "--reset" in sys.argv:
        calib_snapshot.save("before_board_reset")
        if os.path.exists(PTS):
            os.remove(PTS); print("cleared accumulated points")
        if os.path.exists(CORR):
            os.remove(CORR)
            print(f"set {CORR} aside (it is in the snapshot) - re-run the ball "
                  f"calibration after the sweep")

    points = json.load(open(PTS)) if os.path.exists(PTS) else []
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)

    run_pts = []
    n = len(PANS) * len(TILTS)
    print(f"Sweeping {len(PANS)}x{len(TILTS)} = {n} poses at this distance "
          f"(about {n * 0.9 / 60:.0f} min)...")
    for t in TILTS:
        hits = 0
        for p in PANS:
            r = detect_pose(cap, ser, p, t)
            if r:
                run_pts.append(r); hits += 1
        print(f"  tilt={t:3d}: {hits} hits", flush=True)
    cmd(ser, 72, 25, 0)
    cap.release(); ser.close(); cv2.destroyAllWindows()

    # glow filter: a real dot lands on a different pixel each pose
    from collections import Counter
    json.dump(run_pts, open("autocal_debug_raw.json", "w"))
    cells = Counter((p[0] // 8, p[1] // 8) for p in run_pts)
    print("top repeated cells:", cells.most_common(5))
    run_pts = [p for p in run_pts if cells[(p[0] // 8, p[1] // 8)] <= 3]

    points += run_pts
    json.dump(points, open(PTS, "w"))
    print(f"\nthis run: {len(run_pts)} clean points | total {len(points)}")
    if len(points) < 12:
        print("collect more before fitting"); return

    # outlier cleaning + fit
    P = list(points)
    for _ in range(len(P) // 3):
        pc, tc, ep, et = fit(P)
        if max(ep.max(), et.max()) <= 2.5 or len(P) <= 30:
            break
        P.pop(int(np.argmax(ep + et)))
    pc, tc, ep, et = fit(P)
    D = np.array(P, float)[:, 2]
    json.dump({"model": "quad3d", "pan": pc.tolist(), "tilt": tc.tolist(),
               "points": P, "n": len(P)}, open(OUT, "w"), indent=2)
    print(f"kept {len(P)}/{len(points)} | disp coverage {int(D.min())}..{int(D.max())}")
    print(f"PAN mean={ep.mean():.2f} max={ep.max():.1f} | TILT mean={et.mean():.2f} max={et.max():.1f} deg")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
