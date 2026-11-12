r"""
AeroEye-3D — ground-truth capture for detector tuning (vision only, no servos).

Saves what the pipeline actually sees, so the detector can be tuned against real
images instead of aggregate statistics. Writes to debug_images/ball/:

    NN_raw_left.png    NN_raw_right.png   the eyes as captured
    NN_mask.png        the reference-difference mask the detector thresholds
    NN_panel.png       raw | mask | candidates, side by side with the numbers

Timeline (no keyboard needed — it runs unattended so both hands are free):
    0-2 s    settle, then build the reference   -> KEEP THE SCENE EMPTY
    2-6 s    grace period                       -> bring the ball in now
    6-26 s   captures a frame every ~1.3 s      -> let the ball swing

    ..\.venv\Scripts\python.exe capture_ball.py [cam]
"""
import os
import sys
import time

import cv2
import numpy as np

import stereo
from detector_ref import RefDetector, MIN_AREA, MAX_AREA, MIN_CIRC, MIN_FILL

CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
OUT = os.path.join("debug_images", "ball")
SETTLE_S, GRACE_S, SHOTS, SHOT_DT = 2.0, 4.0, 15, 1.3


def panel(left, mask, det):
    """raw | mask | every candidate annotated with its measurements."""
    vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 40:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        (x, y), r = cv2.minEnclosingCircle(c)
        fill = area / (np.pi * r * r) if r > 0 else 0.0
        ok = (MIN_AREA <= area <= MAX_AREA and circ >= MIN_CIRC and fill >= MIN_FILL)
        col = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(vis, (int(x), int(y)), max(int(r), 3), col, 2)
        cv2.putText(vis, f"a{int(area)} c{circ:.2f} f{fill:.2f}",
                    (int(x) - 45, int(y) - int(r) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)
    if det:
        cv2.drawMarker(vis, (det.x, det.y), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    return cv2.hconcat([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis])


def main():
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.2)
    if not cap.read()[0]:
        print("CAMERA FAILED — another process is probably holding it.", flush=True)
        return

    dl, dr = RefDetector(), RefDetector()
    print(f"[{time.strftime('%H:%M:%S')}] settling — KEEP THE SCENE EMPTY", flush=True)
    t0 = time.time()
    while time.time() - t0 < SETTLE_S:
        cap.read()
    while not dl.ready:
        ok, f = cap.read()
        if not ok:
            continue
        l, r = stereo.split_lr(f)
        dl.accumulate_reference(l)
        dr.accumulate_reference(r)
    print(f"reference built (mean L={dl.ref.mean():.0f} R={dr.ref.mean():.0f})", flush=True)
    print(f"BRING THE BALL IN — capturing {SHOTS} shots starting in {GRACE_S:.0f}s",
          flush=True)

    t0 = time.time()
    while time.time() - t0 < GRACE_S:
        cap.read()

    for i in range(SHOTS):
        t1 = time.time()
        ok, f = cap.read()
        if not ok or f is None:
            continue
        left, right = stereo.split_lr(f)
        mask = dl.diff_mask(left)
        det = dl.detect(left)
        detr = dr.detect(right)

        cv2.imwrite(f"{OUT}/{i:02d}_raw_left.png", left)
        cv2.imwrite(f"{OUT}/{i:02d}_raw_right.png", right)
        cv2.imwrite(f"{OUT}/{i:02d}_mask.png", mask)
        cv2.imwrite(f"{OUT}/{i:02d}_panel.png", panel(left, mask, det))

        px = f"L=({det.x},{det.y})a{int(det.area)}" if det else f"L=- rej[{dl.reject}]"
        pr = f"R=({detr.x},{detr.y})" if detr else "R=-"
        dsp = f" disp={det.x - detr.x}" if (det and detr) else ""
        print(f"  shot {i:02d}: {px} {pr}{dsp}  mask_on={int(mask.sum() / 255)}px",
              flush=True)
        while time.time() - t1 < SHOT_DT:
            cap.read()

    cap.release()
    print(f"DONE — {SHOTS} shots in {OUT}", flush=True)


if __name__ == "__main__":
    main()
