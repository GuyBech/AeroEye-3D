r"""
AeroEye-3D — quick AGC check, no window, no keys.

Records the camera for 30 s. During that time, cover ONE side of the picture
with a dark object (or a hand) close to the lens, 3 times for ~3 s each.
The script finds the covered moments by itself (that side goes dark) and
reports what the OTHER side — plain wall that nothing touched — did meanwhile:

  wall brighter every time  ->  the camera raised its own brightness (AGC)
  wall unchanged            ->  no AGC

A beep marks the start and the end. Only reads frames; nothing on the camera
is changed.

    ..\.venv\Scripts\python.exe agc_quick.py [cam] [seconds]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import stereo

try:
    import winsound
except ImportError:
    winsound = None

args = [a for a in sys.argv[1:] if a.isdigit()]
CAM = int(args[0]) if args else 1
SECONDS = int(args[1]) if len(args) > 1 else 30
ROWS = slice(20, 170)      # wall only — the baseboard and floor start lower down
DARK = 25                  # a side this far below its usual level counts as covered
MIN_SHIFT = 2.0            # grey levels — smallest wall change counted as real
OUT = os.path.join("debug_images", "agc_quick_" + time.strftime("%Y%m%d_%H%M%S") + ".json")


def beep(f, ms):
    if winsound:
        try:
            winsound.Beep(f, ms)
        except RuntimeError:
            pass


def sides(eye):
    """Mean grey of the outer third on each side — the middle third is left out
    so a blurry edge of the cover cannot leak into the watched side."""
    W = eye.shape[1]
    return float(eye[ROWS, :W // 3].mean()), float(eye[ROWS, W - W // 3:].mean())


def runs(mask):
    """(start, end) index pairs of consecutive True stretches."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def analyse(a):
    """a columns: t, left-eye left side, left-eye right side, right-eye left, right-eye right."""
    t = a[:, 0]
    usual = np.median(a[:, 1:5], axis=0)
    covered = a[:, 1:5] < usual - DARK
    # Use the eye where one side was dark while its other side still saw the wall.
    # A cover held close can black out one eye completely; then the other eye is
    # the one to read, so neither the eye nor the side has to be chosen by hand.
    best = max(((e, s) for e in (0, 1) for s in (0, 1)),
               key=lambda es: (covered[:, 2 * es[0] + es[1]]
                               & ~covered[:, 2 * es[0] + 1 - es[1]]).sum())
    eye, side = best
    cov, wall = a[:, 1 + 2 * eye + side], a[:, 2 + 2 * eye - side]
    wall_r = a[:, 2 + 2 * (1 - eye) - side]                    # same side, other eye
    events = []
    for s, e in runs(covered[:, 2 * eye + side] & ~covered[:, 2 * eye + 1 - side]):
        if t[e - 1] - t[s] < 1.0:                              # a brush past, not a cover
            continue
        during = (t >= t[s] + 0.7) & (t <= t[e - 1])           # let the camera settle
        before = (t >= t[s] - 2.0) & (t < t[s] - 0.3) & ~covered[:, side]
        after = (t > t[e - 1] + 1.0) & (t <= t[e - 1] + 3.0) & ~covered[:, side]
        ref = before | after
        if during.sum() < 5 or ref.sum() < 5:
            continue
        events.append({"start_s": round(float(t[s]), 2), "length_s": round(float(t[e - 1] - t[s]), 2),
                       "cover_change": round(float(np.median(cov[during]) - np.median(cov[ref])), 1),
                       "wall_change": round(float(np.median(wall[during]) - np.median(wall[ref])), 2),
                       "wall_change_other_eye": round(float(np.median(wall_r[during])
                                                            - np.median(wall_r[ref])), 2)})
    d = [e["wall_change"] for e in events]
    if len(d) < 2:
        verdict = ("UNCLEAR — found fewer than 2 covers of at least 1 s. Cover one side "
                   "of the camera completely, close to the lens, 3 times for ~3 s.")
    elif all(x > MIN_SHIFT for x in d):
        verdict = ("YES — the wall got brighter every time the other side was covered. "
                   "Nothing adds light to the wall when you cover the lens, so the camera "
                   "raised its own brightness: automatic brightness (AGC) is real.")
    elif all(abs(x) < MIN_SHIFT for x in d):
        verdict = ("NO — the wall stayed the same while the other side was covered. "
                   "The camera does not adjust its brightness by itself.")
    elif all(x < -MIN_SHIFT for x in d):
        verdict = ("UNCLEAR — the wall got DARKER: the cover or your body shaded it. "
                   "Hold the cover right at the lens and stand to the side.")
    else:
        verdict = "UNCLEAR — the wall did different things on different covers. Repeat."
    return {"eye": ["left", "right"][eye], "covered_side": ["left", "right"][side],
            "usual_level": usual.round(1).tolist(),
            "events": events, "verdict": verdict}


def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    rows, t0 = [], time.perf_counter()
    beep(1000, 300)
    print(f"RECORDING {SECONDS} s — cover one side of the camera 3 times, ~3 s each", flush=True)
    while time.perf_counter() - t0 < SECONDS:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        rows.append([round(time.perf_counter() - t0, 3), *sides(left), *sides(right)])
    cap.release()
    beep(700, 600)
    if len(rows) < 50:
        print("CAMERA FAILED — no frames. Is another program using it?", flush=True)
        return
    a = np.array(rows)
    res = analyse(a)
    print(f"\nread from the {res['eye']} eye, covered side: {res['covered_side']}"
          f"   usual levels: {res['usual_level']}")
    print("cover at   length   covered side   wall (other side)   wall, other eye")
    for e in res["events"]:
        print(f"{e['start_s']:7.1f}s  {e['length_s']:5.1f}s   {e['cover_change']:+8.1f}"
              f"        {e['wall_change']:+8.2f}           {e['wall_change_other_eye']:+8.2f}")
    print("\n" + res["verdict"], flush=True)
    os.makedirs("debug_images", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({**res, "columns": ["t", "L_left", "L_right", "R_left", "R_right"],
                   "samples": rows}, f, indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
