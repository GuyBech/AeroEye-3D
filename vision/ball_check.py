r"""
AeroEye-3D — BALL CHECK. Measures how well the current detector sees a given
ball, across the working volume. Vision only: no serial, no servos, no laser.

Two things decide aim accuracy, and both are measured here:

  1. Does the detector find the WHOLE ball, or only part of it (e.g. its
     shadowed side)? A partial find puts the centre off the ball centre, and
     the offset changes with lighting, height and distance.
     Test: the image radius r and the disparity d are both proportional to
     1/distance, so r/d must stay constant wherever the ball is. A scattered
     r/d means the detector is grabbing different parts of the ball.

  2. Is the disparity (depth) right? Each eye is detected independently, so
     the two eyes can lock onto different parts of the ball. Compared here
     against a template match: the left-eye ball patch searched along the same
     rows of the right eye. 1 px of disparity error ~ 0.25 deg of pan.

Window: left | right eye, plus the detector's "darkness" map — what the
detector actually sees. A good ball is one solid bright disc in that map.

    ..\.venv\Scripts\python.exe ball_check.py [cam]

SPACE starts a 45 s recording: move the ball slowly near / far / high / low /
left / right. q quits early. The log and sample frames go to
debug_images/check/, and a summary is printed at the end.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import stereo
from detector_dark import DarkBallDetector

CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN = "AeroEye-3D ball check"
OUT = os.path.join("debug_images", "check")
REC_S = 45.0
SAVE_EVERY_S = 1.0
DISP_MIN, DISP_MAX = 3, 70         # same sanity range as the tracker
ROW_TOL = 40
TM_DMAX, TM_DY = 90, 16            # the eyes sit ~8-11 rows apart on this rig


def tm_disparity(left, right, tl):
    """Disparity by finding the left-eye ball patch in the right eye.

    Searches only where the ball can be: the same rows (+/- TM_DY, the eyes
    are not perfectly level) and to the LEFT by DISP_MIN..TM_DMAX px.
    Returns (disparity with sub-pixel x, row offset, match score) or None."""
    H, W = left.shape
    r = max(int(tl.radius * 1.3), 10)
    x0, y0 = max(0, tl.x - r), max(0, tl.y - r)
    x1, y1 = min(W, tl.x + r + 1), min(H, tl.y + r + 1)
    tpl = left[y0:y1, x0:x1]
    sx0, sx1 = max(0, x0 - TM_DMAX), min(W, x1 - DISP_MIN)
    sy0, sy1 = max(0, y0 - TM_DY), min(H, y1 + TM_DY)
    if sx1 - sx0 < tpl.shape[1] or sy1 - sy0 < tpl.shape[0]:
        return None
    res = cv2.matchTemplate(right[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (lx, ly) = cv2.minMaxLoc(res)
    sub = 0.0
    if 0 < lx < res.shape[1] - 1:                   # parabola through the peak
        a, b, c = res[ly, lx - 1], res[ly, lx], res[ly, lx + 1]
        den = a - 2 * b + c
        if den < 0:
            sub = 0.5 * (a - c) / den
    return float(x0 - (sx0 + lx + sub)), int((sy0 + ly) - y0), float(score)


def summarise(log):
    n = len(log)
    if n == 0:
        print("nothing recorded")
        return
    lock = {i for i, e in enumerate(log) if e["L"] and e["R"]
            and DISP_MIN <= e["d_ind"] <= DISP_MAX and e["dy"] <= ROW_TOL}
    print(f"\n==== {n} frames recorded ====")
    print(f"left eye found ball : {sum(1 for e in log if e['L']) / n:6.1%}")
    print(f"right eye found ball: {sum(1 for e in log if e['R']) / n:6.1%}")
    print(f"tracker would LOCK  : {len(lock) / n:6.1%}   (both eyes + sane stereo)")
    tm_set = {i for i, e in enumerate(log) if e["L"] and e["tm"] and e["tm"][2] > 0.6
              and DISP_MIN <= e["tm"][0] <= DISP_MAX}
    tm_ok = [log[i] for i in sorted(tm_set)]
    print(f"lock via template   : {len(tm_set) / n:6.1%}   (left eye + template match)")

    # where it fails — by depth and by height
    print("\nlock rate by region (independent | template), frames:")
    for name, key, bins in (("depth (disp)", "d_any", [(0, 15, "far  <15"), (15, 30, "mid 15-30"), (30, 999, "near >30")]),
                            ("height (y)", "y", [(0, 80, "top"), (80, 160, "middle"), (160, 999, "bottom")])):
        for lo, hi, lab in bins:
            sel = [i for i, e in enumerate(log) if e.get(key) is not None and lo <= e[key] < hi]
            if not sel:
                print(f"  {name:12s} {lab:9s}: no frames")
                continue
            a = sum(1 for i in sel if i in lock) / len(sel)
            b = sum(1 for i in sel if i in tm_set) / len(sel)
            print(f"  {name:12s} {lab:9s}: {a:5.0%} | {b:5.0%}   n={len(sel)}")

    # 1. whole ball?  r = k (d - d0) should hold everywhere
    full = [e for e in tm_ok if not e["edge"]]
    if len(full) >= 8:
        r = np.array([e["L"][2] for e in full], float)
        d = np.array([e["tm"][0] for e in full], float)
        A = np.column_stack([d, np.ones(len(d))])
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        res = r - A @ coef
        print(f"\nradius vs depth: r = {coef[0]:.3f}*d {coef[1]:+.1f}   "
              f"scatter {res.std():.1f} px = {res.std() / r.mean():.0%} of radius")
        print("  (small scatter = the detector finds the whole ball everywhere)")
    # 2. disparity agreement
    pair = [e for e in tm_ok if e["d_ind"] is not None]
    if pair:
        diff = np.array([e["d_ind"] - e["tm"][0] for e in pair])
        print(f"\ndisparity, independent minus template: median {np.median(diff):+.1f} px, "
              f"|diff|>3 px in {np.mean(np.abs(diff) > 3):.0%} of frames, "
              f"worst {diff[np.argmax(np.abs(diff))]:+.1f} px")
        print(f"  (each px ~ 0.25 deg of pan)")
    rej = {}
    for e in log:
        if not e["L"] and e["rejL"]:
            k = e["rejL"].split("[")[0].split("(")[0].strip()[:40]
            rej[k] = rej.get(k, 0) + 1
    if rej:
        print("\nleft-eye reject reasons:")
        for k, v in sorted(rej.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {v:4d}  {k}")


def main():
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    t0 = time.time()
    frame = None
    while time.time() - t0 < 6:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    if frame is None:
        print("camera gives no frames — is another Python window holding it?")
        return
    dl, dr = DarkBallDetector(), DarkBallDetector()
    cv2.namedWindow(WIN)
    print("SPACE = record 45 s (move the ball near/far/high/low/left/right) | q = quit",
          flush=True)

    log, rec_t0, last_save = [], None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tl, tr = dl.detect(left), dr.detect(right)
        tm = tm_disparity(left, right, tl) if tl else None
        d_ind = (tl.x - tr.x) if (tl and tr) else None
        dy = abs(tl.y - tr.y) if (tl and tr) else None

        W = left.shape[1]
        vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
        if tl:
            cv2.circle(vis, (tl.x, tl.y), tl.radius, (0, 255, 0), 1)
        if tr:
            cv2.circle(vis, (W + tr.x, tr.y), tr.radius, (0, 140, 255), 1)
        if tl and tm:
            cv2.circle(vis, (W + int(round(tl.x - tm[0])), tl.y + tm[1]),
                       tl.radius, (255, 0, 255), 1)
        dark = dl.darkness(left)
        dark = cv2.applyColorMap(cv2.normalize(dark, None, 0, 255, cv2.NORM_MINMAX),
                                 cv2.COLORMAP_INFERNO)
        vis = cv2.hconcat([vis, dark])

        now = time.time()
        recording = rec_t0 is not None
        if recording:
            entry = dict(t=round(now - rec_t0, 3),
                         L=[tl.x, tl.y, tl.radius] if tl else None,
                         R=[tr.x, tr.y, tr.radius] if tr else None,
                         d_ind=d_ind, dy=dy,
                         tm=[round(tm[0], 2), tm[1], round(tm[2], 3)] if tm else None,
                         edge=bool(tl and stereo.touches_edge(tl, left.shape)),
                         edgeR=bool(tr and stereo.touches_edge(tr, right.shape)),
                         y=tl.y if tl else None,
                         d_any=(tm[0] if tm else d_ind),
                         rejL=dl.reject)
            log.append(entry)
            if now - last_save >= SAVE_EVERY_S:
                last_save = now
                i = len([f for f in os.listdir(OUT) if f.endswith("_L.png")])
                cv2.imwrite(os.path.join(OUT, f"{i:03d}_L.png"), left)
                cv2.imwrite(os.path.join(OUT, f"{i:03d}_R.png"), right)
            if now - rec_t0 >= REC_S:
                break

        line = (f"r={tl.radius} d_ind={d_ind} d_tm={tm[0]:.1f} match={tm[2]:.2f}"
                if (tl and tm) else (dl.reject or "no ball")[:60])
        state = (f"REC {REC_S - (now - rec_t0):4.1f}s" if recording
                 else "PREVIEW - SPACE to record")
        cv2.putText(vis, state, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255) if recording else (0, 255, 255), 1)
        cv2.putText(vis, line, (8, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow(WIN, cv2.resize(vis, None, fx=1.4, fy=1.4,
                                   interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == 32 and not recording:
            rec_t0, last_save = now, 0.0
            print("recording...", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    summarise(log)
    if log:
        json.dump(log, open(os.path.join(OUT, "log.json"), "w"), default=float)
        print(f"log: {os.path.join(OUT, 'log.json')}")


if __name__ == "__main__":
    main()
