r"""
AeroEye-3D — how well did the laser follow the ball in a recorded live run?

Reads a session made with  track_motion_3d.py COM4 1 --record  (left-eye frames
+ log.jsonl) and, in every frame where the laser was on and the ball was seen
whole, finds the laser dot near the ball and measures how far it is from the
ball centre. A hit is a dot inside the ball's circle in the image.

The ball's speed (from consecutive frames) splits the run into still / slow /
fast, which separates the two causes of a miss: calibration (misses even when
still) and lag (the dot trails behind the ball only while it moves). For moving
frames the error is split into the part ALONG the motion (positive = the dot
is behind the ball, i.e. lag) and the part across it.

    ..\.venv\Scripts\python.exe live_report.py                     # newest session
    ..\.venv\Scripts\python.exe live_report.py debug_images\live_20261017_220000
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

DOT_MIN = 40          # the dot must beat the local median brightness by this
STILL, FAST = 15, 80  # px/s: below STILL the ball counts as still, above FAST as fast
VEL_WIN = 0.2         # s: half-width of the window the ball's velocity is fitted over
COLS, ROWS = ("left", "centre", "right"), ("top", "middle", "bottom")
DEPTH_EDGES = (12, 22)
DEPTH_NAMES = ("far", "mid", "near")


def newest():
    runs = sorted(glob.glob(os.path.join("debug_images", "live_*")))
    return runs[-1] if runs else None


def touches_edge(b, shape):
    x, y, r = b
    H, W = shape[:2]
    return x - r < 2 or y - r < 2 or x + r > W - 3 or y + r > H - 3


def find_dot(gray, x, y, r):
    """Brightest compact spot within 4 ball radii of the ball."""
    pad = int(max(r, 10) * 4)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    sub = gray[y0:min(gray.shape[0], y + pad), x0:min(gray.shape[1], x + pad)]
    if sub.size == 0:
        return None
    _, mv, _, loc = cv2.minMaxLoc(cv2.GaussianBlur(sub, (3, 3), 0))
    if mv < float(np.median(sub)) + DOT_MIN:
        return None
    return x0 + loc[0], y0 + loc[1]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest()
    if not run or not os.path.exists(os.path.join(run, "log.jsonl")):
        print("no recorded session found - run track_motion_3d.py COM4 1 --record")
        return
    log = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"), encoding="utf-8")]
    dur = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 0
    seen = [e for e in log if e["L"]]
    print(f"session {os.path.basename(run)}: {len(log)} frames kept over {dur:.0f} s")
    print(f"ball seen in {len(seen)}/{len(log)} frames, locked in "
          f"{sum(e['locked'] for e in log)}, laser on in {sum(e['laser'] for e in log)}")

    try:
        cfg = json.load(open(os.path.join(run, "settings.json")))
        fast = (f" -> {cfg['LEAD_FAST_S']} s when fast ({cfg['FAST_RAMP'][0]:.0f}-"
                f"{cfg['FAST_RAMP'][1]:.0f} deg/s)" if "LEAD_FAST_S" in cfg else "")
        print(f"settings: look-ahead pan {cfg['LEAD_S']} s{fast}, tilt {cfg['LEAD_TILT_S']} s")
    except (FileNotFoundError, KeyError):
        pass

    # ball velocity: straight-line fit of the left-eye centre over +-VEL_WIN s.
    # Two neighbouring frames alone are too noisy to tell "still" from "slow".
    seen_t = np.array([e["t"] for e in seen])
    seen_xy = np.array([e["L"][:2] for e in seen], float)
    speed = {}
    for k, e in enumerate(seen):
        w = np.abs(seen_t - e["t"]) <= VEL_WIN
        if w.sum() >= 3 and np.ptp(seen_t[w]) > 0:
            speed[e["i"]] = np.array([np.polyfit(seen_t[w], seen_xy[w, a], 1)[0] for a in (0, 1)])

    rows, shots = [], []
    for e in log:
        if not (e["laser"] and e["L"] and e["i"] in speed):
            continue
        img = cv2.imread(os.path.join(run, f"{e['i']:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or touches_edge(e["L"], img.shape):
            continue
        x, y, r = e["L"]
        dot = find_dot(img, x, y, r)
        v = speed[e["i"]]
        sp = float(np.hypot(*v))
        disp = (e["L"][0] - e["R"][0]) if e["R"] else None
        H, W = img.shape[:2]
        if sp < STILL:
            way = None
        elif abs(v[0]) >= abs(v[1]):
            way = "right" if v[0] > 0 else "left"
        else:
            way = "down" if v[1] > 0 else "up"
        row = dict(i=e["i"], speed=sp, r=r, disp=disp, dot=dot, way=way,
                   cell=(min(2, 3 * y // H), min(2, 3 * x // W)))
        if dot is not None:
            err = np.array([x - dot[0], y - dot[1]], float)        # ball - dot
            row["dist"] = float(np.hypot(*err))
            row["hit"] = row["dist"] <= r
            if sp >= STILL:
                u = v / sp
                row["along"] = float(err @ u)          # + = dot behind the ball
                row["across"] = float(abs(err[0] * u[1] - err[1] * u[0]))
        rows.append(row)
        shots.append((e, img, dot))

    if not rows:
        print("no frames with the laser on and the ball fully in view")
        return

    def summary(sel, name):
        if not sel:
            print(f"  {name:22s}   -")
            return
        dots = [r for r in sel if r["dot"] is not None]
        hits = sum(r.get("hit", False) for r in dots)
        med = np.median([r["dist"] / r["r"] for r in dots]) if dots else float("nan")
        print(f"  {name:22s} {len(sel):4d} frames | hits {hits:4d} = {hits / len(sel):4.0%}"
              f" | of frames with a dot {hits / max(1, len(dots)):4.0%}"
              f" | median miss {med:4.2f} ball radii | dot not found {len(sel) - len(dots)}")

    print("\nlaser on the ball (dot inside the ball's circle):")
    summary(rows, "all frames")
    summary([r for r in rows if r["speed"] < STILL], f"still (<{STILL} px/s)")
    summary([r for r in rows if STILL <= r["speed"] < FAST], "slow")
    summary([r for r in rows if r["speed"] >= FAST], f"fast (>{FAST} px/s)")
    for k, nm in enumerate(DEPTH_NAMES):
        lo = DEPTH_EDGES[k - 1] if k else -1
        hi = DEPTH_EDGES[k] if k < 2 else 1e9
        summary([r for r in rows if r["disp"] is not None and lo <= r["disp"] < hi], f"depth {nm}")
    print("by direction of motion (in the image):")
    for way in ("left", "right", "up", "down"):
        summary([r for r in rows if r["way"] == way], f"moving {way}")

    # where in the image: the edges are where an examiner will push it
    print("\nby place in the image: hits of frames with a dot (frames)")
    print("          " + "".join(f"{c:>14s}" for c in COLS))
    for a, rn in enumerate(ROWS):
        line = f"  {rn:8s}"
        for b in range(3):
            sel = [r for r in rows if r["cell"] == (a, b) and r["dot"] is not None]
            line += (f"{sum(r['hit'] for r in sel) / len(sel):9.0%} ({len(sel):3d})" if sel
                     else f"{'-':>14s}")
        print(line)

    mv = [r for r in rows if "along" in r]
    if mv:
        al = np.array([r["along"] / r["r"] for r in mv])
        ac = np.array([r["across"] / r["r"] for r in mv])
        print(f"\nwhile moving ({len(mv)} frames): dot is behind the ball by a median "
              f"{np.median(al):+.2f} ball radii along the motion, "
              f"{np.median(ac):.2f} across it")
        print("  (a clearly positive 'behind' value is lag -> a larger look-ahead would help;"
              " near zero means the misses are calibration, not lag)")

    # contact sheet: up to 8 hits and 8 misses
    pick = ([s for s, r in zip(shots, rows) if r.get("hit")][:8] +
            [s for s, r in zip(shots, rows) if not r.get("hit")][:8])
    tiles = []
    for e, img, dot in pick:
        t = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x, y, r = e["L"]
        cv2.circle(t, (x, y), max(r, 6), (0, 255, 0), 1)
        if dot:
            cv2.drawMarker(t, dot, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(t, f"#{e['i']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = cv2.vconcat([cv2.hconcat(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
    out = os.path.join(run, "report_sheet.png")
    cv2.imwrite(out, sheet)
    print(f"\nexample frames (hits first, then misses): {out}")


if __name__ == "__main__":
    main()
