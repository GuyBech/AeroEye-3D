r"""
AeroEye-3D — what the static aim error looks like, from aim_check.py samples.

The shape of the error decides the fix, so this reports the shape:

  constant offset      the whole map is shifted by the same amount. Something
                       moved since calibration; a fresh ball calibration
                       fixes it.
  depends on position  e.g. worse on the right or at the bottom. The model is
                       extrapolating there; re-sweep the board over the full
                       frame (autocal3d.py), then recalibrate the ball.
  depends on depth     near and far disagree: the parallax term. More board
                       distances / ball points near and far.
  scattered            no pattern left: detection noise, backlash, the dot
                       landing on the wall behind the ball.

Errors are measured in the LEFT eye image: err = ball centre - laser dot, in
pixels, so err_x > 0 means the dot sits LEFT of the ball in the image. A hit is
a dot within one ball radius of the centre. Degrees are approximate (local
slope of the board model); when the dot misses and lands on the wall behind
the ball the pixel offset overstates the angular miss.

    ..\.venv\Scripts\python.exe aim_report.py                    # aim_error.json
    ..\.venv\Scripts\python.exe aim_report.py aim_error_before.json aim_error_after.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

from calib_ball import DEPTH_NAMES, cell, jacobian

EYE = (240, 320)                      # (H, W) of one eye - what aim_check measures in
COLS = ("left", "centre", "right")
ROWS = ("top", "middle", "bottom")


def load(path):
    S = json.load(open(path))
    return [s for s in S if "err_x" in s]


def arrays(S):
    g = lambda k: np.array([s[k] for s in S], float)
    x, y, d = g("ball_x"), g("ball_y"), g("disp")
    ex, ey, r = g("err_x"), g("err_y"), g("radius")
    jac = np.array([jacobian(a, b) for a, b in zip(x, y)])
    return x, y, d, ex, ey, r, np.abs(jac[:, 0]), np.abs(jac[:, 1])


def side(v, pos, neg):
    return pos if v > 0 else neg


def summary(S, name):
    x, y, d, ex, ey, r, kx, ky = arrays(S)
    n = len(S)
    dist = np.hypot(ex, ey)
    hit = dist <= r
    deg = np.hypot(ex * kx, ey * ky)
    manual = sum(s.get("dot_source") == "manual" for s in S)
    print(f"\n=== {name}: {n} samples ({manual} with a hand-marked dot) ===")
    print(f"hits (dot within one ball radius): {hit.sum()}/{n} = {hit.mean():.0%}")
    print(f"miss distance: median {np.median(dist):.1f} px = {np.median(dist / r):.2f} "
          f"ball radii | approx {np.sqrt((deg ** 2).mean()):.2f} deg rms")

    mx, my = ex.mean(), ey.mean()
    print(f"average offset: dot sits {abs(mx):.1f} px "
          f"{side(mx, 'LEFT', 'RIGHT')} and {abs(my):.1f} px "
          f"{side(my, 'ABOVE', 'BELOW')} the ball "
          f"(~{abs(mx) * kx.mean():.2f} deg pan, {abs(my) * ky.mean():.2f} deg tilt)")
    return dict(n=n, hit=hit, dist=dist, deg=deg)


def shape_of_error(S):
    """How much of the error is a constant shift, how much follows x / y / depth."""
    x, y, d, ex, ey, r, kx, ky = arrays(S)
    n = len(S)
    tot = (ex ** 2 + ey ** 2).mean()
    if tot == 0:
        return "no error"
    after_const = ((ex - ex.mean()) ** 2 + (ey - ey.mean()) ** 2).mean()
    const_share = 1 - after_const / tot
    print(f"\nshape of the error:")
    print(f"  a constant shift explains {const_share:.0%} of it")
    if n < 8:
        print(f"  (only {n} samples - at least 8 are needed to test position/depth)")
        return "constant" if const_share >= 0.6 else "too few samples"

    A = np.column_stack([np.ones(n), x, y, d])
    resid = 0.0
    effects = []
    for e, axis in ((ex, "horizontal"), (ey, "vertical")):
        c, *_ = np.linalg.lstsq(A, e, rcond=None)
        resid += ((e - A @ c) ** 2).mean()
        for k, (v, what) in enumerate(((x, "left->right"), (y, "top->bottom"),
                                       (d, "far->near")), start=1):
            span = c[k] * (v.max() - v.min())          # px change across the samples
            effects.append((abs(span), span, axis, what))
    lin_share = 1 - resid / after_const if after_const > 0 else 0.0
    print(f"  of the rest, position + depth explain {lin_share:.0%}")
    effects.sort(reverse=True)
    for mag, span, axis, what in effects[:3]:
        print(f"    {axis} error changes by {span:+.1f} px going {what}")

    if const_share >= 0.6:
        verdict = "constant"
    elif lin_share >= 0.5:
        verdict = "depth" if "far->near" in effects[0][3] else "position"
    else:
        verdict = "scattered"
    return verdict


def regions(S):
    groups = {"depth": defaultdict(list), "column": defaultdict(list),
              "row": defaultdict(list)}
    cells = defaultdict(lambda: [0, 0])
    for s in S:
        di, ci, ri = cell(s["ball_x"], s["ball_y"], s["disp"], EYE)
        h = np.hypot(s["err_x"], s["err_y"]) <= s["radius"]
        for g, k in (("depth", DEPTH_NAMES[di]), ("column", COLS[ci]), ("row", ROWS[ri])):
            groups[g][k].append(s)
        cells[(di, ci, ri)][0] += 1
        cells[(di, ci, ri)][1] += h

    print("\nby region          n   hits   mean err_x  mean err_y  (px; + = dot left / above)")
    for g, names in (("depth", DEPTH_NAMES), ("column", COLS), ("row", ROWS)):
        for k in names:
            L = groups[g][k]
            if not L:
                print(f"  {g:6s} {k:7s}   0      -")
                continue
            ex = np.mean([s["err_x"] for s in L]); ey = np.mean([s["err_y"] for s in L])
            h = sum(np.hypot(s["err_x"], s["err_y"]) <= s["radius"] for s in L)
            print(f"  {g:6s} {k:7s} {len(L):3d}  {h:3d}/{len(L):<3d}  {ex:+9.1f}  {ey:+10.1f}")

    print(f"\ncoverage: {len(cells)}/27 cells   (hits/samples; columns left..right, rows top..bottom)")
    for di, name in enumerate(DEPTH_NAMES):
        rows = []
        for ri in range(3):
            rows.append(" ".join(f"{cells[(di, ci, ri)][1]}/{cells[(di, ci, ri)][0]}"
                                 if cells[(di, ci, ri)][0] else " - "
                                 for ci in range(3)))
        print(f"  {name:4s} | " + " | ".join(rows))


ADVICE = {
    "constant": "Mostly a constant shift: something moved since calibration. "
                "Recalibrate on the ball (calib_ball.py --reset, all 27 cells), "
                "then fit_ball_calib.py.",
    "position": "The error follows the position in the frame: the model is "
                "extrapolating there. Re-sweep the board over the full frame "
                "(autocal3d.py --reset, several distances), then recalibrate the ball.",
    "depth": "The error follows the depth: the parallax term is off. Re-sweep the "
             "board at more distances (near AND far), then recalibrate the ball.",
    "scattered": "No simple pattern is left (constant, or linear in position or "
                 "depth). What remains is noise - detection, backlash, dots landing "
                 "on the wall behind the ball - or a non-linear shape that only more "
                 "points or the physical model would capture. Check the cells that miss.",
    "too few samples": "Too few samples to tell - aim for 20+, spread over all cells.",
    "no error": "No error measured.",
}


def report(path):
    S = load(path)
    if not S:
        print(f"{path}: no samples")
        return None
    res = summary(S, os.path.basename(path))
    verdict = shape_of_error(S)
    regions(S)
    print(f"\nverdict: {verdict.upper()} - {ADVICE[verdict]}")
    return res


def main():
    paths = sys.argv[1:] or ["aim_error.json"]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("not found: " + ", ".join(missing) + "   (run aim_check.py first)")
        return
    results = [(p, report(p)) for p in paths]
    if len(results) == 2 and all(r for _, r in results):
        (pa, a), (pb, b) = results
        print(f"\n=== {os.path.basename(pa)} -> {os.path.basename(pb)} ===")
        print(f"hit rate      {a['hit'].mean():.0%} -> {b['hit'].mean():.0%}")
        print(f"median miss   {np.median(a['dist']):.1f} -> {np.median(b['dist']):.1f} px")
        print(f"rms (approx)  {np.sqrt((a['deg'] ** 2).mean()):.2f} -> "
              f"{np.sqrt((b['deg'] ** 2).mean()):.2f} deg")


if __name__ == "__main__":
    main()
