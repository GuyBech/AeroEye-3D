r"""
AeroEye-3D — fit an aim correction from measured laser-vs-ball errors.

aim_check.py records, for a stationary ball, how far the laser landed from the
ball centre. This fits that residual and writes aim_correction.json, which the
tracker applies on top of calib3d.json.

How the correction is applied (see track_motion_3d.py):
commanding aim(P) puts the dot at P - err(P), so to land the dot ON the ball we
command aim(ball + err(ball)). Correcting the TARGET PIXEL rather than the angle
reuses the existing calibration and needs no separate degrees-per-pixel factor.

Only operator-confirmed samples ("careful": true) are fitted. The earlier
automatic batch is kept in the file for reference but excluded: its dot finder
locked onto frame-border noise whenever the laser missed, and mixing those in
inflated the apparent scatter from 7 px to 18 px and hid the real defect.

Model choice is by leave-one-out cross-validation, and deliberately biased
toward the CONSTANT model. The measured error is a constant tilt offset —
err_y = -34.8 +/- 6.9 px with correlations to x, y and disp of +0.16, +0.18 and
+0.06, i.e. none. A constant extrapolates safely to depths that were never
sampled; a term in disp fitted over disp 14-29 would extrapolate wildly at
disp 60. Richer models are accepted only when they beat the constant clearly
AND there are enough samples to support their parameters.

    ..\.venv\Scripts\python.exe fit_aim_correction.py
"""
import json
import os

import numpy as np

PTS = "aim_error.json"
OUT = "aim_correction.json"
MIN_FOR_LINEAR = 25        # 4 parameters
MIN_FOR_QUAD = 60          # 7 parameters
MARGIN = 0.05              # a richer model must win by this much to be chosen


def feats(x, y, d, model):
    one = np.ones(len(x))
    if model == "constant":
        return np.column_stack([one])
    if model == "linear":
        return np.column_stack([x, y, d, one])
    return np.column_stack([x, y, d, x * y, x * x, y * y, one])


def loo(A, e):
    """Leave-one-out predictions — the honest out-of-sample estimate."""
    n = len(e)
    p = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        c, *_ = np.linalg.lstsq(A[m], e[m], rcond=None)
        p[i] = A[i] @ c
    return p


def main():
    if not os.path.exists(PTS):
        print(f"no {PTS} — run aim_check.py first")
        return
    all_s = json.load(open(PTS))
    S = [s for s in all_s if s.get("careful") and s["err_x"] is not None]
    if len(S) < 8:
        print(f"only {len(S)} operator-confirmed samples — collect more with aim_check.py")
        return

    x = np.array([s["ball_x"] for s in S], float)
    y = np.array([s["ball_y"] for s in S], float)
    d = np.array([s["disp"] for s in S], float)
    r = np.array([s["radius"] for s in S], float)
    ex = np.array([s["err_x"] for s in S], float)
    ey = np.array([s["err_y"] for s in S], float)

    print(f"{len(S)} operator-confirmed samples "
          f"(of {len(all_s)} total; the automatic batch is excluded)")
    print(f"  disp {d.min():.0f}..{d.max():.0f}   "
          f"err_x {ex.mean():+.1f}+/-{ex.std():.1f}   "
          f"err_y {ey.mean():+.1f}+/-{ey.std():.1f} px")
    print(f"  uncorrected hit rate: "
          f"{100 * ((np.abs(ex) <= r) & (np.abs(ey) <= r)).mean():.0f}%\n")

    candidates = ["constant"]
    if len(S) >= MIN_FOR_LINEAR:
        candidates.append("linear")
    if len(S) >= MIN_FOR_QUAD:
        candidates.append("quadratic")

    scores = {}
    for m in candidates:
        A = feats(x, y, d, m)
        px, py = loo(A, ex), loo(A, ey)
        hit = ((np.abs(ex - px) <= r) & (np.abs(ey - py) <= r)).mean()
        scores[m] = hit
        print(f"  {m:10s}: LOO hit={100 * hit:3.0f}%  "
              f"resid=({np.sqrt(((ex - px) ** 2).mean()):.1f},"
              f"{np.sqrt(((ey - py) ** 2).mean()):.1f})px")
    for m in candidates:
        if m != "constant" and scores[m] < scores["constant"] + MARGIN:
            print(f"  ({m} rejected — does not beat constant by {MARGIN:.0%})")

    best = "constant"
    for m in candidates[1:]:
        if scores[m] >= scores[best] + MARGIN:
            best = m
    skipped = [m for m in ("linear", "quadratic") if m not in candidates]
    if skipped:
        print(f"  (not considered, too few samples: {', '.join(skipped)})")

    A = feats(x, y, d, best)
    cx, *_ = np.linalg.lstsq(A, ex, rcond=None)
    cy, *_ = np.linalg.lstsq(A, ey, rcond=None)
    json.dump({"model": best, "x": cx.tolist(), "y": cy.tolist(),
               "n": len(S), "loo_hit": round(float(scores[best]), 3),
               "disp_range": [float(d.min()), float(d.max())]},
              open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}: model={best}, expected hit ~{100 * scores[best]:.0f}%")
    if best == "constant":
        print(f"  pixel offset applied: x{cx[0]:+.1f} y{cy[0]:+.1f}")


if __name__ == "__main__":
    main()
