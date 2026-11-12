r"""
AeroEye-3D — fit a correction (or a full refit) from ball calibration points.

calib_ball.py records poses where the laser was auto-centred ON the ball, so
each point states "these angles put the dot exactly at this (x, y, disparity)".
This compares candidate models on those points and keeps whichever GENERALISES,
judged by leave-one-out cross-validation rather than by fit to its own data —
an earlier 7-parameter fit here scored 97% in-sample but 85% cross-validated,
i.e. it was mostly memorising.

Candidates, cheapest first:
    baseline    calib3d.json as it stands                    0 new parameters
    constant    calib3d + a fixed angle offset               2
    linear      calib3d + offset linear in x, y, disp        8
    refit-quad  a fresh quadratic, ignoring calib3d         14

Corrections live in ANGLE space here, because that is what the ball points
measure directly. A richer model is only accepted when it beats the simpler one
by a real margin, since every extra parameter is another way to extrapolate
badly outside the sampled range.

    ..\.venv\Scripts\python.exe fit_ball_calib.py
"""
import json
import os

import numpy as np

PTS = "ball_calib_points.json"
OUT = "aim_correction.json"
CAL = "calib3d.json"
MIN_LINEAR = 20
MIN_REFIT = 55
MARGIN_DEG = 0.15          # a richer model must cut this much error to be chosen


def quad_feats(x, y, d):
    return np.column_stack([x, y, d, x * y, x * x, y * y, np.ones(len(x))])


def loo_pred(A, e):
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
        print(f"no {PTS} — run calib_ball.py first")
        return
    P = json.load(open(PTS))
    # a point can be kept on file but left out of the fit, with the reason
    # recorded next to it: {"exclude": "why"} — visible, and reversible
    for p in P:
        if p.get("exclude"):
            print(f"excluded ({p['x']},{p['y']}) disp={p['disp']} "
                  f"tilt={p['tilt']:.1f}: {p['exclude']}")
    P = [p for p in P if not p.get("exclude")]
    if len(P) < 8:
        print(f"only {len(P)} points — collect more (aim for 20+, spread over "
              f"left/centre/right, high/low, near/far)")
        return

    cal = json.load(open(CAL))
    PC, TC = np.array(cal["pan"]), np.array(cal["tilt"])
    x = np.array([p["x"] for p in P], float)
    y = np.array([p["y"] for p in P], float)
    d = np.array([p["disp"] for p in P], float)
    pan = np.array([p["pan"] for p in P], float)
    tilt = np.array([p["tilt"] for p in P], float)

    A = quad_feats(x, y, d)
    base_pan, base_tilt = A @ PC, A @ TC
    rp, rt = pan - base_pan, tilt - base_tilt      # residual the model misses

    print(f"{len(P)} ball points | disp {d.min():.0f}..{d.max():.0f} | "
          f"x {x.min():.0f}..{x.max():.0f} | y {y.min():.0f}..{y.max():.0f}")
    print(f"current calib3d residual: pan {rp.mean():+.2f}+/-{rp.std():.2f} deg, "
          f"tilt {rt.mean():+.2f}+/-{rt.std():.2f} deg\n")

    one = np.ones(len(P))
    cands = {"baseline": None, "constant": np.column_stack([one])}
    if len(P) >= MIN_LINEAR:
        cands["linear"] = np.column_stack([x, y, d, one])
    if len(P) >= MIN_REFIT:
        cands["refit-quad"] = A

    results = {}
    for name, M in cands.items():
        if name == "baseline":
            ep, et = rp, rt
        elif name == "refit-quad":
            ep = pan - loo_pred(M, pan)
            et = tilt - loo_pred(M, tilt)
        else:
            ep = rp - loo_pred(M, rp)
            et = rt - loo_pred(M, rt)
        err = float(np.sqrt((ep ** 2 + et ** 2).mean()))
        results[name] = err
        print(f"  {name:11s}: LOO rms = {err:.2f} deg   "
              f"(pan {np.sqrt((ep ** 2).mean()):.2f}, tilt {np.sqrt((et ** 2).mean()):.2f})")

    skipped = [n for n in ("linear", "refit-quad") if n not in cands]
    if skipped:
        need = {"linear": MIN_LINEAR, "refit-quad": MIN_REFIT}
        print("  (not considered: " +
              ", ".join(f"{nm} needs {need[nm]}+ points" for nm in skipped) + ")")

    best = "baseline"
    for name in cands:
        if name != "baseline" and results[name] <= results[best] - MARGIN_DEG:
            best = name
    print(f"\nchosen: {best}")

    if best == "baseline":
        print("calib3d.json already explains these points — nothing written.")
        return
    if best == "refit-quad":
        cp, *_ = np.linalg.lstsq(A, pan, rcond=None)
        ct, *_ = np.linalg.lstsq(A, tilt, rcond=None)
        if os.path.exists(CAL):
            os.replace(CAL, CAL.replace(".json", "_prev.json"))
        json.dump({"model": "quad3d", "pan": cp.tolist(), "tilt": ct.tolist(),
                   "points": [[p["x"], p["y"], p["disp"], p["pan"], p["tilt"]]
                              for p in P], "n": len(P),
                   "source": "ball auto-centring"}, open(CAL, "w"), indent=2)
        if os.path.exists(OUT):
            os.remove(OUT)      # a full refit supersedes any residual patch
        print(f"wrote a fresh {CAL} (previous kept as calib3d_prev.json); "
              f"removed {OUT}")
        return

    M = cands[best]
    cp, *_ = np.linalg.lstsq(M, rp, rcond=None)
    ct, *_ = np.linalg.lstsq(M, rt, rcond=None)
    json.dump({"space": "angle", "model": best,
               "pan": cp.tolist(), "tilt": ct.tolist(),
               "n": len(P), "loo_deg": round(results[best], 3),
               "disp_range": [float(d.min()), float(d.max())]},
              open(OUT, "w"), indent=2)
    print(f"saved {OUT}: angle-space {best} correction, "
          f"expected residual {results[best]:.2f} deg")
    if best == "constant":
        print(f"  offset: pan {cp[0]:+.2f} deg, tilt {ct[0]:+.2f} deg")


if __name__ == "__main__":
    main()
