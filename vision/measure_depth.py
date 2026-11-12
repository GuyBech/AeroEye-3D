r"""
AeroEye-3D — fix the depth scale, so the radar reads in centimetres.

The tracker works entirely in disparity, which is fine for aiming: the
calibration maps (x, y, disparity) straight to angles and never needs a metric
distance. But the radar display, and any statement in the report about how far
the target was, needs real units.

For parallel stereo cameras distance and disparity are inversely proportional:

    Z = f * B / d

with f the focal length in pixels and B the baseline between the lenses.
Neither was ever calibrated here, but their product is a single constant, so
ONE measurement at a known distance determines it:  K = Z * d.

This tool takes that measurement — and takes it twice, at two clearly different
distances, because a single point cannot reveal whether the model is wrong. If
Z*d comes out roughly equal at both, the simple relation holds and the mean is
used. If the two differ by a lot, the parallel-camera assumption does not hold
on this camera and the radar is better left reading raw disparity than reading
a confidently wrong number in centimetres.

    ..\.venv\Scripts\python.exe measure_depth.py [cam]
"""
import sys
import time

import cv2
import numpy as np

import stereo
import radar
from detector_dark import DarkBallDetector

CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN = "AeroEye-3D depth scale"
SAMPLES = 20               # disparity readings medianed per measurement
DISAGREE_PCT = 20          # above this the inverse model is not trustworthy


def read_disparity(cap, dl, dr, n=SAMPLES):
    """Median disparity over n frames where both eyes agree."""
    vals = []
    t0 = time.time()
    while len(vals) < n and time.time() - t0 < 15:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        left, right = stereo.split_lr(f)
        tl, tr = dl.detect(left), dr.detect(right)
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if tl and tr:
            d = tl.x - tr.x
            if 3 <= d <= 70 and abs(tl.y - tr.y) <= 40:
                vals.append(d)
                cv2.circle(vis, (tl.x, tl.y), max(tl.radius, 6), (0, 255, 0), 2)
        cv2.putText(vis, f"samples {len(vals)}/{n}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(WIN, cv2.resize(vis, None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            return None
    return float(np.median(vals)) if vals else None


def ask(prompt):
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if v > 0:
                return v
        except ValueError:
            pass
        print("  enter a positive number, e.g. 50")


def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.2)
    if not cap.read()[0]:
        print("CAMERA FAILED — is another process holding it?")
        return
    cv2.namedWindow(WIN)
    dl, dr = DarkBallDetector(), DarkBallDetector()

    print("Measure the distance from the CAMERA LENSES to the ball with a "
          "tape measure.\nTwo measurements are needed, as different as you can "
          "make them (e.g. 40 cm and 100 cm).\n")

    results = []
    for i, hint in enumerate(["CLOSE (e.g. 40 cm)", "FAR (e.g. 100 cm)"]):
        print(f"--- measurement {i + 1}: place the ball {hint} ---")
        z = ask("  measured distance in cm: ")
        print("  hold it still...", flush=True)
        d = read_disparity(cap, dl, dr)
        if d is None:
            print("  no stable stereo reading — aborted")
            cap.release(); cv2.destroyAllWindows()
            return
        print(f"  distance {z:.0f} cm  ->  disparity {d:.1f}  ->  K = {z * d:.0f}\n")
        results.append((z, d, z * d))

    cap.release()
    cv2.destroyAllWindows()

    Ks = [r[2] for r in results]
    spread = 100 * abs(Ks[0] - Ks[1]) / max(Ks)
    print(f"K values: {Ks[0]:.0f} and {Ks[1]:.0f}  ->  disagreement {spread:.0f}%")

    if spread > DISAGREE_PCT:
        print(f"\nThese disagree by more than {DISAGREE_PCT}%, so distance is NOT")
        print("simply proportional to 1/disparity on this camera. Writing a")
        print("scale from them would put a confidently wrong number on the")
        print("radar — leaving it reading raw disparity instead.")
        print("\nWorth re-checking: were both distances measured from the same")
        print("point (the lenses, not the tripod), and was the ball detected")
        print("cleanly at both?")
        return

    K = float(np.mean(Ks))
    radar.calibrate_depth(1.0, K)      # store K directly: Z = K / d
    print(f"\nsaved depth_scale.json with K = {K:.0f}")
    print(f"the radar now reads in centimetres: "
          f"disparity 20 -> {K / 20:.0f} cm, 40 -> {K / 40:.0f} cm")


if __name__ == "__main__":
    main()
