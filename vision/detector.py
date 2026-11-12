r"""
AeroEye-3D — Phase 5: target detector (SWAPPABLE module).

The rest of the system only calls `detect(gray)` and gets back a target
position. How the position is found lives entirely in here, so the method
(brightness / motion / shape / ML) can be replaced without touching track.py
or the future motor-control code.

The real tracking camera is MONOCHROME, so detection works on a single-channel
grayscale image — never on color. (During development we convert the laptop
webcam's color frames to gray first, so the algorithm is identical.)

Current method: peak-relative brightest-blob (long range).
  1. light blur to kill single-pixel noise
  2. find the brightest PIXEL (minMaxLoc) — this locks on to the light source
     regardless of how dim/small it has become with distance
  3. keep pixels within PEAK_DELTA of that peak -> the dot's region
  4. take the blob holding the peak, return its centroid + radius
Because it thresholds relative to the peak (not a fixed value), a far-away
flashlight that is only a few faint pixels is still found — as long as it is
the brightest thing in view (dim competing lights / windows if needed).
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Target:
    x: int          # centroid pixel X
    y: int          # centroid pixel Y
    radius: int     # approx blob radius (px)
    area: float     # blob area (px^2)


# Tuning knobs (kept here so calibration is one place).
BLUR_KSIZE = 3          # tiny blur — suppress 1px noise without smearing a far dot
ABS_FLOOR  = 90         # if the brightest pixel is below this, there is no real
                        # light source in view -> no target (avoids locking on wall)
PEAK_DELTA = 40         # keep pixels within this many levels of the peak


def _peak_mask(gray):
    blurred = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    _, maxv, _, maxloc = cv2.minMaxLoc(blurred)
    if maxv < ABS_FLOOR:
        return None, None, None
    thr = max(ABS_FLOOR, int(maxv) - PEAK_DELTA)
    mask = (blurred >= thr).astype(np.uint8) * 255
    return mask, maxloc, maxv


def make_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the peak region (for debugging/preview)."""
    mask, _, _ = _peak_mask(gray)
    return mask if mask is not None else np.zeros_like(gray)


def detect(gray: np.ndarray):
    """Return a Target locked on the brightest light source, or None."""
    mask, maxloc, _ = _peak_mask(gray)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # prefer the blob that actually contains the brightest pixel
    best = None
    for c in contours:
        if cv2.pointPolygonTest(c, (float(maxloc[0]), float(maxloc[1])), False) >= 0:
            best = c
            break
    if best is None and contours:
        best = max(contours, key=cv2.contourArea)
    if best is None:
        # peak with no contour (single pixel) — use the peak location itself
        return Target(int(maxloc[0]), int(maxloc[1]), 3, 1.0)
    (x, y), radius = cv2.minEnclosingCircle(best)
    return Target(int(x), int(y), max(int(radius), 3), cv2.contourArea(best))
