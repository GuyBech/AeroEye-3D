r"""
AeroEye-3D — REFERENCE-DIFF ball detector (the one that actually works here).

Why not MOG2: background subtraction detects *change*, so it fails a pendulum
exactly where we need it most. Measured on this rig, at the same three frames:

    MOG2      a=7652 c=0.33 | a=6233 c=0.26 | a=6865 c=0.29   -> all rejected
    ref-diff  a=4504 c=0.78 | a=3687 c=0.79 | a=3971 c=0.80   -> all accepted

MOG2 smears "where the ball was" into "where it is", so a moving ball becomes an
elongated blob and its circularity collapses. It also loses a ball that stops
(both pendulum extremes) and slowly absorbs a lingering one into the background.

This detector instead diffs each frame against a once-captured picture of the
EMPTY scene. That measures presence, not change, so a motionless ball is still
found. The camera is fixed, which is what makes it valid.

Two properties matter on this hardware:
  * |difference| — the ball measured ~53 grey vs ~113 background, i.e. DARKER.
    A signed "brighter than background" test would miss it completely.
  * median subtraction — this camera runs AGC in firmware that OpenCV cannot
    disable (CAP_PROP_AUTO_EXPOSURE reads -1; setting EXPOSURE does not move
    frame_mean). The whole image therefore jumps brightness at once, which a
    plain diff reports as a frame-sized blob (59392 px observed = 77% of the
    eye). The median of the signed difference is set by the background, which is
    most of the pixels, so subtracting it cancels the global shift exactly.

Same Target interface as the other detectors. Stateful -> one instance per eye:
    left_d, right_d = RefDetector(), RefDetector()
    left_d.set_reference(left_gray)      # with the scene EMPTY
    t = left_d.detect(left_gray)
"""
import math

import cv2
import numpy as np

from detector import Target

# Thresholds below are read off measured data (see module docstring): rejected
# blobs scored circ 0.17-0.53 / fill 0.13-0.52, the ball scored circ 0.67-0.83 /
# fill 0.59-0.86 — so the classes separate cleanly in between.
DELTA     = 12         # grey levels a pixel must differ from the reference
MIN_AREA  = 250        # px^2 — below this a blob is too small to judge roundness
MAX_AREA  = 12000      # px^2 — above this it is a lighting shift, not an object
# Roundness is deliberately LOOSE. A swinging ball is genuinely elliptical in the
# raw image: this room needs a long exposure, so ~1 m/s of swing smears the ball
# by ~15 px across a ~40 px diameter. Measured circularity of the real ball
# ranged 0.17-0.83 depending on swing speed — a strict gate accepted only ~6% of
# frames. Shape filtering existed to reject HANDS; with the operator out of frame
# the ball is the only thing differing from the reference, so area bounds plus a
# weak roundness sanity check are enough, and coverage matters far more.
MIN_CIRC  = 0.30
MIN_FILL  = 0.35
REF_FRAMES = 15        # frames medianed together to build a noise-free reference


class RefDetector:
    def __init__(self, delta=DELTA):
        self.delta = delta
        self.ref = None
        self._acc = []
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # A wider CLOSE: the ball's mask breaks into pieces where its shading
        # matches the wall (observed as spurious 341/378/610 px fragments of a
        # ~4500 px ball). Closing with a kernel bigger than those gaps welds the
        # fragments back into one blob before the area test sees them.
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # ---- reference handling -------------------------------------------------
    def set_reference(self, gray):
        """Set the empty-scene reference from a single frame."""
        self.ref = cv2.GaussianBlur(gray, (5, 5), 0).copy()
        self._acc = []

    def accumulate_reference(self, gray):
        """Feed frames one at a time; the reference is the per-pixel MEDIAN of
        REF_FRAMES of them, so sensor noise and a stray frame cannot poison it.
        Returns True once the reference is ready."""
        self._acc.append(cv2.GaussianBlur(gray, (5, 5), 0))
        if len(self._acc) < REF_FRAMES:
            return False
        self.ref = np.median(np.stack(self._acc), axis=0).astype(np.uint8)
        self._acc = []
        return True

    @property
    def ready(self):
        return self.ref is not None

    # ---- detection ----------------------------------------------------------
    def diff_mask(self, gray):
        if self.ref is None:
            return np.zeros_like(gray)
        signed = (cv2.GaussianBlur(gray, (5, 5), 0).astype(np.int16)
                  - self.ref.astype(np.int16))
        signed -= int(np.median(signed))       # cancel the global AGC shift
        mask = (np.abs(signed) >= self.delta).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        return mask

    def detect(self, gray):
        """Return the ball as a Target, or None.

        Also records `self.reject` — the biggest blob that was thrown away and
        the test it failed. Without it a silent None is indistinguishable from
        "saw nothing at all", which is the difference between a threshold that
        is too tight and a ball that is genuinely not in frame."""
        self.reject = None
        if self.ref is None:
            return None
        contours, _ = cv2.findContours(self.diff_mask(gray), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        worst_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            (x, y), r = cv2.minEnclosingCircle(c)
            fill = area / (math.pi * r * r) if r > 0 else 0.0

            why = None
            if area < MIN_AREA:
                why = "area<min"
            elif area > MAX_AREA:
                why = "area>max"
            elif circ < MIN_CIRC:
                why = "not round"
            elif fill < MIN_FILL:
                why = "not filled"
            if why:
                if area > worst_area and area >= 40:
                    worst_area = area
                    self.reject = (f"a{int(area)} c{circ:.2f} f{fill:.2f} "
                                   f"@({int(x)},{int(y)}) {why}")
                continue

            score = circ * fill
            if score > best_score:
                best, best_score = (x, y, r, area), score
        if best is None:
            return None
        x, y, r, area = best
        return Target(int(x), int(y), max(int(r), 4), area)
