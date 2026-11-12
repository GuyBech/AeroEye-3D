r"""
AeroEye-3D — DARK-BLOB ball detector, self-scaling.

Finds a dark round object (the ball) against a lighter background, on a fixed
monochrome camera. Nothing is stored between frames, so there is no reference
to go stale.

WHY LOCAL CONTRAST. Two earlier approaches failed, measured on 15 saved frames:

  MOG2 (detector_motion / detector_shape)
      Detects *change*, so it smears "where the ball was" into "where it is".
      A swinging ball became an elongated blob: circularity fell to 0.15-0.35
      and the roundness gate rejected it. It also lost the ball entirely at
      both pendulum extremes, where velocity is zero.

  Empty-scene reference diff (detector_ref)
      Goes stale. The reference was built at frame mean 132; by capture time
      the camera's firmware AGC — which OpenCV cannot disable here — had
      drifted the scene to 113. With a 34-level illumination gradient across
      the wall, the difference mask lit up half the frame and the ball was lost.

Comparing each pixel to its own local neighbourhood fixes all of that: nothing
to go stale, a global gain shift moves ball and background together, and a
smooth illumination gradient is followed by the local background estimate.

    background = morphological CLOSE of the frame   (closing fills in dark
                 structures smaller than the kernel, leaving the scene as it
                 would look with the ball removed)
    darkness   = background - frame                 (positive only where the
                 image is darker than its surroundings — i.e. the ball)

WHY IT IS SELF-SCALING. An earlier version used a fixed threshold (22 grey
levels) and a fixed rod-removal disc (29 px). Both encode the conditions of one
session: move the rig back and the ball shrinks below the disc, which erases it;
let the daylight change and the contrast falls below the threshold, which finds
nothing. Both failure modes were observed. So neither is fixed here:

  threshold   derived from the frame's own darkness range, as a fraction of its
              peak, with an absolute floor so an empty scene stays empty.
  rod removal by DISTANCE TRANSFORM instead of a fixed-size opening. The value
              at each pixel is its distance to the nearest background pixel, so
              the peak sits at the centre of the largest inscribed circle — the
              ball — and its height IS the ball's radius. A thin rod has a tiny
              inscribed circle whatever its length, so it loses by construction,
              at any ball size and any distance.

Same Target interface as the other detectors; keep one instance per eye:
    left_d, right_d = DarkBallDetector(), DarkBallDetector()
"""
import math

import cv2
import numpy as np

from detector import Target

SCALE = 0.25           # background estimated at quarter res — it is smooth, so
                       # this costs nothing and makes the big kernel affordable
BG_KERNEL = 121        # px at full res; must exceed the ball or the ball would
                       # be absorbed into the background and vanish. 81 lost the
                       # ball once it came closer than ~diameter 80 px: on a 45 s
                       # recording 14 close frames were rejected as "not round".
                       # 121 found all 14, lost none, and cost +0.6 ms/frame.
MIN_RADIUS = 9         # px — below this, roundness cannot be judged reliably,
                       # and tiny blobs are what make the two eyes disagree
MAX_RADIUS = 70        # px — bigger than any plausible ball in a 320-px eye
FLOOR_DELTA = 9        # absolute minimum contrast; keeps an empty scene empty
THR_FRACTIONS = (0.55, 0.42, 0.32, 0.24)   # of the frame's own darkness peak
MIN_FILL = 0.60        # of the inscribed circle actually covered by the mask
MIN_CIRC = 0.55        # roundness of the contour the peak belongs to
MIN_REL_CONTRAST = 0.6 # the candidate must block this fraction of the light of
                       # the wall around it. Measured: black-taped ball 0.84-0.92
                       # (175 detections, near and far); its own shadow on the
                       # wall 0.18-0.36; vignetted frame corners 0.18-0.24. A
                       # shadow is round too and real stereo geometry, so this
                       # is the only gate that tells it apart. Ratio, not grey
                       # levels, so a dimmer room does not shift it. NOTE: the
                       # plain tennis ball measured 0.35-0.58 — set 0 to use it.


class DarkBallDetector:
    def __init__(self, delta=None, open_d=None):
        # delta/open_d are accepted for backward compatibility with older
        # callers and tuning scripts; delta pins the threshold if given.
        self.fixed_delta = delta
        k = max(3, int(BG_KERNEL * SCALE) | 1)
        self.bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.reject = None
        self.last = None          # (threshold, radius) actually used — for logs

    # ---- local contrast -----------------------------------------------------
    def background(self, gray):
        """The scene as it would look with the ball removed."""
        small = cv2.resize(gray, None, fx=SCALE, fy=SCALE,
                           interpolation=cv2.INTER_AREA)
        bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, self.bg_kernel)
        return cv2.resize(bg, (gray.shape[1], gray.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def darkness(self, gray, bg=None):
        """How much darker than its local surroundings each pixel is."""
        if bg is None:
            bg = self.background(gray)
        return cv2.subtract(bg, cv2.GaussianBlur(gray, (5, 5), 0))

    @staticmethod
    def rel_contrast(dark, bg, c):
        """Fraction of the surrounding light the candidate blocks: 0 = same as
        the wall, 1 = black. Measured inside 0.8 r, so the soft rim is ignored."""
        x, y, r = int(c["x"]), int(c["y"]), max(int(c["r"] * 0.8), 2)
        y0, x0 = max(0, y - r), max(0, x - r)
        d, b = dark[y0:y + r + 1, x0:x + r + 1], bg[y0:y + r + 1, x0:x + r + 1]
        m = np.zeros(d.shape, np.uint8)
        cv2.circle(m, (x - x0, y - y0), r, 255, -1)
        return float(np.median(d[m > 0])) / max(float(np.median(b[m > 0])), 1.0)

    def _clean(self, dark, thr):
        m = (dark >= thr).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.small)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.small)
        # fill interior holes. Our OWN laser dot on the ball is bright, so it
        # punches a hole in the dark mask; the distance-transform peak then
        # moves away from the dot, the turret follows, the hole follows — the
        # laser chases itself off the ball. Simulated on the saved frames the
        # centre drifted 14-23 px with the dot on; with holes filled, 2-5 px.
        # With no dot nothing changes: a solid disc has no holes to fill.
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(m, cs, -1, 255, -1)
        return m

    # ---- candidate from one threshold ---------------------------------------
    def _best_at(self, mask):
        """Largest inscribed circle in the mask, if it looks like the ball.

        The distance transform peak is the centre of the biggest circle that
        fits inside the foreground. For a ball-on-a-rod that is the ball: the
        rod is thin, so its inscribed circle is small no matter how long it is.
        This replaces a fixed-size opening and works at any ball size."""
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        _, radius, _, loc = cv2.minMaxLoc(dt)
        H, W = mask.shape
        # a peak ON the image border belongs to something mostly OUTSIDE the
        # frame, whose roundness cannot be judged: the dark lens-vignetted
        # corners (84 grey levels against 131 in the centre), a hand holding
        # the line at the top, or the ball itself more than half out of view.
        # A hand/corner pair once held a false lock, laser on, for 13 s.
        # Reject outright rather than fall through to the next-best blob: that
        # was tried, and with the ball at the border the next-best blob was
        # its own round SHADOW on the wall. No detection = laser off, the safe
        # failure.
        if loc[0] in (0, W - 1) or loc[1] in (0, H - 1):
            return None
        if not (MIN_RADIUS <= radius <= MAX_RADIUS):
            return None

        # the inscribed circle must actually be filled — a long thin arc can
        # score a moderate radius at a bend without being a disc
        r = int(radius)
        y0, y1 = max(0, loc[1] - r), min(mask.shape[0], loc[1] + r + 1)
        x0, x1 = max(0, loc[0] - r), min(mask.shape[1], loc[0] + r + 1)
        patch = mask[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        disc = np.zeros(patch.shape, np.uint8)
        cv2.circle(disc, (loc[0] - x0, loc[1] - y0), max(r - 1, 1), 255, -1)
        inside = int(np.count_nonzero(disc))
        if inside == 0:
            return None
        fill = np.count_nonzero(cv2.bitwise_and(patch, disc)) / inside
        if fill < MIN_FILL:
            return None

        # roundness of the whole component the peak sits in — rejects the case
        # where the ball is fused into a much larger irregular region
        num, lab = cv2.connectedComponents(mask)
        comp = (lab == lab[loc[1], loc[0]]).astype(np.uint8) * 255
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
        circ = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
        return dict(x=loc[0], y=loc[1], r=radius, fill=fill, circ=circ, area=area)

    # ---- detection ----------------------------------------------------------
    def detect(self, gray):
        """Return the ball as a Target, or None.

        Records `self.reject` — the best candidate that failed and why — so a
        silent None can be told apart from a threshold that is merely too tight.
        """
        self.reject = None
        bg = self.background(gray)
        dark = self.darkness(gray, bg)
        peak = int(dark.max())
        if peak < FLOOR_DELTA:
            self.reject = f"no local contrast (peak {peak} < {FLOOR_DELTA})"
            return None

        if self.fixed_delta:
            thresholds = [self.fixed_delta]
        else:
            # scale to the contrast actually present, never below the floor
            thresholds = sorted({max(FLOOR_DELTA, int(peak * f))
                                 for f in THR_FRACTIONS}, reverse=True)

        best, best_thr, near_miss = None, None, None
        for thr in thresholds:
            cand = self._best_at(self._clean(dark, thr))
            if cand is None:
                continue
            if cand["circ"] < MIN_CIRC:
                if near_miss is None:
                    near_miss = (f"a{int(cand['area'])} c{cand['circ']:.2f} "
                                 f"f{cand['fill']:.2f} r{cand['r']:.0f} "
                                 f"@({cand['x']},{cand['y']}) not round "
                                 f"[thr={thr}]")
                continue
            rel = self.rel_contrast(dark, bg, cand)
            if rel < MIN_REL_CONTRAST:
                if near_miss is None:
                    near_miss = (f"r{cand['r']:.0f} @({cand['x']},{cand['y']}) too "
                                 f"faint: blocks {rel:.0%} of the light < "
                                 f"{MIN_REL_CONTRAST:.0%} (shadow?) [thr={thr}]")
                continue
            # prefer the LARGEST well-formed disc, not the tightest one. A high
            # threshold finds only the ball's dark core, which both shrinks the
            # radius and moves the centroid off centre — and when the two eyes
            # pick different-sized cores their rows disagree and the stereo
            # pair is thrown away (dy up to 76 px was observed that way).
            score = cand["r"] * cand["fill"] * cand["circ"]
            if best is None or score > best[0]:
                best, best_thr = (score, cand), thr

        if best is None:
            self.reject = near_miss or (f"nothing round at thresholds "
                                        f"{thresholds} (peak {peak})")
            return None
        c = best[1]
        self.last = (best_thr, c["r"])
        return Target(int(c["x"]), int(c["y"]), max(int(c["r"]), 4),
                      float(math.pi * c["r"] * c["r"]))
