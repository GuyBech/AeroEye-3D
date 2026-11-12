r"""
AeroEye-3D — Radar View. Polar display of the target: bearing and range.

Promised in the project proposal: "ממשק גרפי בתצורת מכ"ם ... מיקום המטרה על
צירים קוטביים (זווית ומרחק), תוך סימון נעילה אדום". This is that display.

It shows what the tracker already knows, in the frame that makes it physically
meaningful. The camera window shows pixels; this shows where the target actually
is relative to the turret — bearing from the pan angle, range from the stereo
disparity. A viewer can see the target crossing the sector and closing in, which
is invisible in the raw camera feed.

Range comes from the standard stereo relation, distance being inversely
proportional to disparity:

    Z = f * B / d          f focal length in px, B the baseline between lenses

f and B were never calibrated on this rig, so their product is carried as one
constant K in depth_scale.json:  Z = K / disparity. One measurement fixes it
(hold the ball at a known distance, note the disparity, K = Z * d) — see
calibrate_depth() below. Without that file the display falls back to showing
raw disparity, clearly labelled as such, rather than inventing a number.

Drawn with OpenCV onto a numpy canvas, so it costs no extra dependency and the
static background (rings, ticks, labels) is rendered once and reused.

    import radar
    r = radar.Radar()
    r.update(pan=58, disp=32, locked=True)
    cv2.imshow("radar", r.canvas())
"""
import json
import math
import os
import time

import cv2
import numpy as np

SIZE = 440                 # square canvas, px
MARGIN = 34
SWEEP_DPS = 90.0           # decorative sweep, degrees per second
TRAIL_S = 2.5              # how long a blip stays visible, seconds
PAN_CENTRE = 90.0          # pan angle that points straight ahead
DEPTH_FILE = "depth_scale.json"

BG = (18, 22, 18)
GRID = (0, 90, 0)
GRID_HI = (0, 140, 0)
INK = (0, 235, 0)
LOCK = (0, 0, 255)
TEXT = (140, 255, 140)


def load_depth_scale():
    """K such that distance_cm = K / disparity, or None if never measured."""
    if os.path.exists(DEPTH_FILE):
        try:
            return float(json.load(open(DEPTH_FILE))["K"])
        except (KeyError, ValueError):
            return None
    return None


def calibrate_depth(distance_cm, disparity):
    """Fix the depth scale from one known (distance, disparity) pair.

    Z = f*B/d means Z*d is constant, so a single measurement determines it.
    A second measurement at a very different distance is worth taking as a
    check: if Z*d differs a lot between them, the stereo geometry is not the
    simple parallel-camera model and the reading should stay as disparity."""
    K = float(distance_cm) * float(disparity)
    json.dump({"K": K, "from": {"distance_cm": distance_cm, "disparity": disparity}},
              open(DEPTH_FILE, "w"), indent=2)
    return K


class Radar:
    def __init__(self, size=SIZE, max_cm=140, half_sector=75):
        self.size = size
        self.max_cm = max_cm            # outer ring
        self.half = half_sector         # sector half-width shown, degrees
        self.cx = size // 2
        self.cy = size - MARGIN
        self.R = size - 2 * MARGIN
        self.K = load_depth_scale()
        self.trail = []                 # (t, x, y, locked)
        self.t0 = time.time()
        self._bg = self._draw_background()

    # ---- geometry -----------------------------------------------------------
    def _polar(self, bearing_deg, frac):
        """Bearing (deg, +right) and range fraction 0..1 -> canvas pixel."""
        a = math.radians(bearing_deg)
        r = max(0.0, min(1.0, frac)) * self.R
        return int(round(self.cx + r * math.sin(a))), int(round(self.cy - r * math.cos(a)))

    def _range_frac(self, disp):
        """Range as a fraction of the outer ring."""
        if disp is None or disp <= 0:
            return None
        if self.K:
            return (self.K / disp) / self.max_cm
        # uncalibrated: still monotonic in true distance, since Z ~ 1/disp
        return (1.0 / disp) / (1.0 / 8.0)

    def distance_cm(self, disp):
        if self.K and disp and disp > 0:
            return self.K / disp
        return None

    # ---- static background --------------------------------------------------
    def _draw_background(self):
        img = np.full((self.size, self.size, 3), BG, np.uint8)
        rings = 4
        for i in range(1, rings + 1):
            r = int(self.R * i / rings)
            cv2.ellipse(img, (self.cx, self.cy), (r, r), 0,
                        -90 - self.half, -90 + self.half,
                        GRID_HI if i == rings else GRID, 1)
            # range labels along the centre line, the usual radar convention —
            # at the sector edge they collide with the bearing spokes
            # unit on the outermost ring only — a separate axis label collides
            # with the bearing "0" at the top of the sector
            label = (f"{self.max_cm * i // rings}" + ("cm" if i == rings else "")
                     if self.K else f"{i}/{rings}")
            p = self._polar(0, i / rings)
            cv2.putText(img, label, (p[0] + 5, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRID_HI, 1)
        for b in range(-self.half, self.half + 1, 15):
            p = self._polar(b, 1.0)
            cv2.line(img, (self.cx, self.cy), p, GRID, 1)
            if b % 30 == 0:
                q = self._polar(b, 1.06)
                cv2.putText(img, f"{b:+d}" if b else "0",
                            (q[0] - 10, max(12, q[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT, 1)
        cv2.circle(img, (self.cx, self.cy), 4, GRID_HI, -1)
        cv2.putText(img, "AeroEye-3D  RADAR", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1)
        cv2.putText(img, "TURRET", (self.cx - 26, self.cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, GRID_HI, 1)
        if not self.K:
            cv2.putText(img, "range: uncalibrated (1/disparity) - run measure_depth.py",
                        (10, self.size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (0, 165, 255), 1)
        return img

    # ---- dynamic ------------------------------------------------------------
    def update(self, pan=None, disp=None, locked=False):
        now = time.time()
        self.trail = [t for t in self.trail if now - t[0] < TRAIL_S]
        if pan is None or disp is None:
            return
        frac = self._range_frac(disp)
        if frac is None:
            return
        bearing = PAN_CENTRE - pan          # turret pan rises to the left
        if abs(bearing) > self.half or frac > 1.05:
            return                          # outside the displayed sector
        x, y = self._polar(bearing, frac)
        self.trail.append((now, x, y, locked, bearing, disp))

    def canvas(self):
        img = self._bg.copy()
        now = time.time()

        # decorative sweep, so the display reads as live even with no target
        ang = ((now - self.t0) * SWEEP_DPS) % (2 * self.half) - self.half
        cv2.line(img, (self.cx, self.cy), self._polar(ang, 1.0), (0, 70, 0), 1)

        for t, x, y, lk, _, _ in self.trail[:-1]:
            age = (now - t) / TRAIL_S
            c = int(200 * (1 - age))
            cv2.circle(img, (x, y), 2, (0, c, 0), -1)

        if self.trail:
            _, x, y, lk, bearing, disp = self.trail[-1]
            col = LOCK if lk else INK
            cv2.circle(img, (x, y), 7, col, 2)
            cv2.drawMarker(img, (x, y), col, cv2.MARKER_CROSS, 14, 1)
            d = self.distance_cm(disp)
            txt = (f"BRG {bearing:+.0f}   "
                   + (f"RNG {d:.0f}cm" if d else f"DISP {disp}"))
            cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
            if lk:
                cv2.putText(img, "LOCK", (self.size - 68, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK, 2)
        else:
            cv2.putText(img, "NO TARGET", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 120, 0), 1)
        return img
