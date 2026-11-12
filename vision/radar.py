r"""
AeroEye-3D — Radar console. Where the ball really is, as a control-room display.

Promised in the project proposal: "ממשק גרפי בתצורת מכ"ם ... מיקום המטרה על
צירים קוטביים (זווית ומרחק), תוך סימון נעילה אדום". This is that display, grown
into a full console: a top view, a side view, live target data, a range history,
the laser drawn on the map, sound, and a boot sequence. It exists to make the
demo impressive, but the ball is drawn where the cameras actually see it.

The camera window shows pixels; this shows the ball in centimetres. Position
comes from the stereo pair, NOT from the servo angle, because the servo angle is
where the laser points (including the look-ahead and the offset between turret
and camera), not where the ball is:

    d' = disparity - DISP_ZERO
    Z  = K / d'                      distance in front of the lenses
    X  = BASELINE * (xc - 160) / d'  to the right of the lenses
    Y  = BASELINE * (120 - yc) / d'  above the camera axis

xc, yc are the midpoint between the two eyes. Both views use one px/cm on both
axes, so each is a scale drawing: 10 cm sideways is as long as 10 cm of depth.

Why the plain Z = K / d is not used: on this camera the eyes have a fixed
disparity offset. Measured on 5,395 whole-ball frames from four recorded runs,
disparity against ball radius in pixels is a straight line, d = 1.73 r - 15.8
(residual 1.5 px). If Z = K / d held, that line would pass through zero; instead
the ball grows x3.2 while disparity grows x10. The slope times the ball radius
gives the lens spacing, 5.8 cm, which nothing here assumed.

Accuracy: DISP_ZERO and BASELINE_CM come from that measurement, so proportions
are sound. FOCAL_PX is only known to 265-320 px (from the pan and tilt fits, whose
servo gains are unknown), so distances in cm are estimates, good to roughly 15%.
A depth_scale.json with K and d0, if it exists, replaces the estimate.

The console only ever receives data. Nothing here feeds back into aiming, and
track_motion_3d.py shuts the display off rather than crash if it ever fails.

    import radar
    r = radar.Radar()
    r.open_window("radar")
    r.update(x=180, y=100, disp=20, pan=60, tilt=30, locked=True, laser=True)
    cv2.imshow("radar", r.canvas())

Keys, passed in through r.key():  f full screen | m mute.
"""
import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np

from radar_sound import Sound

# ---- stereo geometry (see the module docstring) -----------------------------
BASELINE_CM = 5.8          # spacing between the lenses
DISP_ZERO = -15.8          # disparity of a target at infinity
FOCAL_PX = 290.0           # 265-320 measured; sets the cm scale of Z
IMG_CX, IMG_CY = 160, 120
ROW_OFFSET = 9             # the right eye sees the ball this many rows lower
DEPTH_FILE = "depth_scale.json"

HALF_FOV = math.degrees(math.atan(IMG_CX / FOCAL_PX))
HALF_VFOV = math.degrees(math.atan(IMG_CY / FOCAL_PX))

# ---- where the laser is: fitted on calib3d.json and ball_calib_points.json ---
# Turret position relative to the lenses, consistent between the board fit
# (-11 cm) and the ball fit (-12 cm). Gains are servo degrees per real degree.
TURRET_X, TURRET_Y = -11.5, 5.0     # cm: left of and above the lenses
PAN_ZERO, PAN_GAIN = 78.25, -1.15   # beam misses the ball by ~1 cm (median)
TILT_ZERO, TILT_GAIN = 34.88, 0.82  # ~3 cm (median); tilt has backlash

# ---- behaviour --------------------------------------------------------------
SMOOTH = 0.5               # display smoothing of the blip, not used for aiming
TRAIL_S = 2.5              # how long the track stays visible
HISTORY_S = 10.0           # range history window
ACQUIRE_S = 0.5            # reticle closes in for this long before "LOCK"
ARROW_S = 0.5              # the motion arrow shows this many seconds ahead
ARROW_MIN = 6.0            # cm/s; slower than this draws no arrow
SWEEP_DPS = 70.0
RING_S = 0.8               # life of the ping ring around the target
FLASH_S = 0.9              # "TARGET LOCKED" banner
GAP_S = 0.5                # a target gone this long starts a new track

BOOT_STEP_S = 0.36
BOOT_HOLD_S = 0.8
REVEAL_S = 0.6

# ---- layout -----------------------------------------------------------------
W, H = 1280, 720
SCOPE = (4, 40, 784, 716)           # x0, y0, x1, y1
OX, OY = 394, 698                   # the lenses, on the canvas
PX_CM = 6.05
MAX_CM = 100
FAN = 34                            # degrees drawn; beyond HALF_FOV is hatched
SIDE = (793, 61, 1276, 290)
SX, SY = 25, 114                    # the lenses, inside the side view
SIDE_PX_CM = 4.38
CHART = (804, 526, 462, 180)
PILLS = (("SEARCH", 830), ("TRACK", 942), ("LOCK", 1054), ("LASER", 1166))
PILL_W = 104

BG = (6, 10, 7)
SCOPE_BG = (9, 15, 10)
INK = (120, 255, 130)
SOFT = (90, 200, 110)
FAINT = (40, 90, 50)
GLOW = (90, 240, 110)
WHITE = (230, 255, 235)
LOCK = (50, 40, 255)
AMBER = (40, 190, 255)
LASER = (70, 70, 255)
OFF = (60, 100, 70)

F1 = cv2.FONT_HERSHEY_SIMPLEX
F2 = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


def load_depth_scale():
    """(K, d0) such that distance_cm = K / (disparity - d0). A measured
    depth_scale.json wins; without one, the estimate from the recordings."""
    if os.path.exists(DEPTH_FILE):
        try:
            d = json.load(open(DEPTH_FILE))
            return float(d["K"]), float(d.get("d0", 0.0))
        except (KeyError, ValueError):
            pass
    return BASELINE_CM * FOCAL_PX, DISP_ZERO


def calibrate_depth(distance_cm, disparity):
    """Fix the depth scale from one known (distance, disparity) pair, for the
    simple model Z = K / d. Kept for measure_depth.py."""
    K = float(distance_cm) * float(disparity)
    json.dump({"K": K, "from": {"distance_cm": distance_cm, "disparity": disparity}},
              open(DEPTH_FILE, "w"), indent=2)
    return K


# ---- drawing helpers --------------------------------------------------------
def _dim(c, a):
    return tuple(int(v * a) for v in c)


def _text(img, s, p, sc=.42, col=SOFT, font=F1):
    cv2.putText(img, s, p, font, sc, col, 1, AA)


def _sprite(R, col):
    y, x = np.mgrid[-R:R + 1, -R:R + 1]
    g = np.exp(-(x * x + y * y) / (2 * (R / 2.6) ** 2))
    return np.dstack([(g * c).astype(np.uint8) for c in col])


def _add_sprite(img, s, cx, cy):
    R = s.shape[0] // 2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, cx - R), max(0, cy - R), min(w, cx + R + 1), min(h, cy + R + 1)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = cv2.add(img[y0:y1, x0:x1],
                                    s[y0 - cy + R:y1 - cy + R, x0 - cx + R:x1 - cx + R])


def _brackets(img, c, s, col, th=2):
    x, y = c
    l = max(4, s // 2)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        p = (x + sx * s, y + sy * s)
        cv2.line(img, p, (p[0] - sx * l, p[1]), col, th, AA)
        cv2.line(img, p, (p[0], p[1] - sy * l), col, th, AA)


def _pill(img, x, y, w, h, col, s, sc=.42, fill=.25):
    cv2.rectangle(img, (x, y), (x + w, y + h), _dim(col, fill), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col, 1, AA)
    (tw, th), _ = cv2.getTextSize(s, F2, sc, 1)
    _text(img, s, (x + (w - tw) // 2, y + (h + th) // 2), sc, col, F2)


def _panel(img, x, y, w, h, title):
    cv2.rectangle(img, (x, y), (x + w, y + h), _dim(INK, .45), 1, AA)
    cv2.rectangle(img, (x, y), (x + w, y + 20), _dim(INK, .2), -1)
    _text(img, title, (x + 8, y + 15), .4, INK)


def _top(X, Z):
    """World cm -> canvas pixel, top view."""
    return int(round(OX + X * PX_CM)), int(round(OY - Z * PX_CM))


def _polar(az, r):
    a = math.radians(az)
    return _top(r * math.sin(a), r * math.cos(a))


def _side(Y, Z):
    """World cm -> pixel inside the side-view panel."""
    return int(SX + Z * SIDE_PX_CM), int(SY - Y * SIDE_PX_CM)


class Radar:
    def __init__(self, port="COM4", sound=True, boot=True):
        self.K, self.d0 = load_depth_scale()
        self.sound = Sound(sound)
        self.port = port
        self.t0 = time.time()           # restarted when the boot sequence ends
        self.boot_end = self.t0 + (6 * BOOT_STEP_S + BOOT_HOLD_S if boot else 0)
        self.boot_n = 0
        self.ready = False
        self.state = "SEARCH"           # SEARCH / TRACK / LOCK / COAST
        self.pos = None                 # smoothed [X, Y, Z], cm
        self.vel = np.zeros(3)          # cm/s
        self.seen_t = 0.0
        self.lock_t = None
        self.flash_t = -9.0
        self.pan = self.tilt = None
        self.laser = False
        self.trail = deque()            # (t, X, Y, Z, top px, side px)
        self.hist = deque()             # (t, range)
        self.rings = []                 # ping rings: (t, px)
        self.prev_sweep = None
        self.win = None
        self.full = False
        self._glow = _sprite(26, GLOW)
        self._glow_small = _sprite(8, GLOW)
        self._hit = _sprite(15, (70, 70, 255))
        self._base = self._draw_background()
        self._boot_bg = self._draw_boot_background()
        self._make_clutter()

    # ---- window -------------------------------------------------------------
    def open_window(self, name, fullscreen=False):
        self.win = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, W, H)
        if fullscreen:
            self.toggle_fullscreen()

    def toggle_fullscreen(self):
        self.full = not self.full
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.full else cv2.WINDOW_NORMAL)

    def key(self, k):
        if k == ord('f') and self.win:
            try:
                self.toggle_fullscreen()
            except cv2.error:           # the window was closed by hand
                self.full = not self.full
        elif k == ord('m'):
            self.sound.toggle_mute()

    # ---- data ---------------------------------------------------------------
    def world(self, x, y, disp):
        """Left-eye pixel + disparity -> (X, Y, Z) in cm, from the lenses."""
        dd = disp - self.d0
        if dd <= 1:
            return None
        return np.array([BASELINE_CM * (x - disp / 2 - IMG_CX) / dd,
                         BASELINE_CM * (IMG_CY - (y + ROW_OFFSET / 2)) / dd,
                         self.K / dd])

    def update(self, x=None, y=None, disp=None, pan=None, tilt=None,
               locked=False, coasting=False, laser=False):
        """x, y are left-eye pixels of a detection this frame (None if none);
        pan, tilt the commanded aim. Call once per frame."""
        now = time.time()
        self.pan, self.tilt, self.laser = pan, tilt, bool(laser)
        p = self.world(x, y, disp) if (locked and x is not None and disp) else None

        if p is not None:
            fresh = self.pos is None or now - self.seen_t > GAP_S
            self.pos = p if fresh else SMOOTH * p + (1 - SMOOTH) * self.pos
            self.seen_t = now
            X, Y, Z = self.pos
            self.trail.append((now, X, Y, Z, _top(X, Z), _side(Y, Z)))
            self.hist.append((now, math.hypot(X, Z)))
            self._estimate_velocity(now)
            if self.lock_t is None:
                self.lock_t = now

        was = self.state
        if p is not None:
            self.state = "LOCK" if now - self.lock_t >= ACQUIRE_S else "TRACK"
        elif coasting and self.pos is not None:
            self.state = "COAST"
        else:
            self.state = "SEARCH"
            self.lock_t = None

        if self.ready and self.state != was:
            if self.state == "LOCK" and was == "TRACK":
                self.sound.play("lock")
                self.flash_t = now
            elif self.state == "SEARCH":
                self.sound.play("lost")

        while self.trail and now - self.trail[0][0] > TRAIL_S:
            self.trail.popleft()
        while self.hist and now - self.hist[0][0] > HISTORY_S:
            self.hist.popleft()

    def _estimate_velocity(self, now):
        recent = [s for s in self.trail if now - s[0] < 0.3]
        if len(recent) >= 3:
            a, b = recent[0], recent[-1]
            dt = b[0] - a[0]
            if dt > 0.05:
                v = np.array([b[1] - a[1], b[2] - a[2], b[3] - a[3]]) / dt
                self.vel = 0.5 * v + 0.5 * self.vel

    # ---- static layers ------------------------------------------------------
    def _draw_background(self):
        img = np.full((H, W, 3), BG, np.uint8)
        x0, y0, x1, y1 = SCOPE
        img[y0:y1, x0:x1] = SCOPE_BG
        o = (OX, OY)
        R = int(MAX_CM * PX_CM)
        cv2.ellipse(img, o, (R, R), 0, -90 - FAN, -90 + FAN, _dim(INK, .07), -1, AA)
        cv2.ellipse(img, o, (R, R), 0, -90 - HALF_FOV, -90 + HALF_FOV, _dim(INK, .17), -1, AA)
        for a in np.arange(HALF_FOV + 2, FAN, 3.0):     # outside the cameras' view
            for sg in (-1, 1):
                cv2.line(img, _polar(sg * a, 20), _polar(sg * a, MAX_CM), _dim(FAINT, .5), 1, AA)
        for r in range(20, MAX_CM + 1, 10):
            major = r % 20 == 0
            cv2.ellipse(img, o, (int(r * PX_CM),) * 2, 0, -90 - HALF_FOV, -90 + HALF_FOV,
                        _dim(INK, .60 if major else .24), 1, AA)
            if major:
                for a0, a1 in ((-90 - FAN, -90 - HALF_FOV), (-90 + HALF_FOV, -90 + FAN)):
                    cv2.ellipse(img, o, (int(r * PX_CM),) * 2, 0, a0, a1, _dim(FAINT, .7), 1, AA)
                p = _polar(HALF_FOV + .6, r)
                _text(img, f"{r}", (p[0] + 4, p[1] + 4), .38)
        for a in range(-20, 21, 10):
            cv2.line(img, o, _polar(a, MAX_CM), _dim(INK, .5 if a == 0 else .24), 1, AA)
            q = _polar(a, MAX_CM + 3.2)
            _text(img, f"{a:+d}" if a else "0", (q[0] - 10, q[1]), .38)
        for sg in (-1, 1):
            cv2.line(img, o, _polar(sg * HALF_FOV, MAX_CM), SOFT, 1, AA)
        cv2.ellipse(img, o, (int(19 * PX_CM),) * 2, 0, -90 - FAN, -90 + FAN, _dim(FAINT, .8), -1, AA)
        cv2.circle(img, o, 6, SOFT, -1, AA)
        _text(img, "CAMERA", (OX - 22, 712), .38)
        t = _top(TURRET_X, 0)
        cv2.drawMarker(img, t, LASER, cv2.MARKER_DIAMOND, 11, 1, AA)
        _text(img, "TURRET", (t[0] - 62, t[1] + 4), .38, _dim(LASER, .9))
        cv2.rectangle(img, (x0, y0), (x1, y1), (50, 110, 60), 1)
        _text(img, "TOP VIEW  -  TO SCALE, cm", (14, 62), .42)

        cv2.rectangle(img, (0, 0), (W, 36), (14, 30, 18), -1)
        _text(img, "AeroEye-3D", (14, 26), .72, INK, F2)
        _text(img, "STEREO TARGET TRACKING CONSOLE", (190, 25), .46)

        _panel(img, 792, 40, 484, 250, "SIDE VIEW  -  height vs distance, cm")
        self._draw_side(img)
        _panel(img, 792, 296, 484, 196, "TARGET DATA")
        _panel(img, 792, 498, 484, 218, "RANGE HISTORY  -  last 10 s")
        x, y, w, h = CHART
        for r in range(20, MAX_CM + 1, 20):
            yy = int(y + h - (r / MAX_CM) * h)
            cv2.line(img, (x, yy), (x + w, yy), (30, 60, 36), 1)
            _text(img, str(r), (x + w - 22, yy - 3), .32, (70, 150, 85))

        img[::3] = (img[::3] * 0.86).astype(np.uint8)   # faint CRT scan lines
        return img

    def _draw_side(self, img):
        x0, y0, x1, y1 = SIDE
        s = img[y0:y1, x0:x1]
        o = (SX, SY)
        R = int(MAX_CM * SIDE_PX_CM)
        cv2.ellipse(s, o, (R, R), 0, -HALF_VFOV, HALF_VFOV, _dim(INK, .13), -1, AA)
        for r in range(20, MAX_CM + 1, 20):
            cv2.ellipse(s, o, (int(r * SIDE_PX_CM),) * 2, 0, -HALF_VFOV, HALF_VFOV, _dim(INK, .45), 1, AA)
            _text(s, f"{r}", (SX + int(r * SIDE_PX_CM) - 10, y1 - y0 - 6), .34, _dim(INK, .8))
        for sg in (-1, 1):
            a = math.radians(sg * HALF_VFOV)
            cv2.line(s, o, (int(SX + R * math.cos(a)), int(SY + R * math.sin(a))), INK, 1, AA)
        cv2.line(s, o, (SX + R, SY), _dim(INK, .4), 1, AA)
        cv2.circle(s, o, 4, INK, -1, AA)
        cv2.drawMarker(s, _side(TURRET_Y, 0), LASER, cv2.MARKER_DIAMOND, 9, 1, AA)

    def _draw_boot_background(self):
        img = np.full((H, W, 3), BG, np.uint8)
        for r in range(120, 900, 120):
            cv2.circle(img, (W // 2, H // 2 - 110), r, (16, 30, 19), 1, AA)
        for a in range(0, 360, 30):
            q = (int(W / 2 + 900 * math.cos(math.radians(a))),
                 int(H / 2 - 110 + 900 * math.sin(math.radians(a))))
            cv2.line(img, (W // 2, H // 2 - 110), q, (12, 24, 15), 1, AA)
        (tw, _), _ = cv2.getTextSize("AeroEye-3D", F2, 2.4, 2)
        cv2.putText(img, "AeroEye-3D", ((W - tw) // 2, 190), F2, 2.4, INK, 2, AA)
        s = "STEREO VISION  //  LASER TRACKING SYSTEM"
        (tw, _), _ = cv2.getTextSize(s, F1, .6, 1)
        _text(img, s, ((W - tw) // 2, 232), .6, SOFT)
        return img

    def _make_clutter(self):
        """Faint specks the sweep lights up, so the scope reads as a live scan.
        Purely decorative; they are never mistaken for the target by the eye
        because they are dim, small and gone within a second."""
        n = 150
        rng = np.random.default_rng(7)
        az = rng.uniform(-HALF_FOV, HALF_FOV, n)
        r = 22 + 76 * rng.random(n) ** 0.7
        xs = np.round(OX + r * np.sin(np.radians(az)) * PX_CM).astype(int)
        ys = np.round(OY - r * np.cos(np.radians(az)) * PX_CM).astype(int)
        self._cl_az = np.repeat(az, 4)                  # each speck is 2x2 px
        self._cl_x = np.repeat(xs, 4) + np.tile((0, 1, 0, 1), n)
        self._cl_y = np.repeat(ys, 4) + np.tile((0, 0, 1, 1), n)
        self._cl_gain = np.repeat(rng.uniform(.35, 1.0, n), 4)

    # ---- boot ---------------------------------------------------------------
    def _boot(self, now):
        lines = (("STEREO CAMERA", "2 x 320x240"), ("DEPTH MODEL", "LOADED"),
                 ("SERVO LINK", self.port), ("AIM CALIBRATION", "LOADED"),
                 ("LASER INTERLOCK", "ARMED"), ("RADAR DISPLAY", "ONLINE"))
        t = now - self.t0
        n = min(len(lines), int(t / BOOT_STEP_S))
        if n > self.boot_n:
            self.boot_n = n
            self.sound.play("tick")
        img = self._boot_bg.copy()
        for i in range(n):
            y = 310 + i * 40
            _text(img, lines[i][0], (400, y), .62, INK, F2)
            (tw, _), _ = cv2.getTextSize(lines[i][1], F2, .62, 1)
            for x in range(640, 880 - tw - 12, 8):      # dots run up to the value
                cv2.circle(img, (x, y - 2), 1, _dim(SOFT, .6), -1, AA)
            _text(img, lines[i][1], (880 - tw, y), .62, WHITE, F2)
        x, y, w = 400, 590, 480
        cv2.rectangle(img, (x, y), (x + w, y + 12), SOFT, 1, AA)
        frac = min(1.0, t / (len(lines) * BOOT_STEP_S))
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + int((w - 4) * frac), y + 10), INK, -1)
        if n == len(lines):
            s = "SYSTEM READY"
            (tw, _), _ = cv2.getTextSize(s, F2, .9, 1)
            _text(img, s, ((W - tw) // 2, 650), .9, INK if int(t * 8) % 2 else WHITE, F2)
        return img

    # ---- dynamic layers -----------------------------------------------------
    def _sweep(self, img, now):
        span = HALF_FOV + 5
        ph = (now * SWEEP_DPS) % (4 * span)
        a = ph - span if ph < 2 * span else 3 * span - ph
        d = 1 if ph < 2 * span else -1
        x0, y0, x1, y1 = SCOPE
        layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        o = (OX - x0, OY - y0)
        for i in range(13, 0, -1):
            a0, a1 = a - d * i * 1.8, a - d * (i - 1) * 1.8
            if abs(a0) > span or abs(a1) > span:
                continue
            p0, p1 = _polar(a0, MAX_CM), _polar(a1, MAX_CM)
            cv2.fillConvexPoly(layer, np.array([o, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0)]),
                               _dim(INK, .22 * (1 - i / 14) ** 1.6), AA)
        roi = img[y0:y1, x0:x1]
        cv2.add(roi, layer, roi)
        cv2.line(img, (OX, OY), _polar(a, MAX_CM), _dim(INK, .95), 1, AA)

        behind = (a - self._cl_az) * d
        g = np.clip(1 - behind / 45.0, 0, 1) * (behind > 0) * self._cl_gain * 150
        c = np.stack((g * .45, g, g * .5), 1).astype(np.uint8)
        img[self._cl_y, self._cl_x] = np.maximum(img[self._cl_y, self._cl_x], c)

        self._on_sweep(a, d, now)

    def _on_sweep(self, a, d, now):
        """Ping as the beam crosses the target; a soft tick at each end of the
        scan when there is nothing to find."""
        prev = self.prev_sweep
        self.prev_sweep = (a, d)
        if prev is None:
            return
        if self.state in ("LOCK", "TRACK"):
            az = math.degrees(math.atan2(self.pos[0], self.pos[2]))
            if (prev[0] - az) * (a - az) <= 0 and prev[0] != a:
                self.rings.append((now, self.trail[-1][4]))
                self.sound.play("ping")
        elif self.state == "SEARCH" and prev[1] != d:
            self.sound.play("scan")

    def _track(self, img, now):
        pts = list(self.trail)
        side = img[SIDE[1]:SIDE[3], SIDE[0]:SIDE[2]]
        for a, b in zip(pts[:-1], pts[1:]):
            if b[0] - a[0] > GAP_S:
                continue
            age = (now - b[0]) / TRAIL_S
            c = _dim(INK, max(0.0, 1 - age) ** 1.3)
            cv2.line(img, a[4], b[4], c, 2 if age < .35 else 1, AA)
            cv2.line(side, a[5], b[5], c, 1, AA)

    def _beam(self, img, side):
        if not self.laser or self.pan is None or self.pos is None:
            return
        X, Y, Z = self.pos
        a = math.radians((self.pan - PAN_ZERO) / PAN_GAIN)
        L = math.hypot(X - TURRET_X, Z)
        p0, p1 = _top(TURRET_X, 0), _top(TURRET_X + L * math.sin(a), L * math.cos(a))
        cv2.line(img, p0, p1, _dim(LASER, .3), 3, AA)
        cv2.line(img, p0, p1, LASER, 1, AA)
        _add_sprite(img, self._hit, *p1)
        if self.tilt is not None:
            e = math.radians((self.tilt - TILT_ZERO) / TILT_GAIN)
            L = math.hypot(Y - TURRET_Y, Z)
            q0, q1 = _side(TURRET_Y, 0), _side(TURRET_Y + L * math.sin(e), L * math.cos(e))
            cv2.line(side, q0, q1, _dim(LASER, .3), 2, AA)
            cv2.line(side, q0, q1, LASER, 1, AA)

    def _target(self, img, now):
        X, Y, Z = self.pos
        vx, vy, vz = self.vel
        side = img[SIDE[1]:SIDE[3], SIDE[0]:SIDE[2]]
        p, q = self.trail[-1][4], self.trail[-1][5]
        self._beam(img, side)

        if self.state == "LOCK":
            col = LOCK
        else:
            col = AMBER
        moving = self.state != "COAST" and math.hypot(vx, vy, vz) > ARROW_MIN
        if moving:
            tip = _top(X + vx * ARROW_S, Z + vz * ARROW_S)
            cv2.arrowedLine(img, p, tip, WHITE, 2, AA, tipLength=.22)
            tip = _side(Y + vy * ARROW_S, Z + vz * ARROW_S)
            cv2.arrowedLine(side, q, tip, WHITE, 1, AA, tipLength=.25)

        _add_sprite(img, self._glow, *p)
        cv2.circle(img, p, 5, WHITE, -1, AA)
        _add_sprite(side, self._glow, *q)
        cv2.circle(side, q, 4, WHITE, -1, AA)

        size = 17
        if self.state == "TRACK":       # reticle closes in on a new target
            size = 17 + int(30 * max(0.0, 1 - (now - self.lock_t) / ACQUIRE_S))
        _brackets(img, p, size, col)
        _brackets(side, q, 12, col, 1)

        self.rings = [r for r in self.rings if now - r[0] < RING_S]
        for t, c in self.rings:
            age = (now - t) / RING_S
            cv2.circle(img, c, int(14 + 70 * age), _dim(INK, 1 - age), 1, AA)

        rng = math.hypot(X, Z)
        az = math.degrees(math.atan2(X, Z))
        flip = p[0] > SCOPE[2] - 170        # keep the label on the scope
        a = (p[0] + (-13 if flip else 13), p[1] - 13)
        b = (p[0] + (-26 if flip else 26), p[1] - 30)
        e = (b[0] + (-110 if flip else 110), b[1])
        cv2.line(img, a, b, WHITE, 1, AA)
        cv2.line(img, b, e, WHITE, 1, AA)
        tx = min(b[0], e[0]) + 4
        _text(img, "TGT-01", (tx, b[1] - 20), .4, WHITE)
        _text(img, f"{rng:.0f} cm  {az:+.0f} deg", (tx, b[1] - 6), .4, WHITE)

    def _data(self, img):
        x0, x1, y = 808, 1040, 332
        if self.state == "SEARCH" or self.pos is None:
            vals = ("---",) * 6
        else:
            X, Y, Z = self.pos
            vx, vy, vz = self.vel
            r = math.hypot(X, Z)
            vals = (f"{r:.1f} cm", f"{math.degrees(math.atan2(X, Z)):+.1f} deg",
                    f"{math.degrees(math.atan2(Y, r)):+.1f} deg", f"{Y:+.1f} cm",
                    f"{math.hypot(vx, vy, vz):.0f} cm/s", f"{round(-(vx * X + vz * Z) / max(r, 1)):+d} cm/s")
        labels = ("RANGE", "AZIMUTH", "ELEVATION", "HEIGHT", "SPEED", "CLOSING")
        col = INK if self.state != "COAST" else AMBER
        for i in range(6):
            xx, yy = (x1 if i % 2 else x0), y + (i // 2) * 62
            _text(img, labels[i], (xx, yy), .38)
            _text(img, vals[i], (xx, yy + 27), .7, col, F2)

    def _chart(self, img, now):
        x, y, w, h = CHART
        if len(self.hist) < 2:
            return
        t = np.array([s[0] for s in self.hist])
        r = np.array([s[1] for s in self.hist])
        pts = np.stack((x + w - (now - t) / HISTORY_S * w, y + h - r / MAX_CM * h), 1).astype(np.int32)
        cuts = np.where(np.diff(t) > GAP_S)[0] + 1
        cv2.polylines(img, [c for c in np.split(pts, cuts) if len(c) > 1], False, INK, 1, AA)
        if now - t[-1] < GAP_S:
            _add_sprite(img, self._glow_small, int(pts[-1][0]), int(pts[-1][1]))

    def _header(self, img, now):
        lit = {"SEARCH": SOFT, "TRACK": AMBER, "LOCK": LOCK, "LASER": LASER}
        for name, x in PILLS:
            on = (self.state == name or (name == "TRACK" and self.state == "COAST")
                  or (name == "LASER" and self.laser))
            label = "COAST" if (name == "TRACK" and self.state == "COAST") else name
            _pill(img, x, 6, PILL_W, 25, lit[name] if on else OFF, label, fill=.3 if on else .12)
        t = now - self.t0
        _text(img, f"T+ {int(t // 60):02d}:{t % 60:04.1f}", (620, 26), .55, SOFT, F2)
        if self.sound.muted:
            _text(img, "MUTE", (760, 25), .4, AMBER)

    def _console(self, now):
        img = self._base.copy()
        self._sweep(img, now)
        if self.pos is not None and self.trail:
            self._track(img, now)
            if self.state != "SEARCH":
                self._target(img, now)
        if self.state == "SEARCH" and int(now * 1.6) % 2:
            s = "SCANNING FOR TARGET"
            (tw, _), _ = cv2.getTextSize(s, F2, .8, 1)
            _text(img, s, (OX - tw // 2, 330), .8, INK, F2)
        self._data(img)
        self._chart(img, now)
        self._header(img, now)
        age = now - self.flash_t
        if age < FLASH_S:
            cv2.rectangle(img, SCOPE[:2], SCOPE[2:], _dim(LOCK, 1 - age / FLASH_S), 3)
            if int(age * 9) % 2 == 0:
                _pill(img, OX - 150, 76, 300, 44, LOCK, "TARGET LOCKED", .85, .4)
        return img

    def canvas(self):
        now = time.time()
        if now < self.boot_end:
            return self._boot(now)
        if not self.ready:
            self.ready = True
            self.t0 = now
            self.sound.play("ready")
        img = self._console(now)
        age = now - self.t0
        if age < REVEAL_S and self.boot_n:      # the console opens out of the lenses
            mask = np.zeros((H, W), np.uint8)
            cv2.circle(mask, (OX, OY), int(1500 * age / REVEAL_S), 255, -1)
            img = cv2.bitwise_and(img, img, mask=mask)
        return img
