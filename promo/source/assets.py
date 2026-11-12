# AeroEye-3D promo: footage objects (phone clip, rig photos, sensor recording, real detector stages)
import glob
import json
import os
import pickle
import sys

import cv2
import numpy as np

from core import *

VISION = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision"))
sys.path.insert(0, VISION)
import detector_dark as dd

ROOT = os.path.join(VISION, "debug_images")
PH = DATA["phone"]
TE = DATA["tele"]
RAWP = pickle.load(open(os.path.join(HERE, "phone.pkl"), "rb"))
_reader = Reader(os.path.join(HERE, "phone.mp4"))
_plates = {}


def plate_for(i):
    """Empty-wall plate: median of frames around i, the ball moves so it drops out."""
    key = (i // 45) * 45
    if key not in _plates:
        if len(_plates) > 3:
            _plates.pop(next(iter(_plates)))
        c = cv2.VideoCapture(os.path.join(HERE, "phone.mp4"))
        fr = []
        for off in (-80, -55, -30, 30, 55, 80, 105):
            c.set(cv2.CAP_PROP_POS_FRAMES, max(0, key + off))
            ok, f = c.read()
            if ok:
                fr.append(f)
        c.release()
        _plates[key] = np.median(np.stack(fr), axis=0).astype(np.uint8)
    return _plates[key]


def red_mask(f):
    b, g, r = [f[:, :, k].astype(np.int16) for k in range(3)]
    m = ((r - np.maximum(g, b)) > 28).astype(np.uint8)
    return cv2.dilate(m, np.ones((9, 9), np.uint8))


def strip_laser(f):
    """Remove the dot (for story beats before the laser is introduced)."""
    m = red_mask(f)
    if m.sum() == 0:
        return f
    ys, xs = np.where(m)
    x0, x1, y0, y1 = max(xs.min() - 14, 0), min(xs.max() + 14, W), max(ys.min() - 14, 0), min(ys.max() + 14, H)
    out = f.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(f[y0:y1, x0:x1], m[y0:y1, x0:x1], 5, cv2.INPAINT_TELEA)
    return out


def disk(shape, x, y, r, feather=5):
    m = np.zeros(shape, np.float32)
    cv2.circle(m, (int(x), int(y)), int(r), 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), feather)


def dot_at(i):
    r = RAWP[i]["dot"]
    return None if r is None else (r[0], r[1])


def phone_ball(fi):
    i = int(fi)
    u = fi - i
    j = min(i + 1, len(PH["bx"]) - 1)
    return (lerp(PH["bx"][i], PH["bx"][j], u), lerp(PH["by"][i], PH["by"][j], u), lerp(PH["br"][i], PH["br"][j], u))


def phone_at(fi, laser=True, snap=False):
    """Frame fi (float). Between frames: ball moved as a sprite over a clean wall
    so slow motion stays sharp instead of ghosting. snap holds frame i's pixels
    (no cross-fade) for the instant the dot jumps from the wall to the ball."""
    i = int(fi)
    u = fi - i
    if snap:
        return _reader.get(i).copy()
    a = _reader.get(i)
    if u < 0.04:
        return a.copy() if laser else strip_laser(a)
    b = _reader.get(i + 1)
    if not laser:
        a, b = strip_laser(a), strip_laser(b)
    xa, ya, ra = PH["bx"][i], PH["by"][i], PH["br"][i]
    xb, yb, rb = PH["bx"][i + 1], PH["by"][i + 1], PH["br"][i + 1]
    if not (PH["seen"][i] and PH["seen"][i + 1]):
        return cv2.addWeighted(a, 1 - u, b, u, 0)
    xm, ym = lerp(xa, xb, u), lerp(ya, yb, u)
    pl = plate_for(i)
    pad = int(max(ra, rb) * 1.55)
    x0, x1 = int(max(min(xa, xb) - pad, 0)), int(min(max(xa, xb) + pad, W))
    y0, y1 = int(max(min(ya, yb) - pad, 0)), int(min(max(ya, yb) + pad, H))
    A, B, P = [v[y0:y1, x0:x1].astype(np.float32) for v in (a, b, pl)]
    sh = A.shape[:2]
    Ma = disk(sh, xa - x0, ya - y0, ra * 1.16)
    Mb = disk(sh, xb - x0, yb - y0, rb * 1.16)
    bg = (A * (1 - Ma[..., None]) + P * Ma[..., None]) * (1 - u) + (B * (1 - Mb[..., None]) + P * Mb[..., None]) * u
    Ta = np.float32([[1, 0, xm - xa], [0, 1, ym - ya]])
    Tb = np.float32([[1, 0, xm - xb], [0, 1, ym - yb]])
    As = cv2.warpAffine(A, Ta, (sh[1], sh[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Bs = cv2.warpAffine(B, Tb, (sh[1], sh[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Mas = cv2.warpAffine(Ma, Ta, (sh[1], sh[0]))
    Mbs = cv2.warpAffine(Mb, Tb, (sh[1], sh[0]))
    M = ((1 - u) * Mas + u * Mbs)[..., None]
    out = a.copy()
    out[y0:y1, x0:x1] = np.clip(bg * (1 - M) + ((1 - u) * As + u * Bs) * M, 0, 255).astype(np.uint8)
    return out


# ---------- rig photos ----------
class Photo:
    def __init__(s, n):
        im = cv2.imread(os.path.join(HERE, "rig", f"rig_{n}.jpg"))
        im = cv2.fastNlMeansDenoisingColored(im, None, 2.5, 3, 5, 15)
        s.k = 2.0
        up = cv2.resize(im, None, fx=s.k, fy=s.k, interpolation=cv2.INTER_LANCZOS4)
        bl = cv2.GaussianBlur(up, (0, 0), 2.2)
        s.im = cv2.addWeighted(up, 1.55, bl, -0.55, 0)

    def view(s, cx, cy, z, sx=W / 2, sy=H / 2):
        """Original-photo point (cx, cy) lands on screen point (sx, sy) at z screen px per photo px."""
        q = z / s.k
        M = np.float32([[q, 0, sx - q * cx * s.k], [0, q, sy - q * cy * s.k]])
        return cv2.warpAffine(s.im, M, (W, H), flags=cv2.INTER_AREA if q < 1 else cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    @staticmethod
    def pt(p, cx, cy, z, sx=W / 2, sy=H / 2):
        return (sx + (p[0] - cx) * z, sy + (p[1] - cy) * z)


def photo_grade(f, light, sat=0.8, dark=0.92, dof=None, gam=1.35):
    """Moody low-key relight: keeps the purple mount and copper laser alive."""
    v = f.astype(np.float32) / 255
    if dof is not None:
        small = cv2.GaussianBlur(cv2.resize(v, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4)
        v = v * dof + cv2.resize(small, (W, H)) * (1 - dof)
    L = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)[..., None]
    v = L + (v - L) * sat
    v = np.clip(v, 0, 1) ** gam * dark
    v *= np.array([1.08, 1.0, 0.92], np.float32)
    v = v * light + np.array([0.022, 0.014, 0.006], np.float32) * (1 - np.clip(light, 0, 1))
    return np.clip(v * VIG * 255, 0, 255).astype(np.uint8)


# ---------- sensor recording ----------
class Sensor:
    def __init__(s):
        run = os.path.join(ROOT, DATA["RUN"])
        s.t = TE["t"]
        s.fr = [cv2.imread(f"{run}/{i:06d}.png", 0).astype(np.float32) for i in TE["i"]]
        s.plate = np.median(np.stack(s.fr[::6]), axis=0)

    def _state(s, tt):
        k = int(np.clip(np.searchsorted(s.t, tt) - 1, 0, len(s.t) - 2))
        return k, clamp((tt - s.t[k]) / (s.t[k + 1] - s.t[k]))

    def get(s, key, tt):
        k, u = s._state(tt)
        return lerp(TE[key][k], TE[key][k + 1], u)

    def dot(s, tt):
        k, u = s._state(tt)
        a, b = TE["dot"][k], TE["dot"][k + 1]
        if a is None or b is None:
            return None
        return (lerp(a[0], b[0], u), lerp(a[1], b[1], u))

    def hit(s, tt):
        k, u = s._state(tt)
        return TE["hit"][k + (1 if u > .5 else 0)]

    def at(s, tt):
        k, u = s._state(tt)
        a, b = s.fr[k], s.fr[k + 1]
        xa, ya, ra = TE["x"][k], TE["y"][k], TE["r"][k]
        xb, yb, rb = TE["x"][k + 1], TE["y"][k + 1], TE["r"][k + 1]
        xm, ym = lerp(xa, xb, u), lerp(ya, yb, u)
        g = np.median(a) / np.median(s.plate)
        Ma = disk(a.shape, xa, ya, ra + 6, 1.6)
        Mb = disk(b.shape, xb, yb, rb + 6, 1.6)
        bg = (a * (1 - Ma) + s.plate * g * Ma) * (1 - u) + (b * (1 - Mb) + s.plate * g * Mb) * u
        Ta, Tb = np.float32([[1, 0, xm - xa], [0, 1, ym - ya]]), np.float32([[1, 0, xm - xb], [0, 1, ym - yb]])
        A = cv2.warpAffine(a, Ta, (320, 240), borderMode=cv2.BORDER_REPLICATE)
        B = cv2.warpAffine(b, Tb, (320, 240), borderMode=cv2.BORDER_REPLICATE)
        M = (1 - u) * cv2.warpAffine(Ma, Ta, (320, 240)) + u * cv2.warpAffine(Mb, Tb, (320, 240))
        return np.clip(bg * (1 - M) + ((1 - u) * A + u * B) * M, 0, 255).astype(np.uint8)


def sensor_up(g, w, h, tint=True):
    """Upscale a mono sensor frame to a panel with the teal sensor look."""
    u = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    u = np.clip((u - 0.08) / 0.8, 0, 1) ** 1.25
    c = np.stack([u * 0.98, u * 1.0, u * 0.78], -1)
    c *= (0.93 + 0.07 * np.sin(np.arange(h, dtype=np.float32) * 1.57))[:, None, None]
    return np.clip(c * 255, 0, 255).astype(np.uint8)


def detector_stages():
    """Real pipeline stages from detector_dark.py on one recorded frame."""
    k = 28
    run = os.path.join(ROOT, DATA["RUN"])
    gray = cv2.imread(f"{run}/{TE['i'][k]:06d}.png", 0)
    d = dd.DarkBallDetector()
    bg = d.background(gray)
    dark = d.darkness(gray, bg)
    thr = max(dd.FLOOR_DELTA, 0.42 * float(dark.max()))
    mask = d._clean(dark, thr) if True else None
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    t = d.detect(gray)
    return dict(gray=gray, bg=bg, dark=dark, mask=mask, dist=dist, t=t, k=k)


def stereo_pair():
    _, f, xl, yl, rl, xr, yr, rr = DATA["stereo"]
    return cv2.imread(f, 0), cv2.imread(f.replace("_L", "_R"), 0), (xl, yl, rl), (xr, yr, rr)


CODE = """def aim(x, y, disp):
    feats = [x, y, disp, x*y, x*x, y*y, 1.0]
    pan  = sum(c*f for c, f in zip(PAN_C, feats))
    tilt = sum(c*f for c, f in zip(TILT_C, feats))
    pan  += CORR["pan"][-1]
    tilt += CORR["tilt"][-1]
    return int(round(pan)), int(round(tilt))

gray = stereo.split_lr(frame)
target = detector.detect(gray)
disp = target_L.x - target_R.x
lead = velocity * LEAD_S   # 0.15 s
ser.write(f"{pan},{tilt},{laser}\\n")""".split("\n")
