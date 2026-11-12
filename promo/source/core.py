# AeroEye-3D promo: shared drawing, grading and footage helpers
import math
import os
import pickle
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bidi import get_display

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1920, 1080, 30
BEAT, BAR = 0.6, 2.4

# palette (BGR)
INK = (240, 246, 244)
DIM = (150, 150, 135)
CY = (255, 222, 70)
CYD = (120, 105, 30)
RED = (60, 40, 255)
AMB = (40, 176, 255)
GRN = (150, 255, 90)
VIO = (235, 90, 140)
BG = (16, 11, 7)

FONTS = {
    "head": r"C:\Windows\Fonts\gishabd.ttf",
    "body": r"C:\Windows\Fonts\gisha.ttf",
    "ui": r"C:\Windows\Fonts\seguisb.ttf",
    "mono": r"C:\Windows\Fonts\consola.ttf",
    "monob": r"C:\Windows\Fonts\consolab.ttf",
    "logo": r"C:\Windows\Fonts\bahnschrift.ttf",
}


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, u):
    return a + (b - a) * u


def ss(a, b, x):
    u = clamp((x - a) / (b - a)) if b != a else float(x >= b)
    return u * u * (3 - 2 * u)


def eout(u):
    u = clamp(u)
    return 1 - (1 - u) ** 3


def eback(u, k=1.7):
    u = clamp(u) - 1
    return u * u * ((k + 1) * u + k) + 1


# ---------- text ----------
_tcache = {}


def tsprite(s, size, font="head", var=None):
    key = (s, size, font, var)
    if key in _tcache:
        return _tcache[key]
    f = ImageFont.truetype(FONTS[font], size)
    if var:
        try:
            f.set_variation_by_name(var)
        except Exception:
            pass
    shown = get_display(s)
    x0, y0, x1, y1 = f.getbbox(shown)
    pad = 4
    im = Image.new("L", (x1 - x0 + pad * 2, y1 - y0 + pad * 2), 0)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), shown, fill=255, font=f)
    a = np.array(im)
    _tcache[key] = a
    return a


def blit_a(fr, a, x, y, color, op=1.0):
    """Blend colour through alpha sprite `a` with its top-left at (x, y)."""
    h, w = a.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or op <= 0:
        return
    al = a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] / 255 * op
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (roi * (1 - al) + np.array(color, np.float32) * al).astype(np.uint8)


def text(fr, s, x, y, size, color=INK, font="head", anchor="r", op=1.0, var=None):
    """anchor: r/l/c horizontally; (x, y) is the top edge of the text."""
    a = tsprite(s, size, font, var)
    ox = {"r": x - a.shape[1], "l": x, "c": x - a.shape[1] // 2}[anchor]
    blit_a(fr, a, ox, y, color, op)
    return a.shape[1], a.shape[0]


def shadow_text(fr, s, x, y, size, color=INK, font="head", anchor="r", op=1.0, sh=0.8, var=None):
    a = tsprite(s, size, font, var)
    b = cv2.GaussianBlur(cv2.copyMakeBorder(a, 18, 18, 18, 18, cv2.BORDER_CONSTANT), (0, 0), 7)
    ox = {"r": x - a.shape[1], "l": x, "c": x - a.shape[1] // 2}[anchor]
    blit_a(fr, np.clip(b.astype(np.float32) * 1.7, 0, 255).astype(np.uint8), ox - 18, y - 18, (0, 0, 0), sh * op)
    blit_a(fr, a, ox, y, color, op)


def title(fr, s, x, y, size, lt, dur, color=INK, anchor="c", font="head", rise=26, fade=0.22):
    """Text that rises in and fades out over its lifetime lt in [0, dur]."""
    if lt < 0 or lt > dur:
        return
    op = ss(0, fade, lt) * (1 - ss(dur - fade, dur, lt))
    shadow_text(fr, s, x, y + (1 - eout(lt / 0.45)) * rise, size, color, font, anchor, op)


HEB = "אבגדהוזחטיכלמנסעפצקרשת"
HEX = "0123456789ABCDEF"


def scramble(s, u, seed=3):
    """Text resolving from noise, right-to-left for Hebrew. u in 0..1."""
    if u >= 1:
        return s
    rnd = random.Random(int(seed + u * 40))
    n = len(s)
    done = int(n * clamp(u))
    out = []
    for k, c in enumerate(s):
        if k < done or c == " ":
            out.append(c)
        else:
            out.append(rnd.choice(HEB if "\u0590" <= c <= "\u05ff" else HEX))
    return "".join(out)


# ---------- layers ----------
def add_glow(fr, ov, k=0.9, sigma=5):
    small = cv2.resize(ov, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(small, (0, 0), sigma)
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(fr, ov)
    return cv2.add(out, cv2.convertScaleAbs(b, alpha=k * 2.2))


def blank():
    return np.zeros((H, W, 3), np.uint8)


def bgfill(c=BG):
    f = np.empty((H, W, 3), np.uint8)
    f[:] = c
    return f


_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2)
VIG = np.clip(1.08 - 0.62 * _r ** 2.2, 0.08, 1).astype(np.float32)[..., None]
_rng = np.random.default_rng(7)
GRAIN = [cv2.GaussianBlur(_rng.normal(0, 5.5, (H, W)).astype(np.float32), (0, 0), 0.7) for _ in range(6)]


def finish(fr, k, grain=1.0):
    f = fr.astype(np.float32)
    f += (GRAIN[k % 6] * grain)[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def spot(cx, cy, rx, ry, lo=0.1, power=2.0):
    d = np.sqrt(((_xx - cx) / rx) ** 2 + ((_yy - cy) / ry) ** 2)
    return (lo + (1 - lo) * np.exp(-d ** power)).astype(np.float32)[..., None]


def aline(ov, p, q, c, th=1):
    cv2.line(ov, (int(p[0] * 16), int(p[1] * 16)), (int(q[0] * 16), int(q[1] * 16)), c, th, cv2.LINE_AA, 4)


def acircle(ov, p, r, c, th=1):
    cv2.circle(ov, (int(p[0] * 16), int(p[1] * 16)), int(r * 16), c, th, cv2.LINE_AA, 4)


def aarc(ov, p, r, a0, a1, c, th=2):
    cv2.ellipse(ov, (int(p[0] * 16), int(p[1] * 16)), (int(r * 16), int(r * 16)), 0, a0, a1, c, th, cv2.LINE_AA, 4)


def scale(c, k):
    return tuple(int(v * k) for v in c)


def reticle(ov, x, y, r, t, c, u=1.0, ticks=True):
    """Rotating lock reticle: u=0 wide and thin, u=1 tight on the target."""
    rr = r * lerp(2.1, 1.28, eout(u))
    rot = t * 60
    for k in range(3):
        a = rot + k * 120
        aarc(ov, (x, y), rr, a, a + 78, c, 3)
    aarc(ov, (x, y), rr * 1.16, -rot * 1.4, -rot * 1.4 + 40, scale(c, .55), 2)
    if ticks:
        for a in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            aline(ov, (x + ca * rr * 1.2, y + sa * rr * 1.2), (x + ca * rr * 1.46, y + sa * rr * 1.46), c, 2)


def brackets(ov, x0, y0, x1, y1, L, c, th=2):
    for (a, b, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        aline(ov, (a, b), (a + sx * L, b), c, th)
        aline(ov, (a, b), (a, b + sy * L), c, th)


def panel(fr, x0, y0, x1, y1, op=0.72, edge=CYD, fillc=(14, 10, 6)):
    roi = fr[y0:y1, x0:x1].astype(np.float32)
    fr[y0:y1, x0:x1] = (roi * (1 - op) + np.array(fillc, np.float32) * op).astype(np.uint8)
    cv2.rectangle(fr, (x0, y0), (x1 - 1, y1 - 1), edge, 1, cv2.LINE_AA)


def chrome(fr, ov, t, state, state_c=CY, tag=None, op=1.0):
    """Frame furniture for live-feed shots."""
    m = 44
    brackets(ov, m, m, W - m, H - m, 54, scale(CY, .55 * op), 2)
    text(fr, "AEROEYE-3D", m + 22, m + 18, 24, CY, "monob", "l", .85 * op)
    text(fr, f"T+{t:06.2f}", m + 22, m + 52, 22, INK, "mono", "l", .75 * op)
    if state:
        shadow_text(fr, state, W // 2, H - m - 74, 46, state_c, "head", "c", op)
    if tag:
        text(fr, tag, m + 22, m + 84, 24, INK, "ui", "l", .8 * op)


# ---------- footage ----------
class Reader:
    def __init__(s, path):
        s.c = cv2.VideoCapture(path)
        s.pos = -1
        s.cache = {}

    def get(s, i):
        i = int(i)
        if i in s.cache:
            return s.cache[i]
        if i != s.pos + 1:
            s.c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = s.c.read()
        s.pos = i
        s.cache[i] = f
        if len(s.cache) > 8:
            del s.cache[min(s.cache, key=lambda k: abs(k - i) * -1)]
        return f


def grade(bgr, dark=1.0, laser=True, sat_wall=0.0):
    """Night-teal look; keeps the real laser's red as the only hot colour."""
    f = bgr.astype(np.float32) / 255
    b, g, r = cv2.split(f)
    L = 0.114 * b + 0.587 * g + 0.299 * r
    Lc = np.clip((L - 0.07) / 0.74, 0, 1) ** 1.45 * 0.66 * dark
    out = cv2.merge([Lc * 1.14 + 0.012, Lc * 1.0 + 0.006, Lc * 0.8])
    if laser:
        m = np.clip((r - np.maximum(g, b) - 0.15) * 5, 0, 1)
        m = np.maximum(m, np.clip((L - 0.76) * 11, 0, 1))
        small = cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2), (W, H)) * 2.4
        bloom2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 9), (W, H)) * 2.2
        out = out * (1 - m[..., None])
        out += m[..., None] * np.array([0.86, 0.80, 1.0], np.float32)
        out += bloom[..., None] * np.array([0.20, 0.08, 0.95], np.float32)
        out += bloom2[..., None] * np.array([0.10, 0.02, 0.50], np.float32)
    return np.clip(out * VIG * 255, 0, 255).astype(np.uint8)


def cam(fr, cx, cy, z, shake=(0, 0)):
    z = max(z, 1.0)
    hw, hh = W / 2 / z, H / 2 / z
    cx, cy = clamp(cx, hw, W - hw), clamp(cy, hh, H - hh)
    M = np.float32([[z, 0, W / 2 - z * cx + shake[0]], [0, z, H / 2 - z * cy + shake[1]]])
    return cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC if z > 1.05 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def to_view(p, cx, cy, z):
    z = max(z, 1.0)
    hw, hh = W / 2 / z, H / 2 / z
    cx, cy = clamp(cx, hw, W - hw), clamp(cy, hh, H - hh)
    return (W / 2 + (p[0] - cx) * z, H / 2 + (p[1] - cy) * z)


DATA = pickle.load(open(os.path.join(HERE, "data.pkl"), "rb"))
