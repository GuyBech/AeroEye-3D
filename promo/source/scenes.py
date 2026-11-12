# AeroEye-3D promo: the scenes, on a 100 BPM grid (bar = 2.4 s)
import math
import random

import cv2
import numpy as np

from core import *
from assets import *

K = DATA["K"][DATA["RUN"]]
F_PX, KZ = DATA["F"], DATA["Kz"]
SENSOR = Sensor()
PHOTOS = {}


def ph(n):
    if n not in PHOTOS:
        PHOTOS[n] = Photo(n)
    return PHOTOS[n]


STAGES = detector_stages()
STEREO = stereo_pair()


# ---------- HOOK 0.0-7.2 ----------
def s_hook(lt, t, fn):
    if lt < 2.4:      # tight on the ball, out of the dark
        f = phone_at(1870 + lt * 30 * .6, laser=False)
        x, y, r = phone_ball(1870 + lt * 30 * .6)
        z = 2.3 + lt * .12
        g = grade(cam(f, x + 180, y, z), dark=.8 * ss(0, 1.1, lt) + .03, laser=False)
        title(g, "מטרה קטנה.", W - 330, 470, 92, lt - .5, 1.9)
        return g
    if lt < 4.2:      # fast sideways swing
        fi = 1356 + (lt - 2.4) * 30 * 1.0
        f = phone_at(fi, laser=False)
        x, y, r = phone_ball(fi)
        z = 1.75
        g = grade(cam(f, x, y, z, ((lt * 61 % 3) - 1.5, (lt * 47 % 3) - 1.5)), dark=.95, laser=False)
        title(g, "תנועה מהירה.", W // 2, 800, 92, lt - 2.55, 1.6)
        return g
    fr = bgfill((6, 4, 2))
    if lt < 5.4:
        title(fr, "אין זמן לחשוב.", W // 2, 470, 104, lt - 4.3, 1.1, fade=.12)
        return fr
    u = lt - 5.4
    op = ss(0, .2, u)
    shadow_text(fr, "אז המערכת חושבת", W // 2, 196, 84, INK, "head", "c", op)
    n = min(20, int(ss(.1, 1.25, u) * 20 + .001))
    shadow_text(fr, f"~{n}", W // 2, 440, 170, CY, "monob", "c", op)
    shadow_text(fr, "פעם בשנייה", W // 2, 770, 72, INK, "head", "c", op)
    ov = blank()
    ang = u * 900
    aarc(ov, (W / 2, 525), 196, ang, ang + 300 * ss(.1, 1.2, u), scale(CY, .9), 3)
    return add_glow(fr, ov, .7)


# ---------- logo ----------
def logomark(ov, cx, cy, s, u, c=CY):
    a = s * .36
    acircle(ov, (cx - a, cy), s * .62 * eout(u), c, 3)
    acircle(ov, (cx + a, cy), s * .62 * eout(u), c, 3)
    acircle(ov, (cx, cy), s * .14 * eout(u), RED, -1)
    L = s * 1.02 * eout(u)
    aline(ov, (cx - L - a, cy), (cx - s * .3, cy), scale(c, .8), 2)
    aline(ov, (cx + s * .3, cy), (cx + L + a, cy), scale(c, .8), 2)
    aline(ov, (cx, cy - L * .82), (cx, cy - s * .3), scale(c, .8), 2)
    aline(ov, (cx, cy + s * .3), (cx, cy + L * .82), scale(c, .8), 2)


def logo(fr, ov, cx, cy, size, u, sub=None):
    a = tsprite("AeroEye-3D", size, "logo", "Bold SemiCondensed")
    w = a.shape[1]
    logomark(ov, cx - w / 2 - size * .86, cy, size * .68, u)
    x = int(cx - w / 2 + size * .05)
    reveal = int(w * eout(clamp((u - .12) / .7)))
    if reveal > 4:
        blit_a(fr, a[:, :reveal], x, int(cy - a.shape[0] / 2), INK, ss(.05, .3, u))
        aline(ov, (x + reveal, cy - size * .6), (x + reveal, cy + size * .6), scale(CY, 1 - ss(.7, .95, u)), 2)
    if sub:
        shadow_text(fr, sub, int(cx), int(cy + size * .78), int(size * .36), DIM, "ui", "c", ss(.6, .95, u))


# ---------- REVEAL 7.2-9.6 ----------
def s_reveal(lt, t, fn):
    p = ph(1)
    z = 1.5 + lt * .035
    v = p.view(470, 690, z, 1350, 585)
    sweep = lerp(-300, 2300, ss(0, .95, lt))
    light = spot(1320, 600, 760, 640, .05) * (0.34 + .66 * ss(.15, 1.1, lt))
    d = (_xx - sweep) / 240
    light = light + (np.exp(-d * d)[..., None] * .55 * (1 - ss(.7, 1.4, lt))).astype(np.float32)
    fade = np.clip((_xx - 660) / 420, 0, 1)[..., None].astype(np.float32)
    g = photo_grade(v, np.clip(light, 0, 1.3) * fade, sat=.9)
    ov = blank()
    logo(g, ov, 560, 490, 104, ss(.22, 1.3, lt), "מערכת ביות ועקיבה אופטית בזמן אמת")
    g = add_glow(g, ov, .8)
    text(g, "תצלום אמיתי של המערכת", W - 70, H - 78, 22, INK, "ui", "r", .6 * ss(.6, 1, lt))
    if lt < .14:
        g = cv2.addWeighted(g, 1, bgfill((255, 255, 255)), (1 - lt / .14) * .9, 0)
    return g


_xx = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
_yy = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)


# ---------- PARTS 9.6-19.2 ----------
CHAIN = ["ראייה", "עיבוד", "בקרה", "ביצוע"]


def chainbar(fr, ov, k, u):
    xs = [1500, 1110, 720, 330]        # right to left, Hebrew reading order
    y = 972
    for j, x in enumerate(xs):
        c = CY if j == k else (RED if False else scale(CY, .35) if j > k else scale(CY, .75))
        if j < 3:
            aline(ov, (x - 92, y), (xs[j + 1] + 92, y), scale(CY, .28), 2)
            if j == k:
                aline(ov, (x - 92, y), (x - 92 - (x - xs[j + 1] - 184) * eout(u * 1.2), y), CY, 2)
        text(fr, CHAIN[j], x, y - 21, 34, INK if j <= k else DIM, "head", "c", 1 if j <= k else .55)
        acircle(ov, (x + 78, y), 6, c, -1 if j <= k else 2)


def part_text(fr, lt, head, sub, sub2=None, x=W - 90, y=330):
    u = lt
    op = ss(.05, .3, u) * (1 - ss(2.2, 2.4, u))
    text(fr, scramble(head, ss(.05, .5, u), 5), x, y, 74, CY, "head", "r", op)
    aline(fr, (x - 460 * eout(u / .6), y + 106), (x, y + 106), scale(CY, .7), 2)
    text(fr, sub, x, y + 128, 46, INK, "body", "r", ss(.3, .55, u) * (1 - ss(2.2, 2.4, u)))
    if sub2:
        text(fr, sub2, x, y + 192, 46, INK, "body", "r", ss(.45, .7, u) * (1 - ss(2.2, 2.4, u)))


def s_parts(lt, t, fn):
    k = int(lt // BAR)
    u = lt - k * BAR
    ov = blank()
    fadeR = np.clip((1500 - _xx) / 520, 0, 1)[..., None].astype(np.float32)
    if k == 0:      # stereo camera
        p = ph(8)
        z, c = 2.15 + u * .06, (372, 640)
        v = p.view(c[0], c[1], z, 640, 520)
        g = photo_grade(v, spot(640, 520, 700, 430, .1) * fadeR, sat=1.0, dof=spot(640, 540, 620, 300, 0, 3))
        for q in ((330, 656), (531, 519)):
            sp = Photo.pt(q, c[0], c[1], z, 640, 520)
            r = 96 + 40 * (1 - eout(u / .5))
            brackets(ov, sp[0] - r, sp[1] - r, sp[0] + r, sp[1] + r, 22, scale(CY, ss(.2, .5, u)), 2)
        part_text(g, u, "מצלמת סטריאו", "שתי עיניים. תמונת עומק אחת.")
    elif k == 1:    # the real code + the real detection
        g = bgfill((14, 10, 6))
        for j, line in enumerate(CODE):
            yy = 90 + j * 38 - u * 12
            vis = int(len(line) * ss(j * .05, j * .05 + .22, u))
            if vis:
                text(g, line[:vis], 80, int(yy), 27, (215, 235, 190) if j < 7 else CY, "mono", "l", .95)
        s = sensor_up(STAGES["gray"], 480, 360)
        g[625:625 + 300, 80:80 + 400] = cv2.resize(s, (400, 300))
        tk = STAGES["t"]
        q = (80 + tk.x * 400 / 320, 625 + tk.y * 300 / 240)
        reticle(ov, q[0], q[1], tk.radius * 400 / 320, u, GRN, ss(.2, 1, u))
        part_text(g, u, "מחשב + OpenCV", "זיהוי, עומק וחיזוי", "בכל פריים.")
    elif k == 2:    # Arduino
        p = ph(4)
        z, c = 2.1 + u * .06, (205, 590)
        v = p.view(c[0], c[1], z, 680, 520)
        g = photo_grade(v, spot(660, 520, 760, 500, .1) * fadeR, sat=.95)
        sp = Photo.pt((210, 575), c[0], c[1], z, 680, 520)
        brackets(ov, sp[0] - 340, sp[1] - 190, sp[0] + 320, sp[1] + 210, 30, scale(CY, ss(.2, .5, u)), 2)
        part_text(g, u, "Arduino UNO", "הופך זוויות לתנועה.", "UART · 115200 bps")
    else:           # turret
        p = ph(6)
        z, c = 2.0 + u * .08, (232, 505)
        v = p.view(c[0], c[1], z, 660, 500)
        g = photo_grade(v, spot(660, 480, 640, 520, .08) * fadeR, sat=1.0, dof=spot(640, 470, 560, 460, 0, 3))
        sp = Photo.pt((190, 384), c[0], c[1], z, 660, 500)
        reticle(ov, sp[0], sp[1], 30, u, RED, ss(.2, .9, u), False)
        part_text(g, u, "צריח Pan-Tilt", "שני צירים. לייזר אחד.", "נדלק רק בנעילה.")
    chainbar(g, ov, k, u)
    g = add_glow(g, ov, .6)
    if u < .1:
        g = cv2.addWeighted(g, 1, bgfill((255, 240, 200)), (1 - u / .1) * .35, 0)
    return g


# ---------- SCAN 19.2-21.6 ----------
def s_scan(lt, t, fn):
    fi = 72 + lt * 30
    f = phone_at(fi, laser=False)
    z = 1.04 + lt * .01
    g = grade(cam(f, W / 2, H / 2, z), dark=.74 * ss(0, .5, lt) + .06, laser=False)
    ov = blank()
    sx = (lt * 840) % (W + 300) - 150
    # scan sweep with a fading wake
    wake = np.clip(1 - ((sx - _xx) % W) / 380, 0, 1)[..., None].astype(np.float32) * .3
    g = np.clip(g.astype(np.float32) + wake * np.array([60, 48, 14], np.float32), 0, 255).astype(np.uint8)
    aline(ov, (sx, 44), (sx, H - 44), scale(CY, .9), 2)
    for yy in range(160, 940, 130):
        aline(ov, (W / 2 - 14, yy), (W / 2 + 14, yy), scale(CY, .25), 1)
    chrome(g, ov, lt, scramble("סורק…", ss(.15, .6, lt)), CY, "צילום אמיתי של ההצבה", ss(.1, .4, lt))
    title(g, "הכול שקט.", W // 2, 470, 72, lt - .3, 1.9, color=(225, 235, 230), font="body")
    return add_glow(g, ov, .6)


# ---------- DETECT 21.6-26.4 ----------
def detect_fi(lt):
    if lt < 1.2:
        return 140 + lt * 10
    return 152 + (lt - 1.2) * 30 * .84


def s_detect(lt, t, fn):
    fi = detect_fi(lt)
    f = phone_at(fi, laser=False)
    x, y, r = phone_ball(fi)
    u = ss(1.15, 4.6, lt)
    z = lerp(1.05, 1.7, u)
    cx, cy = lerp(W / 2, x, u * .9), lerp(H / 2, y + 120, u * .9)
    g = grade(cam(f, cx, cy, z), dark=.8, laser=False)
    ov = blank()
    seen = lt > 1.16
    if seen:
        p = to_view((x, y), cx, cy, z)
        a = lt - 1.16
        rr = r * z
        c = AMB if a < 1.6 else RED
        if a < .5 and int(a * 16) % 2:
            c = scale(AMB, .35)
        reticle(ov, p[0], p[1], rr, lt, c, ss(.0, 1.5, a))
        lab = "מטרה זוהתה" if a < 1.6 else "נעילה"
        qx, qy = p[0] - rr * 1.75, p[1] + rr * .75
        aline(ov, (p[0] - rr * .95, p[1] + rr * .5), (qx, qy), c, 2)
        aline(ov, (qx, qy), (qx - 250, qy), c, 2)
        text(g, lab, int(qx), int(qy + 8), 42, c, "head", "r", ss(0, .2, a))
    chrome(g, ov, 2.4 + lt, "מטרה בשדה הראייה" if seen else "סורק…", AMB if seen else CY, "צילום אמיתי של ההצבה")
    return add_glow(g, ov, .7)


# ---------- PIPELINE 26.4-31.2 ----------
def colormap(a, kind=cv2.COLORMAP_INFERNO):
    return cv2.applyColorMap(np.clip(a, 0, 255).astype(np.uint8), kind)


_stage_imgs = None


def stage_imgs():
    global _stage_imgs
    if _stage_imgs is None:
        S = STAGES
        Pw, Ph_ = 960, 720
        up = lambda a: cv2.resize(a, (Pw, Ph_), interpolation=cv2.INTER_CUBIC)
        dark = S["dark"].astype(np.float32)
        d = dark / max(1, dark.max()) * 255
        mask = (S["mask"] > 0).astype(np.uint8) * 255
        dist = S["dist"] / max(1, S["dist"].max()) * 255
        _stage_imgs = [sensor_up(S["gray"], Pw, Ph_), up(colormap(d)), up(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
                       up(colormap(dist, cv2.COLORMAP_INFERNO)), sensor_up(S["gray"], Pw, Ph_)]
    return _stage_imgs


PIPE = [("מה המצלמה רואה", "320x240 · שחור־לבן"), ("מפת ניגודיות", "מה כהה מהסביבה שלו?"),
        ("סינון", "רק מה שעובר את הסף"), ("מרכז ורדיוס", "הנקודה העמוקה ביותר"),
        ("מטרה נעולה", "עגולה · מלאה · לא צל")]


def s_pipe(lt, t, fn):
    g = bgfill((12, 9, 6))
    imgs = stage_imgs()
    k = min(4, int(lt / .96))
    u = lt - k * .96
    x0, y0 = 90, 180
    Pw, Ph_ = 960, 720
    cur = imgs[k]
    if k > 0 and u < .22:
        cut = int(Pw * eout(u / .22))
        view = imgs[k - 1].copy()
        view[:, :cut] = cur[:, :cut]
    else:
        cut = None
        view = cur
    g[y0:y0 + Ph_, x0:x0 + Pw] = view
    ov = blank()
    if cut is not None:
        aline(ov, (x0 + cut, y0), (x0 + cut, y0 + Ph_), CY, 2)
    brackets(ov, x0 - 12, y0 - 12, x0 + Pw + 12, y0 + Ph_ + 12, 40, scale(CY, .7), 2)
    tk = STAGES["t"]
    q = (x0 + tk.x * 3, y0 + tk.y * 3)
    if k >= 3:
        aline(ov, (q[0] - 38, q[1]), (q[0] + 38, q[1]), RED if k == 4 else INK, 2)
        aline(ov, (q[0], q[1] - 38), (q[0], q[1] + 38), RED if k == 4 else INK, 2)
        acircle(ov, q, tk.radius * 3, GRN if k == 4 else INK, 2)
    if k == 4:
        reticle(ov, q[0], q[1], tk.radius * 3, lt, RED, ss(0, .5, u))
    xr = W - 90
    text(g, "כך המערכת מוצאת אותה", xr, 186, 34, CY, "ui", "r")
    for j, (a, b) in enumerate(PIPE):
        yy = 262 + j * 134
        on = j == k
        op = 1 if j <= k else .22
        text(g, a, xr - 66, yy, 50 if on else 42, INK if on else DIM, "head", "r", op)
        if on:
            text(g, b, xr - 66, yy + 62, 30, CY, "body", "r", ss(0, .2, u))
        acircle(ov, (xr - 24, yy + 26), 9, CY if j <= k else scale(CY, .3), -1 if j <= k else 2)
        if j < 4:
            aline(ov, (xr - 24, yy + 42), (xr - 24, yy + 126), scale(CY, .5 if j < k else .18), 2)
    text(g, "detector_dark.py · פלט אמיתי", x0 + Pw, y0 + Ph_ + 24, 22, DIM, "ui", "r")
    return add_glow(g, ov, .6)


# ---------- STEREO 31.2-36.0 ----------
def s_stereo(lt, t, fn):
    g = bgfill((12, 9, 6))
    L, R, tl, tr = STEREO
    Pw, Ph_ = 840, 630
    y0 = 200
    mix = ss(2.4, 3.3, lt)
    gap = lerp(60, 0, mix)
    xl = W / 2 - gap / 2 - Pw + lerp(0, Pw / 2 + gap / 2, mix)
    xr = W / 2 + gap / 2 + lerp(0, -Pw / 2 - gap / 2, mix)
    a = sensor_up(L, Pw, Ph_)
    b = sensor_up(R, Pw, Ph_)
    xl, xr = int(xl), int(xr)
    if mix >= 1:
        g[y0:y0 + Ph_, xl:xl + Pw] = cv2.addWeighted(a, .5, b, .5, 0)
    else:
        g[y0:y0 + Ph_, xr:xr + Pw] = b
        o = .5 + .5 * (1 - mix)
        roi = g[y0:y0 + Ph_, xl:xl + Pw]
        g[y0:y0 + Ph_, xl:xl + Pw] = cv2.addWeighted(a, o, roi, 1 - o, 0)
    s = Pw / 320
    ov = blank()
    pa = (xl + tl[0] * s, y0 + tl[1] * s)
    pb = (xr + tr[0] * s, y0 + tr[1] * s)
    ring = ss(.35, .8, lt)
    if ring > 0:
        acircle(ov, pa, tl[2] * s, scale(GRN, ring), 3)
        acircle(ov, pb, tr[2] * s, scale(AMB, ring), 3)
    if mix < .05:
        text(g, "עין שמאל", xl + Pw // 2, y0 + Ph_ + 20, 36, GRN, "ui", "c", ss(.2, .5, lt))
        text(g, "עין ימין", xr + Pw // 2, y0 + Ph_ + 20, 36, AMB, "ui", "c", ss(.2, .5, lt))
    dsp = tl[0] - tr[0]
    if mix >= 1:
        o = ss(3.2, 3.5, lt)
        yy = (pa[1] + pb[1]) / 2 + tl[2] * s + 60
        aline(ov, (pa[0], pa[1]), (pa[0], yy + 16), scale(GRN, o), 2)
        aline(ov, (pb[0], pb[1]), (pb[0], yy + 16), scale(AMB, o), 2)
        aline(ov, (pb[0], yy), (pa[0], yy), scale(INK, o), 3)
        shadow_text(g, f"Δ = {dsp} px", int((pa[0] + pb[0]) / 2), int(yy + 16), 36, INK, "monob", "c", o)
    head = "שתי עיניים רואות אותה קצת אחרת" if lt < 2.5 else "ההפרש ביניהן הוא העומק"
    shadow_text(g, head, W // 2, 72, 62, INK, "head", "c", ss(.05, .3, lt) * (1 - ss(2.25, 2.4, lt)) + ss(2.5, 2.8, lt))
    shadow_text(g, f"קרוב יותר ← הפרש גדול יותר ← טווח ≈ {KZ / dsp:.0f} ס״מ", W // 2, 880, 40, CY, "ui", "c", ss(3.5, 3.8, lt))
    brackets(ov, 60, 150, W - 60, 860, 40, scale(CY, .4), 2)
    return add_glow(g, ov, .6)


# ---------- DASH 36.0-45.6 ----------
FOCUS = [("מיקום", 0), ("גובה", 1), ("כיוון התקדמות", 2), ("חיזוי", 3)]


def proj3(p, yaw, ox, oy, sc):
    x, y, z = p[0], p[1], p[2] - 26
    c, s = math.cos(yaw), math.sin(yaw)
    x, z = x * c - z * s, x * s + z * c
    c2, s2 = math.cos(.42), math.sin(.42)
    y, z = y * c2 - z * s2, y * s2 + z * c2
    d = 95 / (95 + z)
    return ox + x * sc * d, oy - y * sc * d, d


def panel3d(g, ov, tt, lt, x0, y0, x1, y1, hi):
    panel(g, x0, y0, x1, y1, 1.0, CY if hi else CYD)
    ox, oy = (x0 + x1) / 2 + 20, (y0 + y1) / 2 + 66
    yaw = .55 * math.sin(lt * .5) + .3
    sc = 12.2
    for gx in range(-30, 31, 10):
        aline(ov, proj3((gx, -12, 0), yaw, ox, oy, sc), proj3((gx, -12, 50), yaw, ox, oy, sc), scale(CY, .3), 1)
    for gz in range(0, 51, 10):
        aline(ov, proj3((-30, -12, gz), yaw, ox, oy, sc), proj3((30, -12, gz), yaw, ox, oy, sc), scale(CY, .3), 1)
    # turret at the origin and the line of fire
    o = proj3((0, -12, 0), yaw, ox, oy, sc)
    acircle(ov, o, 7, CY, -1)
    t_ = SENSOR.t
    idx = np.where((t_ <= tt) & (t_ >= tt - 3.2))[0]
    prev = None
    for j in idx:
        P = (TE["X"][j], TE["Y"][j], TE["Z"][j])
        q = proj3(P, yaw, ox, oy, sc)
        if prev is not None:
            a = 1 - (tt - t_[j]) / 3.2
            aline(ov, prev, q, scale(CY, .25 + .75 * a), 3)
        prev = q
    P = (SENSOR.get("X", tt), SENSOR.get("Y", tt), SENSOR.get("Z", tt))
    V = (SENSOR.get("vX", tt), SENSOR.get("vY", tt), SENSOR.get("vZ", tt))
    q = proj3(P, yaw, ox, oy, sc)
    fl = proj3((P[0], -12, P[2]), yaw, ox, oy, sc)
    aline(ov, q, fl, scale(CY, .55), 1)
    acircle(ov, fl, 5, scale(CY, .5), 2)
    aline(ov, o, q, scale(RED, .8), 2)
    tip = proj3((P[0] + V[0] * .55, P[1] + V[1] * .55, P[2] + V[2] * .55), yaw, ox, oy, sc)
    aline(ov, q, tip, GRN, 3)
    acircle(ov, q, 9, RED, -1)
    text(g, "מסלול תלת־ממדי", x1 - 20, y0 + 14, 30, INK, "ui", "r")
    text(g, "3D TRACK", x0 + 18, y0 + 18, 22, CY, "mono", "l", .8)


def panel_radar(g, ov, tt, lt, x0, y0, x1, y1, hi):
    panel(g, x0, y0, x1, y1, 1.0, CY if hi else CYD)
    ox, oy = (x0 + x1) / 2, y1 - 24
    R = (y1 - y0) - 76
    for k in range(1, 5):
        aarc(ov, (ox, oy), R * k / 4, -135, -45, scale(GRN, .3), 1)
    for a in (-45, -22.5, 0, 22.5, 45):
        e = (ox + R * math.sin(math.radians(a)), oy - R * math.cos(math.radians(a)))
        aline(ov, (ox, oy), e, scale(GRN, .3), 1)
    sw = 45 * math.sin(lt * 2.2)
    aline(ov, (ox, oy), (ox + R * math.sin(math.radians(sw)), oy - R * math.cos(math.radians(sw))), scale(GRN, .75), 2)
    def pos(j_t):
        x, z = SENSOR.get("X", j_t), SENSOR.get("Z", j_t)
        rng = math.hypot(x, z)
        b = math.atan2(x, z)
        k = min(rng / 40, 1.02) * R
        return ox + k * math.sin(b), oy - k * math.cos(b)
    for j in range(16):
        tj = tt - j * .14
        if tj >= 0:
            acircle(ov, pos(tj), 3, scale(GRN, 1 - j / 16), -1)
    p = pos(tt)
    acircle(ov, p, 7, RED, -1)
    acircle(ov, p, 15 + 3 * math.sin(lt * 8), RED, 2)
    text(g, "מכ״ם", x1 - 20, y0 + 12, 30, INK, "ui", "r")
    text(g, "BRG/RNG", x0 + 16, y0 + 16, 20, GRN, "mono", "l", .8)


def panel_height(g, ov, tt, lt, x0, y0, x1, y1, hi):
    panel(g, x0, y0, x1, y1, 1.0, CY if hi else CYD)
    px0, px1, py0, py1 = x0 + 56, x1 - 22, y0 + 62, y1 - 30
    for k in range(5):
        yy = lerp(py0, py1, k / 4)
        aline(ov, (px0, yy), (px1, yy), scale(CY, .14), 1)
    ymid = (py0 + py1) / 2
    aline(ov, (px0, ymid), (px1, ymid), scale(CY, .35), 1)
    def yp(h):
        return ymid - h / 14 * (py1 - py0) / 2
    span = 4.0
    prev = None
    for j in range(80):
        tj = tt - j * span / 80
        if tj < 0:
            break
        q = (px1 - j / 79 * (px1 - px0), yp(SENSOR.get("Y", tj)))
        if prev is not None:
            aline(ov, prev, q, scale(AMB, 1 - j / 100), 2)
        prev = q
    acircle(ov, (px1, yp(SENSOR.get("Y", tt))), 7, AMB, -1)
    text(g, "גובה", x1 - 20, y0 + 12, 30, INK, "ui", "r")
    text(g, "+14", x0 + 8, int(py0 - 12), 18, DIM, "mono", "l")
    text(g, "-14", x0 + 8, int(py1 - 12), 18, DIM, "mono", "l")
    text(g, "HEIGHT cm", x0 + 16, y0 + 16, 20, AMB, "mono", "l", .8)


def tile(g, x0, y0, x1, y1, label, value, unit, c, hi):
    panel(g, x0, y0, x1, y1, 1.0, c if hi else CYD)
    text(g, label, x1 - 14, y0 + 8, 26, INK if hi else DIM, "ui", "r")
    text(g, value, x0 + 14, y0 + 48, 54, c, "monob", "l")
    a = tsprite(value, 54, "monob")
    text(g, unit, x0 + 24 + a.shape[1], y0 + 68, 28, DIM, "ui", "l")


def s_dash(lt, t, fn):
    g = bgfill((12, 9, 6))
    tt = lt * (9.5 / 9.6) + .05
    k = min(3, int(lt / BAR))
    u = lt - k * BAR
    Pw, Ph_ = 960, 720
    x0, y0 = 60, 118
    s = sensor_up(SENSOR.at(tt), Pw, Ph_)
    g[y0:y0 + Ph_, x0:x0 + Pw] = s
    ov = blank()
    bx, by, r = SENSOR.get("x", tt), SENSOR.get("y", tt), SENSOR.get("r", tt)
    p = (x0 + bx * 3, y0 + by * 3)
    reticle(ov, p[0], p[1], r * 3, lt, RED, 1)
    dot = SENSOR.dot(tt)
    if dot:
        q = (x0 + dot[0] * 3, y0 + dot[1] * 3)
        acircle(ov, q, 5, (200, 180, 255), -1)
        acircle(ov, q, 11, RED, 2)
    vx = (SENSOR.get("x", min(tt + .1, 9.8)) - SENSOR.get("x", max(tt - .1, 0))) / .2
    vy = (SENSOR.get("y", min(tt + .1, 9.8)) - SENSOR.get("y", max(tt - .1, 0))) / .2
    if k >= 2:
        L = .9
        e = (p[0] + vx * 3 * L, p[1] + vy * 3 * L)
        cv2.arrowedLine(ov, (int(p[0]), int(p[1])), (int(e[0]), int(e[1])), GRN, 4, cv2.LINE_AA, tipLength=.16)
    if k == 3:
        q = (p[0] + vx * 3 * .15, p[1] + vy * 3 * .3)
        acircle(ov, q, r * 3 * .9, scale(CY, .8), 2)
        aline(ov, (q[0] - 14, q[1]), (q[0] + 14, q[1]), CY, 2)
        aline(ov, (q[0], q[1] - 14), (q[0], q[1] + 14), CY, 2)
        text(g, "עוד 0.15 שנייה", int(q[0] + r * 3 + 16), int(q[1] - 22), 30, CY, "ui", "l")
    pan, tilt = SENSOR.get("sent", tt)[:2] if False else (0, 0)
    X, Y, Z = SENSOR.get("X", tt), SENSOR.get("Y", tt), SENSOR.get("Z", tt)
    brg = math.degrees(math.atan2(X, Z))
    rate = math.hypot(vx, vy) * .21
    panel3d(g, ov, tt, lt, 1050, 118, 1860, 590, k == 2)
    panel_radar(g, ov, tt, lt, 1050, 606, 1450, 998, k == 0)
    panel_height(g, ov, tt, lt, 1466, 606, 1860, 998, k == 1)
    tw = 232
    ty = 852
    tile(g, x0, ty, x0 + tw, ty + 146, "כיוון", f"{brg:+.0f}", "°", CY, k == 0)
    tile(g, x0 + tw + 10, ty, x0 + 2 * tw + 10, ty + 146, "טווח משוער", f"≈{Z:.0f}".replace("≈", "~"), "ס״מ", CY, k == 0)
    tile(g, x0 + 2 * (tw + 10), ty, x0 + 3 * tw + 20, ty + 146, "גובה", f"{Y:+.1f}", "ס״מ", AMB, k == 1)
    tile(g, x0 + 3 * (tw + 10), ty, x0 + 4 * tw + 30, ty + 146, "קצב", f"{rate:.0f}", "°/ש׳", GRN, k >= 2)
    names = ["מיקום", "גובה", "כיוון התקדמות", "חיזוי"]
    w0, _ = text(g, "המערכת מבינה:", W - 60, 34, 46, INK, "head", "r")
    xr = W - 60 - w0 - 56
    for j, n in enumerate(names):
        w, _ = text(g, n, xr, 28, 58 if j == k else 36, CY if j == k else DIM, "head", "r", 1 if j == k else .6)
        xr -= w + 56
    text(g, "נתוני טלמטריה אמיתיים מההקלטה · 17.10.2026", 66, 1014, 20, DIM, "ui", "l", .9)
    text(g, "LIVE", x0 + 16, y0 + 12, 22, RED, "monob", "l", .6 + .4 * math.sin(lt * 7))
    return add_glow(g, ov, .55)


# ---------- COMMAND 45.6-50.4 ----------
USB_PATH = [(30, 1205), (92, 1228), (185, 1290), (270, 1310), (372, 1262), (470, 1192), (507, 1100), (490, 1005), (440, 945), (375, 925), (300, 926)]
SERVO_PATH = [(262, 905), (305, 878), (360, 858), (398, 850)]
LASER_PATH = [(250, 925), (330, 900), (372, 870), (412, 835)]


def pulses(ov, pts, u, c, n=3, tail=.14):
    pts = np.array(pts, np.float32)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    S = np.r_[0, np.cumsum(seg)] / seg.sum()
    def at(v):
        return (np.interp(v, S, pts[:, 0]), np.interp(v, S, pts[:, 1]))
    for k in range(n):
        h = u - k * .28
        if h < 0 or h - tail > 1:
            continue
        m = 12
        for j in range(m):
            a = clamp(h - tail * j / m)
            b = clamp(h - tail * (j + 1) / m)
            aline(ov, at(a), at(b), scale(c, 1 - j / m), 4)
        if h <= 1:
            acircle(ov, at(h), 7, INK, -1)


def s_cmd(lt, t, fn):
    ov = blank()
    if lt < BAR:
        p = ph(2)
        z, c = 1.82, (360, 960)
        S = (W / 2 + 330, H / 2)
        v = p.view(c[0], c[1], z, S[0], S[1])
        fadeL = np.clip((_xx - 300) / 500, 0, 1)[..., None].astype(np.float32)
        g = photo_grade(v, spot(S[0], S[1], 1000, 680, .4) * fadeL, sat=1.05, dark=1.0, gam=1.5)
        q = lambda pts: [Photo.pt(a, c[0], c[1], z, S[0], S[1]) for a in pts]
        pulses(ov, q(USB_PATH), ss(.1, 1.3, lt), CY, 3)
        pulses(ov, q(SERVO_PATH), ss(1.1, 1.9, lt), AMB, 2, .25)
        pulses(ov, q(LASER_PATH), ss(1.5, 2.2, lt), RED, 2, .25)
        panel(g, 70, 170, 640, 860, .82)
        text(g, "COM4 · 115200", 96, 190, 24, CY, "mono", "l")
        tt = 3.0 + lt * .35
        kk = int(np.searchsorted(SENSOR.t, tt))
        for j in range(12):
            jj = kk - 11 + j
            if jj < 0:
                continue
            sn = TE["sent"][jj]
            yy = 240 + j * 50
            a = f"{int(sn[0])},{int(sn[1])},{int(sn[2])}"
            text(g, ">", 96, yy, 26, CY, "monob", "l", .7)
            text(g, a, 124, yy, 26, INK, "mono", "l")
            text(g, f"OK {int(sn[0])} {int(sn[1])} {int(sn[2])}", 330, yy, 26, GRN, "mono", "l", .85)
        shadow_text(g, "הפקודה יוצאת", W - 90, 90, 72, INK, "head", "r", ss(.05, .3, lt))
        text(g, "מחשב ← Arduino ← מנועים ולייזר", W - 90, 186, 36, CY, "ui", "r", ss(.3, .6, lt))
        text(g, "pan, tilt, laser", 96, 806, 22, DIM, "mono", "l")
    else:
        u = lt - BAR
        p = ph(7)
        z, c = 2.45 + u * .1, (445, 880)
        S = (W / 2 - 200, H / 2 + 60)
        v = p.view(c[0], c[1], z, S[0], S[1])
        fadeR = np.clip((1640 - _xx) / 460, 0, 1)[..., None].astype(np.float32)
        g = photo_grade(v, spot(S[0], S[1] - 60, 640, 500, .06) * fadeR, sat=1.15, dark=.62, gam=2.2, dof=spot(S[0], S[1] - 60, 520, 480, 0, 3))
        tip = Photo.pt((442, 824), c[0], c[1], z, S[0], S[1])
        ign = ss(.9, 1.15, u)
        if ign > 0:
            fl = 1 + .08 * math.sin(lt * 40)
            acircle(ov, tip, 12 * fl, scale((220, 200, 255), ign), -1)
            acircle(ov, tip, 30 * fl, scale(RED, ign), 3)
            aline(ov, (tip[0] - 240 * ign, tip[1]), (tip[0] + 240 * ign, tip[1]), scale(RED, .6 * ign), 2)
            aline(ov, (tip[0], tip[1] - 120 * ign), (tip[0], tip[1] + 120 * ign), scale(RED, .5 * ign), 2)
            aline(ov, tip, (tip[0] + 120 * ign, -40), scale(RED, ign), 4)
        # real commanded angles from the log
        tt = 3.0 + lt * .9
        pan = float(np.interp(tt, SENSOR.t, TE["sent"][:, 0]))
        tlt = float(np.interp(tt, SENSOR.t, TE["sent"][:, 1]))
        cx, cy = 1560, 540
        aarc(ov, (cx, cy), 150, 180, 360, scale(CY, .3), 2)
        aarc(ov, (cx, cy), 150, 270 - 45, 270 + (pan - 90) * .95, CY, 5)
        aarc(ov, (cx, cy), 104, 180, 300, scale(AMB, .3), 2)
        aarc(ov, (cx, cy), 104, 180, 180 + tlt * 1.8, AMB, 5)
        text(g, f"PAN {pan:03.0f}°", cx, cy + 22, 34, CY, "monob", "c")
        text(g, f"TILT {tlt:03.0f}°", cx, cy + 66, 34, AMB, "monob", "c")
        shadow_text(g, "הצריח מסתובב", W - 90, 90, 72, INK, "head", "r", ss(.05, .3, u) * (1 - ss(.82, .9, u)))
        shadow_text(g, "הלייזר נדלק", W - 90, 90, 72, RED, "head", "r", ss(.9, 1.0, u))
    return add_glow(g, ov, .75)


# ---------- CLIMAX 50.4-55.2 ----------
HIT_LT = 3.0
JUMP = 1027


def climax_fi(lt):
    if lt < 1.2:
        return 1000 + lt * 15
    if lt < HIT_LT:
        return 1018 + (lt - 1.2) / 1.8 * 8.96
    return JUMP + (lt - HIT_LT) * 30


def s_climax(lt, t, fn):
    fi = climax_fi(lt)
    i = int(fi)
    f = phone_at(fi, snap=JUMP - 1 <= fi < JUMP)
    x, y, r = phone_ball(fi)
    hitu = lt - HIT_LT
    zu = ss(.0, HIT_LT, lt)
    z = lerp(1.25, 1.9, zu ** 1.4)
    cx, cy = lerp(1500, x + 40, zu), lerp(300, y + 60, zu)
    shake = (0, 0)
    if 0 <= hitu < .7:
        a = (1 - hitu / .7) ** 2 * 26
        shake = (a * math.sin(hitu * 91), a * math.cos(hitu * 73))
    g = grade(cam(f, cx, cy, z, shake), dark=.92)
    ov = blank()
    p = to_view((x, y), cx, cy, z)
    rr = r * z
    locked = hitu >= 0
    reticle(ov, p[0], p[1], rr, lt, RED, 1)
    if -.1 < hitu < .05:      # the snap itself: wall -> ball between two camera frames
        d0 = to_view(dot_at(JUMP - 1) or (1874, 180), cx, cy, z)
        d1 = to_view(dot_at(JUMP) or (x, y), cx, cy, z)
        aline(ov, d0, d1, (210, 190, 255), 6)
    if 0 <= hitu < 1.0:
        for k, (spd, c) in enumerate(((1500, INK), (900, RED), (520, scale(RED, .7)))):
            rad = rr + spd * hitu ** .75
            o = max(0, 1 - hitu / (.75 - k * .1))
            acircle(ov, p, rad, scale(c, o), 4 - k)
    chrome(g, ov, 31.2 + lt, None if locked else "לייזר פעיל", AMB, "צילום אמיתי · הילוך איטי" if lt < HIT_LT else "צילום אמיתי")
    g = add_glow(g, ov, .9)
    if 0 <= hitu < .16:
        g = cv2.addWeighted(g, 1, bgfill((230, 215, 255)), (1 - hitu / .16) * .95, 0)
    if hitu > 0:
        u = hitu
        size = 210
        a = tsprite("פגיעה", size, "head")
        sc = lerp(1.5, 1.0, eout(u / .22))
        s2 = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        tx, ty = int(clamp(p[0] - rr * 1.6 - 340, 400, W - 400)), int(p[1])
        op = ss(0.02, .12, u)
        b = cv2.GaussianBlur(cv2.copyMakeBorder(s2, 30, 30, 30, 30, cv2.BORDER_CONSTANT), (0, 0), 16)
        blit_a(g, b, tx - s2.shape[1] // 2 - 30, ty - s2.shape[0] // 2 - 30, (20, 0, 90), .85 * op)
        blit_a(g, s2, tx - s2.shape[1] // 2, ty - s2.shape[0] // 2, INK, op)
        shadow_text(g, "הנקודה האדומה: הלייזר של המערכת, על מטרה בתנועה", W // 2, H - 240, 42, INK, "ui", "c", ss(.4, .7, u))
    return g


# ---------- MONTAGE 55.2-60.0 ----------
MONT = [(2568, 1.5), (1542, 1.45), (1218, 1.5), (2386, 1.5)]


def s_mont(lt, t, fn):
    k = min(3, int(lt / 1.2))
    u = lt - k * 1.2
    i0, z0 = MONT[k]
    fi = i0 + u * 30
    f = phone_at(fi)
    x, y, r = phone_ball(fi)
    z = z0 + u * .1
    g = grade(cam(f, x, y + 30, z), dark=.95)
    ov = blank()
    p = to_view((x, y), x, y + 30, z)
    reticle(ov, p[0], p[1], r * z, lt, RED, 1)
    chrome(g, ov, 36 + lt, "עוקבת", RED, "צילום אמיתי")
    words = ["עוקבת.", "מתקנת.", "מחזיקה.", "פוגעת."]
    shadow_text(g, words[k], W - 130, 800, 110, INK, "head", "r", ss(0, .1, u))
    g = add_glow(g, ov, .7)
    if u < .08:
        g = cv2.addWeighted(g, 1, bgfill((255, 255, 255)), (1 - u / .08) * .5, 0)
    return g


# ---------- KPI 60.0-67.2 ----------
_kbg = None


def s_kpi(lt, t, fn):
    global _kbg
    if _kbg is None:
        f = grade(phone_at(2600), dark=.5)
        _kbg = cv2.GaussianBlur(f, (0, 0), 16)
    g = (cam(_kbg, W / 2, H / 2, 1.05 + lt * .012).astype(np.float32) * .55).astype(np.uint8)
    ov = blank()
    shadow_text(g, "לא הבטחות. מדידות.", W // 2, 92, 76, INK, "head", "c", ss(.05, .35, lt))
    items = [(K["lock_seen"], "%", "נעילה", "כשהמטרה בשדה הראייה", 0, CY),
             (K["near"][0], "%", "פגיעה בטווח קרוב", "מטרה קרובה למערכת", 0, RED),
             (K["slow"][0], "%", "פגיעה בתנועה איטית", "מטרה במעקב רציף", 0, RED),
             (20, "", "החלטות בשנייה", "זיהוי ← עומק ← פקודה", 0, GRN)]
    tw, gap = 410, 30
    x0 = (W - 4 * tw - 3 * gap) // 2
    for j, (val, unit, a, b, _, c) in enumerate(items):
        # Hebrew order: first tile on the right
        x = x0 + (3 - j) * (tw + gap)
        d = lt - .25 - j * .55
        op = ss(0, .25, d)
        if op <= 0:
            continue
        yy = 300 + int((1 - eout(d / .4)) * 40)
        panel(g, x, yy, x + tw, yy + 420, .78 * op, scale(c, .8 * op))
        v = val * ss(0, .85, d)
        s = f"{v:.0f}"
        pre = "~" if unit == "" else ""
        text(g, pre + s + unit, x + tw // 2, yy + 58, 128, c, "monob", "c", op)
        text(g, a, x + tw // 2, yy + 234, 42, INK, "head", "c", op)
        text(g, b, x + tw // 2, yy + 296, 30, DIM, "ui", "c", op)
    fp = (f"נמדד על המתקן, 17.10.2026 · פגיעה = נקודת הלייזר בתוך הכדור בתמונה · "
          f"בכל התנאים יחד {K['all'][0]:.0f}%, בתנועה מהירה {K['fast'][0]:.0f}%")
    text(g, fp, W // 2, 776, 28, INK, "ui", "c", .78 * ss(3.0, 3.5, lt))
    return add_glow(g, ov, .5)


# ---------- END 67.2-74.4 ----------
def s_end(lt, t, fn):
    p = ph(1)
    z = 1.5 + lt * .02
    v = p.view(470, 690, z, 1350, 585)
    fade = np.clip((_xx - 660) / 420, 0, 1)[..., None].astype(np.float32)
    g = photo_grade(v, spot(1320, 600, 760, 640, .05) * fade * (.5 * ss(2.4, 3.6, lt)), sat=.8) if lt > 2.4 else bgfill((6, 4, 2))
    ov = blank()
    if lt < 2.4:
        words = ["רואה.", "מבין.", "פוגע."]
        cols = [INK, INK, RED]
        for j, w in enumerate(words):
            u = lt - j * .6 - .1
            if u > 0:
                s = eback(u / .26)
                x = W - 500 - j * 460
                a = tsprite(w, 150, "head")
                text(g, w, x, 440, 150, cols[j], "head", "c", ss(0, .08, u))
    else:
        u = lt - 2.4
        logo(g, ov, 690, 440, 118, ss(0, 1.1, u), "מערכת ביות ועקיבה אופטית בזמן אמת")
        text(g, "רואה. מבין. פוגע.", 690, 620, 56, INK, "head", "c", ss(.8, 1.2, u))
        text(g, "גיא בכר · רז יחיאלי", 690, 730, 34, INK, "ui", "c", ss(1.2, 1.6, u))
        text(g, "פרויקט מעבדה בחומרה · אוניברסיטת תל אביב", 690, 780, 28, DIM, "ui", "c", ss(1.3, 1.7, u))
        text(g, "הרחפן, הנוף והצריח התלת־ממדי: הדמיית מחשב להמחשה", 690, 910, 24, DIM, "ui", "c", .9 * ss(1.6, 2.0, u))
        text(g, "תצלומי החומרה והמדדים: מהמערכת האמיתית, מודל הדגמה מעבדתי", 690, 946, 24, DIM, "ui", "c", .9 * ss(1.6, 2.0, u))
    g = add_glow(g, ov, .8)
    k = 1 - ss(6.5, 7.2, lt)
    return (g.astype(np.float32) * k).astype(np.uint8)


SCENES = [(0.0, 7.2, s_hook), (7.2, 9.6, s_reveal), (9.6, 19.2, s_parts), (19.2, 21.6, s_scan), (21.6, 26.4, s_detect),
          (26.4, 31.2, s_pipe), (31.2, 36.0, s_stereo), (36.0, 45.6, s_dash), (45.6, 50.4, s_cmd), (50.4, 55.2, s_climax),
          (55.2, 60.0, s_mont), (60.0, 67.2, s_kpi), (67.2, 74.4, s_end)]
DUR = SCENES[-1][1]


def frame(fn):
    t = fn / FPS
    for a, b, f in SCENES:
        if a <= t < b:
            return finish(f(t - a, t, fn), fn)
    return finish(bgfill((0, 0, 0)), fn)
