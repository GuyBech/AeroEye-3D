# AeroEye-3D promo, final cut: real photos for the hardware, full CGI for the engagement, the system's FLOW as the spine
import json
import math
import os

import cv2
import numpy as np

from core import *
import scenes as base
from scenes import ph, logo, s_reveal, s_parts, s_end, pulses, USB_PATH, SERVO_PATH, LASER_PATH, K, tile
from assets import Photo, photo_grade, sensor_up, TE

CGD = os.path.join(HERE, "cg", "frames")
_xx = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
_meta = {}
SIM = "הדמיה להמחשה"

# the seven steps of the tracker's loop, coloured by the board that does them
FLOW = [("לכידה", CY), ("זיהוי", GRN), ("עומק", GRN), ("מעקב", GRN), ("חיזוי", GRN), ("פקודה", AMB), ("ביצוע", RED)]
LAYERS = [("מצלמה", 0, 0), ("מחשב", 1, 4), ("Arduino", 5, 5), ("צריח + לייזר", 6, 6)]


def meta(sh):
    if sh not in _meta:
        _meta[sh] = json.load(open(os.path.join(CGD, sh, "meta.json")))
    return _meta[sh]


def post(im, b=.8):
    small = cv2.resize(im, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hi = cv2.subtract(small, np.full_like(small, 150))
    up = cv2.resize(cv2.GaussianBlur(hi, (0, 0), 7), (W, H), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(im, 1, up, b * .62, 0)
    return (out.astype(np.float32) * VIG).astype(np.uint8)


def cg(sh, f):
    n = meta(sh)
    f = int(clamp(f, 0, len(n) - 1))
    return post(cv2.imread(os.path.join(CGD, sh, f"{f}.jpg"))), n[f]


def flowbar(fr, ov, k, u=1.0):
    y = 1006
    x0, dx = W - 196, 254
    strip = fr[934:1072].astype(np.float32)
    ramp = np.linspace(1, .34, 40, dtype=np.float32)
    mask = np.r_[ramp, np.full(138 - 40, .34, np.float32)][:, None, None]
    fr[934:1072] = (strip * mask).astype(np.uint8)
    for j, (n, c) in enumerate(FLOW):
        x = x0 - j * dx
        on = j <= k
        if j < 6:
            aline(ov, (x - 14, y), (x - dx + 14, y), scale(c, .28), 2)
            if j < k:
                aline(ov, (x - 14, y), (x - dx + 14, y), c, 2)
            elif j == k:
                aline(ov, (x - 14, y), (x - 14 - (dx - 28) * clamp(u), y), c, 2)
        acircle(ov, (x, y), 9, c if on else scale(c, .45), -1 if on else 2)
        if j == k:
            acircle(ov, (x, y), 16 + 3 * math.sin(u * 12), c, 2)
        text(fr, n, x, y - 50, 34 if j == k else 28, INK if on else DIM, "head", "c", 1 if on else .62)
    for n, a, b in LAYERS:
        x = x0 - (a + b) / 2 * dx
        text(fr, n, int(x), y + 22, 20, DIM, "ui", "c", .85)


def step(fr, ov, lt, n, head, sub, dur, c=CY):
    op = ss(.02, .22, lt) * (1 - ss(dur - .16, dur, lt))
    if op <= 0:
        return
    x, y = W - 72, 48
    cx = x - 34
    acircle(ov, (cx, y + 40), 34, scale(c, op), 3)
    text(fr, str(n), cx, y + 19, 44, c, "monob", "c", op)
    a = tsprite(head, 72, "head")
    shadow_text(fr, head, x - 92, y, 72, INK, "head", "r", op)
    shadow_text(fr, sub, x - 92, y + 92, 38, INK if c == RED else c, "body", "r", op * ss(.14, .36, lt))


def cg_chrome(g, ov, t, state, c=CY):
    chrome(g, ov, t, state, c, SIM)


# ---------- HOOK 0.0-7.2 ----------
def s_hook(lt, t, fn):
    if lt < 2.4:
        g, _ = cg("hook1", lt * 30)
        g = (g.astype(np.float32) * ss(0, 1.1, lt)).astype(np.uint8)
        title(g, "מטרה קטנה.", W // 2, 760, 100, lt - .5, 1.9)
        text(g, SIM, W - 66, H - 70, 24, INK, "ui", "r", .8 * ss(.8, 1.2, lt))
        return g
    if lt < 4.2:
        g, _ = cg("hook2", (lt - 2.4) * 30)
        title(g, "תנועה מהירה.", W // 2, 780, 100, lt - 2.55, 1.6)
        text(g, SIM, W - 66, H - 70, 24, INK, "ui", "r", .8)
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


# ---------- SCAN 19.2-21.6 ----------
def s_scan(lt, t, fn):
    g, m = cg("scan", lt * 30)
    g = (g.astype(np.float32) * ss(0, .3, lt)).astype(np.uint8)
    ov = blank()
    tip = m["c"]["tip"]
    if tip[2] > 0:
        a = lt * 2.6
        for k in range(3):
            r = ((lt * 340 + k * 260) % 780) + 40
            aarc(ov, tip[:2], r, -150, -30, scale(CY, .55 * (1 - r / 820)), 2)
    cg_chrome(g, ov, lt, scramble("סורק…", ss(.15, .6, lt)))
    title(g, "הכול שקט.", W // 2, 150, 76, lt - .2, 2.0, color=(225, 235, 230))
    return add_glow(g, ov, .6)


# ---------- DETECT 21.6-26.4 ----------
def s_detect(lt, t, fn):
    g, m = cg("detect", lt * 30)
    c = m["c"]
    ov = blank()
    p = c["drone"]
    seen = lt > 1.1
    if seen and p[2] > 0:
        a = lt - 1.1
        rr = max(c["rpx"] * 1.2, 30)
        col = AMB if a < 1.6 else RED
        if a < .5 and int(a * 16) % 2:
            col = scale(AMB, .35)
        reticle(ov, p[0], p[1], rr, lt, col, ss(0, 1.5, a))
        lab = "מטרה זוהתה" if a < 1.6 else "נעילה"
        qx, qy = p[0] + rr * 2.1, p[1] - rr * 1.5
        aline(ov, (p[0] + rr * .9, p[1] - rr * .6), (qx, qy), col, 2)
        aline(ov, (qx, qy), (qx + 190, qy), col, 2)
        shadow_text(g, lab, int(qx + 190), int(qy - 56), 46, col, "head", "r", ss(0, .2, a))
    cg_chrome(g, ov, 2.4 + lt, "מטרה בשדה הראייה" if seen else "סורק…", AMB if seen else CY)
    title(g, "ואז היא מופיעה.", W // 2, 130, 76, lt - .1, 1.05, color=(225, 235, 230))
    return add_glow(g, ov, .7)


# ---------- EYES 26.4-31.2: steps 1 + 2 ----------
def dark_map(gray):
    """Same idea as detector_dark.py: closing erases small dark things, so background - frame lights them up."""
    s = cv2.resize(gray, (480, 360), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    d = cv2.subtract(bg, s)
    d[-20:], d[:, -20:] = 0, 0        # closing is undefined at the frame edge
    return d


def s_eyes(lt, t, fn):
    g = bgfill((12, 9, 6))
    f = int(clamp(lt * 30, 0, 143))
    n = meta("eyes")[f]
    Pw, Ph_ = 860, 645
    y0 = 214
    gap = 56 * ss(.25, .8, lt)
    xl = int(W / 2 - gap / 2 - Pw)
    xr = int(W / 2 + gap / 2)
    s = Pw / 960
    ov = blank()
    for tag, x0, col, name in (("L", xl, GRN, "עין שמאל"), ("R", xr, AMB, "עין ימין")):
        im = cv2.imread(os.path.join(CGD, "eyes", f"{f}_{tag}.jpg"))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        panel_im = sensor_up(cv2.convertScaleAbs(gray, alpha=1.5, beta=8), Pw, Ph_)
        d = dark_map(gray)
        heat = ss(1.75, 2.15, lt)
        if heat > 0:
            top = float(max(d.max(), 40))
            hm = cv2.resize(cv2.applyColorMap(np.clip(d.astype(np.float32) / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), (Pw, Ph_), interpolation=cv2.INTER_CUBIC)
            cut = int(Pw * heat)
            panel_im[:, :cut] = hm[:, :cut]
        if lt > 2.9:
            top = float(max(d.max(), 40))
            mask = ((d.astype(np.float32) / top) > .42).astype(np.uint8) * 255
            keep = np.zeros_like(mask)
            cv2.circle(keep, (int(n[tag]["drone"][0] / 2), int(n[tag]["drone"][1] / 2)), 34, 255, -1)
            mask = cv2.dilate(cv2.bitwise_and(mask, keep), np.ones((3, 3), np.uint8))
            panel_im = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (Pw, Ph_), interpolation=cv2.INTER_LINEAR)
        if lt > 3.8:
            panel_im = sensor_up(cv2.convertScaleAbs(gray, alpha=1.5, beta=8), Pw, Ph_)
        g[y0:y0 + Ph_, x0:x0 + Pw] = panel_im
        q = n[tag]["drone"]
        px = (x0 + q[0] * s, y0 + q[1] * s)
        rr = max(n[tag]["rpx"] * s * 1.25, 30)
        if lt > 3.3:
            acircle(ov, px, rr, col, 3)
            aline(ov, (px[0] - 16, px[1]), (px[0] + 16, px[1]), RED, 2)
            aline(ov, (px[0], px[1] - 16), (px[0], px[1] + 16), RED, 2)
        if gap > 10:
            text(g, name, x0 + Pw // 2, y0 + Ph_ + 14, 32, col, "ui", "c", ss(.5, .8, lt))
    brackets(ov, xl - 12, y0 - 12, xr + Pw + 12, y0 + Ph_ + 12, 40, scale(CY, .55), 2)
    # one sensor frame, 640x240 side by side: that's what the camera really delivers
    if lt < 1.6:
        text(g, f"frame {12840 + int(lt * 20):06d}  640x240 → 2 × 320x240", 66, 150, 26, CY, "mono", "l", ss(.1, .3, lt))
    elif lt < 2.9:
        text(g, "מפת ניגודיות: מה כהה מהסביבה שלו?", 66, 146, 32, INK, "ui", "l", ss(1.8, 2.0, lt))
    elif lt < 3.8:
        text(g, "סינון: נשאר רק מה שעובר את הסף", 66, 146, 32, INK, "ui", "l")
    else:
        text(g, "נמצאה בשתי העיניים", 66, 146, 32, GRN, "ui", "l")
    if lt < 1.6:
        step(g, ov, lt, 1, "לכידה", "מצלמה אחת, שתי עיניים.", 1.6, CY)
        flowbar(g, ov, 0, lt / 1.6)
    else:
        step(g, ov, lt - 1.6, 2, "זיהוי", "בכל עין בנפרד.", 3.2, GRN)
        flowbar(g, ov, 1, (lt - 1.6) / 3.2)
    return add_glow(g, ov, .6)


# ---------- DEPTH 31.2-36.0: step 3 ----------
def s_depth(lt, t, fn):
    g, m = cg("depth", lt * 30)
    c = m["c"]
    ov = blank()
    d = c["drone"]
    L, R = c["lensL"], c["lensR"]
    u = ss(.3, 1.3, lt)
    for q, col in ((L, GRN), (R, AMB)):
        e = (lerp(q[0], d[0], u), lerp(q[1], d[1], u))
        aline(ov, q[:2], e, col, 2)
        for k in range(4):
            h = ((lt * .9 + k * .25) % 1) * u
            pp = (lerp(q[0], d[0], h), lerp(q[1], d[1], h))
            acircle(ov, pp, 4, col, -1)
        acircle(ov, q[:2], 15, col, 2)
    rr = max(c["rpx"] * 1.2, 24)
    if u >= 1:
        reticle(ov, d[0], d[1], rr, lt, RED, 1)
        o = ss(1.5, 1.8, lt)
        x, y = int(d[0] + 80), int(d[1] - 24)
        shadow_text(g, f"טווח ≈ {m['range']:.0f} מ׳", x + 300, y - 20, 50, INK, "head", "r", o)
    # inset: both eyes overlaid, the offset between them is the depth
    n = meta("eyes")[143]
    a = cv2.imread(os.path.join(CGD, "eyes", "143_L.jpg"), 0)
    b = cv2.imread(os.path.join(CGD, "eyes", "143_R.jpg"), 0)
    pl, pr = n["L"]["drone"], n["R"]["drone"]
    cx, cy = (pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2
    sw = 520
    sc = 460 / sw
    M = np.float32([[sc, 0, 230 - cx * sc], [0, sc, 130 - cy * sc]])
    mix = cv2.addWeighted(a, .5, b, .5, 0)
    ins = sensor_up(cv2.convertScaleAbs(cv2.warpAffine(mix, M, (460, 260), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), alpha=1.5, beta=8), 460, 260)
    x0, y0 = 66, 230
    o = ss(1.8, 2.2, lt)
    if o > 0:
        roi = g[y0:y0 + 260, x0:x0 + 460]
        g[y0:y0 + 260, x0:x0 + 460] = cv2.addWeighted(ins, o, roi, 1 - o, 0)
        cv2.rectangle(g, (x0, y0), (x0 + 460, y0 + 260), scale(CY, o), 1, cv2.LINE_AA)
        qa = (x0 + 230 + (pl[0] - cx) * sc, y0 + 130 + (pl[1] - cy) * sc)
        qb = (x0 + 230 + (pr[0] - cx) * sc, y0 + 130 + (pr[1] - cy) * sc)
        acircle(ov, qa, 18, scale(GRN, o), 2)
        acircle(ov, qb, 18, scale(AMB, o), 2)
        aline(ov, qa, qb, scale(INK, o), 2)
        text(g, f"Δ = {abs(pl[0] - pr[0]) / 3:.0f} px", x0 + 230, y0 + 212, 30, INK, "monob", "c", o)
        text(g, "שתי העיניים, זו על זו", x0 + 450, y0 - 42, 30, INK, "ui", "r", o)
    step(g, ov, lt, 3, "עומק", "אותה מטרה, שתי זוויות. ההפרש נותן את הטווח.", 4.8, GRN)
    cg_chrome(g, ov, 12 + lt, None)
    flowbar(g, ov, 2, lt / 4.8)
    return add_glow(g, ov, .75)


# ---------- TRACK 36.0-45.6: steps 4 + 5 ----------
def small_tile(g, x0, y0, w, label, value, unit, c, hi):
    panel(g, x0, y0, x0 + w, y0 + 96, .72, c if hi else CYD)
    text(g, label, x0 + w - 14, y0 + 8, 24, INK if hi else DIM, "ui", "r")
    wv, _ = text(g, value, x0 + 14, y0 + 38, 46, c, "monob", "l")
    text(g, unit, x0 + 22 + wv, y0 + 52, 26, DIM, "ui", "l")


def s_track(lt, t, fn):
    g, m = cg("track", lt * 30)
    c = m["c"]
    ov = blank()
    k = min(3, int(lt / BAR))
    u = lt - k * BAR
    d = c["drone"]
    rr = max(c["rpx"] * 1.15, 44)
    # past path
    prev = d
    for j, q in enumerate(c["trail"]):
        if q[2] > 1 and prev[2] > 1 and min(q[0], prev[0]) > 400:
            aline(ov, prev[:2], q[:2], scale(CY, 1 - j / 28), 3)
        prev = q
    reticle(ov, d[0], d[1], rr, lt, RED, 1)
    if k == 0:
        aline(ov, (d[0], d[1]), c["piv"][:2], scale(RED, .55), 1)
    if k == 1:
        g0 = c["ground"]
        aline(ov, (d[0], d[1] + rr), g0[:2], AMB, 2)
        for j in range(1, 8):
            yy = lerp(d[1] + rr, g0[1], j / 8)
            xx = lerp(d[0], g0[0], j / 8)
            aline(ov, (xx - 10, yy), (xx + 10, yy), AMB, 2)
        acircle(ov, g0[:2], 12, AMB, 2)
        shadow_text(g, f"{m['alt']:.0f} מ׳ מעל הקרקע", int(clamp(d[0] + 34, 440, 1300)), int(clamp((d[1] + g0[1]) / 2, 300, 700)), 46, AMB, "head", "l", ss(.1, .4, u))
    if k >= 2:
        pts = c["pred"]
        e = pts[3]
        if e[2] > 1:
            vx, vy = e[0] - d[0], e[1] - d[1]
            ln = max(math.hypot(vx, vy), 1)
            f = min(ln, min(340, max(d[0] - 430, 60))) / ln
            cv2.arrowedLine(ov, (int(d[0]), int(d[1])), (int(d[0] + vx * f), int(d[1] + vy * f)), GRN, 5, cv2.LINE_AA, tipLength=.16)
            if k == 2:
                shadow_text(g, f"{m['speed']:.0f} מ׳/ש׳", int(clamp(d[0], 520, 1400)), int(clamp(d[1] - rr - 84, 220, 800)), 46, GRN, "head", "c", ss(.1, .4, u))
    if k == 3:
        a = c["pred"][3]
        if a[2] > 1:
            ax = max(a[0], 520)
            ay = a[1] + (a[0] - ax) * -.1
            acircle(ov, (ax, ay), 46, CY, 2)
            aline(ov, (ax - 18, ay), (ax + 18, ay), CY, 2)
            aline(ov, (ax, ay - 18), (ax, ay + 18), CY, 2)
            shadow_text(g, "לכאן מכוונים", int(ax), int(ay + 62), 44, CY, "head", "c", ss(.1, .4, u))
    # radar
    x0, y0, side = 60, 200, 330
    panel(g, x0, y0, x0 + side, y0 + side, .68, CY if k == 0 else CYD)
    ox, oy, R = x0 + side / 2, y0 + side - 22, side - 66
    for j in range(1, 5):
        aarc(ov, (ox, oy), R * j / 4, -135, -45, scale(GRN, .35), 1)
    for a in (-45, -22.5, 0, 22.5, 45):
        aline(ov, (ox, oy), (ox + R * math.sin(math.radians(a)), oy - R * math.cos(math.radians(a))), scale(GRN, .35), 1)
    sw = 45 * math.sin(lt * 2.2)
    aline(ov, (ox, oy), (ox + R * math.sin(math.radians(sw)), oy - R * math.cos(math.radians(sw))), scale(GRN, .8), 2)
    nn = meta("track")
    idx = int(clamp(lt * 30, 0, len(nn) - 1))

    def rp(q):
        rng = math.hypot(q["x"], q["z"])
        b = math.atan2(q["x"], q["z"])
        f = min(rng / 460, 1.04) * R
        return ox + f * math.sin(b), oy - f * math.cos(b)
    for j in range(1, 14):
        q = nn[max(idx - j * 6, 0)]
        acircle(ov, rp(q), 3, scale(GRN, 1 - j / 14), -1)
    acircle(ov, rp(m), 7, RED, -1)
    acircle(ov, rp(m), 14 + 3 * math.sin(lt * 8), RED, 2)
    text(g, "מכ״ם", x0 + side - 14, y0 + 10, 28, INK, "ui", "r")
    # height strip
    y1 = y0 + side + 14
    hh = 190
    panel(g, x0, y1, x0 + side, y1 + hh, .68, AMB if k == 1 else CYD)
    px0, px1, py0, py1 = x0 + 16, x0 + side - 18, y1 + 48, y1 + hh - 18
    lo, hi = 70, 110
    pv = None
    for j in range(60):
        q = nn[max(idx - j * 2, 0)]
        p_ = (px1 - j / 59 * (px1 - px0), py1 - (q["y"] - lo) / (hi - lo) * (py1 - py0))
        if pv is not None:
            aline(ov, pv, p_, scale(AMB, 1 - j / 75), 2)
        pv = p_
    text(g, "גובה", x0 + side - 14, y1 + 8, 28, INK, "ui", "r")
    # readouts
    tw, ty = 300, 806
    tx = W - 60 - 3 * tw - 20
    small_tile(g, tx + 2 * (tw + 10), ty, tw, "כיוון", f"{m['brg']:+.0f}", "°", CY, k == 0)
    small_tile(g, tx + 1 * (tw + 10), ty, tw, "טווח", f"{m['range']:.0f}", "מ׳", CY, k == 0)
    small_tile(g, tx, ty, tw, "מהירות", f"{m['speed']:.0f}", "מ׳/ש׳", GRN, k == 2)
    if k < 3:
        name = ["מיקום", "גובה", "כיוון התקדמות"][k]
        step(g, ov, lt, 4, "מעקב", "", 7.2, GRN)
        shadow_text(g, name, W - 164, 136, 56, CY, "head", "r", ss(0, .2, u) * (1 - ss(2.25, 2.4, u)))
        flowbar(g, ov, 3, lt / 7.2)
    else:
        step(g, ov, u, 5, "חיזוי", "מכוונים לאן שהמטרה תהיה, לא לאן שהיא הייתה.", BAR, GRN)
        flowbar(g, ov, 4, u / BAR)
    cg_chrome(g, ov, 17 + lt, None)
    return add_glow(g, ov, .6)


# ---------- COMMAND 45.6-50.4: steps 6 + 7 ----------
def s_cmd(lt, t, fn):
    ov = blank()
    if lt < BAR:
        p = ph(2)
        z, c = 1.82, (360, 960)
        S = (W / 2 + 330, H / 2 + 50)
        v = p.view(c[0], c[1], z, S[0], S[1])
        fadeL = np.clip((_xx - 300) / 500, 0, 1)[..., None].astype(np.float32)
        g = photo_grade(v, spot(S[0], S[1], 1000, 680, .4) * fadeL, sat=1.05, dark=1.0, gam=1.5)
        q = lambda pts: [Photo.pt(a, c[0], c[1], z, S[0], S[1]) for a in pts]
        pulses(ov, q(USB_PATH), ss(.1, 1.3, lt), CY, 3)
        pulses(ov, q(SERVO_PATH), ss(1.1, 1.9, lt), AMB, 2, .25)
        pulses(ov, q(LASER_PATH), ss(1.5, 2.2, lt), RED, 2, .25)
        panel(g, 70, 210, 620, 880, .84)
        text(g, "COM4 · 115200", 96, 228, 24, CY, "mono", "l")
        tt = 3.0 + lt * .35
        kk = int(np.searchsorted(TE["t"], tt))
        for j in range(11):
            jj = kk - 10 + j
            if jj < 0:
                continue
            sn = TE["sent"][jj]
            yy = 276 + j * 50
            text(g, ">", 96, yy, 26, CY, "monob", "l", .7)
            text(g, f"{int(sn[0])},{int(sn[1])},{int(sn[2])}", 124, yy, 26, INK, "mono", "l")
            text(g, f"OK {int(sn[0])} {int(sn[1])} {int(sn[2])}", 330, yy, 26, GRN, "mono", "l", .85)
        text(g, "pan, tilt, laser", 96, 838, 22, DIM, "mono", "l")
        step(g, ov, lt, 6, "פקודה", "שלושה מספרים ב-UART: pan, tilt, laser.", BAR + .02, AMB)
        text(g, "תצלום אמיתי של המערכת", 640, 900, 22, INK, "ui", "l", .6)
        flowbar(g, ov, 5, lt / BAR)
    else:
        u = lt - BAR
        g, m = cg("turret", u * 30)
        tt = u
        pan, tlt = m["pan"], m["tilt"]
        cx, cy = 300, 560
        aarc(ov, (cx, cy), 150, 200, 340, scale(CY, .3), 2)
        aarc(ov, (cx, cy), 150, 270, 270 + clamp(pan, -70, 70), CY, 5)
        aarc(ov, (cx, cy), 104, 180, 300, scale(AMB, .3), 2)
        aarc(ov, (cx, cy), 104, 180, 180 + clamp(tlt * 3, 0, 120), AMB, 5)
        text(g, f"PAN {pan:+04.0f}°", cx, cy + 22, 34, CY, "monob", "c")
        text(g, f"TILT {tlt:+04.0f}°", cx, cy + 66, 34, AMB, "monob", "c")
        step(g, ov, u, 7, "ביצוע", "הצריח מסתובב." if u < .92 else "הלייזר נדלק, רק בנעילה.", BAR + .02, RED)
        cg_chrome(g, ov, 26.4 + u, None)
        flowbar(g, ov, 6, u / BAR)
    return add_glow(g, ov, .75)


# ---------- CLIMAX 50.4-55.2 ----------
HIT_LT = 3.0


def sparks(ov, p, age, sc=1.0, seed=5, n=110):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    spd = rng.uniform(160, 1050, n)
    life = rng.uniform(.25, 1.25, n)
    for a, s_, l in zip(ang, spd, life):
        if age > l:
            continue
        k = age / l
        for_a = [age, max(age - .035, 0)]
        pts = []
        for b in for_a:
            d = s_ * b * (1 - .4 * b / l) * sc
            pts.append((p[0] + math.cos(a) * d, p[1] + math.sin(a) * d + 300 * b * b * sc))
        col = (int(180 * (1 - k) ** 2), int(120 + 110 * (1 - k)) if k < .5 else int(230 * (1 - k)), 255)
        aline(ov, pts[0], pts[1], scale(col, 1 - k * .5), 2)


def smoke(g, pts, r0):
    lay = np.zeros((H // 3, W // 3), np.float32)
    for a, b in zip(pts[:-1], pts[1:]):
        for h in np.linspace(0, 1, 6, endpoint=False):
            x, y, k = lerp(a[0], b[0], h), lerp(a[1], b[1], h), lerp(a[3], b[3], h)
            wob = 7 * math.sin(k * 1.7) * (1 + k * .05)
            rr = (r0 * (1 + k * .09)) / 3
            cv2.circle(lay, (int((x + wob) / 3), int((y - wob * .6) / 3)), max(int(rr), 1), float(.85 * (1 - k / 66)), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 9)
    a = np.clip(cv2.resize(lay, (W, H)), 0, .82)[..., None]
    g[:] = (g.astype(np.float32) * (1 - a) + np.array([20, 22, 26], np.float32) * a).astype(np.uint8)


def s_climax(lt, t, fn):
    ov = blank()
    if lt < HIT_LT:
        g, m = cg("chase", lt * 30)
        c = m["c"]
        d = c["drone"]
        rr = max(c["rpx"] * 1.25, 60)
        reticle(ov, d[0], d[1], rr, lt, RED, 1)
        cg_chrome(g, ov, 28.8 + lt, "לייזר פעיל", AMB)
        return add_glow(g, ov, .8)
    u = lt - HIT_LT
    g, m = cg("hit", u * 30)
    c = m["c"]
    d = c["drone"]
    h = c.get("hitp", d)
    rr = max(c["rpx"] * 1.25, 60)
    if c.get("smoke"):
        smoke(g, c["smoke"], 54)
    sparks(ov, h, u, 1.0)
    if u > .3:
        sparks(ov, d, u - .3, .7, 9, 50)
    if u < 1.0:
        for j, (spd, col) in enumerate(((1500, INK), (900, RED), (520, scale(RED, .7)))):
            acircle(ov, h, rr + spd * u ** .75, scale(col, max(0, 1 - u / (.75 - j * .1))), 4 - j)
    cg_chrome(g, ov, 31.8 + u, None)
    g = add_glow(g, ov, .95)
    if u < .16:
        g = cv2.addWeighted(g, 1, bgfill((230, 215, 255)), (1 - u / .16) * .95, 0)
    a = tsprite("פגיעה", 210, "head")
    sc = lerp(1.5, 1.0, eout(u / .22))
    s2 = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    tx, ty = int(clamp(d[0] - rr * 1.5 - 360, 400, W - 400)), int(clamp(d[1], 240, 760))
    op = ss(.02, .12, u)
    b = cv2.GaussianBlur(cv2.copyMakeBorder(s2, 30, 30, 30, 30, cv2.BORDER_CONSTANT), (0, 0), 16)
    blit_a(g, b, tx - s2.shape[1] // 2 - 30, ty - s2.shape[0] // 2 - 30, (20, 0, 90), .85 * op)
    blit_a(g, s2, tx - s2.shape[1] // 2, ty - s2.shape[0] // 2, INK, op)
    return g


# ---------- RECAP 55.2-60.0: the whole loop ----------
def s_recap(lt, t, fn):
    g, m = cg("recap", lt * 30)
    c = m["c"]
    d = c["drone"]
    ov = blank()
    if c.get("smoke") and d[2] > 0:
        smoke(g, c["smoke"], 20)
        acircle(ov, d[:2], 7, (120, 200, 255), -1)
        acircle(ov, d[:2], 15, (30, 110, 255), -1)
    # dim the upper sky a touch so the diagram reads
    g[:560] = (g[:560].astype(np.float32) * np.r_[np.full(440, .34, np.float32), np.linspace(.34, 1, 120, dtype=np.float32)][:, None, None]).astype(np.uint8)
    y = 250
    x0, dx = W - 250, 237
    run = (lt * 1.25) % 1 * 7
    for j, (n, col) in enumerate(FLOW):
        x = x0 - j * dx
        on = abs(j - run) < .9 or abs(j - run + 7) < .9
        o = 1.0 if on else .45
        if j < 6:
            aline(ov, (x - 62, y), (x - dx + 62, y), scale(col, .75), 2)
            cv2.arrowedLine(ov, (int(x - dx + 90), y), (int(x - dx + 62), y), scale(col, .75), 2, cv2.LINE_AA, tipLength=.5)
        cv2.rectangle(ov, (x - 62, y - 34), (x + 62, y + 34), scale(col, o), 2, cv2.LINE_AA)
        text(g, n, x, y - 22, 34, INK, "head", "c", .6 + .4 * o)
    # loop back
    yb = y + 92
    xr, xl = x0, x0 - 6 * dx
    aline(ov, (xl, y + 34), (xl, yb), CY, 2)
    aline(ov, (xl, yb), (xr, yb), CY, 2)
    cv2.arrowedLine(ov, (int(xr), int(yb)), (int(xr), y + 36), CY, 2, cv2.LINE_AA, tipLength=.3)
    h = (lt * 1.25) % 1
    acircle(ov, (lerp(xl, xr, h), yb), 6, INK, -1)
    shadow_text(g, "כל הלולאה הזו: כ-20 פעם בשנייה", W // 2, 44, 74, INK, "head", "c", ss(.05, .3, lt))
    shadow_text(g, "הפריים הבא", W // 2, yb + 10, 28, CY, "ui", "c", .9)
    shadow_text(g, "המטרה נעלמה לרגע? המערכת מחזיקה כיוון 0.8 שנייה ולא מכבה.", W // 2, 410, 36, INK, "body", "c", ss(1.6, 2.0, lt))
    cg_chrome(g, ov, 33.6 + lt, None)
    return add_glow(g, ov, .6)


# ---------- KPI 60.0-67.2 ----------
_kbg = None


def s_kpi(lt, t, fn):
    global _kbg
    if _kbg is None:
        f, _ = cg("recap", 143)
        _kbg = cv2.GaussianBlur(f, (0, 0), 16)
    g = (cam(_kbg, W / 2, H / 2, 1.05 + lt * .012).astype(np.float32) * .42).astype(np.uint8)
    ov = blank()
    shadow_text(g, "והמערכת האמיתית? לא הבטחות. מדידות.", W // 2, 92, 72, INK, "head", "c", ss(.05, .35, lt))
    items = [(K["lock_seen"], "%", "נעילה", "כשהמטרה בשדה הראייה", CY),
             (K["near"][0], "%", "פגיעה בטווח קרוב", "מטרה קרובה למערכת", RED),
             (K["slow"][0], "%", "פגיעה בתנועה איטית", "מטרה במעקב רציף", RED),
             (20, "", "החלטות בשנייה", "זיהוי ← עומק ← פקודה", GRN)]
    tw, gap = 410, 30
    x0 = (W - 4 * tw - 3 * gap) // 2
    for j, (val, unit, a, b, c) in enumerate(items):
        x = x0 + (3 - j) * (tw + gap)
        d = lt - .25 - j * .55
        op = ss(0, .25, d)
        if op <= 0:
            continue
        yy = 300 + int((1 - eout(d / .4)) * 40)
        panel(g, x, yy, x + tw, yy + 420, .78 * op, scale(c, .8 * op))
        s = f"{val * ss(0, .85, d):.0f}"
        text(g, ("~" if unit == "" else "") + s + unit, x + tw // 2, yy + 58, 128, c, "monob", "c", op)
        text(g, a, x + tw // 2, yy + 234, 42, INK, "head", "c", op)
        text(g, b, x + tw // 2, yy + 296, 30, DIM, "ui", "c", op)
    fp = (f"נמדד על המתקן האמיתי, כדור מתנדנד, 17.10.2026 · פגיעה = נקודת הלייזר בתוך הכדור בתמונה · "
          f"בכל התנאים יחד {K['all'][0]:.0f}%, בתנועה מהירה {K['fast'][0]:.0f}%")
    text(g, fp, W // 2, 776, 28, INK, "ui", "c", .8 * ss(3.0, 3.5, lt))
    return add_glow(g, ov, .5)


SCENES = [(0.0, 7.2, s_hook), (7.2, 9.6, s_reveal), (9.6, 19.2, s_parts), (19.2, 21.6, s_scan), (21.6, 26.4, s_detect),
          (26.4, 31.2, s_eyes), (31.2, 36.0, s_depth), (36.0, 45.6, s_track), (45.6, 50.4, s_cmd), (50.4, 55.2, s_climax),
          (55.2, 60.0, s_recap), (60.0, 67.2, s_kpi), (67.2, 74.4, s_end)]
DUR = SCENES[-1][1]


def frame(fn):
    t = fn / FPS
    for a, b, f in SCENES:
        if a <= t < b:
            return finish(f(t - a, t, fn), fn)
    return finish(bgfill((0, 0, 0)), fn)
