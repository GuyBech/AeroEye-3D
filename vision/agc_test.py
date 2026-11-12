r"""
AeroEye-3D — does the camera change its own brightness? (AGC test)

The documents say the camera runs automatic gain / exposure (AGC) in firmware
that OpenCV cannot switch off. The evidence so far is indirect: the property
reads -1, setting EXPOSURE has no visible effect, and an empty-scene reference
went from frame mean 132 to 113 between capture and use. That last drop could
also have been the room light changing. This test separates the two.

Method: watch a fixed patch of wall that nothing touches, and put a large
object in the OTHER half of the picture, on cue, several times. The light that
falls on the watched wall does not change when the object enters, so:

  wall brightness follows the object  ->  the camera re-gained the image (AGC)
  wall brightness ignores the object  ->  no AGC; the old drift was the room

Two checks make the verdict hard to fool:
  * repetition — 3 cycles per object. A room-light change would not happen to
    line up with the cue three times in a row.
  * direction — a DARK object must make the wall BRIGHTER (the camera boosts
    the image to restore its average), a WHITE object must make it DARKER.
    A shadow or a covered patch makes the wall darker for BOTH objects, so it
    can never fake the dark-object result.
Each object phase is compared to the average of the empty phases on both sides
of it, which also cancels a slow drift of the room light during the test.

Nothing is changed on the camera: the script only reads its properties and
frames. No Arduino, laser or servos are involved.

How to run (about 2 minutes; best in the evening with the lamp on and the
curtains closed, so daylight cannot drift during the test):
  1. Close the tracker (it holds the camera). Start this script.
     A live view opens with a GREEN box on the wall in each eye — the watched
     patch — and a YELLOW frame on the other half: that is where the object
     goes. Click to move the green box to plain wall, near the outer edge.
  2. Get a large DARK object (black folder, jacket, book) and a sheet of WHITE
     paper. Hold the object ~20-30 cm in front of the camera, inside the
     yellow half, so it fills most of that half but never covers or shades the
     green box. Try it now and look at the view.
  3. Stand to the side, out of the picture. Press SPACE to start.
  4. Follow the cue on the screen (a beep marks each change):
        EMPTY      nothing in the picture
        DARK IN    hold the dark object in place, still
        WHITE IN   hold the white paper in place, still
     Q or Esc aborts.
  5. At the end the verdict is printed and saved, with a graph.

    ..\.venv\Scripts\python.exe agc_test.py [cam] [--nowhite]

Output: debug_images/agc_<date>_<time>/
          result.json   every frame, per-phase values, verdict
          graph.png     wall brightness over time, object phases shaded
          compare.png   one frame from an EMPTY, DARK and WHITE phase
"""
import json
import os
import sys
import time

import cv2
import numpy as np

import stereo

try:
    import winsound
except ImportError:            # not Windows: run without the beeps
    winsound = None

CAM   = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WHITE = "--nowhite" not in sys.argv
WIN   = "AeroEye-3D AGC test"
SCALE = 2
FONT  = cv2.FONT_HERSHEY_SIMPLEX

PHASE_S    = 6.0     # length of every EMPTY / object phase
SETTLE_S   = 2.0     # ignored at the start of a phase: your reaction + the camera re-adjusting
LEAD_IN_S  = 3.0     # after SPACE, time to step out of the picture
CYCLES     = 3       # in/out repetitions per object
HALF       = 20      # watched patch = 41x41 px
MIN_SHIFT  = 2.0     # grey levels — smallest wall change counted as real
MIN_OBJECT = 15.0    # grey levels the object must change its half of the picture by
OUT = os.path.join("debug_images", "agc_" + time.strftime("%Y%m%d_%H%M%S"))

PROPS = {"AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE, "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
         "GAIN": cv2.CAP_PROP_GAIN, "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
         "FPS": cv2.CAP_PROP_FPS}


def build_phases():
    seq = ["EMPTY"]
    for obj in ["DARK"] + (["WHITE"] if WHITE else []):
        for _ in range(CYCLES):
            seq += [obj, "EMPTY"]
    return seq


def clamp_box(x, y, W, H):
    """Keep the watched patch wholly inside one half of the eye, so the object
    half never overlaps it."""
    if x < W // 2:
        x = min(max(x, HALF), W // 2 - HALF - 4)
    else:
        x = min(max(x, W // 2 + HALF + 4), W - HALF - 1)
    return x, min(max(y, HALF), H - HALF - 1)


def obj_slice(cx, W):
    return slice(W // 2, W) if cx < W // 2 else slice(0, W // 2)


def measure(left, right, cx, cy):
    """Mean grey of the watched patch and of the object half, in each eye."""
    W = left.shape[1]
    box = (slice(cy - HALF, cy + HALF + 1), slice(cx - HALF, cx + HALF + 1))
    half = obj_slice(cx, W)
    return [float(left[box].mean()), float(right[box].mean()),
            float(left[:, half].mean()), float(right[:, half].mean())]


def beep(name):
    if winsound is None:
        return
    try:
        if name == "EMPTY":
            winsound.Beep(600, 150)
        else:                                   # object in: higher, double
            winsound.Beep(1200, 90); winsound.Beep(1200, 90)
    except RuntimeError:
        pass


def draw_view(left, right, cx, cy, lines, banner=None, banner_col=(255, 255, 255)):
    H, W = left.shape[:2]
    vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
    big = cv2.resize(vis, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    ox0 = W // 2 if cx < W // 2 else 0
    for e in (0, 1):
        x0, x1 = (e * W + ox0) * SCALE, (e * W + ox0 + W // 2) * SCALE
        cv2.rectangle(big, (x0 + 3, 3), (x1 - 4, H * SCALE - 4), (0, 200, 255), 1)
        cv2.putText(big, "object here", (x0 + 10, H * SCALE - 12), FONT, 0.5, (0, 200, 255), 1)
        cv2.rectangle(big, ((e * W + cx - HALF) * SCALE, (cy - HALF) * SCALE),
                      ((e * W + cx + HALF) * SCALE, (cy + HALF) * SCALE), (0, 255, 0), 2)
    cv2.line(big, (W * SCALE, 0), (W * SCALE, H * SCALE), (80, 80, 80), 1)
    for i, t in enumerate(lines):
        cv2.putText(big, t, (8, 20 + 20 * i), FONT, 0.5, (0, 255, 255), 1)
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, FONT, 1.6, 3)
        bx, by = (big.shape[1] - tw) // 2, big.shape[0] // 2
        cv2.rectangle(big, (bx - 12, by - th - 12), (bx + tw + 12, by + 14), (0, 0, 0), -1)
        cv2.putText(big, banner, (bx, by), FONT, 1.6, banner_col, 3)
    cv2.imshow(WIN, big)


def window_closed():
    return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1


# ---------------------------------------------------------------- analysis

def analyse(phases, samples):
    a = np.array(samples, dtype=float)          # t, phase, wallL, wallR, objL, objR
    summ = []
    for i, name in enumerate(phases):
        sel = (a[:, 1] == i) & (a[:, 0] - i * PHASE_S >= SETTLE_S)
        r = a[sel]
        s = {"phase": i, "name": name, "frames": int(sel.sum())}
        for k, col in (("wall_L", 2), ("wall_R", 3), ("obj_L", 4), ("obj_R", 5)):
            s[k] = round(float(np.median(r[:, col])), 2) if len(r) else None
        summ.append(s)

    # Frame-to-frame scatter of the wall inside EMPTY phases (robust sigma).
    resid = []
    for s in summ:
        if s["name"] == "EMPTY" and s["frames"] > 10:
            sel = (a[:, 1] == s["phase"]) & (a[:, 0] - s["phase"] * PHASE_S >= SETTLE_S)
            resid.extend(a[sel, 2] - np.median(a[sel, 2]))
    sigma = 1.4826 * float(np.median(np.abs(resid))) if resid else 0.0
    thresh = max(MIN_SHIFT, 4 * sigma)

    cycles = []
    for i, s in enumerate(summ):
        if s["name"] == "EMPTY":
            continue
        b, n = summ[i - 1], summ[i + 1]         # always EMPTY by construction
        if None in (s["wall_L"], b["wall_L"], n["wall_L"]):
            cycles.append({"object": s["name"], "phase": i, "valid": False})
            continue
        base = lambda k: (b[k] + n[k]) / 2
        eff = s["obj_L"] - base("obj_L")
        cycles.append({"object": s["name"], "phase": i,
                       "wall_delta_L": round(s["wall_L"] - base("wall_L"), 2),
                       "wall_delta_R": round(s["wall_R"] - base("wall_R"), 2),
                       "object_effect_L": round(eff, 2),
                       "valid": bool(eff <= -MIN_OBJECT if s["name"] == "DARK"
                                     else eff >= MIN_OBJECT)})

    def direction(obj, key):
        mine = [c for c in cycles if c["object"] == obj]
        if not mine:
            return "SKIPPED"
        d = [c[key] for c in mine if c["valid"]]
        if len(d) < 2:
            return "INVALID"
        if all(x > thresh for x in d):
            return "UP"
        if all(x < -thresh for x in d):
            return "DOWN"
        if all(abs(x) < thresh for x in d):
            return "NONE"
        return "MIXED"

    res = {"dark": direction("DARK", "wall_delta_L"),
           "white": direction("WHITE", "wall_delta_L"),
           "dark_right_eye": direction("DARK", "wall_delta_R"),
           "white_right_eye": direction("WHITE", "wall_delta_R")}
    verdict, why = decide(res["dark"], res["white"], thresh)
    if (res["dark_right_eye"], res["white_right_eye"]) == (res["dark"], res["white"]):
        why += " Both eyes behave the same."
    else:
        why += (f" Eyes differ: right eye dark={res['dark_right_eye']}"
                f" white={res['white_right_eye']}.")
    res.update(verdict=verdict, explanation=why, noise_sigma=round(sigma, 3),
               threshold=round(thresh, 2), phases=summ, cycles=cycles)
    return res


def decide(dark, white, thresh):
    t = f"{thresh:.1f}"
    if dark == "DOWN":
        return ("INCONCLUSIVE",
                "The dark object made the watched wall DARKER: it covered or shaded "
                "the green box. Hold it further from the box (or move the box) and repeat.")
    if dark == "UP":
        if white == "DOWN":
            return ("AGC CONFIRMED",
                    "The wall got brighter every time the dark object entered and darker "
                    "every time the white paper did. Only the camera re-adjusting its own "
                    "brightness does that; a shadow or a room-light change cannot flip "
                    "direction with the colour of the object.")
        return ("AGC CONFIRMED (dark test)",
                "The wall got brighter every time the dark object entered. Nothing in the "
                "room adds light when a dark object appears, so the camera raised its own "
                f"brightness. White test: {white}.")
    if dark == "NONE" and white == "DOWN":
        return ("AGC PROBABLE (one-sided)",
                "The wall got darker when the white paper entered but did not react to the "
                "dark object. That fits a camera already at its longest exposure in this "
                "light (it can darken, not brighten further), but a shadow from the paper "
                "would look the same. Repeat with the room lamp brighter: if the dark object "
                "then brightens the wall, AGC is confirmed.")
    if dark == "NONE":
        extra = ("" if white != "SKIPPED" else
                 " The white test was not run: in a dim room the camera may already be at "
                 "its maximum and only able to darken, so run it again without --nowhite.")
        return ("NO AGC DETECTED",
                f"The watched wall stayed within +-{t} grey levels while the object changed "
                "its half of the picture. The camera does not re-adjust its brightness to the "
                "scene, so the 132 -> 113 drift in the documents was most likely the room "
                f"light changing. White test: {white}." + extra)
    if dark == "INVALID":
        return ("INCONCLUSIVE",
                f"The dark object did not darken its half of the picture by {MIN_OBJECT:.0f}+ "
                "grey levels in at least 2 cycles. Hold it closer to the camera so it fills "
                "most of the yellow half, and repeat.")
    return ("INCONCLUSIVE",
            "The wall moved differently from cycle to cycle (room light changing, or the "
            "object held in different places). Repeat with steady light.")


# ---------------------------------------------------------------- pictures

def draw_graph(phases, samples, res, path):
    a = np.array(samples, dtype=float)
    Wg, Hg, x0, x1 = 1100, 615, 70, 1080
    img = np.full((Hg, Wg, 3), 255, np.uint8)
    T = len(phases) * PHASE_S
    X = lambda t: int(x0 + (x1 - x0) * t / T)
    panels = [("watched wall patch (grey level)", 2, 3, 60, 360, 10.0),
              ("object half of the picture (grey level)", 4, 5, 410, 570, 20.0)]
    for i, name in enumerate(phases):
        if name != "EMPTY":
            col = (190, 190, 190) if name == "DARK" else (215, 240, 250)
            for _, _, _, ya, yb, _ in panels:
                cv2.rectangle(img, (X(i * PHASE_S), ya), (X((i + 1) * PHASE_S), yb), col, -1)
            cv2.putText(img, name, (X(i * PHASE_S) + 4, 52), FONT, 0.4, (60, 60, 60), 1)
    for title, cl, cr, ya, yb, min_span in panels:
        vals = a[:, [cl, cr]]
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < min_span:                    # never blow noise up into a "signal"
            mid = (hi + lo) / 2
            lo, hi = mid - min_span / 2, mid + min_span / 2
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        Y = lambda v: int(yb - (yb - ya) * (v - lo) / (hi - lo))
        cv2.rectangle(img, (x0, ya), (x1, yb), (120, 120, 120), 1)
        for v in np.linspace(lo, hi, 5)[1:-1]:
            cv2.line(img, (x0, Y(v)), (x1, Y(v)), (225, 225, 225), 1)
            cv2.putText(img, f"{v:.0f}", (x0 - 40, Y(v) + 4), FONT, 0.4, (80, 80, 80), 1)
        for col, colour in ((cr, (0, 140, 255)), (cl, (200, 80, 0))):
            pts = np.array([[X(t), Y(v)] for t, v in zip(a[:, 0], a[:, col])], np.int32)
            cv2.polylines(img, [pts], False, colour, 1, cv2.LINE_AA)
        cv2.putText(img, title, (x0 + 6, ya + 16), FONT, 0.45, (0, 0, 0), 1)
    cv2.putText(img, f"{res['verdict']}   (dark: {res['dark']}, white: {res['white']},"
                f" threshold {res['threshold']} grey levels)", (x0, 24), FONT, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "left eye", (x1 - 170, 24), FONT, 0.45, (200, 80, 0), 1)
    cv2.putText(img, "right eye", (x1 - 85, 24), FONT, 0.45, (0, 140, 255), 1)
    for s in range(0, int(T) + 1, 12):
        cv2.putText(img, str(s), (X(s) - 6, 588), FONT, 0.4, (80, 80, 80), 1)
    cv2.putText(img, "time (s)", ((x0 + x1) // 2 - 30, 607), FONT, 0.45, (80, 80, 80), 1)
    cv2.imwrite(path, img)


def draw_compare(phases, snaps, cx, cy, path):
    tiles = []
    for want in ("EMPTY", "DARK", "WHITE"):
        idx = next((i for i, n in enumerate(phases) if n == want and i in snaps), None)
        if idx is None:
            continue
        left = snaps[idx]
        wall = float(left[cy - HALF:cy + HALF + 1, cx - HALF:cx + HALF + 1].mean())
        t = cv2.resize(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(t, ((cx - HALF) * SCALE, (cy - HALF) * SCALE),
                      ((cx + HALF) * SCALE, (cy + HALF) * SCALE), (0, 255, 0), 2)
        cv2.putText(t, f"{want}   wall = {wall:.1f}", (8, 24), FONT, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    if tiles:
        cv2.imwrite(path, cv2.hconcat(tiles))


# ---------------------------------------------------------------- main

def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    props = {k: cap.get(v) for k, v in PROPS.items()}     # read only — nothing is set
    print("camera properties as OpenCV reports them (-1 / 0 = not exposed):", flush=True)
    for k, v in props.items():
        print(f"  {k:14s} {v}", flush=True)

    cv2.namedWindow(WIN)
    st = {"cx": 40, "cy": 110, "W": 320, "H": 240, "setup": True}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and st["setup"]:
            xi, yi = x // SCALE, y // SCALE
            st["cx"], st["cy"] = clamp_box(xi % st["W"], yi, st["W"], st["H"])
    cv2.setMouseCallback(WIN, on_mouse)

    # ---- setup: live view until SPACE
    last_ok = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if time.time() - last_ok > 5:
                print("CAMERA FAILED — no frames. Is the tracker or another program "
                      "using it?", flush=True)
                cap.release(); cv2.destroyAllWindows()
                return
            continue
        last_ok = time.time()
        left, right = stereo.split_lr(frame)
        st["H"], st["W"] = left.shape[:2]
        wl, wr, ol, orr = measure(left, right, st["cx"], st["cy"])
        draw_view(left, right, st["cx"], st["cy"], [
            "SETUP  click = move the green box   SPACE = start   Q = quit",
            f"wall L {wl:6.1f}  R {wr:6.1f}     object half L {ol:6.1f}  R {orr:6.1f}",
            "try the object in the yellow half now - it must not touch the box"])
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            break
        if k in (ord("q"), 27) or window_closed():
            cap.release(); cv2.destroyAllWindows()
            return
    st["setup"] = False
    cx, cy = st["cx"], st["cy"]

    # ---- timed run
    phases = build_phases()
    samples, snaps = [], {}
    cur, t0 = None, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        t = time.perf_counter() - t0 - LEAD_IN_S
        idx = -1 if t < 0 else int(t // PHASE_S)
        if idx >= len(phases):
            break
        if idx != cur:
            cur = idx
            if idx >= 0:
                beep(phases[idx])
        left, right = stereo.split_lr(frame)
        m = measure(left, right, cx, cy)
        if idx < 0:
            banner, col = f"STEP ASIDE  {-t:.1f}", (255, 255, 255)
            sub = f"first: {phases[0]}"
        else:
            name = phases[idx]
            into = t - idx * PHASE_S
            samples.append([round(t, 3), idx] + [round(v, 2) for v in m])
            if into >= PHASE_S - 0.5 and idx not in snaps:
                snaps[idx] = left.copy()
            banner = {"EMPTY": "EMPTY", "DARK": "DARK IN", "WHITE": "WHITE IN"}[name]
            banner += f"  {PHASE_S - into:.1f}"
            col = {"EMPTY": (255, 255, 255), "DARK": (0, 200, 255), "WHITE": (0, 255, 0)}[name]
            nxt = phases[idx + 1] if idx + 1 < len(phases) else "done"
            sub = f"phase {idx + 1}/{len(phases)}   next: {nxt}"
        draw_view(left, right, cx, cy, [
            f"RUNNING   {sub}   (Q = abort)",
            f"wall L {m[0]:6.1f}  R {m[1]:6.1f}     object half L {m[2]:6.1f}  R {m[3]:6.1f}"],
            banner, col)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27) or window_closed():
            print("aborted — nothing saved.", flush=True)
            cap.release(); cv2.destroyAllWindows()
            return
    cap.release(); cv2.destroyAllWindows()
    if winsound:
        try:
            winsound.Beep(900, 400)
        except RuntimeError:
            pass

    # ---- verdict + files
    res = analyse(phases, samples)
    os.makedirs(OUT, exist_ok=True)
    print("\ncycle  object   wall L   wall R   object half L   counted", flush=True)
    for n, c in enumerate(res["cycles"], 1):
        if "wall_delta_L" not in c:
            print(f"  {n}    {c['object']:6s}   (no data)", flush=True)
            continue
        print(f"  {n}    {c['object']:6s}  {c['wall_delta_L']:+6.1f}   {c['wall_delta_R']:+6.1f}"
              f"      {c['object_effect_L']:+6.1f}        {'yes' if c['valid'] else 'NO'}",
              flush=True)
    print(f"\nwall noise sigma {res['noise_sigma']:.2f}, threshold {res['threshold']:.1f}"
          f" grey levels   (dark: {res['dark']}, white: {res['white']})", flush=True)
    print(f"\nVERDICT: {res['verdict']}\n{res['explanation']}\n", flush=True)

    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "cam": CAM, "camera_properties": props,
           "patch": {"cx": cx, "cy": cy, "half": HALF},
           "settings": {"phase_s": PHASE_S, "settle_s": SETTLE_S, "cycles": CYCLES,
                        "min_shift": MIN_SHIFT, "min_object": MIN_OBJECT},
           **res,
           "samples_columns": ["t", "phase", "wall_L", "wall_R", "obj_L", "obj_R"],
           "samples": samples}
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    draw_graph(phases, samples, res, os.path.join(OUT, "graph.png"))
    draw_compare(phases, snaps, cx, cy, os.path.join(OUT, "compare.png"))
    print(f"saved {OUT}\\result.json, graph.png, compare.png", flush=True)


if __name__ == "__main__":
    main()
