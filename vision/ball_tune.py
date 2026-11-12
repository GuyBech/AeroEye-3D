r"""
AeroEye-3D — BALL TUNING VIEWER (vision only, no servos, no serial).

Shows exactly WHY a candidate blob is accepted or rejected, so the ball
detector can be tuned from data instead of guesswork. Nothing is sent to the
board, so this is safe to run while thinking.

Panels (2x2):
    top-left     left eye, every candidate annotated with area/circ/fill
    top-right    MOG2 motion mask
    bottom-left  brightness mask (adaptive: mean + K*std of the frame)
    bottom-right right eye, annotated the same way

Live trackbars retune MIN_AREA / MIN_CIRC / MIN_FILL / brightness K without
restarting. The console prints the best candidate's numbers a few times a
second, so the values can be read even without looking at the window.

    ..\.venv\Scripts\python.exe ball_tune.py [cam]
Press q (or close the window) to quit.
"""
import sys
import math
import time

import cv2
import numpy as np

import stereo

CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN = "AeroEye-3D ball tuning"


def stats(contour):
    """area, circularity, fill, centre, radius for one contour."""
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    circ = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    (x, y), r = cv2.minEnclosingCircle(contour)
    fill = area / (math.pi * r * r) if r > 0 else 0.0
    return area, circ, fill, (int(x), int(y)), max(int(r), 3)


def candidates(mask, min_area, min_circ, min_fill, max_area):
    """Every blob in the mask, each tagged with why it passed or failed."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area, circ, fill, ctr, r = stats(c)
        if area < 15:
            continue                     # pure noise, not worth reporting
        if area < min_area:
            why = "area<min"
        elif area > max_area:
            why = "area>max"             # a whole-frame gain shift, not an object
        elif circ < min_circ:
            why = "not round"
        elif fill < min_fill:
            why = "not filled"
        else:
            why = None                   # accepted
        out.append(dict(area=area, circ=circ, fill=fill, ctr=ctr, r=r, why=why))
    out.sort(key=lambda d: -d["area"])
    return out


def annotate(gray, cands, title):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in cands[:6]:
        ok = d["why"] is None
        col = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(vis, d["ctr"], d["r"], col, 2)
        label = f"a{int(d['area'])} c{d['circ']:.2f} f{d['fill']:.2f}"
        cv2.putText(vis, label, (d["ctr"][0] - 40, d["ctr"][1] - d["r"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
        if not ok:
            cv2.putText(vis, d["why"], (d["ctr"][0] - 40, d["ctr"][1] + d["r"] + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
    cv2.putText(vis, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return vis


def ref_mask(gray, ref, delta):
    """Static-reference difference mask — the right tool for a FIXED camera.

    MOG2 detects *change*, so it loses an object that stops moving (a pendulum
    stops dead at both extremes) and slowly absorbs a lingering one into the
    background. Diffing against a once-captured empty-scene reference has
    neither problem: a ball that is merely PRESENT still differs from the
    reference. Uses |difference|, so it catches a ball that is darker than the
    wall just as well as a brighter one (the ball measured ~20 grey levels
    DARKER than the wall, so that direction matters).

    Gain-immune: this camera has AGC in firmware that OpenCV cannot disable
    (CAP_PROP_AUTO_EXPOSURE reads -1 and setting EXPOSURE leaves frame_mean
    unchanged), so the whole image periodically shifts brightness at once. A
    plain difference then lights up the entire frame. Subtracting the MEDIAN of
    the signed difference removes exactly that uniform shift — the median is set
    by the background, which is most of the pixels — leaving only genuinely
    different regions.
    """
    if ref is None:
        return np.zeros_like(gray), 0
    signed = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.int16) - ref.astype(np.int16)
    signed -= int(np.median(signed))          # cancel global gain/illumination shift
    d = np.abs(signed).astype(np.uint8)
    mask = (d >= delta).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    return mask, int(d.max())


def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    # Lock auto-exposure. With it enabled, a large object entering the frame
    # makes the sensor re-gain the WHOLE image, so every pixel changes at once
    # and background subtraction flags half the frame as "motion".
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    time.sleep(1.2)
    ok, probe = cap.read()
    if not ok or probe is None:
        print("CAMERA FAILED — no frames. Is another process using it?", flush=True)
        return
    print(f"camera OK {probe.shape} — tuning window open", flush=True)
    print("KEYS:  r = capture empty-scene reference (do this with NO ball in view)"
          "   |   q = quit", flush=True)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 640)
    cv2.createTrackbar("min_area", WIN, 70, 2000, lambda v: None)
    cv2.createTrackbar("min_circ x100", WIN, 55, 100, lambda v: None)
    cv2.createTrackbar("min_fill x100", WIN, 55, 100, lambda v: None)
    cv2.createTrackbar("ref delta", WIN, 18, 90, lambda v: None)
    cv2.createTrackbar("max_area /100", WIN, 120, 700, lambda v: None)

    bg_l = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30,
                                              detectShadows=False)
    bg_r = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30,
                                              detectShadows=False)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_i, last_print = 0, 0.0
    ref_l = ref_r = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_i += 1
        left, right = stereo.split_lr(frame)

        min_area = max(cv2.getTrackbarPos("min_area", WIN), 1)
        min_circ = cv2.getTrackbarPos("min_circ x100", WIN) / 100.0
        min_fill = cv2.getTrackbarPos("min_fill x100", WIN) / 100.0
        delta = max(cv2.getTrackbarPos("ref delta", WIN), 3)
        max_area = max(cv2.getTrackbarPos("max_area /100", WIN), 2) * 100

        def motion(bg, gray):
            fg = bg.apply(gray)
            fg = cv2.medianBlur(fg, 5)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ker)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ker)
            return cv2.dilate(fg, ker, iterations=1)

        m_l = motion(bg_l, left)
        motion(bg_r, right)                    # keep the right model warm
        d_l, peak_l = ref_mask(left, ref_l, delta)
        d_r, _ = ref_mask(right, ref_r, delta)

        c_motion_l = candidates(m_l, min_area, min_circ, min_fill, max_area)
        c_ref_l = candidates(d_l, min_area, min_circ, min_fill, max_area)
        c_ref_r = candidates(d_r, min_area, min_circ, min_fill, max_area)

        top = annotate(left, c_motion_l, "LEFT / MOG2 motion")
        mid = cv2.cvtColor(d_l, cv2.COLOR_GRAY2BGR)
        cv2.putText(mid, "reference-diff mask" + ("" if ref_l is not None else "  [press r]"),
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        bot = annotate(left, c_ref_l, f"LEFT / ref-diff peak={peak_l}")
        rgt = annotate(right, c_ref_r, "RIGHT / ref-diff")

        grid = cv2.vconcat([cv2.hconcat([top, mid]), cv2.hconcat([bot, rgt])])
        cv2.putText(grid, f"area {min_area}..{max_area}  circ>{min_circ:.2f}  fill>{min_fill:.2f}"
                          f"  delta>{delta}  bg_mean={left.mean():.0f}"
                          f"  ref={'SET' if ref_l is not None else 'NOT SET - press r'}",
                    (6, grid.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
        cv2.imshow(WIN, grid)

        now = time.time()
        if now - last_print > 0.7:
            last_print = now

            def brief(cs):
                if not cs:
                    return "none"
                d = cs[0]
                return (f"a={int(d['area'])} c={d['circ']:.2f} f={d['fill']:.2f} "
                        f"@({d['ctr'][0]},{d['ctr'][1]}) {d['why'] or 'ACCEPT'}")

            acc_m = sum(1 for d in c_motion_l if d["why"] is None)
            acc_r = sum(1 for d in c_ref_l if d["why"] is None)
            # contrast of the best ref-diff blob against the reference — tells us
            # whether the ball is separable from the wall at all
            contrast = ""
            if c_ref_l and ref_l is not None:
                d0 = c_ref_l[0]
                x, y, r = d0["ctr"][0], d0["ctr"][1], max(d0["r"] // 2, 2)
                y0, y1 = max(0, y - r), min(left.shape[0], y + r)
                x0, x1 = max(0, x - r), min(left.shape[1], x + r)
                if y1 > y0 and x1 > x0:
                    contrast = (f" ball_grey={left[y0:y1, x0:x1].mean():.0f}"
                                f" vs ref={ref_l[y0:y1, x0:x1].mean():.0f}")
            print(f"[{frame_i:5d}] MOG2({len(c_motion_l)}b,{acc_m}ok): {brief(c_motion_l)}"
                  f" || REF({len(c_ref_l)}b,{acc_r}ok): {brief(c_ref_l)}{contrast}",
                  flush=True)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            ref_l = cv2.GaussianBlur(left, (5, 5), 0).copy()
            ref_r = cv2.GaussianBlur(right, (5, 5), 0).copy()
            print(f"[{frame_i:5d}] REFERENCE CAPTURED "
                  f"(mean L={ref_l.mean():.0f} R={ref_r.mean():.0f})", flush=True)
        if key == ord('q'):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
