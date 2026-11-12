r"""
AeroEye-3D — AIM ERROR METER, manually triggered and manually verifiable.

"Sometimes hits, mostly misses" has two different causes whose fixes are
opposite: a calibration error misses even with the ball STATIONARY, a latency
error misses only while it swings. So samples are taken with the ball held
still, and whatever this reports decides what to fix.

Operator-driven, because the automatic version contaminated its own data: the
"ball is still" trigger fired at the wrong instants, and the automatic dot
finder locked onto noise at the frame border when the laser missed entirely
(2 of 36 samples, with errors large enough to distort the fit). Here nothing is
recorded unless the operator confirms it, and the laser dot can be clicked by
hand when the detector is wrong.

Cycle:
    live      hold the ball still, press SPACE
              -> aims, waits for the servos, freezes the frame
    review    the ball is circled; the auto-detected dot is marked
              click the REAL dot to correct it (or accept the automatic one)
              ENTER saves | X discards | R re-aims and re-freezes
    live      move the ball, repeat
              U removes the last saved sample (a wrong click is not fatal)

Each sample stores dot_source = "manual" or "auto", so the fit can weigh
hand-verified points differently from automatic ones.

    ..\.venv\Scripts\python.exe aim_check.py [COM4] [cam] [--tag=before]

--tag=NAME keeps a measurement round in its own file (aim_error_NAME.json,
images in debug_images/aim_NAME/), so a before/after pair can be compared:
    ..\.venv\Scripts\python.exe aim_report.py aim_error_before.json aim_error_after.json
After each saved sample the coverage of the 27 cells is printed (same cells as
calib_ball.py), and on quit the full aim_report.py summary is printed.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import serial

from collections import Counter

import stereo
from detector_dark import DarkBallDetector
from calib_ball import DEPTH_NAMES, cell, depth_bin

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
TAG = next((a[len("--tag="):] for a in sys.argv[1:] if a.startswith("--tag=")), "")
OUT = os.path.join("debug_images", "aim_" + TAG if TAG else "aim")
PTS = f"aim_error_{TAG}.json" if TAG else "aim_error.json"
WIN = "AeroEye-3D aim error"

# aim with the TRACKER's own function, so this measures exactly what the
# tracker sends to the turret. An earlier private copy here only understood
# the old pixel-space correction and silently went stale when the correction
# moved to angle space.
from track_motion_3d import aim

SETTLE_S = 0.9         # let the servos finish travelling before measuring
SCALE = 3              # display zoom — big enough to click the dot precisely
DOT_MIN = 40           # auto-detect: dot must beat the local median by this

click = {"pt": None}


def on_mouse(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (mx // SCALE, my // SCALE)


def find_dot(gray, ball):
    """Brightest compact spot near the ball. Only a suggestion — the operator
    overrides it with a click whenever it is wrong."""
    r = max(ball[2], 10)
    pad = int(r * 4)
    x0, y0 = max(0, ball[0] - pad), max(0, ball[1] - pad)
    x1, y1 = min(gray.shape[1], ball[0] + pad), min(gray.shape[0], ball[1] + pad)
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    _, mv, _, loc = cv2.minMaxLoc(cv2.GaussianBlur(sub, (3, 3), 0))
    if mv < float(np.median(sub)) + DOT_MIN:
        return None
    return (x0 + loc[0], y0 + loc[1])


link = {}     # the live serial port; replaced if the Arduino drops off USB


def open_link():
    """Open the Arduino port, waiting for it to come back if it just dropped."""
    for _ in range(40):
        try:
            s = serial.Serial(PORT, 115200, timeout=1)
            time.sleep(2.5)              # the UNO resets when the port opens
            s.reset_input_buffer()
            return s
        except serial.SerialException:
            time.sleep(0.5)
    raise SystemExit(f"cannot open {PORT} - check the Arduino USB cable")


def cmd(ser, p, t, l):
    """Send one command. The USB link has dropped mid-session (the board
    re-enumerates and the write fails); reconnect and resend instead of dying,
    because every sample is already saved and the session should go on."""
    for attempt in range(3):
        try:
            link["ser"].write(f"{p},{t},{l}\n".encode())
            link["ser"].readline()
            return
        except serial.SerialException:
            print("Arduino USB link dropped - reconnecting...", flush=True)
            try:
                link["ser"].close()
            except Exception:
                pass
            link["ser"] = open_link()
            print("reconnected", flush=True)
    raise SystemExit("Arduino link keeps dropping - check the USB cable and port")


def show(img, lines, colour=(0, 255, 255), grid=None):
    """grid: {(col, row): count} for the ball's CURRENT depth, drawn as a 3x3
    overlay (red 0 = no sample there yet) — same cells as calib_ball.py."""
    big = cv2.resize(img, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)
    if grid is not None:
        H, W = big.shape[:2]
        for k in (1, 2):
            cv2.line(big, (W * k // 3, 0), (W * k // 3, H), (90, 90, 90), 1)
            cv2.line(big, (0, H * k // 3), (W, H * k // 3), (90, 90, 90), 1)
        for (c, r), n in grid.items():
            col = (0, 0, 255) if n == 0 else ((0, 200, 255) if n == 1 else (0, 220, 0))
            cv2.putText(big, str(n), (W * c // 3 + W // 6 - 10, H * r // 3 + H // 6 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    for i, t in enumerate(lines):
        cv2.putText(big, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1)
    cv2.imshow(WIN, big)


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = json.load(open(PTS)) if os.path.exists(PTS) else []
    print(f"loaded {len(samples)} previous samples", flush=True)

    ser = link["ser"] = open_link()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.2)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    dl, dr = DarkBallDetector(), DarkBallDetector()

    print("SPACE = aim & freeze | click = mark the real dot | "
          "ENTER = save | X = discard | R = re-aim | U = undo last | Q = quit", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tl, tr = dl.detect(left), dr.detect(right)
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)

        disp, whole = None, False
        if tl and tr:
            dsp = tl.x - tr.x
            if 3 <= dsp <= 70 and abs(tl.y - tr.y) <= 40:
                disp = dsp
                # a ball cut by the frame edge has a wrong centre and depth —
                # measuring the aim against it would blame the calibration
                whole = not (stereo.touches_edge(tl, left.shape) or
                             stereo.touches_edge(tr, right.shape))
                cv2.circle(vis, (tl.x, tl.y), max(tl.radius, 6),
                           (0, 255, 0) if whole else (0, 0, 255), 2)

        # coverage: per-depth summary always, the 3x3 grid for the ball's depth
        got = Counter(cell(s["ball_x"], s["ball_y"], s["disp"], left.shape) for s in samples)
        per_depth = "  ".join(
            f"{name} {sum(got[(di, c, r)] > 0 for c in range(3) for r in range(3))}/9"
            for di, name in enumerate(DEPTH_NAMES))
        grid, here = None, ""
        if disp:
            di = depth_bin(disp)
            grid = {(c, r): got[(di, c, r)] for c in range(3) for r in range(3)}
            here = f"  | ball is {DEPTH_NAMES[di].upper()}"
        show(vis, [f"samples={len(samples)}  " +
                   ((f"ball=({tl.x},{tl.y}) disp={disp} - SPACE to sample" if whole
                     else "ball touches the image edge - move it inward")
                    if disp else "no ball / stereo reject"),
                   f"covered: {per_depth}{here}",
                   "U = undo last sample | Q = quit"], grid=grid)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        if k == ord('u') and samples:
            gone = samples.pop()
            json.dump(samples, open(PTS, "w"), indent=1)
            img = f"{OUT}/{len(samples) + 1:02d}.png"
            if os.path.exists(img):
                os.remove(img)
            print(f"undo: removed #{len(samples) + 1:02d} "
                  f"ball=({gone['ball_x']},{gone['ball_y']}) disp={gone['disp']}", flush=True)
            print_coverage(samples)
            continue
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k != 32 or disp is None or not whole:
            continue

        # ---- aim, settle, freeze -------------------------------------------
        while True:
            pan, tilt = aim(tl.x, tl.y, disp)
            cmd(ser, pan, tilt, 1)
            t0 = time.time()
            while time.time() - t0 < SETTLE_S:
                cap.read()
            ok2, f2 = cap.read()
            if not ok2:
                break
            l2, _ = stereo.split_lr(f2)
            ball = dl.detect(l2) or tl
            dot = find_dot(l2, (ball.x, ball.y, ball.radius))
            source = "auto" if dot else None
            click["pt"] = None

            # ---- review ----------------------------------------------------
            decision = None
            while decision is None:
                p = cv2.cvtColor(l2, cv2.COLOR_GRAY2BGR)
                cv2.circle(p, (ball.x, ball.y), max(ball.radius, 6), (0, 255, 0), 2)
                if click["pt"]:
                    dot, source = click["pt"], "manual"
                if dot:
                    cv2.drawMarker(p, dot, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                    cv2.line(p, dot, (ball.x, ball.y), (255, 255, 0), 1)
                    err = (ball.x - dot[0], ball.y - dot[1])
                    hit = abs(err[0]) <= ball.radius and abs(err[1]) <= ball.radius
                    info = (f"err=({err[0]:+d},{err[1]:+d}) r={ball.radius} "
                            f"{'HIT' if hit else 'MISS'} [{source}]")
                else:
                    info = "no dot found - CLICK the laser dot"
                show(p, [f"#{len(samples) + 1}  {info}",
                         "ENTER save | X discard | R re-aim | click = mark dot"],
                     (0, 255, 0) if dot else (0, 165, 255))

                kk = cv2.waitKey(20) & 0xFF
                if kk in (13, 10):
                    decision = "save" if dot else None
                elif kk == ord('x'):
                    decision = "discard"
                elif kk == ord('r'):
                    decision = "reaim"
                elif kk == ord('q'):
                    decision = "quit"
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    decision = "quit"

            if decision == "reaim":
                continue
            if decision in ("discard", "quit"):
                cmd(ser, pan, tilt, 0)
                if decision == "quit":
                    cap.release(); link["ser"].close(); cv2.destroyAllWindows()
                    print(f"saved {len(samples)} samples to {PTS}", flush=True)
                    final_report()
                    return
                break

            rec = dict(ball_x=ball.x, ball_y=ball.y, disp=disp, pan=pan, tilt=tilt,
                       dot_x=dot[0], dot_y=dot[1],
                       err_x=ball.x - dot[0], err_y=ball.y - dot[1],
                       radius=ball.radius, dot_source=source)
            samples.append(rec)
            json.dump(samples, open(PTS, "w"), indent=1)
            p = cv2.cvtColor(l2, cv2.COLOR_GRAY2BGR)
            cv2.circle(p, (ball.x, ball.y), max(ball.radius, 6), (0, 255, 0), 2)
            cv2.drawMarker(p, dot, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.imwrite(f"{OUT}/{len(samples):02d}.png", p)
            print(f"#{len(samples):02d} ball=({ball.x},{ball.y}) disp={disp} "
                  f"-> pan={pan} tilt={tilt} | dot=({dot[0]},{dot[1]}) "
                  f"err=({rec['err_x']:+d},{rec['err_y']:+d}) r={ball.radius} "
                  f"[{source}]", flush=True)
            print_coverage(samples)
            cmd(ser, pan, tilt, 0)
            break

    cmd(ser, 90, 25, 0)
    cap.release()
    link["ser"].close()
    cv2.destroyAllWindows()
    print(f"saved {len(samples)} samples to {PTS}", flush=True)
    final_report()


def print_coverage(samples):
    """Which of the 27 cells (depth x column x row) already have a sample."""
    from collections import Counter
    from calib_ball import DEPTH_NAMES, cell
    got = Counter(cell(s["ball_x"], s["ball_y"], s["disp"], (240, 320)) for s in samples)
    print(f"   coverage {len(got)}/27 cells:  " + "   ".join(
        f"{name}: " + "/".join("".join(str(got[(di, c, r)]) for c in range(3))
                               for r in range(3))
        for di, name in enumerate(DEPTH_NAMES)), flush=True)


def final_report():
    if os.path.exists(PTS):
        import aim_report
        aim_report.report(PTS)


if __name__ == "__main__":
    main()
