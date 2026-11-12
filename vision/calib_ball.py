r"""
AeroEye-3D — recalibration against the BALL itself, by auto-centring.

The original calibration swept a laser across a flat board. This one uses the
actual target at actual working positions, which also folds in whatever has
shifted since (the measured 5.8 deg tilt bias says something moved).

Why auto-centring rather than aim-once-and-measure: a single shot records
"we commanded these angles and missed by 30 px", so the point is only as good
as the model used to derive the error. Here the turret iterates until the dot
sits ON the ball centre, and only then is the pose recorded. Each point then
means "these angles put the laser exactly HERE", which is ground truth
independent of any model — the same reason the board sweep was accurate, but
measured on the real target.

Convergence uses the current calibration's local Jacobian to convert a pixel
error into an angle nudge:
    d(pan)/dx  = PAN_C[0]  + PAN_C[3]*y + 2*PAN_C[4]*x
    d(tilt)/dy = TILT_C[1] + TILT_C[3]*x + 2*TILT_C[5]*y
with a gain below 1 so the loop cannot overshoot into oscillation.

Cycle (default, user-paced):
    live     hold the ball still, press SPACE
    centre   the turret aims once and WAITS. Judge on the screen image - the
             camera's view is what the tracker aims by, so "centre" means the
             centre of the ball in THIS image, not as seen from where you stand.
               C        one automatic correction step (detected or clicked dot)
               W/A/S/D  nudge the dot up / left / down / right by 1 deg
               click    mark the dot when the detector misses or mislabels it
               ENTER    the dot is on the ball centre -> record
               X        cancel
    saved    move the ball, repeat;  U removes the last point

--auto restores the old fully automatic loop (iterate, record on convergence).
It needs the dot detector to be right every step; when it locks onto a glint
the point is recorded at the wrong angles (seen: tilt 3 / 9.7 / 11.3 deg for
the same ball position), which is why it is no longer the default.

Two guards on point quality:
  * EDGE  a ball cut by the image edge (in EITHER eye) has its centre pulled
          onto the visible part, so its position and disparity are wrong.
          5 of the first 20 points were like that. SPACE is refused while the
          ball touches an edge, and a point whose final view is cut is dropped.
  * COVERAGE  a correction fitted on one corner of the working volume only
          guesses everywhere else (the first set covered the top 60% of the
          frame and disparity 18-62, while the demo runs at 5-33). The volume
          is split into 3 depths x 3 columns x 3 rows = 27 cells; the live view
          shows a 3x3 grid for the ball's CURRENT depth with the number of
          points already in each cell (red = none yet). Aim for every cell.

Then fit:
    ..\.venv\Scripts\python.exe fit_ball_calib.py

    ..\.venv\Scripts\python.exe calib_ball.py [COM4] [cam]
    ..\.venv\Scripts\python.exe calib_ball.py --reset     # start a fresh set

--reset snapshots the active calibration (calib_snapshot.py) and starts with no
points. Use it whenever the turret or camera moved: points recorded before the
move describe a different geometry and would pull the fit the wrong way.
"""
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import serial

import stereo
from detector_dark import DarkBallDetector

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
PTS = "ball_calib_points.json"
OUT = os.path.join("debug_images", "ballcal")
WIN = "AeroEye-3D ball calibration"

calib = json.load(open("calib3d.json"))
PAN_C, TILT_C = calib["pan"], calib["tilt"]

SCALE = 3
SETTLE_S = 0.55        # servo travel + camera latency before re-measuring
GAIN = 0.55            # <1 so the correction cannot overshoot into oscillation
TOL_PX = 4             # converged once the dot is this close to the ball centre
MAX_IT = 12
DOT_MIN = 35           # dot must beat the local median brightness by this
PAN_LIM = (0, 180)
TILT_LIM = (3, 150)
DISP_MIN, DISP_MAX, ROW_TOL = 3, 70, 40     # same sanity gate as the tracker
# depth bins by disparity: far < 12 <= mid < 22 <= near. Chosen from a 45 s
# recording of the demo range (disparity 5-33, median 16) so that each bin
# holds about a third of where the ball actually goes.
DEPTH_EDGES = (12, 22)
DEPTH_NAMES = ("far", "mid", "near")

AUTO = "--auto" in sys.argv
NUDGE_DEG = 1          # the firmware takes whole degrees

click = {"pt": None}


def on_mouse(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["pt"] = (mx // SCALE, my // SCALE)


def aim(x, y, disp):
    f = [x, y, disp, x * y, x * x, y * y, 1.0]
    return (sum(c * v for c, v in zip(PAN_C, f)),
            sum(c * v for c, v in zip(TILT_C, f)))


def jacobian(x, y):
    """Degrees per pixel, locally, from the current model."""
    dpan = PAN_C[0] + PAN_C[3] * y + 2 * PAN_C[4] * x
    dtilt = TILT_C[1] + TILT_C[3] * x + 2 * TILT_C[5] * y
    # guard against a degenerate local slope sending the nudge to infinity
    if abs(dpan) < 1e-3:
        dpan = -0.15
    if abs(dtilt) < 1e-3:
        dtilt = -0.166
    return dpan, dtilt


def find_dot(gray, ball):
    r = max(ball[2], 10)
    pad = int(r * 5)
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
    """Send one command; if the USB link dropped (it has, mid-session),
    reconnect and resend rather than lose the session."""
    msg = f"{int(round(p))},{int(round(t))},{l}\n".encode()
    for attempt in range(3):
        try:
            link["ser"].write(msg)
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


def depth_bin(disp):
    return sum(disp >= e for e in DEPTH_EDGES)


def cell(x, y, disp, shape):
    """Coverage cell of a point: (depth, column, row), each 0..2."""
    H, W = shape[:2]
    return depth_bin(disp), min(2, int(x * 3 // W)), min(2, int(y * 3 // H))


def coverage(points, shape):
    return Counter(cell(p["x"], p["y"], p["disp"], shape) for p in points)


def coverage_report(points, shape):
    got = coverage(points, shape)
    print(f"coverage: {len(got)}/27 cells, {len(points)} points   "
          f"(columns left..right, rows top..bottom)", flush=True)
    for di, name in enumerate(DEPTH_NAMES):
        rows = ["  ".join(f"{got[(di, c, r)]:d}" for c in range(3)) for r in range(3)]
        print(f"  {name:4s} | " + " | ".join(rows), flush=True)


def show(img, lines, colour=(0, 255, 255), grid=None):
    """grid: {(col, row): count} for the ball's current depth, drawn as a 3x3
    overlay so empty cells (red) can be seen and filled while calibrating."""
    big = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    if grid is not None:
        H, W = big.shape[:2]
        for k in (1, 2):
            cv2.line(big, (W * k // 3, 0), (W * k // 3, H), (90, 90, 90), 1)
            cv2.line(big, (0, H * k // 3), (W, H * k // 3), (90, 90, 90), 1)
        for (c, r), n in grid.items():
            col = (0, 0, 255) if n == 0 else ((0, 200, 255) if n == 1 else (0, 220, 0))
            cv2.putText(big, str(n), (W * c // 3 + W // 6 - 8, H * r // 3 + H // 6 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    for i, t in enumerate(lines):
        cv2.putText(big, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imshow(WIN, big)
    return cv2.waitKey(20) & 0xFF


def grab(cap, dl, dr):
    ok, f = cap.read()
    if not ok or f is None:
        return None, None, None, None
    l, r = stereo.split_lr(f)
    return l, r, dl.detect(l), dr.detect(r)


def start_aim(x, y, disp):
    """First shot from the tracker's own aim (board model + active correction),
    so the dot starts as close as the live system gets; board model if that
    cannot be loaded."""
    try:
        from track_motion_3d import aim as tracker_aim
        return tracker_aim(x, y, disp)
    except Exception:
        return aim(x, y, disp)


def centre_assisted(cap, dl, dr, tl, disp):
    """User-paced centring: nothing moves unless a key says so, so there is
    always time to check the image and correct the detector by hand.
    Returns (record, image) or (None, None) if cancelled."""
    pan, tilt = start_aim(tl.x, tl.y, disp)
    pan = int(round(min(max(pan, PAN_LIM[0]), PAN_LIM[1])))
    tilt = int(round(min(max(tilt, TILT_LIM[0]), TILT_LIM[1])))
    cmd(None, pan, tilt, 1)
    dpan, dtilt = jacobian(tl.x, tl.y)
    sx = 1 if dpan > 0 else -1          # pan direction that moves the dot RIGHT in the image
    sy = 1 if dtilt > 0 else -1         # tilt direction that moves the dot DOWN in the image
    marked = None                       # a clicked dot; dropped as soon as the turret moves
    click["pt"] = None
    good = None                         # (ball, disparity) last seen whole in both eyes
    p = None
    while True:
        left, right, b, br = grab(cap, dl, dr)
        if left is None:
            continue
        if b and br and DISP_MIN <= (b.x - br.x) <= DISP_MAX and abs(b.y - br.y) <= ROW_TOL \
                and not stereo.touches_edge(b, left.shape) \
                and not stereo.touches_edge(br, right.shape):
            good = (b, b.x - br.x)
        ball = b or (good[0] if good else tl)
        if click["pt"]:
            marked, click["pt"] = click["pt"], None
        dot = marked or find_dot(left, (ball.x, ball.y, ball.radius))
        err = (ball.x - dot[0], ball.y - dot[1]) if dot else None

        p = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        cv2.circle(p, (ball.x, ball.y), max(ball.radius, 6), (0, 255, 0), 2)
        cv2.drawMarker(p, (ball.x, ball.y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
        if dot:
            cv2.drawMarker(p, dot, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.line(p, dot, (ball.x, ball.y), (255, 255, 0), 1)
        k = show(p, [f"pan={pan} tilt={tilt}   " +
                     (f"dot -> centre: ({err[0]:+d},{err[1]:+d}) px [{'clicked' if marked else 'auto'}]"
                      if err else "dot not found - click it"),
                     "ENTER = dot on centre: record | C = auto step | W/A/S/D = nudge 1 deg",
                     "click = mark the real dot | X = cancel"],
                 (0, 255, 0) if dot else (0, 165, 255))

        if k == ord('x'):
            cmd(None, pan, tilt, 0)
            return None, None
        if k in (13, 10):
            if good is None:
                print("  (ball not seen whole in both eyes - not recorded; "
                      "keep it off the image edge)", flush=True)
                continue
            b0, d0 = good
            cmd(None, pan, tilt, 0)
            return dict(x=b0.x, y=b0.y, disp=d0, pan=float(pan), tilt=float(tilt),
                        radius=b0.radius, mode="assisted"), p
        moved = False
        if k == ord('c') and err:
            jp, jt = jacobian(ball.x, ball.y)
            pan = int(round(pan + GAIN * err[0] * jp))
            tilt = int(round(tilt + GAIN * err[1] * jt))
            moved = True
        elif k in (ord('a'), ord('d'), ord('w'), ord('s')):
            pan += {ord('d'): sx, ord('a'): -sx}.get(k, 0) * NUDGE_DEG
            tilt += {ord('s'): sy, ord('w'): -sy}.get(k, 0) * NUDGE_DEG
            moved = True
        if moved:
            pan = min(max(pan, PAN_LIM[0]), PAN_LIM[1])
            tilt = min(max(tilt, TILT_LIM[0]), TILT_LIM[1])
            cmd(None, pan, tilt, 1)
            marked = None


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--reset" in sys.argv:
        import calib_snapshot
        calib_snapshot.save("before_ball_reset")
        if os.path.exists(PTS):
            os.remove(PTS)
            print("cleared the ball points (they are in the snapshot)", flush=True)
    points = json.load(open(PTS)) if os.path.exists(PTS) else []
    print(f"loaded {len(points)} calibration points", flush=True)

    ser = link["ser"] = open_link()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.2)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    dl, dr = DarkBallDetector(), DarkBallDetector()

    print("SPACE = aim at the ball | then C = auto step, W/A/S/D = nudge, "
          "click = mark the dot, ENTER = record, X = cancel | U = undo last point | Q = quit"
          if not AUTO else
          "SPACE = centre & record (auto) | click = mark the dot | X = abort | "
          "U = undo last point | Q = quit", flush=True)
    shape = None

    while True:
        left, right, tl, tr = grab(cap, dl, dr)
        if left is None:
            continue
        shape = left.shape
        vis = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        disp, whole = None, False
        if tl and tr:
            d0 = tl.x - tr.x
            if DISP_MIN <= d0 <= DISP_MAX and abs(tl.y - tr.y) <= ROW_TOL:
                disp = d0
                whole = not (stereo.touches_edge(tl, left.shape) or
                             stereo.touches_edge(tr, right.shape))
                cv2.circle(vis, (tl.x, tl.y), max(tl.radius, 6),
                           (0, 255, 0) if whole else (0, 0, 255), 2)
        got = coverage(points, left.shape)
        grid = None
        if disp is None:
            status = "no ball / stereo reject"
        else:
            db = depth_bin(disp)
            grid = {(c, r): got[(db, c, r)] for c in range(3) for r in range(3)}
            status = (f"ball=({tl.x},{tl.y}) disp={disp} [{DEPTH_NAMES[db]}] - SPACE to centre"
                      if whole else "ball touches the image edge - move it inward")
        k = show(vis, [f"points={len(points)}  cells={len(got)}/27   U = undo last point",
                       status],
                 (0, 255, 255) if (disp is None or whole) else (0, 0, 255), grid)
        if k == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k == ord('u') and points:
            gone = points.pop()
            json.dump(points, open(PTS, "w"), indent=1)
            img = f"{OUT}/{len(points) + 1:02d}.png"
            if os.path.exists(img):
                os.remove(img)
            print(f"undo: removed #{len(points) + 1:02d} ({gone['x']},{gone['y']}) "
                  f"disp={gone['disp']}", flush=True)
            coverage_report(points, left.shape)
            continue
        if k != 32 or disp is None or not whole:
            continue

        if not AUTO:
            rec, p = centre_assisted(cap, dl, dr, tl, disp)
            if rec is None:
                print("  (cancelled - not recorded)", flush=True)
                continue
            points.append(rec)
            json.dump(points, open(PTS, "w"), indent=1)
            cv2.imwrite(f"{OUT}/{len(points):02d}.png", p)
            print(f"#{len(points):02d} ball=({rec['x']},{rec['y']}) disp={rec['disp']} "
                  f"[{DEPTH_NAMES[depth_bin(rec['disp'])]}] -> pan={rec['pan']:.0f} "
                  f"tilt={rec['tilt']:.0f}  (confirmed by hand)", flush=True)
            coverage_report(points, left.shape)
            continue

        # ---- converge the dot onto the ball (--auto) -----------------------
        pan, tilt = aim(tl.x, tl.y, disp)
        aborted, converged, err = False, False, None
        for it in range(MAX_IT):
            pan = min(max(pan, PAN_LIM[0]), PAN_LIM[1])
            tilt = min(max(tilt, TILT_LIM[0]), TILT_LIM[1])
            cmd(ser, pan, tilt, 1)
            t0 = time.time()
            while time.time() - t0 < SETTLE_S:
                cap.read()
            left, right, b, br = grab(cap, dl, dr)
            if left is None:
                continue
            ball = b or tl
            # refresh the depth only from a pair where both eyes see it whole
            if b and br and DISP_MIN <= (b.x - br.x) <= DISP_MAX \
                    and not stereo.touches_edge(b, left.shape) \
                    and not stereo.touches_edge(br, right.shape):
                disp = b.x - br.x
            click["pt"] = None
            dot = find_dot(left, (ball.x, ball.y, ball.radius))

            p = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
            cv2.circle(p, (ball.x, ball.y), max(ball.radius, 6), (0, 255, 0), 2)
            if dot:
                cv2.drawMarker(p, dot, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
                err = (ball.x - dot[0], ball.y - dot[1])
            k = show(p, [f"iter {it + 1}/{MAX_IT}  pan={pan:.1f} tilt={tilt:.1f}"
                         + (f"  err=({err[0]:+d},{err[1]:+d})" if err else "  NO DOT"),
                         "click = mark the dot | X = abort"],
                     (0, 255, 0) if dot else (0, 165, 255))
            if k == ord('x'):
                aborted = True
                break
            if click["pt"]:
                dot = click["pt"]
                err = (ball.x - dot[0], ball.y - dot[1])
            if dot is None:
                continue
            if abs(err[0]) <= TOL_PX and abs(err[1]) <= TOL_PX:
                converged = True
                break
            dpan, dtilt = jacobian(ball.x, ball.y)
            pan += GAIN * err[0] * dpan
            tilt += GAIN * err[1] * dtilt

        cmd(ser, pan, tilt, 0)
        if aborted or not converged:
            print(f"  ({'aborted' if aborted else 'did NOT converge'} — "
                  f"not recorded)", flush=True)
            continue
        if b is None or stereo.touches_edge(b, left.shape):
            print("  (ball not seen whole at the end — touching the image edge "
                  "or lost — not recorded; move it inward)", flush=True)
            continue

        rec = dict(x=ball.x, y=ball.y, disp=disp, pan=float(pan), tilt=float(tilt),
                   radius=ball.radius)
        points.append(rec)
        json.dump(points, open(PTS, "w"), indent=1)
        cv2.imwrite(f"{OUT}/{len(points):02d}.png", p)
        print(f"#{len(points):02d} ball=({ball.x},{ball.y}) disp={disp} "
              f"[{DEPTH_NAMES[depth_bin(disp)]}] -> pan={pan:.1f} tilt={tilt:.1f}  "
              f"(converged in {it + 1} steps)", flush=True)
        coverage_report(points, left.shape)

    cmd(ser, 90, 25, 0)
    cap.release()
    link["ser"].close()
    cv2.destroyAllWindows()
    print(f"saved {len(points)} points to {PTS}", flush=True)
    if points and shape is not None:
        coverage_report(points, shape)


if __name__ == "__main__":
    main()
