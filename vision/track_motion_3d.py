r"""
AeroEye-3D — THE TRACKER. Depth-aware real-time aiming at a moving target.

Per frame: detect the target in both eyes -> stereo disparity gives depth ->
calib3d.json maps (x, y, disparity) to servo angles -> aim_correction.json
applies the measured residual -> smooth, predict slightly ahead, send over UART.

  * COAST: when the target briefly vanishes (a pendulum is momentarily still at
    each swing extreme) the laser holds the last aim for COAST_S rather than
    cutting out and re-acquiring.
  * The laser is only ever on while locked or coasting, never when no target.

Detector is selectable; the DEFAULT is the one that measured best on this rig.
The others are kept because each failed in an instructive way (see the module
docstring of detector_dark.py for the measurements):

    (default)  detector_dark   dark ball on a light wall — local contrast
    --ref      detector_ref    difference against an empty-scene reference
    --shape    detector_shape  MOG2 motion + roundness
    --motion   detector_motion MOG2 motion, any moving object
    --flash    detector        brightest spot, for a flashlight beacon

    ..\.venv\Scripts\python.exe track_motion_3d.py [COM4] [cam] [--flag]
Keys: r re-captures the reference (--ref only) | q or close the window to quit.
"""
import os
import sys
import time
import json
from collections import deque

import cv2
import numpy as np
import serial

import stereo
import radar as radar_mod

if "--ref" in sys.argv:
    from detector_ref import RefDetector as Det      # round object vs empty-scene reference
elif "--shape" in sys.argv:
    from detector_shape import BallDetector as Det   # round object only (tennis ball)
elif "--flash" in sys.argv:
    import detector as _bright                        # brightest spot (flashlight)
    class Det:
        """Brightest-spot detector + saturation gate: a real beacon aimed at the
        camera SATURATES the sensor (255); our own laser dot reflected off a
        wall peaks ~220. Without this gate the tracker locks onto its own dot
        and keeps the laser on forever."""
        SAT = 245
        def detect(self, gray):
            t = _bright.detect(gray)
            if t is None:
                return None
            y0, y1 = max(0, t.y - 2), t.y + 3
            x0, x1 = max(0, t.x - 2), t.x + 3
            if gray[y0:y1, x0:x1].max() < self.SAT:
                return None                            # not a beacon — ignore
            return t
elif "--motion" in sys.argv:
    from detector_motion import MotionDetector as Det  # any moving object
else:
    from detector_dark import DarkBallDetector as Det  # DEFAULT — best measured

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM  = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)
WIN  = "AeroEye-3D motion tracking 3D"
RADAR_WIN = "AeroEye-3D radar"

calib = json.load(open("calib3d.json"))
PAN_C, TILT_C = calib["pan"], calib["tilt"]
QUAD = calib.get("model") == "quad3d"

# calib3d.json covers disparity 2..57, so a floor of 8 silently threw away valid
# FAR targets and a ceiling of 200 let the quadratic extrapolate wildly beyond
# anything it was fitted on. Track the calibrated range, with a small margin.
DISP_MIN, DISP_MAX = 3, 70
ROW_TOL = 40
WARMUP  = 25          # frames to let the background model settle
COAST_S = 0.8         # keep aiming this long after motion is lost
SEND_DT = 0.03        # command rate (~33 Hz) — firmware acks immediately now
ALPHA   = 0.7         # aim smoothing (higher = snappier, lower = steadier)
LEAD_S  = 0.15        # velocity prediction: keep SMALL — CV velocity is noisy,
                      # a large lead amplifies that noise into overshoot.
                      # measure_latency.py (2026-10-17): dead time 62 ms, but a
                      # recorded swinging run (live_20261017_221331) showed the
                      # dot trailing the ball by ~150-200 ms — the servo's full
                      # response (20-deg settle 175 ms), not just its start.
                      # History: 0.05 estimate -> 0.06 dead time -> 0.15.
                      # 0.15 lifted live hits 29% -> 71% (live_20261017_223922).
LEAD_TILT_S = 0.30    # tilt gets its own lead: in that same run the dot still
                      # trailed by ~70 ms in pan but ~220 ms in tilt (the tilt
                      # servo lifts the laser against gravity), and half the
                      # remaining misses were the dot BELOW a rising ball.
LEAD_FAST_S = 0.15    # pan lead for a FAST ball — OFF (= LEAD_S). In the
                      # examiner-style run (live_20261017_225129) the pan lag
                      # left over after 0.15 grew with speed (66 ms at 80-150
                      # px/s, 84 ms at 150-250, 133 ms above), so 0.20 was tried
                      # when fast (live_20261017_230456): the lag shrank but the
                      # hits did not rise (80-150 px/s 67% -> 65%, 150-250
                      # 39% -> 26%, more overshoot) — a bigger lead also
                      # amplifies velocity noise. Kept for A/B: --lead-fast=0.20
FAST_RAMP = (15.0, 35.0)  # deg/s of pan: below -> LEAD_S, above -> LEAD_FAST_S,
                      # linear in between (a hard switch would jump the aim).
                      # 1 px/s of the ball ~ 0.22 deg/s, so ~70..160 px/s.
                      # The lead also shrinks by itself at each swing's end,
                      # where the ball slows down and overshoot would show.
# A/B overrides without editing:  --lead=0.15  --lead-tilt=0.15  --lead-fast=0.20
def _arg(name, default):
    return next((float(a.split("=", 1)[1]) for a in sys.argv
                 if a.startswith(name + "=")), default)
LEAD_S, LEAD_TILT_S = _arg("--lead", LEAD_S), _arg("--lead-tilt", LEAD_TILT_S)
LEAD_FAST_S = _arg("--lead-fast", LEAD_FAST_S)


def pan_lead(v):
    """Look-ahead for a pan velocity v (deg/s): LEAD_S when slow, rising
    linearly to LEAD_FAST_S across FAST_RAMP."""
    lo, hi = FAST_RAMP
    a = min(1.0, max(0.0, (abs(v) - lo) / (hi - lo)))
    return LEAD_S + a * (LEAD_FAST_S - LEAD_S)
HOLD_S  = 0.5         # how long a trusted disparity may be held (StereoLock)
DISP_N  = 3           # trusted disparity = median of the last DISP_N
if "--oldstereo" in sys.argv:   # the previous behaviour exactly, for A/B tests:
    HOLD_S, DISP_N = 0.0, 1     # nothing held, nothing filtered -> raw pair only


try:
    CORR = json.load(open("aim_correction.json"))
    if CORR.get("space") == "angle":
        print(f"aim correction: angle-space {CORR['model']} from {CORR['n']} "
              f"ball points (pan{CORR['pan'][-1]:+.2f} tilt{CORR['tilt'][-1]:+.2f} deg)")
    else:
        print(f"aim correction: pixel-space {CORR['model']} from {CORR['n']} "
              f"samples (x{CORR['x'][-1]:+.1f} y{CORR['y'][-1]:+.1f} px)")
except FileNotFoundError:
    CORR = None


def _corr_terms(x, y, disp, model):
    if model == "constant":
        return [1.0]
    if model == "linear":
        return [x, y, disp, 1.0]
    return [x, y, disp, x * y, x * x, y * y, 1.0]


def aim(x, y, disp):
    """Pixel + disparity -> servo angles, with the measured residual applied.

    Two correction spaces are supported. ANGLE-space comes from calib_ball.py,
    which auto-centres the laser on the ball and so measures the angle error
    directly. PIXEL-space is the older form: commanding aim(P) lands the dot at
    P - err(P), so aiming at ball + err(ball) lands it on the ball."""
    if CORR is not None and CORR.get("space") != "angle":
        f0 = _corr_terms(x, y, disp, CORR["model"])
        x += sum(c * v for c, v in zip(CORR["x"], f0))
        y += sum(c * v for c, v in zip(CORR["y"], f0))
    if QUAD:
        feats = [x, y, disp, x * y, x * x, y * y, 1.0]
    else:
        feats = [x, y, disp, 1.0]
    pan  = sum(c * f for c, f in zip(PAN_C, feats))
    tilt = sum(c * f for c, f in zip(TILT_C, feats))
    if CORR is not None and CORR.get("space") == "angle":
        f0 = _corr_terms(x, y, disp, CORR["model"])
        pan  += sum(c * v for c, v in zip(CORR["pan"], f0))
        tilt += sum(c * v for c, v in zip(CORR["tilt"], f0))
    return int(round(pan)), int(round(tilt))


class Recorder:
    """--record: keep every REC_EVERY-th left-eye frame plus one log line per
    kept frame, so a live session can be judged afterwards (live_report.py):
    where the laser dot actually was relative to the ball, still and moving.
    Off by default; the normal run (the .bat) never writes anything."""
    REC_EVERY = 2

    def __init__(self, root=os.path.join("debug_images", "live")):
        self.dir = root + "_" + time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.dir, exist_ok=True)
        self.log = open(os.path.join(self.dir, "log.jsonl"), "w", encoding="utf-8")
        # the settings this run used, so runs can be compared afterwards
        json.dump(dict(LEAD_S=LEAD_S, LEAD_TILT_S=LEAD_TILT_S, LEAD_FAST_S=LEAD_FAST_S,
                       FAST_RAMP=FAST_RAMP, ALPHA=ALPHA, argv=sys.argv[1:]),
                  open(os.path.join(self.dir, "settings.json"), "w"), indent=1)
        print(f"recording to {self.dir}  (lead pan {LEAD_S}->{LEAD_FAST_S} s "
              f"over {FAST_RAMP[0]:.0f}-{FAST_RAMP[1]:.0f} deg/s, tilt {LEAD_TILT_S} s)",
              flush=True)

    def add(self, i, t, left, tl, tr, locked, how, coasting, laser_on_at_capture, sent):
        if i % self.REC_EVERY:
            return
        cv2.imwrite(os.path.join(self.dir, f"{i:06d}.png"), left,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        box = lambda b: [b.x, b.y, b.radius] if b else None
        self.log.write(json.dumps(dict(
            i=i, t=round(t, 4), L=box(tl), R=box(tr), locked=bool(locked),
            how=how if locked else None, coasting=bool(coasting),
            laser=int(bool(laser_on_at_capture)), sent=list(sent))) + "\n")
        self.log.flush()

    def close(self):
        self.log.close()
        print(f"recording saved: {self.dir}", flush=True)


class StereoLock:
    """Where to aim (left-eye x, y) and at what depth, from one stereo pair.

    Depth is only TRUSTED when both eyes see the whole ball. A ball cut by the
    frame edge has its centre pulled inward onto the visible part, so the two
    eyes disagree and the disparity is wrong — 40% of the frames on a
    close-range recording, and every px of disparity is ~0.25 deg of pan.
    Depth changes slowly, so instead the last trusted disparity is held for
    HOLD_S, and the aim point comes from the best view available:

        stereo   both eyes whole     fresh disparity (median of DISP_N)
        hold     left eye whole      left centre + held disparity
        right    only right whole    right centre shifted back by held disparity
        cut      no whole view       left centre, i.e. on the visible part

    With nothing trusted yet it uses the raw pair ("raw"), which is exactly
    what the tracker did before — so it never locks less often than that."""

    def __init__(self):
        self.disps = deque(maxlen=DISP_N)
        self.row_off = deque(maxlen=15)     # right-eye row minus left-eye row
        self.t = 0.0

    def update(self, tl, tr, shape, now):
        whole_l = tl is not None and not stereo.touches_edge(tl, shape)
        whole_r = tr is not None and not stereo.touches_edge(tr, shape)
        raw = None
        if tl and tr:
            d = tl.x - tr.x
            if DISP_MIN <= d <= DISP_MAX and abs(tl.y - tr.y) <= ROW_TOL:
                raw = d
        if whole_l and whole_r and raw is not None:
            self.disps.append(raw)
            self.row_off.append(tr.y - tl.y)
            self.t = now
            return tl.x, tl.y, float(np.median(self.disps)), "stereo"

        held = (float(np.median(self.disps))
                if self.disps and now - self.t < HOLD_S else None)
        if held is None and raw is None:
            return None
        d = held if held is not None else raw
        if whole_l:
            return tl.x, tl.y, d, ("hold" if held is not None else "raw")
        if whole_r and held is not None:
            x = tr.x + d
            if x < shape[1]:                # stay inside the calibrated image
                off = float(np.median(self.row_off)) if self.row_off else 0.0
                return int(round(x)), int(round(tr.y - off)), d, "right"
        if tl:
            return tl.x, tl.y, d, ("cut" if held is not None else "raw")
        return None


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5); ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    cv2.namedWindow(WIN)
    left_md, right_md = Det(), Det()
    # Radar console — the camera window shows pixels, this shows where the target
    # physically is, in cm, from the stereo pair. Keys: f full screen, m mute.
    show_radar = "--noradar" not in sys.argv
    rdr = radar_mod.Radar(port=PORT, sound="--nosound" not in sys.argv) if show_radar else None
    if show_radar:
        rdr.open_window(RADAR_WIN, fullscreen="--fullscreen" in sys.argv)
    # RefDetector must first photograph the EMPTY scene; the others are stateless
    # at startup. hasattr keeps every detector interchangeable behind detect().
    needs_ref = hasattr(left_md, "accumulate_reference")
    if needs_ref:
        print("CAPTURING REFERENCE — keep the scene EMPTY (no ball, no people)."
              "  Press r to recapture at any time.", flush=True)

    frame_i = 0
    last_lock = None            # smoothed (pan, tilt)
    sm = None                  # smoothing state [pan, tilt]
    vel = [0.0, 0.0]           # deg/s estimate for prediction
    prev_sm_t = None
    last_lock_t = 0.0
    last_sent, last_t = (None, None, None), 0.0
    stereo_lock = StereoLock()
    rec = Recorder() if "--record" in sys.argv else None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_i += 1
        left, right = stereo.split_lr(frame)

        if needs_ref and not left_md.ready:
            left_md.accumulate_reference(left)
            if right_md.accumulate_reference(right):
                print(f"reference captured at frame {frame_i} — tracking live", flush=True)
            vis0 = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
            cv2.putText(vis0, "CAPTURING REFERENCE - keep scene empty",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow(WIN, cv2.resize(vis0, None, fx=1.6, fy=1.6,
                                       interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)
            continue

        tl = left_md.detect(left)
        tr = right_md.detect(right)
        vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
        w = left.shape[1]
        if tl: cv2.circle(vis, (tl.x, tl.y), max(tl.radius, 6), (0, 255, 0), 2)
        if tr: cv2.circle(vis, (w + tr.x, tr.y), max(tr.radius, 6), (0, 128, 255), 2)

        now = time.time()
        locked = False
        disp = None
        hit = stereo_lock.update(tl, tr, left.shape, now) if frame_i > WARMUP else None
        if hit:
            ax, ay, disp, how = hit
            cv2.drawMarker(vis, (ax, ay), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
            raw = aim(ax, ay, disp)
            if sm is None:
                sm = [float(raw[0]), float(raw[1])]
            else:
                prev = list(sm)
                sm[0] = ALPHA * raw[0] + (1 - ALPHA) * sm[0]
                sm[1] = ALPHA * raw[1] + (1 - ALPHA) * sm[1]
                if prev_sm_t is not None and (now - prev_sm_t) > 0.015:
                    dt = now - prev_sm_t
                    # blend velocity estimate; predict ahead to cancel lag
                    # (dt guard: a tiny dt would explode the estimate)
                    vel[0] = 0.6 * ((sm[0] - prev[0]) / dt) + 0.4 * vel[0]
                    vel[1] = 0.6 * ((sm[1] - prev[1]) / dt) + 0.4 * vel[1]
            prev_sm_t = now
            last_lock = (int(round(sm[0] + vel[0] * pan_lead(vel[0]))),
                         int(round(sm[1] + vel[1] * LEAD_TILT_S)))
            last_lock_t = now
            locked = True
            cv2.putText(vis, f"LOCK[{how}] disp={disp:.0f} -> pan={last_lock[0]} tilt={last_lock[1]}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if now - getattr(main, "_log_t", 0.0) > 0.7:
            main._log_t = now
            d = (tl.x - tr.x) if (tl and tr) else None
            rej = getattr(left_md, "reject", None)
            print(f"[{frame_i:5d}] L={'(%d,%d)a%d' % (tl.x, tl.y, tl.area) if tl else '-':>18}"
                  f" R={'(%d,%d)a%d' % (tr.x, tr.y, tr.area) if tr else '-':>18}"
                  f" disp={d if d is not None else '-'}"
                  f" {'LOCK[%s]->%s' % (how, last_lock) if locked else 'no-lock'}"
                  f"{'  rejL: ' + rej if (rej and not tl) else ''}", flush=True)

        coasting = (not locked) and last_lock and (now - last_lock_t < COAST_S)
        if not locked:
            msg = "COAST (holding)" if coasting else ("warming up" if frame_i <= WARMUP else "no motion (laser off)")
            cv2.putText(vis, msg, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        laser_at_capture = last_sent[2] == 1      # state while this frame was exposed
        if now - last_t > SEND_DT:
            if locked or coasting:
                cmd = (last_lock[0], last_lock[1], 1)
            else:
                cmd = (last_sent[0], last_sent[1], 0) if last_sent[0] else (90, 20, 0)
            if cmd != last_sent:
                ser.write(f"{cmd[0]},{cmd[1]},{cmd[2]}\n".encode()); ser.readline()
                last_sent = cmd
            last_t = now

        if rec is not None:
            rec.add(frame_i, now, left, tl, tr, locked, how if locked else None,
                    coasting, laser_at_capture, last_sent)

        cv2.imshow(WIN, cv2.resize(vis, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST))
        if rdr is not None:
            # display only: nothing here feeds back into aiming, and a radar
            # failure must not take the tracker down in the middle of a demo
            try:
                aimed = last_lock if (locked or coasting) else (None, None)
                rdr.update(x=ax if locked else None, y=ay if locked else None, disp=disp,
                           pan=aimed[0], tilt=aimed[1], locked=locked,
                           coasting=bool(coasting), laser=last_sent[2] == 1)
                cv2.imshow(RADAR_WIN, rdr.canvas())
            except Exception as e:
                print(f"radar display stopped ({e}); tracking continues", flush=True)
                rdr = None
        # 25 ms, not 30: the radar console costs about 5 ms a frame to draw, and
        # the wait gives that back so the loop keeps the rate LEAD_S was tuned at
        key = cv2.waitKey(25) & 0xFF
        if rdr is not None:
            rdr.key(key)
        if key == ord('r') and needs_ref:
            left_md.set_reference(left); right_md.set_reference(right)
            print(f"reference RE-captured at frame {frame_i}", flush=True)
        if key == ord('q'):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    ser.write(b"90,20,0\n")
    cap.release(); cv2.destroyAllWindows(); ser.close()
    if rec is not None:
        rec.close()


if __name__ == "__main__":
    main()
