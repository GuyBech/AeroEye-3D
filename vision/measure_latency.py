r"""
AeroEye-3D — end-to-end latency measurement.

Fast-swing tracking lags, and LEAD_S (the velocity look-ahead in
track_motion_3d.py) was chosen conservatively rather than from data. This
measures the real number so it can be set from evidence.

Method: point the laser at a wall, note where the dot sits, then command a step
in pan and watch, frame by frame, when the dot actually moves. The interval
between issuing the command and seeing the first movement is the system's DEAD
TIME — it contains everything in the loop:

    serial transmit + firmware parse + servo start-up + mechanical inertia
    + camera exposure + USB transfer + driver buffering + our own read()

Two numbers come out, and they answer different questions:

  dead time    command -> first observed movement. Pure transport delay. This
               is what a velocity look-ahead must cancel, because during it the
               turret is still pointing at where the target used to be.
  settle time  command -> dot stops moving. Adds the mechanical travel, which
               depends on step size and so matters less for the small
               continuous corrections tracking actually makes.

Frames are read exactly as the tracker reads them, buffering included, so the
result is the latency the tracker really experiences rather than a best case.

The laser must be visible on a surface — aim at a wall about 0.5-1 m away, with
nothing brighter in view.

    ..\.venv\Scripts\python.exe measure_latency.py [COM4] [cam]
"""
import statistics
import sys
import time

import cv2
import numpy as np
import serial

import stereo

PORT = next((a for a in sys.argv[1:] if a.upper().startswith("COM")), "COM4")
CAM = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1)

BASE_PAN, BASE_TILT = 65, 32   # centre of the camera view (ball calibration: pan 65 at x~160),
                               # so a +/-20 deg step keeps the dot in the frame both ways
STEP_DEG = 20              # big enough to see clearly, small enough to stay in view
TRIALS = 8
MOVE_PX = 6                # displacement that counts as "it started moving"
STILL_PX = 2               # per-frame change below this counts as settled
STILL_N = 3
TIMEOUT_S = 1.5
SETTLE_BEFORE = 1.2        # let everything come to rest before each trial


def find_dot(gray):
    """Brightest spot — the laser on the wall."""
    b = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mv, _, loc = cv2.minMaxLoc(b)
    return loc, mv


def cmd(ser, p, t, l):
    ser.write(f"{p},{t},{l}\n".encode())
    ser.readline()


def baseline(cap, n=8):
    pts = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            continue
        left, _ = stereo.split_lr(f)
        loc, mv = find_dot(left)
        pts.append(loc)
    if not pts:
        return None
    return (float(np.median([p[0] for p in pts])),
            float(np.median([p[1] for p in pts])))


def one_trial(cap, ser, direction):
    """Returns (dead_time_s, settle_time_s, frame_interval_s) or None."""
    target = BASE_PAN + direction * STEP_DEG
    cmd(ser, BASE_PAN, BASE_TILT, 1)
    time.sleep(SETTLE_BEFORE)
    base = baseline(cap)
    if base is None:
        return None

    # drain whatever the driver already holds, so the first frame we time is
    # genuinely captured after the command went out
    for _ in range(2):
        cap.read()

    t_cmd = time.time()
    ser.write(f"{target},{BASE_TILT},1\n".encode())

    dead = None
    settle = None
    prev = None
    still = 0
    stamps = []
    while time.time() - t_cmd < TIMEOUT_S:
        ok, f = cap.read()
        t = time.time()
        if not ok or f is None:
            continue
        stamps.append(t)
        left, _ = stereo.split_lr(f)
        loc, mv = find_dot(left)
        d = math_hypot(loc[0] - base[0], loc[1] - base[1])
        if dead is None and d > MOVE_PX:
            dead = t - t_cmd
        if dead is not None:
            if prev is not None and math_hypot(loc[0] - prev[0], loc[1] - prev[1]) <= STILL_PX:
                still += 1
                if still >= STILL_N:
                    settle = t - t_cmd
                    break
            else:
                still = 0
        prev = loc
    ser.readline()
    if dead is None:
        return None
    dt = (statistics.median([b - a for a, b in zip(stamps, stamps[1:])])
          if len(stamps) > 2 else 0.0)
    return dead, settle, dt


def math_hypot(a, b):
    return (a * a + b * b) ** 0.5


def main():
    ser = serial.Serial(PORT, 115200, timeout=1)
    time.sleep(2.5)
    ser.reset_input_buffer()
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.2)

    ok, f = cap.read()
    if not ok:
        print("CAMERA FAILED — is another process holding it?")
        return
    left, _ = stereo.split_lr(f)
    _, mv = find_dot(left)
    print(f"brightest pixel in view: {mv:.0f}")
    if mv < 150:
        print("WARNING: no bright dot detected. Point the laser at a wall "
              "0.5-1 m away, with nothing brighter in view.")
    print(f"\nrunning {TRIALS} trials, +/-{STEP_DEG} deg steps...\n", flush=True)

    deads, settles, dts = [], [], []
    for i in range(TRIALS):
        r = one_trial(cap, ser, 1 if i % 2 == 0 else -1)
        if r is None:
            print(f"  trial {i + 1}: no movement detected — skipped", flush=True)
            continue
        dead, settle, dt = r
        deads.append(dead)
        if settle:
            settles.append(settle)
        if dt:
            dts.append(dt)
        print(f"  trial {i + 1}: dead={dead * 1000:5.0f} ms   "
              f"settle={settle * 1000:5.0f} ms" if settle else
              f"  trial {i + 1}: dead={dead * 1000:5.0f} ms   settle=n/a",
              flush=True)

    cmd(ser, BASE_PAN, BASE_TILT, 0)
    cap.release()
    ser.close()

    if not deads:
        print("\nno usable trials — is the laser dot visible to the camera?")
        return

    md = statistics.median(deads)
    print(f"\n{'=' * 52}")
    print(f"DEAD TIME    median {md * 1000:.0f} ms   "
          f"(min {min(deads) * 1000:.0f}, max {max(deads) * 1000:.0f}, n={len(deads)})")
    if settles:
        ms = statistics.median(settles)
        print(f"SETTLE TIME  median {ms * 1000:.0f} ms   "
              f"for a {STEP_DEG} deg step")
    if dts:
        fi = statistics.median(dts)
        print(f"FRAME PERIOD median {fi * 1000:.0f} ms  "
              f"({1 / fi:.0f} fps) — the measurement resolution")
    print(f"{'=' * 52}")
    print(f"\nSuggested LEAD_S in track_motion_3d.py: {md:.3f}")
    print("  This is the dead time: during it the turret is still aimed where")
    print("  the target used to be, so the look-ahead should cancel exactly it.")
    print("  Raise it only if fast tracking still lags; too large a lead")
    print("  amplifies the noise in the velocity estimate into overshoot.")
    if dts and md < statistics.median(dts):
        print("\n  NOTE: dead time is below one frame period, so this is an")
        print("  upper bound limited by the frame rate, not a precise value.")


if __name__ == "__main__":
    main()
