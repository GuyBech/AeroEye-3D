r"""
AeroEye-3D — radar replay. Runs the radar console from a recorded session, so it
can be rehearsed, or shown, with no camera, Arduino or laser connected.

It feeds the same radar.py the tracker uses, from a log.jsonl written by
track_motion_3d.py --record, so what appears here is what a live run looks like.
The log has about 10 lines a second; the ball is interpolated between them.

    ..\.venv\Scripts\python.exe radar_replay.py [run_folder] [--fullscreen] [--nosound]
Keys: f full screen | m mute | q or close the window to quit.
"""
import json
import os
import sys
import time

import cv2

import radar as radar_mod

DEFAULT_RUN = os.path.join("debug_images", "live_20261017_225129")   # still, swings, in and out of view
WIN = "AeroEye-3D radar"
ROW_OFFSET = radar_mod.ROW_OFFSET
DISP_MIN, DISP_MAX = 3, 70
MAX_INTERP_S = 0.3         # do not glide across a real gap in tracking


def _arg(name, default):
    return next((float(a.split("=", 1)[1]) for a in sys.argv
                 if a.startswith(name + "=")), default)


def load(run):
    """One sample per log line. The ball comes from whichever eye saw it, and
    disparity is held across frames that had only one eye, as the tracker does."""
    rows, last_d = [], None
    for ln in open(os.path.join(run, "log.jsonl"), encoding="utf-8"):
        o = json.loads(ln)
        L, R = o["L"], o["R"]
        if L and R and DISP_MIN <= L[0] - R[0] <= DISP_MAX:
            last_d = L[0] - R[0]
        x = y = None
        if o["locked"] and last_d is not None:
            if L:
                x, y = L[0], L[1]
            elif R:
                x, y = R[0] + last_d, R[1] - ROW_OFFSET
        rows.append(dict(t=o["t"], x=x, y=y, d=last_d, pan=o["sent"][0], tilt=o["sent"][1],
                         laser=o["sent"][2] == 1, coasting=o["coasting"]))
    return rows


def between(a, b, t):
    """The sample at time t, gliding from a to b when both saw the ball."""
    s = dict(a)
    if a["x"] is not None and b["x"] is not None and 0 < b["t"] - a["t"] < MAX_INTERP_S:
        w = (t - a["t"]) / (b["t"] - a["t"])
        for k in ("x", "y", "d", "pan", "tilt"):
            s[k] = a[k] + w * (b[k] - a[k])
    return s


def main():
    run = next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT_RUN)
    rows = load(run)
    seconds = _arg("--seconds", None)      # stop after this long (for testing)
    print(f"replaying {run}: {len(rows)} samples, "
          f"{rows[-1]['t'] - rows[0]['t']:.0f} s, looping", flush=True)

    rdr = radar_mod.Radar(port="REPLAY", sound="--nosound" not in sys.argv)
    rdr.open_window(WIN, fullscreen="--fullscreen" in sys.argv)
    start = t0 = time.time()
    i = 0
    while True:
        now = time.time()
        rec_t = rows[0]["t"] + (now - t0)
        while i + 2 < len(rows) and rows[i + 1]["t"] <= rec_t:
            i += 1
        if i + 2 >= len(rows):              # end of the recording: go around
            i, t0 = 0, now
            continue
        s = between(rows[i], rows[i + 1], rec_t)
        seen = s["x"] is not None
        rdr.update(x=s["x"], y=s["y"], disp=s["d"], pan=s["pan"], tilt=s["tilt"],
                   locked=seen, coasting=s["coasting"], laser=s["laser"])
        cv2.imshow(WIN, rdr.canvas())
        key = cv2.waitKey(25) & 0xFF
        rdr.key(key)
        if key == ord('q') or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if seconds is not None and now - start > seconds:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
