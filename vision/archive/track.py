r"""
AeroEye-3D — Phase 5: live tracking loop.

Captures from the camera, converts each frame to grayscale (matching the real
monochrome camera), asks detector.detect() for the target, and draws a
crosshair + the normalized error from frame center. That error (ex, ey in
[-1..1]) is exactly what will drive the pan/tilt motors in Phase 6.

Modes:
  live (default):  ..\.venv\Scripts\python.exe track.py [index]
      Opens a window. Keys: q=quit, m=toggle mask view.
  snapshot:        ..\.venv\Scripts\python.exe track.py [index] --snapshot
      Grabs one annotated frame to track_annotated.png and exits (no window).
"""
import sys
import cv2

import detector
import stereo

CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
SNAPSHOT = "--snapshot" in sys.argv
STEREO = "--stereo" in sys.argv      # split the side-by-side frame, use the left eye


def to_gray(frame):
    """Return the grayscale image we run detection on (left eye if stereo)."""
    if STEREO:
        left, _ = stereo.split_lr(frame)
        return left
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def annotate(frame, gray, target):
    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2
    # frame center reference
    cv2.drawMarker(frame, (cx, cy), (180, 180, 180), cv2.MARKER_CROSS, 16, 1)
    if target is None:
        cv2.putText(frame, "no target", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame, None
    # lock on the target
    cv2.circle(frame, (target.x, target.y), max(target.radius, 6), (0, 255, 0), 2)
    cv2.drawMarker(frame, (target.x, target.y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.line(frame, (cx, cy), (target.x, target.y), (0, 255, 0), 1)
    # normalized error from center: +x = right, +y = down
    ex = (target.x - cx) / (w / 2)
    ey = (target.y - cy) / (h / 2)
    cv2.putText(frame, f"x={target.x} y={target.y}  ex={ex:+.2f} ey={ey:+.2f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, (ex, ey)


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {CAM_INDEX}")
        sys.exit(1)

    if SNAPSHOT:
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("ERROR: failed to read a frame")
            sys.exit(1)
        gray = to_gray(frame)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        target = detector.detect(gray)
        out, err = annotate(disp, gray, target)
        cv2.imwrite("track_annotated.png", out)
        print(f"target={target}  error={err}")
        print("Saved track_annotated.png")
        return

    print("Live tracking. Keys: q=quit, m=toggle mask.  Camera index", CAM_INDEX)
    show_mask = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = to_gray(frame)
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        target = detector.detect(gray)
        out, _ = annotate(disp, gray, target)
        if show_mask:
            cv2.imshow("AeroEye-3D tracking", detector.make_mask(gray))
        else:
            cv2.imshow("AeroEye-3D tracking", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            show_mask = not show_mask
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
