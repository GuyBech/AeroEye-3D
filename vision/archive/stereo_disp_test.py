r"""
Diagnostic: detect the bright target in BOTH stereo eyes and report the
horizontal disparity (x_left - x_right). Disparity grows as the target gets
CLOSER, so it is our depth signal.

Hold a flashlight in view and run this at a few distances:
    ..\.venv\Scripts\python.exe stereo_disp_test.py [cam]
"""
import sys
import time
import cv2
import stereo
import detector

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def grab(cap, n=8):
    frame = None
    got = 0
    for _ in range(60):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f; got += 1
            if got >= n: break
        time.sleep(0.03)
    return frame


def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    time.sleep(1.0)
    frame = grab(cap)
    cap.release()
    if frame is None:
        print("ERROR: no frame"); return

    left, right = stereo.split_lr(frame)
    tl = detector.detect(left)
    tr = detector.detect(right)
    print("left  target:", tl)
    print("right target:", tr)
    if tl and tr:
        disp = tl.x - tr.x
        print(f"DISPARITY = {disp}  (x_left={tl.x}, x_right={tr.x})")
    else:
        print("target not found in both eyes — aim a bright light in view")

    # save a side-by-side annotated image
    dispimg = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
    if tl: cv2.drawMarker(dispimg, (tl.x, tl.y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    if tr: cv2.drawMarker(dispimg, (right.shape[1] + tr.x, tr.y), (0, 128, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite("disp_test.png", dispimg)
    print("Saved disp_test.png")


if __name__ == "__main__":
    main()
