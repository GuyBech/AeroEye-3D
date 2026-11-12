r"""Diagnostic: find and mark the single brightest spot in the stereo left eye.
Tells us whether the laser dot registers as the brightest thing at all.
    ..\.venv\Scripts\python.exe find_brightest.py [cam_index]
"""
import sys
import time
import cv2
import stereo

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
time.sleep(1.0)
frame = None
for _ in range(30):                 # warm up / retry until a valid frame
    ok, f = cap.read()
    if ok and f is not None:
        frame = f
    time.sleep(0.05)
cap.release()
if frame is None:
    print("ERROR: no frame"); sys.exit(1)

left, _ = stereo.split_lr(frame)
blur = cv2.GaussianBlur(left, (9, 9), 0)
minv, maxv, minloc, maxloc = cv2.minMaxLoc(blur)
print(f"brightest value={maxv:.0f} at {maxloc}   (dimmest={minv:.0f})")

disp = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
cv2.circle(disp, maxloc, 18, (0, 0, 255), 2)
cv2.drawMarker(disp, maxloc, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
cv2.putText(disp, f"max={maxv:.0f} @ {maxloc}", (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite("brightest.png", disp)
print("Saved brightest.png")
