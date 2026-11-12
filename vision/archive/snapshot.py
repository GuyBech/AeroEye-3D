r"""
AeroEye-3D — Phase 5, step 2: grab one frame and save it to snapshot.png.
Lets us verify the camera works and confirm which camera it is.

Run:  ..\.venv\Scripts\python.exe snapshot.py [index]
"""
import sys
import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = "snapshot.png"

cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"ERROR: could not open camera index {index}")
    sys.exit(1)

# Discard a few warm-up frames (first frames are often dark/garbage).
for _ in range(5):
    cap.read()

ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print("ERROR: opened the camera but failed to read a frame")
    sys.exit(1)

cv2.imwrite(OUT, frame)
h, w = frame.shape[:2]
print(f"Saved {OUT}  ({w}x{h}) from camera index {index}")
