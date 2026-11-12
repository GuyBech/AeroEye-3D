r"""
AeroEye-3D — Phase 5, step 1: probe which camera indices are available.
Tries indices 0..5 on the Windows DirectShow backend (fast, avoids the slow
MSMF startup) and reports resolution + FPS for each one that opens.

Run:  ..\.venv\Scripts\python.exe probe_cameras.py
"""
import cv2

MAX_INDEX = 6

def probe(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    info = None
    if ok and frame is not None:
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = (w, h, fps)
    cap.release()
    return info

def main():
    print("Probing camera indices 0..{} (DirectShow)...\n".format(MAX_INDEX - 1))
    found = []
    for i in range(MAX_INDEX):
        info = probe(i)
        if info:
            w, h, fps = info
            print(f"  [index {i}]  OPEN   {w}x{h} @ {fps:.0f} fps")
            found.append(i)
        else:
            print(f"  [index {i}]  --     (no camera)")
    print()
    if found:
        print(f"Available camera indices: {found}")
        print(f"Use index {found[0]} as the default for the next step.")
    else:
        print("No cameras found. Check the USB connection and Windows camera privacy settings.")

if __name__ == "__main__":
    main()
