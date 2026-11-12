r"""
Simple enlarged LIVE view of the stereo camera (both eyes), with the detected
bright target marked in each eye and the live disparity. No motors involved —
just for seeing what the camera sees. Close the window to quit.
    ..\.venv\Scripts\python.exe stereo_view.py [cam] [scale]
"""
import sys
import cv2
import stereo
import detector

CAM   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
WIN   = "AeroEye-3D stereo view"


def main():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    cv2.namedWindow(WIN)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        left, right = stereo.split_lr(frame)
        tl, tr = detector.detect(left), detector.detect(right)
        vis = cv2.cvtColor(cv2.hconcat([left, right]), cv2.COLOR_GRAY2BGR)
        w = left.shape[1]
        if tl:
            cv2.drawMarker(vis, (tl.x, tl.y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if tr:
            cv2.drawMarker(vis, (w + tr.x, tr.y), (0, 128, 255), cv2.MARKER_CROSS, 20, 2)
        disp = (tl.x - tr.x) if (tl and tr) else None
        cv2.line(vis, (w, 0), (w, left.shape[0]), (80, 80, 80), 1)
        cv2.putText(vis, f"LEFT eye        RIGHT eye     disparity={disp}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        big = cv2.resize(vis, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(WIN, big)
        cv2.waitKey(30)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
