r"""
AeroEye-3D — SHAPE detector: track a specific ROUND object (e.g. a tennis ball)
and ignore hands / arbitrary objects.

Motion (background subtraction) first isolates what is moving, then each blob
is filtered by SIZE and ROUNDNESS:
    circularity = 4*pi*area / perimeter^2     (perfect circle = 1.0)
    fill        = area / minEnclosingCircle area
Only a sufficiently round blob of the right size is returned.

Same Target interface as detector.py. Stateful -> one instance per eye:
    left_bd, right_bd = BallDetector(), BallDetector()

Tuning: raise MIN_CIRC / MIN_FILL to be stricter about roundness; adjust
MIN_AREA / MAX_AREA to the ball's size in pixels at your working distance.
"""
import math
import cv2

from detector import Target

MIN_AREA = 70
MAX_AREA = 12000
MIN_CIRC = 0.55        # 1.0 = perfect circle; hands/irregular shapes score lower
MIN_FILL = 0.55        # how completely the blob fills its bounding circle
HISTORY  = 200
VAR_THRESH = 30


class BallDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=HISTORY, varThreshold=VAR_THRESH, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, gray):
        fg = self.bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)
        fg = cv2.dilate(fg, self.kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            (x, y), r = cv2.minEnclosingCircle(c)
            fill = area / (math.pi * r * r) if r > 0 else 0
            if circ < MIN_CIRC or fill < MIN_FILL:
                continue
            score = circ * fill           # prefer the roundest, most filled blob
            if score > best_score:
                best, best_score = (x, y, r, area), score
        if best is None:
            return None
        x, y, r, area = best
        return Target(int(x), int(y), max(int(r), 4), area)
