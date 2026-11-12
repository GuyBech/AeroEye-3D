r"""
AeroEye-3D — motion detector (SWAPPABLE, for tracking a passive moving object).

Same Target interface as detector.py, but finds the largest MOVING region via
background subtraction instead of the brightest spot. A static scene produces
no motion -> detect() returns None -> the laser stays off. When an object
enters and moves, it is detected and tracked.

Stateful (keeps a background model), so make ONE instance per camera eye:
    left_md, right_md = MotionDetector(), MotionDetector()
    tl = left_md.detect(left_gray);  tr = right_md.detect(right_gray)
"""
import cv2
import numpy as np

from detector import Target

MIN_AREA = 60          # ignore motion blobs smaller than this (px^2) — noise
HISTORY  = 200         # frames of background the model remembers
VAR_THRESH = 30        # MOG2 sensitivity (higher = less sensitive)


class MotionDetector:
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=HISTORY, varThreshold=VAR_THRESH, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, gray):
        fg = self.bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.dilate(fg, self.kernel, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            return None
        (x, y), radius = cv2.minEnclosingCircle(c)
        return Target(int(x), int(y), max(int(radius), 4), area)

    def mask(self, gray):
        return self.bg.apply(gray, learningRate=0)   # for preview, no model update
