r"""
AeroEye-3D — stereo camera helper.

The USB camera is a dual-lens stereo unit that shows up as ONE UVC device
(index 1) delivering a 640x240 frame = two side-by-side monochrome eyes,
~320x240 each (left | right).

  - Phase 5 (2D detection): use one eye (left) -> target (x, y).
  - Phase 6 (depth): match the target in both eyes -> disparity -> depth Z.

split_lr() returns grayscale (left, right). Frames are already monochrome,
but we force single-channel so downstream code is always gray.
"""
import cv2


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def split_lr(frame):
    """Split a side-by-side stereo frame into grayscale (left, right) halves."""
    w = frame.shape[1]
    half = w // 2
    left = to_gray(frame[:, :half])
    right = to_gray(frame[:, half:])
    return left, right


def touches_edge(t, shape):
    """True when a detected ball runs off the image. Its centre is then pulled
    inward onto the visible part, so it is not a trustworthy stereo match."""
    H, W = shape[:2]
    return t.x - t.radius < 2 or t.y - t.radius < 2 or \
        t.x + t.radius > W - 3 or t.y + t.radius > H - 3
