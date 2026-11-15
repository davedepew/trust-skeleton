"""Frame-level detectors. Pure functions of frames: no I/O, easy to test."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Detection:
    motion_boxes: list[Box] = field(default_factory=list)
    person_boxes: list[Box] = field(default_factory=list)
    motion_fraction: float = 0.0  # share of the (zone) area that changed

    @property
    def motion(self) -> bool:
        return bool(self.motion_boxes)

    @property
    def person(self) -> bool:
        return bool(self.person_boxes)


class Zone:
    """Optional region of interest. Points are normalised (0..1) polygon vertices.

    Everything outside the polygon is ignored (windows, TVs, fans, a street view).
    """

    def __init__(self, points: Sequence[tuple[float, float]] | None = None):
        self.points = list(points) if points else None
        self._mask: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if not self.points:
            return None
        if self._mask is None or self._shape != shape:
            h, w = shape
            poly = np.array([[int(x * w), int(y * h)] for x, y in self.points], np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            self._mask, self._shape = m, shape
        return self._mask

    def apply(self, gray: np.ndarray) -> np.ndarray:
        m = self.mask(gray.shape[:2])
        return gray if m is None else cv2.bitwise_and(gray, gray, mask=m)


class MotionDetector:
    """MOG2 background subtraction. Adapts to slow lighting change; shadows are discarded."""

    def __init__(self, min_area: int = 800, history: int = 300, var_threshold: float = 32,
                 warmup_frames: int = 30, zone: Zone | None = None, width: int = 640):
        self.min_area = min_area
        self.warmup_frames = warmup_frames
        self.zone = zone or Zone()
        self.width = width
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self._seen = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def scale(self, frame: np.ndarray) -> float:
        return min(1.0, self.width / frame.shape[1])

    def detect(self, frame: np.ndarray) -> tuple[list[Box], float]:
        s = self.scale(frame)
        small = cv2.resize(frame, None, fx=s, fy=s) if s < 1 else frame
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        fg = self._bg.apply(gray)
        self._seen += 1
        if self._seen <= self.warmup_frames:  # let the background model settle
            return [], 0.0
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # drops shadows (127)
        m = self.zone.mask(fg.shape[:2])
        if m is not None:
            fg = cv2.bitwise_and(fg, m)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fg = cv2.dilate(fg, self._kernel, iterations=2)
        area_total = float(cv2.countNonZero(m)) if m is not None else float(fg.size)
        fraction = cv2.countNonZero(fg) / max(area_total, 1.0)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area * s * s
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((int(x / s), int(y / s), int(w / s), int(h / s)))
        return boxes, fraction


class PersonDetector:
    """OpenCV's built-in HOG + linear SVM pedestrian detector (no model download).

    Works best on upright, mostly-visible people ~100px+ tall. For seated/partial/
    overhead views swap in a YOLO/ONNX model by subclassing and overriding `detect`.
    """

    def __init__(self, width: int = 640, hit_threshold: float = 0.0,
                 min_weight: float = 0.6, zone: Zone | None = None):
        self.width = width
        self.hit_threshold = hit_threshold
        self.min_weight = min_weight
        self.zone = zone or Zone()
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Box]:
        s = min(1.0, self.width / frame.shape[1])
        small = cv2.resize(frame, None, fx=s, fy=s) if s < 1 else frame
        gray = self.zone.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        rects, weights = self._hog.detectMultiScale(
            gray, hitThreshold=self.hit_threshold, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return []
        keep = [i for i, w in enumerate(np.ravel(weights)) if w >= self.min_weight]
        rects = [list(map(int, rects[i])) for i in keep]
        idx = cv2.dnn.NMSBoxes(rects, [1.0] * len(rects), 0.0, 0.4) if rects else []
        return [tuple(int(v / s) for v in rects[i]) for i in np.ravel(idx)]
