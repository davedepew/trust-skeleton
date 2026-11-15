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


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, ratio, (pad_x, pad_y))."""
    h, w = frame.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(frame, (nw, nh)) if (nw, nh) != (w, h) else frame
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


def decode_yolo_person(out: np.ndarray, conf: float, iou: float, ratio: float,
                       pad: tuple[int, int], frame_shape: tuple[int, ...]) -> list[Box]:
    """Decode YOLOv8/YOLO11 ONNX output (1, 4+classes, N) into person boxes (COCO class 0).

    Pure function so it can be tested without a model file.
    """
    pred = np.squeeze(out, 0)
    if pred.shape[0] < pred.shape[1]:          # (features, N) -> (N, features)
        pred = pred.T
    scores = pred[:, 4]                         # class 0 = person
    keep = scores >= conf
    if not keep.any():
        return []
    pred, scores = pred[keep], scores[keep]
    h, w = frame_shape[:2]
    px, py = pad
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = np.clip((cx - bw / 2 - px) / ratio, 0, w - 1)
    y1 = np.clip((cy - bh / 2 - py) / ratio, 0, h - 1)
    x2 = np.clip((cx + bw / 2 - px) / ratio, 0, w - 1)
    y2 = np.clip((cy + bh / 2 - py) / ratio, 0, h - 1)
    rects = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), conf, iou)
    return [tuple(rects[i]) for i in np.ravel(idx) if rects[i][2] > 0 and rects[i][3] > 0]


class YoloPersonDetector:
    """Person detector using a YOLOv8 / YOLO11 model exported to ONNX (COCO classes; 'person' = 0).

    Handles seated, partly hidden and angled people far better than HOG. Needs
    `pip install onnxruntime` and a model file, e.g. exported with
    `yolo export model=yolov8n.pt format=onnx` (check the model's licence before redistributing).
    """

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.45,
                 zone: Zone | None = None):
        import onnxruntime as ort  # lazy: only needed when YOLO is used
        self.conf, self.iou = conf, iou
        self.zone = zone or Zone()
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 640

    def detect(self, frame: np.ndarray) -> list[Box]:
        m = self.zone.mask(frame.shape[:2])
        if m is not None:
            frame = cv2.bitwise_and(frame, frame, mask=m)
        img, ratio, pad = letterbox(frame, self.size)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: blob})[0]
        return decode_yolo_person(out, self.conf, self.iou, ratio, pad, frame.shape)
