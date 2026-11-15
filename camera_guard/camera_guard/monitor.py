"""Capture loop: read frames, detect, track events, save snapshots, write a JSONL log."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .detectors import Detection, MotionDetector, PersonDetector, YoloPersonDetector, Zone
from .events import Event, EventTracker

log = logging.getLogger("camera_guard")


@dataclass
class Config:
    source: str | int = 0                 # device index, RTSP/HTTP URL, or video file
    out_dir: Path = Path("camera_guard_events")
    min_area: int = 800                   # px^2 at full-res; raise to ignore small movers (pets, bugs)
    person_every: int = 5                 # run the (slower) person detector every N frames
    clear_seconds: float = 5.0
    min_frames: int = 3
    zone: list[tuple[float, float]] | None = None
    snapshot_cooldown: float = 10.0       # seconds between snapshots of the same kind
    max_snapshots: int = 2000             # oldest are deleted beyond this (disk safety)
    detect_people: bool = True
    yolo_model: str | None = None         # path to YOLOv8/YOLO11 .onnx; default is the built-in HOG detector
    yolo_conf: float = 0.4
    reconnect_seconds: float = 5.0
    on_event: Callable[[str, Event], None] | None = None  # hook: ("start"|"end", event)
    display: bool = False


def annotate(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    for x, y, w, h in det.motion_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
    for x, y, w, h in det.person_boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, "PERSON", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, time.strftime("%Y-%m-%d %H:%M:%S"), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


class Monitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        zone = Zone(cfg.zone)
        self.motion = MotionDetector(min_area=cfg.min_area, zone=zone)
        self.people = None
        if cfg.detect_people:
            self.people = (YoloPersonDetector(cfg.yolo_model, conf=cfg.yolo_conf, zone=zone)
                           if cfg.yolo_model else PersonDetector(zone=zone))
        self.tracker = EventTracker(cfg.clear_seconds, cfg.min_frames)
        self.snap_dir = cfg.out_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = cfg.out_dir / "events.jsonl"
        self._last_snap: dict[str, float] = {}
        self._n = 0
        self._person_boxes: list = []

    # -- per-frame ---------------------------------------------------------
    def process(self, frame: np.ndarray, now: float | None = None) -> Detection:
        now = time.time() if now is None else now
        self._n += 1
        boxes, frac = self.motion.detect(frame)
        # Only pay for the person detector when something is actually moving.
        if self.people and boxes and self._n % self.cfg.person_every == 0:
            self._person_boxes = self.people.detect(frame)
        elif not boxes:
            self._person_boxes = []
        det = Detection(boxes, list(self._person_boxes), frac)
        opened, closed = self.tracker.update(det, now)
        for ev in opened:
            ev.snapshot = self._snapshot(frame, det, ev.kind, now, force=True)
            self._emit("start", ev)
        for kind in {"motion", "person"}:  # periodic refresh while an event stays open
            if kind in self.tracker._open and getattr(det, kind):
                self._snapshot(frame, det, kind, now)
        for ev in closed:
            self._emit("end", ev)
        return det

    def _snapshot(self, frame, det, kind, now, force=False) -> str | None:
        if not force and now - self._last_snap.get(kind, 0) < self.cfg.snapshot_cooldown:
            return None
        self._last_snap[kind] = now
        name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{kind}.jpg"
        path = self.snap_dir / name
        cv2.imwrite(str(path), annotate(frame, det), [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._prune()
        return str(path)

    def _prune(self) -> None:
        files = sorted(self.snap_dir.glob("*.jpg"))
        for f in files[: max(0, len(files) - self.cfg.max_snapshots)]:
            f.unlink(missing_ok=True)

    def _emit(self, phase: str, ev: Event) -> None:
        rec = {"phase": phase, "kind": ev.kind,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ev.started if phase == "start" else ev.ended)),
               "duration_s": round(ev.duration, 1), "peak_boxes": ev.peak_boxes, "snapshot": ev.snapshot}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("%s %s", phase.upper(), rec)
        if self.cfg.on_event:
            try:
                self.cfg.on_event(phase, ev)
            except Exception:  # a broken hook must never stop the camera loop
                log.exception("on_event hook failed")

    # -- capture loop ------------------------------------------------------
    def frames(self) -> Iterator[np.ndarray]:
        is_file = isinstance(self.cfg.source, str) and Path(self.cfg.source).is_file()
        while True:
            cap = cv2.VideoCapture(self.cfg.source)
            if not cap.isOpened():
                if is_file:
                    raise FileNotFoundError(f"cannot open {self.cfg.source}")
                log.warning("cannot open %s; retrying in %ss", self.cfg.source, self.cfg.reconnect_seconds)
                time.sleep(self.cfg.reconnect_seconds)
                continue
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()
            if is_file:
                return
            log.warning("stream lost; reconnecting")
            time.sleep(self.cfg.reconnect_seconds)

    def run(self) -> None:
        try:
            for frame in self.frames():
                det = self.process(frame)
                if self.cfg.display:
                    cv2.imshow("camera_guard (q to quit)", annotate(frame, det))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            for ev in self.tracker.flush(time.time()):
                self._emit("end", ev)
            if self.cfg.display:
                cv2.destroyAllWindows()
