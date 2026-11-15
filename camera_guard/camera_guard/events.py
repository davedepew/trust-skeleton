"""Turns noisy per-frame detections into discrete events (start / end) with debouncing."""
from __future__ import annotations

from dataclasses import dataclass, field

from .detectors import Detection


@dataclass
class Event:
    kind: str                 # "motion" | "person"
    started: float
    ended: float | None = None
    peak_boxes: int = 0
    snapshot: str | None = None

    @property
    def duration(self) -> float:
        return (self.ended or self.started) - self.started


@dataclass
class EventTracker:
    """An event opens after `min_frames` consecutive hits and closes after `clear_seconds` of nothing.

    `update` returns (opened, closed) event lists for this frame.
    """
    clear_seconds: float = 5.0
    min_frames: int = 3
    _open: dict[str, Event] = field(default_factory=dict)
    _streak: dict[str, int] = field(default_factory=dict)
    _last_seen: dict[str, float] = field(default_factory=dict)

    def update(self, det: Detection, now: float) -> tuple[list[Event], list[Event]]:
        opened, closed = [], []
        hits = {"motion": len(det.motion_boxes), "person": len(det.person_boxes)}
        for kind, n in hits.items():
            if n:
                self._streak[kind] = self._streak.get(kind, 0) + 1
                self._last_seen[kind] = now
                ev = self._open.get(kind)
                if ev:
                    ev.peak_boxes = max(ev.peak_boxes, n)
                elif self._streak[kind] >= self.min_frames:
                    ev = self._open[kind] = Event(kind, now, peak_boxes=n)
                    opened.append(ev)
            else:
                self._streak[kind] = 0
                ev = self._open.get(kind)
                if ev and now - self._last_seen.get(kind, now) >= self.clear_seconds:
                    ev.ended = self._last_seen[kind]
                    closed.append(self._open.pop(kind))
        return opened, closed

    def flush(self, now: float) -> list[Event]:
        out = []
        for kind, ev in list(self._open.items()):
            ev.ended = self._last_seen.get(kind, now)
            out.append(self._open.pop(kind))
        return out
