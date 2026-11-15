"""camera_guard: person + motion detection for a camera watching an indoor space."""
from .detectors import MotionDetector, PersonDetector, YoloPersonDetector, Zone, Detection
from .events import EventTracker, Event
from .alerts import EmailAlerter
from .monitor import Monitor, Config

__all__ = ["EmailAlerter", "MotionDetector", "PersonDetector", "YoloPersonDetector", "YoloPersonDetector", "Zone", "Detection",
           "EventTracker", "Event", "Monitor", "Config"]
