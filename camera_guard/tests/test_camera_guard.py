import json
import numpy as np
import cv2
import pytest

from camera_guard import Config, Monitor, MotionDetector, Zone, Detection, EventTracker


def room(w=640, h=480):
    f = np.full((h, w, 3), 90, np.uint8)
    cv2.rectangle(f, (50, 300), (250, 450), (60, 60, 120), -1)  # static furniture
    return f


def with_blob(x, w=80, h=200):
    f = room()
    cv2.rectangle(f, (x, 200), (x + w, 200 + h), (200, 180, 160), -1)
    return f


def warm(det, n=40):
    for _ in range(n):
        det.detect(room())


def test_static_scene_no_motion():
    d = MotionDetector(); warm(d)
    boxes, frac = d.detect(room())
    assert boxes == [] and frac == 0.0


def test_moving_object_detected():
    d = MotionDetector(); warm(d)
    boxes = []
    for x in (300, 330, 360):
        boxes, _ = d.detect(with_blob(x))
    assert boxes, "moving person-sized blob must register"


def test_small_noise_ignored():
    d = MotionDetector(min_area=800); warm(d)
    f = room(); cv2.circle(f, (500, 100), 5, (255, 255, 255), -1)
    assert d.detect(f)[0] == []


def test_zone_masks_outside_motion():
    d = MotionDetector(zone=Zone([(0, 0), (0.4, 0), (0.4, 1), (0, 1)])); warm(d)
    for x in (400, 430, 460):  # blob lives entirely in the ignored right side
        boxes, _ = d.detect(with_blob(x))
    assert boxes == []


def test_tracker_debounce_and_close():
    t = EventTracker(clear_seconds=2, min_frames=3)
    hit, miss = Detection([(0, 0, 9, 9)]), Detection()
    assert t.update(hit, 0)[0] == [] and t.update(hit, 1)[0] == []   # not yet
    opened, _ = t.update(hit, 2)
    assert [e.kind for e in opened] == ["motion"]
    assert t.update(miss, 3)[1] == []                                 # still within clear window
    _, closed = t.update(miss, 5)
    assert len(closed) == 1 and closed[0].ended == 2


def test_monitor_end_to_end_logs_and_snapshots(tmp_path):
    seen = []
    m = Monitor(Config(out_dir=tmp_path, detect_people=False, on_event=lambda p, e: seen.append(p)))
    t = 1000.0
    for _ in range(40):
        m.process(room(), t); t += 0.1
    for i in range(10):
        m.process(with_blob(200 + i * 25), t); t += 0.1
    for _ in range(80):
        m.process(room(), t); t += 0.1
    recs = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in recs][:2] == ["start", "end"]
    assert list((tmp_path / "snapshots").glob("*.jpg"))
    assert seen[:2] == ["start", "end"]


def test_hook_failure_does_not_crash(tmp_path):
    def boom(*_): raise RuntimeError("x")
    m = Monitor(Config(out_dir=tmp_path, detect_people=False, on_event=boom))
    t = 0.0
    for _ in range(40): m.process(room(), t); t += 0.1
    for i in range(6): m.process(with_blob(200 + i * 25), t); t += 0.1


def test_person_detector_runs_on_blank_frame():
    from camera_guard import PersonDetector
    assert PersonDetector().detect(room()) == []


# ---- YOLO (tested without a model file: pure pre/post-processing + a stub session) ----
from camera_guard.detectors import letterbox, decode_yolo_person, YoloPersonDetector


def fake_yolo_output(boxes, n=8400, classes=80):
    """(1, 4+classes, n) tensor with the given (cx, cy, w, h, person_score) rows; rest ~0."""
    out = np.zeros((1, 4 + classes, n), np.float32)
    for i, (cx, cy, w, h, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4, i] = s
    return out


def test_letterbox_geometry():
    img, r, (px, py) = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert img.shape == (640, 640, 3) and r == 1.0 and (px, py) == (0, 80)


def test_decode_maps_back_to_frame_and_filters():
    # person centred at (320, 320) in 640 letterbox space; 480x640 frame => pad_y=80
    out = fake_yolo_output([(320, 320, 100, 200, 0.9), (50, 50, 20, 20, 0.1)])
    boxes = decode_yolo_person(out, 0.4, 0.45, 1.0, (0, 80), (480, 640, 3))
    assert boxes == [(270, 140, 100, 200)]


def test_decode_nms_merges_duplicates_and_handles_empty():
    out = fake_yolo_output([(320, 320, 100, 200, 0.9), (322, 322, 100, 200, 0.8)])
    assert len(decode_yolo_person(out, 0.4, 0.45, 1.0, (0, 80), (480, 640, 3))) == 1
    assert decode_yolo_person(fake_yolo_output([]), 0.4, 0.45, 1.0, (0, 0), (480, 640, 3)) == []


def test_yolo_detector_end_to_end_with_stub_session():
    class Stub:
        def run(self, _, feed):
            assert feed["images"].shape == (1, 3, 640, 640) and feed["images"].dtype == np.float32
            return [fake_yolo_output([(320, 320, 100, 200, 0.9)])]
    d = object.__new__(YoloPersonDetector)
    d.conf, d.iou, d.zone, d.sess, d.input_name, d.size = 0.4, 0.45, Zone(), Stub(), "images", 640
    assert d.detect(room()) == [(270, 140, 100, 200)]
