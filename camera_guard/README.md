# camera_guard

Detects **movement** and **people** from a camera watching an indoor space, logs events, and saves annotated snapshots. Runs on a laptop, mini-PC or Raspberry Pi 4+; no cloud, no model download.

## Run

```bash
pip install -r requirements.txt
python -m camera_guard --source 0                       # USB/built-in webcam
python -m camera_guard --source rtsp://user:pw@192.168.1.20/stream1   # IP camera
python -m camera_guard --source clip.mp4 --out out/     # test on a recording
python -m camera_guard --source 0 --yolo-model yolov8n.onnx   # better person detection (see below)
python -m camera_guard --source 0 --display             # live window (install opencv-python, not -headless)
```

Output in `--out` (default `camera_guard_events/`): `events.jsonl` (one line per event start/end) and `snapshots/*.jpg` with boxes drawn (orange = motion, red = person).

## Options that matter

| Flag | Use |
|---|---|
| `--zone "0.1,0.2;0.9,0.2;0.9,0.95;0.1,0.95"` | Only watch this polygon (normalised 0..1). Mask out windows, TVs, fans. |
| `--min-area 800` | Raise to ignore pets / small movers. |
| `--yolo-model path.onnx` | Use a YOLOv8/YOLO11 model for person detection instead of HOG. |
| `--yolo-conf 0.4` | YOLO confidence threshold; raise for fewer false alarms, lower to miss fewer people. |
| `--no-people` | Motion only; much lower CPU. |
| `--clear-seconds 5` | Quiet time before an event is closed. |

## How it works

1. **MotionDetector**: MOG2 background subtraction with shadow removal; adapts to slow lighting changes, 30-frame warm-up on start.
2. **PersonDetector**: OpenCV HOG pedestrian detector, run only when something is moving, every 5th frame.
3. **EventTracker**: debounces (3 consecutive hits to open, quiet period to close) so flicker doesn't spam events.
4. **Monitor**: snapshot cooldown, snapshot cap (`max_snapshots`, oldest pruned), auto-reconnect for dropped streams, and an `on_event(phase, event)` hook to send alerts (email, push, webhook). A failing hook never stops the camera loop.

```python
from camera_guard import Config, Monitor
Monitor(Config(source=0, on_event=lambda phase, ev: print(phase, ev.kind))).run()
```

## Better person detection with YOLO (optional)

Export a COCO-trained YOLOv8 or YOLO11 model to ONNX (`yolo export model=yolov8n.pt format=onnx`, needs the `ultralytics` package) and pass it with `--yolo-model`. Needs `onnxruntime`. Check the model's licence (Ultralytics models are AGPL-3.0) before redistributing. The pre/post-processing is unit-tested; **accuracy on a real model and real footage is untested here.**

## Known limits (be honest about these)

- HOG is a classic detector: reliable for upright, mostly visible people ~100px+ tall; it **misses** seated, partially hidden, overhead or low-light people and can false-positive on odd shapes. **Motion events are the dependable signal; "person" is a best-effort label.** Use `--yolo-model` for better accuracy.
- Not tested against a real camera or real footage here, only synthetic video. Tune `--min-area` and `--zone` on your own feed first.
- A still person won't trigger motion; an event ends after `--clear-seconds` of stillness.
- Use only where you have the right to record, and tell people in the space. Snapshots are stored unencrypted on local disk.

## Tests

`python -m pytest -q tests`
