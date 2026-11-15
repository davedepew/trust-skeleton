import argparse
import logging
from pathlib import Path

from .monitor import Config, Monitor


def main() -> None:
    p = argparse.ArgumentParser(prog="camera_guard", description="Detect people and movement from a camera.")
    p.add_argument("--source", default="0", help="camera index (0), RTSP/HTTP URL, or video file")
    p.add_argument("--out", default="camera_guard_events", help="output dir for events.jsonl + snapshots/")
    p.add_argument("--min-area", type=int, default=800, help="min moving blob area in px (raise to ignore pets/noise)")
    p.add_argument("--zone", help='polygon of normalised points to watch, e.g. "0.1,0.2;0.9,0.2;0.9,0.95;0.1,0.95"')
    p.add_argument("--no-people", action="store_true", help="motion only (faster, lower CPU)")
    p.add_argument("--display", action="store_true", help="show a live window (needs opencv-python, not headless)")
    p.add_argument("--clear-seconds", type=float, default=5.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    zone = [tuple(float(v) for v in pt.split(",")) for pt in a.zone.split(";")] if a.zone else None
    src = int(a.source) if a.source.isdigit() else a.source
    Monitor(Config(source=src, out_dir=Path(a.out), min_area=a.min_area, zone=zone,
                   detect_people=not a.no_people, display=a.display, clear_seconds=a.clear_seconds)).run()


main()
