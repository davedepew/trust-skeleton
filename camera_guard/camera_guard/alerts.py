"""Email alerts. Plug into Config(on_event=...). Standard library only (smtplib).

Credentials come from environment variables, never from code or the command line:
  CG_SMTP_HOST (required)  CG_SMTP_PORT (default 587; 465 = implicit SSL)
  CG_SMTP_USER             CG_SMTP_PASSWORD   (Gmail/Outlook: use an app password)
  CG_EMAIL_FROM (defaults to CG_SMTP_USER)
"""
from __future__ import annotations

import logging
import mimetypes
import os
import queue
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterable

from .events import Event

log = logging.getLogger("camera_guard.alerts")


class EmailAlerter:
    """on_event hook that emails a snapshot when an event starts.

    Sending happens on a background thread so a slow/unreachable mail server never
    stalls the camera loop. Alerts are rate-limited per kind so a busy room can't
    flood your inbox.
    """

    def __init__(self, host: str, recipients: Iterable[str], sender: str,
                 port: int = 587, user: str | None = None, password: str | None = None,
                 kinds: Iterable[str] = ("person",), min_interval: float = 60.0,
                 send_end: bool = False, timeout: float = 20.0,
                 smtp_factory: Callable | None = None, clock: Callable[[], float] = time.monotonic):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.recipients = sender, list(recipients)
        if not self.recipients:
            raise ValueError("at least one recipient is required")
        self.kinds, self.min_interval, self.send_end = set(kinds), min_interval, send_end
        self.timeout = timeout
        self._smtp_factory = smtp_factory
        self._clock = clock
        self._last: dict[str, float] = {}
        self._q: queue.Queue[EmailMessage | None] = queue.Queue(maxsize=20)
        self._worker = threading.Thread(target=self._run, daemon=True, name="email-alerts")
        self._worker.start()

    @classmethod
    def from_env(cls, recipients: Iterable[str], env=os.environ, **kw) -> "EmailAlerter":
        host = env.get("CG_SMTP_HOST")
        if not host:
            raise RuntimeError("CG_SMTP_HOST is not set (see camera_guard/alerts.py for the variables)")
        user = env.get("CG_SMTP_USER")
        sender = env.get("CG_EMAIL_FROM") or user
        if not sender:
            raise RuntimeError("set CG_EMAIL_FROM or CG_SMTP_USER")
        return cls(host, recipients, sender, port=int(env.get("CG_SMTP_PORT", "587")),
                   user=user, password=env.get("CG_SMTP_PASSWORD"), **kw)

    # -- hook ----------------------------------------------------------------
    def __call__(self, phase: str, ev: Event) -> None:
        if ev.kind not in self.kinds or (phase == "end" and not self.send_end):
            return
        if phase == "start":
            now = self._clock()
            if now - self._last.get(ev.kind, -1e18) < self.min_interval:
                log.info("alert for %s suppressed (rate limit)", ev.kind)
                return
            self._last[ev.kind] = now
        try:
            self._q.put_nowait(self.build_message(phase, ev))
        except queue.Full:
            log.warning("alert queue full; dropping %s alert", ev.kind)

    def close(self, timeout: float = 15.0) -> None:
        """Send anything still queued, then stop the worker."""
        try:
            self._q.put(None, timeout=1)
        except queue.Full:
            return
        self._worker.join(timeout)

    # -- message / transport -------------------------------------------------
    def build_message(self, phase: str, ev: Event) -> EmailMessage:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.started))
        what = "Person detected" if ev.kind == "person" else "Motion detected"
        msg = EmailMessage()
        msg["From"], msg["To"] = self.sender, ", ".join(self.recipients)
        msg["Subject"] = f"[camera_guard] {what} at {when}" if phase == "start" \
            else f"[camera_guard] {ev.kind} event ended ({ev.duration:.0f}s)"
        msg.set_content(
            f"{what}.\nTime: {when}\nType: {ev.kind}\nObjects: {ev.peak_boxes}\n"
            + (f"Duration: {ev.duration:.0f}s\n" if phase == "end" else "")
            + ("Snapshot attached.\n" if ev.snapshot else ""))
        if phase == "start" and ev.snapshot and Path(ev.snapshot).is_file():
            ctype = mimetypes.guess_type(ev.snapshot)[0] or "image/jpeg"
            maintype, subtype = ctype.split("/")
            msg.add_attachment(Path(ev.snapshot).read_bytes(), maintype=maintype,
                               subtype=subtype, filename=Path(ev.snapshot).name)
        return msg

    def _connect(self):
        if self._smtp_factory:
            return self._smtp_factory()
        if self.port == 465:
            return smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout,
                                    context=ssl.create_default_context())
        return smtplib.SMTP(self.host, self.port, timeout=self.timeout)

    def send(self, msg: EmailMessage) -> None:
        with self._connect() as s:
            if not self._smtp_factory and self.port != 465:
                s.starttls(context=ssl.create_default_context())  # never send credentials in clear
            if self.user:
                s.login(self.user, self.password or "")
            s.send_message(msg)

    def send_test(self) -> None:
        """Synchronous test email; raises on failure so you see the real SMTP error."""
        msg = EmailMessage()
        msg["From"], msg["To"] = self.sender, ", ".join(self.recipients)
        msg["Subject"] = "[camera_guard] test alert"
        msg.set_content("If you can read this, camera_guard email alerts are working.")
        self.send(msg)

    def _run(self) -> None:
        while True:
            msg = self._q.get()
            if msg is None:
                return
            try:
                self.send(msg)
                log.info("alert emailed: %s", msg["Subject"])
            except Exception:  # never let a mail failure kill the worker
                log.exception("failed to send alert email")
