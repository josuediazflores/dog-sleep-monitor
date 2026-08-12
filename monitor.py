#!/usr/bin/env python3
"""Stillness monitor for a fixed camera pointed at a dog playpen.

No model, no training. Samples the camera on an interval, measures how much
changed since the previous sample, and reports asleep / awake using a rolling
window with hysteresis.

Supports a TP-Link Tapo (or any RTSP camera) over the network, and USB webcams.

Subcommands:
    preview    grab one frame, draw the ROI, write preview.jpg
    calibrate  collect motion scores for a while, suggest thresholds
    watch      run the monitor
"""

import argparse
import csv
import hmac
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

# Must be set before the first VideoCapture. UDP loses packets over Wi-Fi and
# produces torn frames that read as motion; TCP does not.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
CALIB_PATH = os.path.join(HERE, "calibration.csv")
ENV_PATH = os.path.join(HERE, ".env")


def load_env(path=ENV_PATH):
    """Read KEY=VALUE lines from .env without overriding the real environment.

    Keeps the camera password in a gitignored file instead of config.json, and
    means nothing has to be typed into a shell on every run.
    """
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val

DEFAULTS = {
    # "rtsp" for a Tapo / IP camera, "usb" for a webcam
    "source": "rtsp",

    "rtsp": {
        "host": "192.168.1.100",     # the camera's LAN IP
        "port": 554,
        "path": "/stream2",          # stream1 = full res, stream2 = 640x360
        "user_env": "TAPO_USER",     # credentials come from the environment,
        "pass_env": "TAPO_PASS",     # never from this file
        "connect_timeout": 20.0,
        "stale_after": 10.0,         # a frame older than this means the feed died
    },

    "usb": {
        "camera_index": 0,
        "capture_width": 1280,
        "capture_height": 720,
        "flush_frames": 5,           # webcams buffer; drop stale frames first
        "warmup_seconds": 2.0,
    },

    # region of interest as fractions of the frame [x, y, w, h], each 0..1,
    # so it survives a resolution or stream change. null means full frame.
    "roi": None,

    # preprocessing
    "work_size": [64, 48],           # downscale target; small is the point
    "blur_kernel": 5,                # odd number
    "frames_per_sample": 5,          # averaged per sample; cuts noise by sqrt(n)
    "frame_spacing": 0.2,            # seconds between those frames

    # scoring
    "pixel_threshold": 0.30,         # per-pixel change, normalized contrast units
    "quiet_score": 0.010,            # below this, the frame pair counts as still
    "active_score": 0.030,           # above this, it counts as movement
    "scene_change_score": 0.60,      # a scene change needs BOTH a score above
    "scene_corr_max": 0.50,          # this AND correlation below this, so a dog
                                     # filling the frame is not mistaken for the
                                     # day/night IR switch

    # timing
    "sample_seconds": 5.0,
    "quiet_samples_to_sleep": 12,    # 12 x 5s = 1 minute of stillness
    "active_samples_to_wake": 2,     # waking should register fast

    # output
    "log_path": "sleep_log.csv",
    "events_path": "events.csv",
    "print_every_sample": True,

    # visual evidence, so a logged transition can be checked against a picture
    "snapshot_dir": "snapshots",
    "snapshot_on_event": True,       # every state change and scene change
    "snapshot_every_minutes": 15,    # also a periodic one; 0 disables
    "snapshot_scale": 0.5,           # 0.5 of 1280x720 is ~54KB per jpeg
    "snapshot_retention_days": 14,   # older files are pruned at startup

    # every-sample archive, for offline review of a whole session.
    # ~54KB x 720/hour = ~39MB/hour. Temporary by design; see `purge`.
    "archive_all_samples": False,
    "archive_dir": "archive",
    "archive_scale": 0.5,
    "archive_quality": 80,
    "archive_max_mb": 3000,          # hard stop, so it cannot fill the disk

    # read-only JSON API for the iOS app to pull from
    "api_bind": "127.0.0.1",         # set to the tailnet IP to reach the phone
    "api_port": 8787,
    "api_token_env": "MONITOR_API_TOKEN",
    "api_min_session_minutes": 10,   # shorter stretches are not "a sleep"
    "api_merge_stirs_minutes": 10,   # awake gaps below this merge into one sleep
}


def load_config(path=CONFIG_PATH):
    """Merge config.json over DEFAULTS, one level deep for nested blocks."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if os.path.exists(path):
        with open(path) as fh:
            user = json.load(fh)
        for key, val in user.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def stamp():
    return datetime.now().isoformat(timespec="seconds")


def pace(next_at, interval):
    """Sleep until next_at, then return the following deadline.

    Keeps the sample interval honest over long runs instead of letting the
    per-sample processing time accumulate into drift.
    """
    time.sleep(max(0.0, next_at - time.monotonic()))
    return next_at + interval


# --- capture -----------------------------------------------------------------

class RtspCamera:
    """RTSP camera with a background reader that always holds the newest frame.

    An RTSP stream cannot be sampled on demand. If you open it and read once
    every 5 seconds you get frames from the socket backlog, not from now, so
    every comparison looks still. A drain thread consumes the stream
    continuously and keeps only the latest frame; frame() takes a copy of it.
    The thread also reconnects on its own, because a Wi-Fi camera watched for
    eight hours will drop at some point.
    """

    def __init__(self, cfg):
        rt = cfg["rtsp"]
        user = os.environ.get(rt["user_env"])
        password = os.environ.get(rt["pass_env"])
        if not user or not password:
            raise SystemExit(
                f"No camera credentials found in {rt['user_env']} / "
                f"{rt['pass_env']}.\n\n"
                f"Write them into {ENV_PATH} (gitignored):\n"
                f"  {rt['user_env']}=camuser\n"
                f"  {rt['pass_env']}=campass\n\n"
                "These are the Tapo *Camera Account* credentials, which you create "
                "in the Tapo app\nunder Device Settings > Advanced Settings > "
                "Camera Account.\nThey are not your TP-Link login, and creating "
                "them is what switches the\ncamera's RTSP server on."
            )

        self.url = (
            f"rtsp://{urllib.parse.quote(user, safe='')}:"
            f"{urllib.parse.quote(password, safe='')}@"
            f"{rt['host']}:{rt['port']}{rt['path']}"
        )
        self.safe_url = f"rtsp://{user}:***@{rt['host']}:{rt['port']}{rt['path']}"
        self.stale_after = float(rt["stale_after"])

        self._lock = threading.Lock()
        self._frame = None
        self._at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + float(rt["connect_timeout"])
        while True:
            with self._lock:
                if self._frame is not None:
                    return
            if time.monotonic() > deadline:
                self.close()
                raise SystemExit(
                    f"No frames from {self.safe_url} within "
                    f"{rt['connect_timeout']:.0f}s.\n"
                    "Check, in order:\n"
                    "  1. the host IP is right (Tapo app > Device Settings > "
                    "Device Info)\n"
                    "  2. the Camera Account exists and the credentials match\n"
                    "  3. the path is /stream1 or /stream2 for your model\n"
                    f"  4. it works outside this script: ffplay '{self.safe_url}'"
                )
            time.sleep(0.25)

    def _drain(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self._stop.wait(2.0)
                continue
            misses = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 30:
                        break          # feed is gone, drop out and reconnect
                    self._stop.wait(0.1)
                    continue
                misses = 0
                with self._lock:
                    self._frame = frame
                    self._at = time.monotonic()
            cap.release()

    def frame(self):
        with self._lock:
            if self._frame is None:
                raise RuntimeError("No frame available; the stream is reconnecting.")
            age = time.monotonic() - self._at
            if age > self.stale_after:
                raise RuntimeError(
                    f"Newest frame is {age:.0f}s old; the camera feed dropped."
                )
            return self._frame.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)


class UsbCamera:
    """USB webcam. Discards the buffered backlog before every real read."""

    def __init__(self, cfg):
        ub = cfg["usb"]
        index = ub["camera_index"]
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(
                f"Cannot open camera index {index}. "
                "Try another index (0, 1, 2) in config.json under \"usb\"."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ub["capture_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ub["capture_height"])
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.flush = int(ub["flush_frames"])
        deadline = time.monotonic() + float(ub["warmup_seconds"])
        while time.monotonic() < deadline:
            self.cap.read()

    def frame(self):
        for _ in range(self.flush):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Frame grab failed. Camera unplugged or in use?")
        return frame

    def close(self):
        self.cap.release()


def open_camera(cfg):
    source = cfg["source"]
    if source == "rtsp":
        return RtspCamera(cfg)
    if source == "usb":
        return UsbCamera(cfg)
    raise SystemExit(f'Unknown source {source!r}. Use "rtsp" or "usb".')


# --- the math ----------------------------------------------------------------

def roi_pixels(roi, width, height):
    """Fractional ROI [x, y, w, h] in 0..1 -> pixel box, clamped to the frame."""
    fx, fy, fw, fh = (float(v) for v in roi)
    x = max(0, min(width - 1, int(round(fx * width))))
    y = max(0, min(height - 1, int(round(fy * height))))
    w = max(1, min(width - x, int(round(fw * width))))
    h = max(1, min(height - y, int(round(fh * height))))
    return x, y, w, h


def prepare(frame, cfg):
    """Turn a raw BGR frame into a small, brightness-normalized float image.

    Downscale plus blur kills sensor noise. Mean/std normalization removes
    global brightness and gain shifts, which is what makes this survive a
    camera that quietly re-tunes its own exposure.
    """
    if cfg["roi"]:
        h, w = frame.shape[:2]
        x, y, rw, rh = roi_pixels(cfg["roi"], w, h)
        frame = frame[y:y + rh, x:x + rw]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, tuple(cfg["work_size"]), interpolation=cv2.INTER_AREA)
    k = int(cfg["blur_kernel"]) | 1  # force odd
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    out = gray.astype(np.float32)
    out -= out.mean()
    # Floor the divisor so a flat dark frame does not get its noise amplified.
    out /= max(float(out.std()), 8.0)
    return out


def motion_score(a, b, pixel_threshold):
    """Fraction of pixels that changed by more than pixel_threshold."""
    diff = np.abs(a - b)
    return float((diff > pixel_threshold).mean())


def correlation(a, b):
    """Pearson correlation between two prepared frames, in [-1, 1].

    Distinguishes a big local change from a global one, which magnitude alone
    cannot. A dog moving across the pen leaves most of the scene intact, so
    correlation stays high even when the score is large. A day/night IR switch
    re-lights every surface and inverts relative brightness, so correlation
    collapses toward zero or goes negative.

    This exists because a dog close to the camera can score 0.82, above any
    plausible magnitude-only scene-change threshold, and was being dismissed as
    a lighting change while she was visibly moving.
    """
    x, y = a.ravel(), b.ravel()
    sx, sy = x.std(), y.std()
    if sx < 1e-6 or sy < 1e-6:
        return 1.0  # a flat frame has no structure to disagree about
    return float(np.clip(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy),
                         -1.0, 1.0))


def grab_sample(cam, cfg, keep_raw=False):
    """One sample: the mean of several frames, prepared.

    Sensor and h264 noise are independent between frames, so averaging n of
    them cuts the noise by sqrt(n). A sleeping dog does not move over the
    ~1 second this spans, so the signal survives intact. This buys sensitivity
    that raising pixel_threshold would have spent.

    With keep_raw, also returns the averaged full-size frame, so a snapshot can
    show the exact image the score was computed from rather than a later one.
    """
    n = max(1, int(cfg["frames_per_sample"]))
    gap = float(cfg["frame_spacing"])
    acc = cam.frame().astype(np.float32)
    for _ in range(n - 1):
        time.sleep(gap)
        acc += cam.frame().astype(np.float32)
    acc /= n
    raw = acc.astype(np.uint8)
    return (prepare(raw, cfg), raw) if keep_raw else prepare(raw, cfg)


# --- snapshots ---------------------------------------------------------------

def snapshot_path(cfg, kind, score):
    d = os.path.join(HERE, cfg["snapshot_dir"])
    os.makedirs(d, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return os.path.join(d, f"{ts}_{kind}_{score:.4f}.jpg")


def save_snapshot(frame, cfg, kind, score):
    """Write the frame behind an event, with the ROI drawn for review."""
    scale = float(cfg["snapshot_scale"])
    img = frame if scale >= 0.999 else cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if cfg["roi"]:
        h, w = img.shape[:2]
        x, y, rw, rh = roi_pixels(cfg["roi"], w, h)
        cv2.rectangle(img, (x, y), (x + rw, y + rh), (0, 255, 0), 1)
    path = snapshot_path(cfg, kind, score)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return path


def dir_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for name in os.listdir(path):
            try:
                total += os.path.getsize(os.path.join(path, name))
            except OSError:
                pass
    return total / (1024.0 * 1024.0)


class FrameArchive:
    """Writes one jpeg per sample for offline review, under a hard size cap.

    Meant to be temporary: capture a session, have something review it, then
    `purge`. The cap exists because 720 frames an hour fills a disk quietly.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg["archive_all_samples"])
        self.dir = os.path.join(HERE, cfg["archive_dir"])
        self.scale = float(cfg["archive_scale"])
        self.quality = int(cfg["archive_quality"])
        self.cap_mb = float(cfg["archive_max_mb"])
        self.written = 0
        self.stopped = False
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)
            self.used_mb = dir_size_mb(self.dir)

    def save(self, frame, score):
        if not self.enabled or self.stopped:
            return None
        # Recheck the real total periodically rather than trusting the running
        # estimate, since jpeg sizes vary with scene complexity.
        if self.written % 200 == 0:
            self.used_mb = dir_size_mb(self.dir)
        if self.used_mb >= self.cap_mb:
            self.stopped = True
            print(f"{stamp()}  ARCHIVE FULL at {self.used_mb:.0f}MB "
                  f"(cap {self.cap_mb:.0f}MB). Archiving off; monitoring continues.")
            return None

        img = frame if self.scale >= 0.999 else cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        path = os.path.join(self.dir, f"{ts}_{score:.4f}.jpg")
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        self.written += 1
        try:
            self.used_mb += os.path.getsize(path) / (1024.0 * 1024.0)
        except OSError:
            pass
        return path


def prune_snapshots(cfg):
    """Drop snapshots older than the retention window. Returns count removed."""
    days = int(cfg["snapshot_retention_days"])
    d = os.path.join(HERE, cfg["snapshot_dir"])
    if days <= 0 or not os.path.isdir(d):
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for name in os.listdir(d):
        p = os.path.join(d, name)
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
                removed += 1
        except OSError:
            pass
    return removed


# --- doctor ------------------------------------------------------------------

def tcp_open(host, port, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def cmd_doctor(cfg, args):
    """Check the boring things in order, so failures name their own cause."""
    ok = True
    if cfg["source"] != "rtsp":
        print(f'source is "{cfg["source"]}", nothing to check here.')
        return

    rt = cfg["rtsp"]
    host, port = rt["host"], int(rt["port"])

    print(f"1. TCP {host}:{port} (RTSP) ...", end=" ")
    if tcp_open(host, port):
        print("open")
    else:
        ok = False
        print("CLOSED")
        print(f"   Reachable at all? TCP {host}:443 ...", end=" ")
        print("yes, so the camera is online" if tcp_open(host, 443)
              else "no, check the IP and that the camera is powered")
        print("   A Tapo camera keeps its RTSP server switched off until you\n"
              "   create a Camera Account: Tapo app > Device Settings >\n"
              "   Advanced Settings > Camera Account. Do that first.")

    print("2. credentials ...", end=" ")
    user, password = os.environ.get(rt["user_env"]), os.environ.get(rt["pass_env"])
    if user and password:
        print(f"{rt['user_env']}={user}, {rt['pass_env']}=({len(password)} chars)")
    else:
        ok = False
        missing = [k for k in (rt["user_env"], rt["pass_env"]) if not os.environ.get(k)]
        print(f"MISSING {', '.join(missing)}")
        print(f"   Add them to {ENV_PATH}")

    if not ok:
        print("\nFix the above, then run doctor again.")
        return

    print("3. pulling a frame ...", end=" ", flush=True)
    cam = open_camera(cfg)
    try:
        frame = cam.frame()
    finally:
        cam.close()
    h, w = frame.shape[:2]
    print(f"got {w}x{h}")

    print("4. ROI ...", end=" ")
    if cfg["roi"]:
        x, y, rw, rh = roi_pixels(cfg["roi"], w, h)
        print(f"{rw}x{rh}px at ({x},{y}), {100.0 * rw * rh / (w * h):.0f}% of frame")
    else:
        print("not set. Run `preview --pick-roi` before calibrating.")

    print("\nAll good.")


# --- preview -----------------------------------------------------------------

def cmd_preview(cfg, args):
    cam = open_camera(cfg)
    try:
        frame = cam.frame()
    finally:
        cam.close()

    h, w = frame.shape[:2]
    print(f"Connected. Frame is {w}x{h}.")

    if args.pick_roi:
        print("Drag a box around the playpen, then press ENTER. Press C to cancel.")
        box = cv2.selectROI("pick playpen", frame, showCrosshair=False)
        cv2.destroyAllWindows()
        if box[2] > 0 and box[3] > 0:
            x, y, bw, bh = (float(v) for v in box)
            cfg["roi"] = [round(x / w, 4), round(y / h, 4),
                          round(bw / w, 4), round(bh / h, 4)]
            with open(CONFIG_PATH, "w") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"Saved roi {cfg['roi']} (frame fractions) to config.json")
        else:
            print("No ROI selected, leaving config.json alone.")

    shown = frame.copy()
    if cfg["roi"]:
        x, y, rw, rh = roi_pixels(cfg["roi"], w, h)
        cv2.rectangle(shown, (x, y), (x + rw, y + rh), (0, 255, 0), 3)
        print(f"ROI covers {rw}x{rh}px, {100.0 * rw * rh / (w * h):.0f}% of the frame.")
    out = os.path.join(HERE, "preview.jpg")
    cv2.imwrite(out, shown)
    print(f"Wrote {out}" + (" (green box = ROI)" if cfg["roi"] else " (no ROI set)"))


# --- calibrate ---------------------------------------------------------------

def percentile_report(label, scores):
    arr = np.array(sorted(scores))
    ps = {p: float(np.percentile(arr, p)) for p in (5, 50, 90, 95, 99)}
    print(f"\n[{label}]  n={len(arr)}")
    print(f"  min {arr[0]:.4f}   p50 {ps[50]:.4f}   p90 {ps[90]:.4f}   "
          f"p95 {ps[95]:.4f}   max {arr[-1]:.4f}")
    return ps


class SleepState:
    """The policy layer, isolated so it can be replayed against logged scores.

    Feed it scores in order and it returns the same decisions `watch` makes
    live. That is what makes threshold tuning measurable instead of a guess.
    """

    def __init__(self, cfg, state="unknown"):
        self.quiet_needed = int(cfg["quiet_samples_to_sleep"])
        self.active_needed = int(cfg["active_samples_to_wake"])
        self.quiet_score = float(cfg["quiet_score"])
        self.active_score = float(cfg["active_score"])
        self.scene_score = float(cfg["scene_change_score"])
        self.scene_corr_max = float(cfg["scene_corr_max"])
        self.state = state
        self.quiet_run = 0
        self.active_run = 0

    def update(self, score, corr=1.0):
        """Returns (state, tag, changed).

        corr defaults to 1.0, meaning "structurally the same scene". Replays of
        logged scores use that default, which is correct because scene-change
        rows are excluded from labeled data anyway.
        """
        if score > self.scene_score and corr < self.scene_corr_max:
            # Both conditions required. A big change that stays correlated is a
            # dog filling the frame, not the room re-lighting; treating that as
            # a scene change would freeze the state while she is visibly moving.
            self.quiet_run = self.active_run = 0
            return self.state, "scene", False

        if score < self.quiet_score:
            self.quiet_run += 1
            self.active_run = 0
            tag = "still"
        elif score > self.active_score:
            self.active_run += 1
            self.quiet_run = 0
            tag = "MOVING"
        else:
            tag = "..."  # deadband: counts toward neither, prevents flapping

        changed = False
        if self.state != "asleep" and self.quiet_run >= self.quiet_needed:
            self.state, changed = "asleep", True
        elif self.state != "awake" and self.active_run >= self.active_needed:
            self.state, changed = "awake", True
        return self.state, tag, changed


def append_calibration(label, rows):
    """rows: iterable of (timestamp_string, score)."""
    calib_is_new = not os.path.exists(CALIB_PATH)
    with open(CALIB_PATH, "a", newline="") as fh:
        wr = csv.writer(fh)
        if calib_is_new:
            wr.writerow(["timestamp", "label", "score"])
        for ts, s in rows:
            wr.writerow([ts, label, f"{s:.6f}"])


def summarize_calibration():
    """Report every label collected so far and suggest a threshold split."""
    by_label = {}
    with open(CALIB_PATH) as fh:
        for row in csv.DictReader(fh):
            by_label.setdefault(row["label"], []).append(float(row["score"]))

    stats = {lab: percentile_report(lab, vals)
             for lab, vals in sorted(by_label.items()) if len(vals) >= 2}

    if "quiet" in stats and "active" in stats:
        quiet = round(stats["quiet"][95], 4)
        active = round(max(stats["active"][5], quiet * 2), 4)
        print("\nSuggested config.json values:")
        print(f'  "quiet_score": {quiet},')
        print(f'  "active_score": {active}')
        if stats["active"][5] < stats["quiet"][95]:
            print("\nNote: the two distributions overlap. The quietest movement in"
                  "\nthe 'active' data was smaller than the noisiest sample in the"
                  "\n'quiet' data, so no single threshold separates them cleanly."
                  "\nTighten the ROI, or accept that small movements read as still.")
    else:
        print("\nNeed both labels before a threshold can be suggested.")


def cmd_label(cfg, args):
    """Tag a time window of the live log as quiet or active.

    Lets a running `watch` double as the calibration source, instead of opening
    a second RTSP connection to the same camera while it is mid-run.
    """
    log_path = os.path.join(HERE, cfg["log_path"])
    if not os.path.exists(log_path):
        raise SystemExit(f"No log at {log_path}. Run `watch` first.")

    def parse(text):
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = datetime.strptime(text, fmt).time()
                return datetime.combine(datetime.now().date(), t)
            except ValueError:
                continue
        raise SystemExit(f"Cannot parse time {text!r}. Use HH:MM, HH:MM:SS, or ISO.")

    start, end = parse(args.start), parse(args.end)
    if end <= start:
        raise SystemExit("--to must be after --from.")

    rows, skipped = [], 0
    with open(log_path) as fh:
        for row in csv.DictReader(fh):
            when = datetime.fromisoformat(row["timestamp"])
            if not (start <= when <= end):
                continue
            if row["changed"] == "scene-change":
                skipped += 1  # not the dog, do not teach a threshold with it
                continue
            rows.append((row["timestamp"], float(row["score"])))

    if not rows:
        raise SystemExit(f"No samples between {start} and {end}.")

    append_calibration(args.label, rows)
    print(f"Labeled {len(rows)} samples as '{args.label}'"
          + (f", skipped {skipped} scene-change rows" if skipped else ""))
    summarize_calibration()


def cmd_calibrate(cfg, args):
    interval = float(cfg["sample_seconds"])
    n = max(2, int(args.seconds / interval))
    print(f"Calibrating label '{args.label}' for ~{args.seconds:.0f}s "
          f"({n} samples, {interval:.0f}s apart). Leave the scene as it is.")

    cam = open_camera(cfg)
    scores = []
    try:
        prev = grab_sample(cam, cfg)
        next_at = time.monotonic() + interval
        for i in range(n):
            next_at = pace(next_at, interval)
            cur = grab_sample(cam, cfg)
            s = motion_score(prev, cur, cfg["pixel_threshold"])
            scores.append(s)
            prev = cur
            print(f"  {i + 1}/{n}  score {s:.4f}")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        cam.close()

    if len(scores) < 2:
        print("Not enough samples to report. Run it longer.")
        return

    append_calibration(args.label, [(stamp(), s) for s in scores])
    print(f"\nAppended {len(scores)} samples to {CALIB_PATH}")
    summarize_calibration()


# --- tune --------------------------------------------------------------------

def replay(scores, cfg, start_state):
    """Run scores through the real state machine. Returns list of transitions."""
    m = SleepState(cfg, state=start_state)
    out = []
    for i, s in enumerate(scores):
        state, _, changed = m.update(s)
        if changed:
            out.append((i, state))
    return out


def cmd_tune(cfg, args):
    """Grid-search thresholds against labeled data, scored on run behaviour.

    Per-sample separability is the wrong measure: a rowdy dog pauses, and a
    quiet scene spikes. What matters is whether a *run* long enough to flip the
    state gets misclassified. So this replays each labeled sequence through the
    actual state machine and counts wrong transitions.
    """
    if not os.path.exists(CALIB_PATH):
        raise SystemExit(f"No labeled data at {CALIB_PATH}. Use `label` first.")

    seqs = {}
    with open(CALIB_PATH) as fh:
        for row in sorted(csv.DictReader(fh), key=lambda r: r["timestamp"]):
            seqs.setdefault(row["label"], []).append(float(row["score"]))
    for lab in ("quiet", "active"):
        if len(seqs.get(lab, [])) < 5:
            raise SystemExit(f"Need at least 5 '{lab}' samples; have "
                             f"{len(seqs.get(lab, []))}.")

    print(f"quiet: {len(seqs['quiet'])} samples   active: {len(seqs['active'])}"
          f"   (asleep needs {cfg['quiet_samples_to_sleep']} quiet in a row, "
          f"awake needs {cfg['active_samples_to_wake']})\n")

    QUIETS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
    ACTIVES = [0.020, 0.030, 0.040, 0.060, 0.080, 0.120]

    results = []
    for q in QUIETS:
        for a in ACTIVES:
            if a <= q:
                continue
            trial = {**cfg, "quiet_score": q, "active_score": a}
            # A quiet stretch must not produce a wake.
            false_wakes = sum(1 for _, st in replay(seqs["quiet"], trial, "asleep")
                              if st == "awake")
            # An active stretch must not produce a sleep.
            false_sleeps = sum(1 for _, st in replay(seqs["active"], trial, "awake")
                               if st == "asleep")
            # How fast does a wake get noticed from a standing start?
            wake = replay(seqs["active"], trial, "asleep")
            latency = next((i + 1 for i, st in wake if st == "awake"), None)
            results.append((false_wakes, false_sleeps,
                            latency if latency is not None else 999, q, a))

    results.sort()
    print(f"{'quiet':>7}{'active':>8}{'false wakes':>13}{'false sleeps':>14}"
          f"{'wake latency':>14}")
    for fw, fs, lat, q, a in results[:10]:
        lat_s = f"{lat} samples" if lat != 999 else "never"
        print(f"{q:>7.3f}{a:>8.3f}{fw:>13}{fs:>14}{lat_s:>14}")

    fw, fs, lat, q, a = results[0]
    print(f"\nBest: quiet_score {q}, active_score {a}")
    print(f"  false wakes {fw}, false sleeps {fs}, "
          f"wake detected in {lat} samples "
          f"({lat * cfg['sample_seconds']:.0f}s)" if lat != 999 else "  never wakes")
    if fw or fs:
        print("\nNo threshold pair is clean on this data. More labeled data, "
              "especially\na genuinely sleeping dog rather than an empty pen, "
              "should be collected\nbefore trusting these numbers.")


# --- report ------------------------------------------------------------------

def read_log(cfg, start=None, end=None):
    """Load the sample log as (when, score, state, changed) tuples."""
    path = os.path.join(HERE, cfg["log_path"])
    if not os.path.exists(path):
        raise SystemExit(f"No log at {path}. Run `watch` first.")
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            when = datetime.fromisoformat(r["timestamp"])
            if (start and when < start) or (end and when > end):
                continue
            rows.append((when, float(r["score"]), r["state"], r["changed"]))
    rows.sort(key=lambda t: t[0])
    return rows


def sessionize(rows, interval, gap_factor=3.0):
    """Collapse per-sample rows into runs of one state, and find data gaps.

    A gap is a stretch with no samples at all, which means the monitor was not
    running: the laptop slept, the process died, the feed was down. Gaps are
    reported separately and never bridged, because silently joining two sleep
    sessions across a four-hour outage would invent four hours of sleep.
    """
    sessions, gaps = [], []
    cur = None
    prev_when = None

    def close(end_when):
        if cur and cur["samples"]:
            cur["end"] = end_when
            cur["duration"] = (end_when - cur["start"]).total_seconds()
            sessions.append(cur)

    for when, score, state, _changed in rows:
        if prev_when is not None:
            delta = (when - prev_when).total_seconds()
            if delta > gap_factor * interval:
                close(prev_when)
                cur = None
                gaps.append({"start": prev_when, "end": when, "duration": delta})
        if cur is None or cur["state"] != state:
            close(prev_when if prev_when else when)
            cur = {"state": state, "start": when, "samples": [], "scores": []}
        cur["samples"].append(when)
        cur["scores"].append(score)
        prev_when = when

    close(prev_when if prev_when else datetime.now())
    for s in sessions:
        s["mean"] = sum(s["scores"]) / len(s["scores"])
        s["max"] = max(s["scores"])
    return sessions, gaps


def human(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def timeline_bar(sessions, gaps, width=64):
    """One-line ASCII timeline. Each cell is the dominant state in that slice."""
    spans = ([(s["start"], s["end"], s["state"]) for s in sessions]
             + [(g["start"], g["end"], "gap") for g in gaps])
    if not spans:
        return "", None, None
    t0 = min(s[0] for s in spans)
    t1 = max(s[1] for s in spans)
    total = max((t1 - t0).total_seconds(), 1.0)
    glyph = {"asleep": "#", "awake": "~", "unknown": ".", "gap": " "}
    cells = []
    for i in range(width):
        a = t0 + timedelta(seconds=total * i / width)
        b = t0 + timedelta(seconds=total * (i + 1) / width)
        best, best_overlap = "gap", 0.0
        for ss, se, st in spans:
            ov = (min(b, se) - max(a, ss)).total_seconds()
            if ov > best_overlap:
                best, best_overlap = st, ov
        cells.append(glyph.get(best, "?"))
    return "".join(cells), t0, t1


def cmd_report(cfg, args):
    def parse(text):
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    t = datetime.strptime(text, fmt).time()
                    return datetime.combine(datetime.now().date(), t)
                except ValueError:
                    continue
        raise SystemExit(f"Cannot parse time {text!r}.")

    interval = float(cfg["sample_seconds"])
    rows = read_log(cfg, parse(args.start), parse(args.end))
    if not rows:
        raise SystemExit("No samples in that window.")
    sessions, gaps = sessionize(rows, interval)

    stir_s = args.min_wake * 60.0
    totals = {}
    for s in sessions:
        totals[s["state"]] = totals.get(s["state"], 0.0) + s["duration"]
    gap_total = sum(g["duration"] for g in gaps)
    asleep = totals.get("asleep", 0.0)
    awake = totals.get("awake", 0.0)
    monitored = asleep + awake + totals.get("unknown", 0.0)

    sleeps = [s for s in sessions if s["state"] == "asleep"]
    wakes = [s for s in sessions if s["state"] == "awake"]
    real_wakes = [w for w in wakes if w["duration"] >= stir_s]
    stirs = [w for w in wakes if w["duration"] < stir_s]

    print(f"\n  {rows[0][0]:%Y-%m-%d %H:%M}  to  {rows[-1][0]:%H:%M}"
          f"   ({human((rows[-1][0] - rows[0][0]).total_seconds())} wall clock, "
          f"{len(rows)} samples)\n")

    bar, t0, t1 = timeline_bar(sessions, gaps)
    print(f"  {bar}")
    print(f"  {t0:%H:%M}{' ' * max(0, len(bar) - 11)}{t1:%H:%M}")
    print("  # asleep   ~ awake   . warming up   (blank) no data\n")

    print(f"  asleep        {human(asleep):>10}"
          f"   {100.0 * asleep / monitored if monitored else 0:>5.1f}% of monitored")
    print(f"  awake         {human(awake):>10}"
          f"   {100.0 * awake / monitored if monitored else 0:>5.1f}%")
    if gap_total:
        print(f"  NO DATA       {human(gap_total):>10}"
              f"   {len(gaps)} gap(s), monitor was not running")
    print()
    if sleeps:
        longest = max(sleeps, key=lambda s: s["duration"])
        print(f"  sleep sessions {len(sleeps):>9}"
              f"   longest {human(longest['duration'])} "
              f"from {longest['start']:%H:%M}")
    print(f"  wakes         {len(real_wakes):>10}"
          f"   (over {args.min_wake:g} min)")
    print(f"  brief stirs   {len(stirs):>10}   (under {args.min_wake:g} min)")

    scene = sum(1 for r in rows if r[3] == "scene-change")
    if scene:
        print(f"  scene changes {scene:>10}   (light shifts, IR switch, camera moved)")

    print(f"\n  {'state':<10}{'from':>9}{'to':>9}{'duration':>11}"
          f"{'mean':>9}{'max':>9}")
    shown = sessions if args.all else [s for s in sessions
                                       if s["duration"] >= stir_s
                                       or s["state"] != "awake"]
    lines = []
    for s in shown:
        tag = s["state"]
        if s["state"] == "awake" and s["duration"] < stir_s:
            tag = "stir"
        lines.append((s["start"],
                      f"  {tag:<10}"
                      + f"{s['start']:%H:%M:%S}".rjust(9)
                      + f"{s['end']:%H:%M:%S}".rjust(9)
                      + f"{human(s['duration']):>11}"
                      + f"{s['mean']:>9.4f}{s['max']:>9.4f}"))
    for g in gaps:
        lines.append((g["start"],
                      f"  {'NO DATA':<10}"
                      + f"{g['start']:%H:%M:%S}".rjust(9)
                      + f"{g['end']:%H:%M:%S}".rjust(9)
                      + f"{human(g['duration']):>11}{'-':>9}{'-':>9}"))
    for _, line in sorted(lines, key=lambda t: t[0]):
        print(line)
    if not args.all and len(shown) < len(sessions):
        print(f"\n  {len(sessions) - len(shown)} brief stirs hidden; "
              f"--all shows every session.")
    print("\n  A high max on an asleep session is the movement that ended it:"
          "\n  waking needs 2 consecutive samples to confirm, so the first one"
          "\n  is still filed under the old state.")

    if args.html:
        out = write_html_report(cfg, rows, sessions, gaps, args)
        print(f"\n  Wrote {out}")


def write_html_report(cfg, rows, sessions, gaps, args):
    """Self-contained HTML: a hypnogram band over the raw score trace."""
    spans = ([(s["start"], s["end"], s["state"], s["duration"]) for s in sessions]
             + [(g["start"], g["end"], "gap", g["duration"]) for g in gaps])
    t0 = min(s[0] for s in spans)
    t1 = max(s[1] for s in spans)
    total = max((t1 - t0).total_seconds(), 1.0)
    W, BAND_H, TRACE_H = 1000.0, 46.0, 150.0

    def x_of(when):
        return W * (when - t0).total_seconds() / total

    colors = {"asleep": "var(--asleep)", "awake": "var(--awake)",
              "unknown": "var(--unknown)", "gap": "var(--gap)"}
    bands = "".join(
        f'<rect x="{x_of(a):.2f}" y="0" width="{max(x_of(b) - x_of(a), 0.6):.2f}" '
        f'height="{BAND_H}" fill="{colors.get(st, "var(--gap)")}">'
        f'<title>{st} {a:%H:%M} to {b:%H:%M} ({human(d)})</title></rect>'
        for a, b, st, d in spans)

    # Square-root scale against the observed maximum. Scores span three orders
    # of magnitude (0.0000 to 0.82), so a linear axis either clips the peaks or
    # flattens everything interesting into the bottom pixel row.
    cap = max(max((s for _w, s, _st, _c in rows), default=0.0), 0.02)

    def y_of(score):
        return TRACE_H - TRACE_H * math.sqrt(max(score, 0.0) / cap)

    pts = " ".join(f"{x_of(w):.2f},{y_of(s):.2f}" for w, s, _st, _c in rows)
    q = y_of(float(cfg["quiet_score"]))
    a_ = y_of(float(cfg["active_score"]))

    ticks = ""
    hour = t0.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while hour < t1:
        hx = x_of(hour)
        ticks += (f'<line x1="{hx:.1f}" y1="0" x2="{hx:.1f}" y2="{TRACE_H}" '
                  f'class="tick"/><text x="{hx:.1f}" y="{TRACE_H + 14}" '
                  f'class="tlabel">{hour:%H:%M}</text>')
        hour += timedelta(hours=1)

    asleep = sum(s["duration"] for s in sessions if s["state"] == "asleep")
    awake = sum(s["duration"] for s in sessions if s["state"] == "awake")
    gap_total = sum(g["duration"] for g in gaps)
    sleeps = [s for s in sessions if s["state"] == "asleep"]
    stir_s = args.min_wake * 60.0
    real_wakes = [s for s in sessions
                  if s["state"] == "awake" and s["duration"] >= stir_s]
    stirs = [s for s in sessions
             if s["state"] == "awake" and s["duration"] < stir_s]

    def stat(label, value, sub=""):
        return (f'<div class="stat"><div class="k">{label}</div>'
                f'<div class="v">{value}</div><div class="s">{sub}</div></div>')

    stats = "".join([
        stat("asleep", human(asleep),
             f"{100.0 * asleep / (asleep + awake):.0f}% of monitored time"
             if asleep + awake else ""),
        stat("awake", human(awake), f"{len(real_wakes)} wakes"),
        stat("longest sleep",
             human(max((s["duration"] for s in sleeps), default=0)),
             f"{len(sleeps)} sessions"),
        stat("brief stirs", str(len(stirs)), f"under {args.min_wake:g} min"),
        stat("no data", human(gap_total) if gap_total else "none",
             f"{len(gaps)} gap(s)" if gaps else "continuous"),
    ])

    rows_html = "".join(
        f'<tr class="{st}"><td>{st if not (st == "awake" and d < stir_s) else "stir"}'
        f'</td><td>{a:%H:%M:%S}</td><td>{b:%H:%M:%S}</td>'
        f'<td class="num">{human(d)}</td></tr>'
        for a, b, st, d in sorted(spans, key=lambda s: s[0]))

    html = f"""<title>Sleep report {t0:%Y-%m-%d}</title>
<style>
  :root {{
    --bg:#fbfaf8; --fg:#1a1a1a; --muted:#6b6b6b; --line:#e2ded8; --card:#fff;
    --asleep:#3d5a80; --awake:#e8a33d; --unknown:#c9c4bc; --gap:#f0ece6;
    --trace:#98a8bd;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#16171a; --fg:#ececec; --muted:#9a9a9a; --line:#2c2e33; --card:#1e2024;
      --asleep:#7fa1cc; --awake:#f0b459; --unknown:#4a4d54; --gap:#232529;
      --trace:#5c6b80;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#16171a; --fg:#ececec; --muted:#9a9a9a; --line:#2c2e33; --card:#1e2024;
    --asleep:#7fa1cc; --awake:#f0b459; --unknown:#4a4d54; --gap:#232529;
    --trace:#5c6b80;
  }}
  body {{ background:var(--bg); color:var(--fg); margin:0; padding:32px 24px;
    font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1060px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; font-weight:650; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:26px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin-bottom:26px; }}
  .stat {{ background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:13px 15px; }}
  .stat .k {{ font-size:11px; text-transform:uppercase; letter-spacing:.07em;
    color:var(--muted); }}
  .stat .v {{ font-size:23px; font-weight:600; margin:3px 0 1px;
    font-variant-numeric:tabular-nums; }}
  .stat .s {{ font-size:12px; color:var(--muted); }}
  .chart {{ background:var(--card); border:1px solid var(--line);
    border-radius:10px; padding:18px; margin-bottom:26px; overflow-x:auto; }}
  svg {{ display:block; width:100%; height:auto; min-width:640px; }}
  .tick {{ stroke:var(--line); stroke-width:1; }}
  .tlabel {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
  .thr {{ stroke-dasharray:3 3; stroke-width:1; }}
  .legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px;
    color:var(--muted); margin-top:12px; }}
  .legend i {{ width:11px; height:11px; border-radius:2px; display:inline-block;
    vertical-align:-1px; margin-right:5px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px;
    font-variant-numeric:tabular-nums; }}
  th {{ text-align:left; font-weight:600; font-size:11px; text-transform:uppercase;
    letter-spacing:.07em; color:var(--muted); padding:0 10px 7px; }}
  td {{ padding:6px 10px; border-top:1px solid var(--line); }}
  td.num {{ text-align:right; }}
  tr.asleep td:first-child {{ color:var(--asleep); font-weight:600; }}
  tr.awake td:first-child {{ color:var(--awake); font-weight:600; }}
  tr.gap td {{ color:var(--muted); font-style:italic; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:20px;
    border-top:1px solid var(--line); padding-top:14px; }}
</style>
<div class="wrap">
  <h1>Sleep report</h1>
  <div class="sub">{t0:%A %d %B %Y, %H:%M} to {t1:%H:%M} &middot;
    {len(rows)} samples at {cfg['sample_seconds']:.0f}s intervals &middot;
    thresholds quiet {cfg['quiet_score']} / active {cfg['active_score']}</div>
  <div class="stats">{stats}</div>
  <div class="chart">
    <svg viewBox="0 0 {W} {BAND_H + TRACE_H + 30}"
         preserveAspectRatio="none" role="img">
      <g>{bands}</g>
      <g transform="translate(0,{BAND_H + 10})">
        {ticks}
        <line x1="0" y1="{q:.1f}" x2="{W}" y2="{q:.1f}" class="thr"
              stroke="var(--asleep)"/>
        <line x1="0" y1="{a_:.1f}" x2="{W}" y2="{a_:.1f}" class="thr"
              stroke="var(--awake)"/>
        <polyline points="{pts}" fill="none" stroke="var(--trace)"
                  stroke-width="1" vector-effect="non-scaling-stroke"/>
      </g>
    </svg>
    <div class="legend">
      <span><i style="background:var(--asleep)"></i>asleep</span>
      <span><i style="background:var(--awake)"></i>awake</span>
      <span><i style="background:var(--unknown)"></i>warming up</span>
      <span><i style="background:var(--gap)"></i>no data</span>
      <span>lower trace: raw motion score on a sqrt scale, 0 to
        {cap:.2f}; dashed lines are the quiet and active thresholds</span>
    </div>
  </div>
  <table>
    <thead><tr><th>state</th><th>from</th><th>to</th>
      <th style="text-align:right">duration</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="note">
    This measures stillness, not sleep. A motionless awake dog reads as asleep,
    and an empty pen reads as asleep. Gaps are stretches where the monitor was
    not running and are never bridged.
  </div>
</div>
"""
    out = os.path.join(HERE, args.html)
    with open(out, "w") as fh:
        fh.write(html)
    return out


# --- serve -------------------------------------------------------------------

def iso(when):
    """UTC ISO-8601 with a Z suffix, which Swift's .iso8601 strategy decodes."""
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def consolidate_sleep(sessions, gaps, merge_s):
    """Merge asleep runs separated by a brief stir into one sleep span.

    A dog-health log wants "slept 7h04m with 5 stirs", not 6 separate sleep
    entries. A long awake span closes the sleep. A data gap also closes it and
    marks it partial, because the end time is then unknown: the monitor was not
    watching when it actually ended.
    """
    spans = sorted(
        [(s["start"], s["end"], s["state"], s["duration"]) for s in sessions]
        + [(g["start"], g["end"], "gap", g["duration"]) for g in gaps],
        key=lambda t: t[0])

    out, cur = [], None
    for start, end, state, duration in spans:
        if state == "asleep":
            if cur is None:
                cur = {"start": start, "end": end, "stirs": 0, "partial": False}
            else:
                cur["end"] = end
                cur["stirs"] += 1
        elif state == "gap":
            if cur is not None:
                cur["partial"] = True
                out.append(cur)
                cur = None
        elif state == "awake" and cur is not None and duration > merge_s:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def sleep_events(cfg, since=None, min_minutes=None, merge_minutes=None):
    """Completed sleep sessions, shaped so Swift decodes them into PupEvent.

    Extra keys (source, partial, stirs, duration_s) are ignored by a synthesized
    Codable decoder, so the same payload serves a plain PupEvent client and a
    richer one.

    The id is derived from the start instant, so pulling the same session twice
    yields the same id. PupLog's conflict rule for events is set union by id,
    which makes repeated pulls idempotent with no cursor bookkeeping required.
    """
    min_s = 60.0 * (float(cfg["api_min_session_minutes"])
                    if min_minutes is None else float(min_minutes))
    merge_s = 60.0 * (float(cfg["api_merge_stirs_minutes"])
                      if merge_minutes is None else float(merge_minutes))

    rows = read_log(cfg)
    sessions, gaps = sessionize(rows, float(cfg["sample_seconds"]))
    events = []
    for sp in consolidate_sleep(sessions, gaps, merge_s):
        duration = (sp["end"] - sp["start"]).total_seconds()
        if duration < min_s:
            continue
        if since is not None and sp["end"] <= since:
            continue
        events.append({
            "id": f"c-{int(sp['start'].timestamp() * 1000)}",
            "kind": "sleep",
            "ts": iso(sp["start"]),
            "start": iso(sp["start"]),
            "end": iso(sp["end"]),
            "source": "camera",
            "partial": sp["partial"],
            "stirs": sp["stirs"],
            "duration_s": int(duration),
        })
    return events


def live_state(cfg):
    """Current state and when it began, for an in-progress sleep banner."""
    rows = read_log(cfg)
    if not rows:
        return {"state": "unknown", "since": None, "elapsed_s": 0,
                "last_sample": None, "stale": True}
    sessions, _gaps = sessionize(rows, float(cfg["sample_seconds"]))
    last = sessions[-1] if sessions else None
    age = (datetime.now() - rows[-1][0]).total_seconds()
    return {
        "state": last["state"] if last else "unknown",
        "since": iso(last["start"]) if last else None,
        "elapsed_s": int((datetime.now() - last["start"]).total_seconds())
                     if last else 0,
        "last_sample": iso(rows[-1][0]),
        "last_score": rows[-1][1],
        # The monitor may have died while this server keeps answering. Say so
        # rather than letting a client believe a four-hour-old state is current.
        "stale": age > 4 * float(cfg["sample_seconds"]),
        "stale_for_s": int(age) if age > 4 * float(cfg["sample_seconds"]) else 0,
    }


def make_handler(cfg, token):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "dog-sleep-monitor/1"
        protocol_version = "HTTP/1.1"

        def _send(self, code, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            header = self.headers.get("Authorization", "")
            prefix = "Bearer "
            if not header.startswith(prefix):
                return False
            # Constant-time, so a wrong token cannot be discovered byte by byte
            # from response timing.
            return hmac.compare_digest(header[len(prefix):], token)

        def log_message(self, fmt, *a):
            # Default logs the full request line. Keep it, but never headers.
            sys.stderr.write(f"{stamp()}  {self.address_string()}  {fmt % a}\n")

        def do_POST(self):
            self._send(405, {"error": "read-only server"})

        do_PUT = do_DELETE = do_PATCH = do_POST

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = urllib.parse.parse_qs(parsed.query)

            if path == "/health":
                # Unauthenticated on purpose: proves the process is alive and
                # nothing more. No dog data, no config, no version of anything
                # an attacker could pivot on.
                return self._send(200, {"ok": True})

            if not self._authorized():
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Bearer realm="monitor"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if path == "/v1/state":
                return self._send(200, live_state(cfg))

            if path == "/v1/events":
                since = None
                raw = (query.get("since") or [None])[0]
                if raw:
                    try:
                        since = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        if since.tzinfo:
                            since = since.astimezone().replace(tzinfo=None)
                    except ValueError:
                        return self._send(400, {"error": "since must be ISO-8601"})
                try:
                    events = sleep_events(
                        cfg, since=since,
                        min_minutes=(query.get("min_minutes") or [None])[0],
                        merge_minutes=(query.get("merge_minutes") or [None])[0])
                except ValueError:
                    return self._send(400, {"error": "bad numeric parameter"})
                return self._send(200, {
                    "events": events,
                    "count": len(events),
                    "server_time": iso(datetime.now()),
                    "state": live_state(cfg),
                })

            return self._send(404, {"error": "not found"})

    return Handler


def cmd_serve(cfg, args):
    from http.server import ThreadingHTTPServer

    token = os.environ.get(cfg["api_token_env"])
    if not token or len(token) < 24:
        raise SystemExit(
            f"Set {cfg['api_token_env']} to a long random value before serving.\n\n"
            f"  echo \"{cfg['api_token_env']}=$(openssl rand -hex 32)\" >> "
            f"{ENV_PATH}\n\n"
            "Refusing to start without one: this endpoint reports when someone's\n"
            "dog is unattended, which is not something to serve unauthenticated."
        )

    host = args.bind or cfg["api_bind"]
    port = int(args.port or cfg["api_port"])
    httpd = ThreadingHTTPServer((host, port), make_handler(cfg, token))
    print(f"Serving on http://{host}:{port}")
    print("  GET /health              no auth, liveness only")
    print("  GET /v1/state            current state")
    print("  GET /v1/events?since=    completed sleep sessions as PupEvent JSON")
    print("Auth: Authorization: Bearer <token>. Read-only; POST/PUT/DELETE "
          "return 405.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\nNote: bound to {host}, so this is reachable beyond this machine.\n"
              "Plain HTTP means the token crosses the network in the clear. Put it\n"
              "on a tailnet (or `tailscale serve --https`) rather than a bare LAN.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


# --- purge -------------------------------------------------------------------

PURGE_TARGETS = {
    "archive": ("archive_dir", "every-sample frame archive"),
    "snapshots": ("snapshot_dir", "event snapshots"),
}


def cmd_purge(cfg, args):
    """Delete captured image data. Reports first, deletes only with --yes."""
    plan = []
    for name in (args.targets or list(PURGE_TARGETS)):
        if name not in PURGE_TARGETS:
            raise SystemExit(f"Unknown target {name!r}. "
                             f"Choose from: {', '.join(PURGE_TARGETS)}")
        key, desc = PURGE_TARGETS[name]
        d = os.path.join(HERE, cfg[key])
        files = [f for f in os.listdir(d)
                 if os.path.isfile(os.path.join(d, f))] if os.path.isdir(d) else []
        plan.append((name, d, desc, files, dir_size_mb(d)))

    total_files = sum(len(f) for _, _, _, f, _ in plan)
    total_mb = sum(mb for _, _, _, _, mb in plan)

    print("Would delete:")
    for name, d, desc, files, mb in plan:
        oldest = min(files) if files else "-"
        newest = max(files) if files else "-"
        print(f"  {name:<11} {len(files):>6} files  {mb:>8.1f} MB  ({desc})")
        if files:
            print(f"              oldest {oldest}\n              newest {newest}")
    print(f"  {'TOTAL':<11} {total_files:>6} files  {total_mb:>8.1f} MB")

    if total_files == 0:
        print("\nNothing to delete.")
        return
    if not args.yes:
        print("\nNothing deleted. Re-run with --yes to actually delete.")
        return

    removed = 0
    for _, d, _, files, _ in plan:
        for f in files:
            try:
                os.remove(os.path.join(d, f))
                removed += 1
            except OSError as exc:
                print(f"  could not remove {f}: {exc}")
    print(f"\nDeleted {removed} files, freed {total_mb:.1f} MB.")
    print("Kept: sleep_log.csv, calibration.csv, events.csv "
          "(the numbers, which are small and worth keeping).")


# --- watch -------------------------------------------------------------------

def cmd_watch(cfg, args):
    interval = float(cfg["sample_seconds"])
    quiet_needed = int(cfg["quiet_samples_to_sleep"])
    active_needed = int(cfg["active_samples_to_wake"])
    log_path = os.path.join(HERE, cfg["log_path"])

    print(f"Watching via {cfg['source']}. Sample every {interval:.0f}s. "
          f"asleep after {quiet_needed} quiet samples "
          f"({quiet_needed * interval / 60:.1f} min), "
          f"awake after {active_needed}.")
    print(f"Logging to {log_path}. Ctrl-C to stop.\n")

    log_is_new = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if log_is_new:
        writer.writerow(["timestamp", "score", "state", "changed"])

    events_path = os.path.join(HERE, cfg["events_path"])
    events_is_new = not os.path.exists(events_path)
    events = open(events_path, "a", newline="")
    ev_writer = csv.writer(events)
    if events_is_new:
        ev_writer.writerow(["timestamp", "kind", "score", "image"])

    pruned = prune_snapshots(cfg)
    if pruned:
        print(f"Pruned {pruned} snapshots older than "
              f"{cfg['snapshot_retention_days']} days.")
    snap_every = float(cfg["snapshot_every_minutes"]) * 60.0
    next_snap = time.monotonic() + snap_every if snap_every > 0 else float("inf")

    def record(kind, score, frame):
        """Log an event and, if enabled, the picture behind it."""
        path = ""
        if cfg["snapshot_on_event"] and frame is not None:
            path = os.path.relpath(save_snapshot(frame, cfg, kind, score), HERE)
        ev_writer.writerow([stamp(), kind, f"{score:.6f}", path])
        events.flush()
        return path

    archive = FrameArchive(cfg)
    if archive.enabled:
        print(f"Archiving every sample to {cfg['archive_dir']}/ "
              f"(~39MB/hour, hard cap {archive.cap_mb:.0f}MB, "
              f"{archive.used_mb:.0f}MB already there).")

    cam = open_camera(cfg)
    machine = SleepState(cfg)
    prev = None
    next_at = time.monotonic()
    taken = 0
    limit = int(getattr(args, "samples", 0) or 0)

    try:
        while limit == 0 or taken < limit:
            next_at = pace(next_at, interval)
            taken += 1
            try:
                cur, raw = grab_sample(cam, cfg, keep_raw=True)
            except RuntimeError as exc:
                # Feed hiccup. The reader thread reconnects on its own; drop the
                # stale reference so the next good frame starts a fresh pair.
                print(f"{stamp()}  feed: {exc}")
                record("feed-drop", 0.0, None)
                prev = None
                continue

            if prev is None:
                prev = cur
                continue

            score = motion_score(prev, cur, cfg["pixel_threshold"])
            corr = correlation(prev, cur)
            prev = cur

            state, tag, changed = machine.update(score, corr)
            archive.save(raw, score)

            if time.monotonic() >= next_snap:
                record("periodic", score, raw)
                next_snap = time.monotonic() + snap_every

            if tag == "scene":
                writer.writerow([stamp(), f"{score:.6f}", state, "scene-change"])
                log.flush()
                shot = record("scene-change", score, raw)
                print(f"{stamp()}  score {score:.4f}  SCENE CHANGE, counters reset"
                      + (f"  [{shot}]" if shot else ""))
                continue

            writer.writerow([stamp(), f"{score:.6f}", state, "yes" if changed else ""])
            log.flush()

            if changed:
                shot = record(state, score, raw)
                print(f"{stamp()}  --> {state.upper()}"
                      + (f"  [{shot}]" if shot else ""))
            elif cfg["print_every_sample"]:
                print(f"{stamp()}  score {score:.4f}  {tag:<6} "
                      f"quiet {machine.quiet_run}/{quiet_needed}  state {state}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cam.close()
        log.close()
        events.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preview", help="test the connection and set the ROI")
    p.add_argument("--pick-roi", action="store_true",
                   help="drag a box around the playpen (needs a display)")

    p = sub.add_parser("calibrate", help="collect motion scores, suggest thresholds")
    p.add_argument("--label", choices=["quiet", "active"], required=True)
    p.add_argument("--seconds", type=float, default=180.0)

    p = sub.add_parser("watch", help="run the monitor")
    p.add_argument("--samples", type=int, default=0,
                   help="stop after N samples (0 = run forever)")

    p = sub.add_parser("label", help="tag a window of the live log quiet/active")
    p.add_argument("--from", dest="start", required=True, help="HH:MM, HH:MM:SS or ISO")
    p.add_argument("--to", dest="end", required=True, help="HH:MM, HH:MM:SS or ISO")
    p.add_argument("--label", choices=["quiet", "active"], required=True)

    sub.add_parser("tune", help="grid-search thresholds against labeled data")

    p = sub.add_parser("serve", help="read-only JSON API for the iOS app")
    p.add_argument("--bind", help="interface to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8787)")

    p = sub.add_parser("report", help="summarize the log into sleep sessions")
    p.add_argument("--from", dest="start", help="HH:MM or ISO; default all")
    p.add_argument("--to", dest="end", help="HH:MM or ISO; default all")
    p.add_argument("--min-wake", type=float, default=2.0,
                   help="awake spans shorter than this many minutes are stirs")
    p.add_argument("--all", action="store_true", help="list every session")
    p.add_argument("--html", nargs="?", const="report.html",
                   help="also write a visual report (default report.html)")

    p = sub.add_parser("purge", help="delete captured images (reports unless --yes)")
    p.add_argument("targets", nargs="*", choices=["archive", "snapshots"],
                   help="what to delete; default both")
    p.add_argument("--yes", action="store_true", help="actually delete")
    sub.add_parser("doctor", help="check network, credentials, stream, ROI")

    args = ap.parse_args()
    load_env()
    cfg = load_config()
    {"preview": cmd_preview,
     "calibrate": cmd_calibrate,
     "watch": cmd_watch,
     "label": cmd_label,
     "tune": cmd_tune,
     "purge": cmd_purge,
     "report": cmd_report,
     "serve": cmd_serve,
     "doctor": cmd_doctor}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
