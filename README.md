# dog-sleep-monitor

Detects whether the dog in the playpen is still or moving, using frame
differencing from a fixed camera. No model, no training, no dataset.

Works with a TP-Link Tapo (or any RTSP camera) over the network, and with USB
webcams. Set `source` in `config.json` to `"rtsp"` or `"usb"`.

It reports **stillness**, which for a fixed camera on a playpen is a decent
proxy for sleep. It cannot tell "asleep" from "awake but not moving", and it
cannot tell "asleep" from "pen is empty". Those need a model.

## Setup

```bash
cd ~/Projects/dog-sleep-monitor
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On the Pi, if `pip install opencv-python` stalls building from source:

```bash
sudo apt install -y python3-opencv python3-numpy ffmpeg
```

...then run `python3 monitor.py` directly instead of using the venv.

## This camera

Developed against a Tapo C100 (hardware 5.0, firmware 1.5.4). Put your own
camera's LAN address in `config.json` under `rtsp.host`.

- 1080p on `/stream1`, 640x360 on `/stream2`. This project uses stream2, since
  640x360 is far more than the math needs and costs nothing to decode.
- 2.4 GHz Wi-Fi only, so RTSP stability tracks the 2.4 GHz signal at the pen.
- IR night vision, which is good news. See the RTSP specifics below.
- No pan or tilt, so the fixed camera angle this whole approach depends on is
  guaranteed by the hardware.

Give it a DHCP reservation in your router. Otherwise the IP changes on some
future lease renewal and the monitor silently stops finding it.

## Tapo setup, in order

### 1. Create a Camera Account in the Tapo app

Device Settings > Advanced Settings > Camera Account. Set a username and
password there.

This is **not** your TP-Link account login, and it is not optional: a Tapo
camera keeps its RTSP server switched off until this account exists. Until then
port 554 is closed and nothing can connect.

### 2. Put the credentials in `.env`

```bash
cp .env.example .env
```

Then fill in the two values. `.env` is gitignored, so the password never reaches
`config.json` or a commit. Special characters are fine, they get URL-escaped.

### 3. Check everything at once

```bash
.venv/bin/python monitor.py doctor
```

Checks port 554, then credentials, then pulls an actual frame, then reports ROI
coverage. It stops at the first failure and names the cause rather than making
you guess.

If 554 is still closed after creating the Camera Account, update the camera
firmware in the Tapo app and try again.

### 4. Optional sanity check outside Python

```bash
ffplay "rtsp://USER:PASS@YOUR_CAMERA_IP:554/stream2"
```

## Three steps to running

### 1. Connect and crop to the playpen

```bash
.venv/bin/python monitor.py preview --pick-roi
```

Doubles as the connection test. Drag a box around the playpen, press ENTER. It
saves the box to `config.json` and writes `preview.jpg` so you can confirm the
framing.

**Do not skip the ROI.** With the full frame, the dog is a small percentage of
the pixels, so real movement produces a score barely above sensor noise.
Cropping to the pen multiplies the signal by roughly 3.5x. `test_math.py` prints
both columns so you can see it.

The ROI here is set to the whole visible pen, 67% of the frame, because Bailey
sleeps anywhere she likes and not just in the crate. That is a deliberate
trade: covering more area dilutes the score, but missing a dog asleep on the
left side would report her as awake, and missing one awake outside the crate
would report her as asleep. Coverage beats sensitivity here.

**Changing the camera angle invalidates calibration data.** The score is a
fraction of the ROI that changed, so a different ROI is a different scale. The
camera was re-aimed on 2026-08-12 to bring her whole body into frame; the
pre-move labels were set aside as `calibration.old-angle.csv` rather than
reused. Archive frames from before the move are still on disk and separable by
timestamp.

The one hard requirement is that the ROI must exclude the burnt-in timestamp
strip along the top. Measured on its own, the ticking digits score 0.0365 mean
and 0.0599 max, above the movement threshold, forever. The current ROI starts at
y=48px for that reason.

The ROI is stored as **fractions** of the frame, not pixels, so switching
between stream1 and stream2 does not invalidate it.

On a headless Pi there is no display for the picker. Run `preview --pick-roi` on
your Mac against the same camera and copy the `roi` line into the Pi's
`config.json`.

### 2. Calibrate the thresholds

The log records raw scores, which are threshold-independent, so you can leave
`watch` running and label windows of it afterwards. That avoids opening a second
RTSP connection to a camera that is already streaming:

```bash
.venv/bin/python monitor.py label --from 22:10 --to 23:40 --label quiet
.venv/bin/python monitor.py label --from 07:15 --to 07:30 --label active
```

Then grid-search thresholds against that labeled data:

```bash
.venv/bin/python monitor.py tune
```

`tune` replays each labeled sequence through the real `SleepState` machine and
counts **false wakes**, **false sleeps**, and wake latency. That is the metric
that matters, not per-sample separability: a rowdy dog pauses and a quiet scene
spikes, so individual samples overlap even when runs of 12 never do.

`calibrate --label quiet|active --seconds N` still exists for collecting a fresh
window directly, when nothing else is holding the camera.

**Label honestly.** `tune` optimizes against whatever labels it is given, so a
wrong label produces a confidently wrong threshold. An empty pen is not a
sleeping dog.

That said, one worry turned out to be unfounded here. The assumption was that a
sleeping dog's breathing would put its quiet floor well above an empty room's.
Measured on a settled dog at this ROI size, the scores were 0.0000, 0.0000,
0.0010, 0.0000, 0.0000. Breathing does not register when the chest is a tiny
fraction of a 930x667 region. Expect that to change if you crop tightly to a
sleeping spot.

### 3. Watch

```bash
.venv/bin/python monitor.py watch
```

Prints a line per sample, appends to `sleep_log.csv`
(`timestamp, score, state, changed`). Ctrl-C to stop.

## The noise floor moves, so calibrate long

Measured on this camera, empty pen, identical ROI and threshold each time:

| time | mean score | max |
| --- | --- | --- |
| 14:48 | 0.0040 | 0.0163 |
| 14:53 | 0.0404 | 0.2021 |
| 15:05 | 0.0000 | 0.0000 |

A 10x swing with nothing in the room moving. A diff heatmap over that window
showed diffuse speckle across every textured surface and no localized hot spot,
so it is sensor and h264 noise rising as ambient light drops and the camera
raises gain, not anything real.

Two consequences:

1. **A threshold calibrated over 30 seconds is worthless.** Calibrate across
   hours, ideally spanning the day/night transition.
2. **Each sample averages `frames_per_sample` frames** (default 5, 200ms apart).
   Noise is independent between frames so averaging cuts it by roughly sqrt(n),
   while a sleeping dog does not move over that second, so the signal survives.
   This buys sensitivity that raising `pixel_threshold` would have spent.
   Honest caveat: the floor happened to be back at zero when this was added, so
   the improvement is theoretically sound but not yet demonstrated on real
   noisy footage.

The way to get real thresholds is an overnight `watch` run with the dog in the
pen, then read the score distribution out of `sleep_log.csv`.

## How it works

Per sample, four operations on the cropped region:

1. Grayscale, so color noise cannot register as motion.
2. Downscale to 64x48, which averages sensor grain away.
3. Gaussian blur, for the grain that survived step 2.
4. Normalize by mean and standard deviation. **This is the important one.** It
   removes global brightness and gain shifts, so a camera re-tuning its own
   exposure does not read as the whole room moving. Verified in `test_math.py`:
   a +40 brightness jump and a 1.3x gain jump both score 0.0000.

The score is the fraction of pixels that changed by more than
`pixel_threshold`. Then a state machine:

- above `scene_change_score` (0.60): the whole scene changed, so resync and
  count it as nothing
- below `quiet_score` (0.010): still
- above `active_score` (0.030): movement
- **in between: neither**, a deadband that stops the state flapping
- asleep after 12 consecutive still samples (1 minute), awake after 2

Waking is deliberately fast to detect and sleeping deliberately slow, since a
dog holding still for 10 seconds is common and a dog still for a minute is not.

## RTSP specifics

**A background thread drains the stream.** An RTSP feed cannot be sampled on
demand: read it once every 5 seconds and you get frames out of the socket
backlog rather than from now, so every comparison looks still. The reader thread
consumes continuously and keeps only the newest frame. It also reconnects by
itself, because a Wi-Fi camera watched for eight hours will drop at some point.

**Transport is forced to TCP.** RTSP over UDP loses packets on Wi-Fi and
produces torn frames, which read as motion.

**Night is handled, the switch into it is the problem.** Tapo IR night vision is
a stable, evenly lit grayscale image, which is easier for this than a dim
ambient-light webcam. But the moment the IR cut filter drops out, every pixel in
the frame changes at once. In testing that scores 0.97 while the largest real
dog movement scores 0.33, so `scene_change_score` at 0.60 separates them
cleanly and you get a logged `scene-change` instead of a false wake at dusk and
dawn.

## Tuning

| Symptom | Fix |
| --- | --- |
| Always reads awake | Tighten the ROI to just the pen, then raise `quiet_score`. |
| Always reads asleep | Lower `active_score`. Check `preview.jpg` actually frames the dog. |
| Flaps between states | Widen the gap between `quiet_score` and `active_score`. |
| Slow to notice waking | Set `active_samples_to_wake` to 1. |
| `scene-change` firing often | Raise `scene_change_score`, or stop the camera from being nudged. |
| Feed drops constantly | Move to `/stream2`, or improve the camera's Wi-Fi signal. |

## Reading the results

```bash
.venv/bin/python monitor.py report                 # terminal summary
.venv/bin/python monitor.py report --html          # also writes report.html
.venv/bin/python monitor.py report --from 22:00 --to 08:00 --all
```

`report` collapses per-sample rows into **sessions**: runs of one state with a
start, an end, a duration, and the mean and max score over the run. On top of
that:

- **Stirs vs wakes.** An awake span shorter than `--min-wake` (default 2 min) is
  a stir, not a wake. Counted separately and hidden from the table unless
  `--all`, because "she woke 14 times" and "she stirred 12 times and woke twice"
  describe the same night very differently.
- **Gaps are never bridged.** A stretch with no samples means the monitor was not
  running: laptop asleep, process dead, feed down. Bridging two sleep sessions
  across a four-hour outage would invent four hours of sleep, so gaps are their
  own row type and are excluded from the percentages.
- **A high max on an asleep session is normal.** Waking needs 2 consecutive
  samples to confirm, so the first movement sample is still filed under the
  previous state. That max is the movement that ended the session.

`--html` writes a self-contained page: stat cards, a hypnogram band, and the raw
score trace beneath it with the thresholds drawn in. The trace uses a sqrt scale
because scores span three orders of magnitude, from 0.0000 to 0.82, and a linear
axis either clips the peaks or flattens the sleep periods into one pixel row.

## Feeding an app

```bash
echo "MONITOR_API_TOKEN=$(openssl rand -hex 32)" >> .env
.venv/bin/python monitor.py serve --bind 100.x.y.z
```

A read-only JSON API. `GET /v1/events` returns completed sleep sessions with
data-derived ids, so repeated pulls are idempotent; `GET /v1/state` returns the
current state with a `stale` flag so a client cannot mistake a dead monitor for a
sleeping dog. Bearer-token auth, constant-time compared, and the server refuses
to start without a token. Anything other than GET returns 405.

See [INTEGRATION.md](INTEGRATION.md) for the full contract, the reasoning behind
pulling rather than pushing, and the security model.

## Stored data

Numbers are small and permanent. Images are large and temporary.

| file or dir | what | notes |
| --- | --- | --- |
| `sleep_log.csv` | timestamp, score, state, transition, every 5s | keep, flushed per sample |
| `events.csv` | timestamp, kind, score, image path | keep, the index into snapshots |
| `calibration.csv` | labeled score windows | keep |
| `snapshots/` | one jpeg per state change, scene change, and every 15 min | ~54KB each, pruned after `snapshot_retention_days` |
| `archive/` | one jpeg per **sample** when `archive_all_samples` is on | ~56KB each, ~40MB/hour |
| `watch.out` | console output | appended across restarts |

Snapshot and archive filenames are `TIMESTAMP_kind_score.jpg` and
`TIMESTAMP_score.jpg`, so they sort chronologically and join to `sleep_log.csv`
on the timestamp without needing a schema change.

### Frame archive

Meant for capturing a session, having something review it offline, then throwing
it away. Off by default. `archive_max_mb` (3000) is a hard stop: archiving
switches itself off and says so, while monitoring continues uninterrupted. At
40MB/hour that cap is about 75 hours.

### Disposal

```bash
.venv/bin/python monitor.py purge              # reports, deletes nothing
.venv/bin/python monitor.py purge archive --yes
```

Reports file counts, total MB, and the oldest and newest filenames, then deletes
only with `--yes`. Never touches the CSVs, since the numbers are what you keep.

## Tests

```bash
.venv/bin/python test_math.py && .venv/bin/python test_policy.py
```

Both camera-free.

- `test_math.py` (10 cases): synthetic room, noise immunity, exposure and gain
  immunity, movement detection, the IR scene-change guard, and resolution
  independence of the fractional ROI.
- `test_policy.py` (12 cases): the `SleepState` machine. Pins the 12-quiet and
  2-active run lengths, that the deadband advances nothing but preserves a run
  across itself, and that a scene change resets both counters rather than
  registering as movement.

## Measured on this setup, 2026-08-12

| what | value |
| --- | --- |
| empty pen, whole-pen ROI | p50 0.0000, p95 0.0169, max 0.0521 |
| Bailey awake and rowdy | p50 0.0495, p95 0.1949, max 0.3421 |
| ticking timestamp strip alone | mean 0.0365, max 0.0599 |
| day to night IR switch (synthetic) | 0.9740 |
| wake detection latency | 2 samples, 10s |

`active_score` was raised from 0.030 to 0.040 because the empty-pen data
contained exactly one pair of consecutive samples above 0.030 (0.0521 then
0.0358), which produced one false wake. 0.040 clears that pair and still catches
Bailey in 2 samples. `quiet_score` remains at the unvalidated default of 0.010,
pending a labeled window of an actually sleeping dog.

## Moving to the Pi 5

Nothing about the camera changes, since it is on the network rather than plugged
into anything. The Pi only needs to be on the same LAN.

```bash
sudo apt install -y python3-opencv python3-numpy ffmpeg
```

Then copy `monitor.py`, `config.json`, and `.env` across, and run
`python3 monitor.py doctor` to confirm the Pi can reach the camera.

The one thing that does not work headless is `preview --pick-roi`, which needs a
display. Pick the ROI on the Mac and copy the `roi` line from `config.json`. It
is stored as frame fractions, so it transfers as-is.

Resource use: decoding 640x360 h264 continuously is a few percent of one core,
plus about 60 MB of RAM. The 16 GB is irrelevant to this workload.

For running it unattended across reboots you want a systemd service. Not
included here on purpose; ask for it when you are ready.
