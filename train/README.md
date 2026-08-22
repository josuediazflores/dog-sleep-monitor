# Training the presence classifier

A dog/no-dog classifier for the pen ROI. Trained here on the Mac from the
monitor's own archived frames, exported to ONNX, and run on the Pi through the
`python3-opencv` that is already installed there. **The Pi never installs
anything for this.** That constraint is why the pipeline ends at an ONNX file
instead of a checkpoint.

## Why, in one paragraph

Presence started as reference-differencing: keep a few frames of the empty pen,
diff the current frame against the closest one, call it occupied if enough
pixels moved. It works until the pen is rearranged, and then it does not
degrade, it inverts. A toy dragged three inches produces the same contiguous
blob a curled-up dog does. Measured overnight on this camera, an
empty-but-rearranged pen scored 0.045 while a sleeping dog scored 0.023, so
every threshold that calls the first one empty calls the second one empty too.
No cutoff fixes an ordering. The reference layer answers "does this frame
differ from an empty pen", and the actual question is "is there a dog in this
picture", which is what a classifier answers and what it does not care about
the blanket for.

References stay as the fallback. With `presence_model` unset the monitor
behaves exactly as it did before any of this existed.

## Setup, once

```bash
uv venv --python 3.12 train/.venv
uv pip install --python train/.venv/bin/python -r train/requirements-train.txt
```

Mac only. Nothing here is pinned on purpose; the versions that produced a given
model are recorded in the sidecar json next to it.

## The workflow

### 1. Pull the data off the Pi

```bash
./train/pull.sh                      # or: PI=josue@10.0.0.5 ./train/pull.sh
```

Rsyncs the archive, `sleep_log.csv`, `events.csv`, `presence_labels.csv`,
`markers.csv` and `config.json` into `.local/pi/`. Read-only on the Pi: nothing
is pushed, nothing under `~/dog-sleep-monitor` is touched, and the live monitor
keeps running throughout.

`.local/` is gitignored, and that matters more than it sounds. The archive is
pictures of the inside of a home and this repo is public.

The Pi only has frames if `archive_all_samples` is on. At every-sample
archiving it writes roughly 39 MB an hour and prunes to `archive_max_mb`, so
pull before the ring buffer laps the window you meant to label.

### 2. Label windows

You are claiming that a stretch of the archive had a dog in it, or did not.
Look at the frames first: `snapshots/`, the archive itself, or
`monitor.py report --html` to find the stretches worth looking at.

```bash
python monitor.py label-presence --from 2026-08-20T22:00 --to 2026-08-21T06:00 \
    --label dog   --notes "overnight, crate door shut"
python monitor.py label-presence --from 08:00 --to 09:30 \
    --label empty --notes "morning walk, pen empty the whole time"
```

`--from` / `--to` take `HH:MM`, `HH:MM:SS` or a full ISO timestamp; bare times
mean today. Windows append to `presence_labels.csv` at the repo root
(gitignored, like every other csv here). Nothing is merged or deduplicated: a
later window overrides an earlier one where they overlap, so correcting a
mistake means labeling the smaller window again rather than editing the file.
That is the vet-trip case -- label the whole night `dog`, then label the twenty
minutes she was out `empty` on top of it.

Label honestly and label boring stretches. The empty class is the one that
decides whether an empty pen ever reads as sleep again, and eight hours of a
motionless empty pen at 3am is exactly the data the reference layer could not
use.

If the labels were written on the Pi, `pull.sh` brought them down to
`.local/pi/presence_labels.csv`; copy them into place or re-record them here.

### 3. Export a manifest

```bash
python monitor.py dataset --archive .local/pi/archive --labeled-only
```

Writes `dataset/manifest.csv`: one row per sampled frame, with
`label_dog_present` filled in from the windows above and a `split` column.

- `--archive DIR` reads frames from somewhere other than the configured
  `archive_dir`. Needed here because the Mac's own `archive/` was shot at a
  different camera angle and would poison the manifest.
- `--labeled-only` drops frames outside every window, so the `--limit` budget
  is spent on frames that can actually be trained on.
- `--labels FILE` points at a different label file. For throwaway label sets;
  the real one should not be moved around.

**Do not re-split the rows.** The `split` column is assigned by time block, not
by frame, and that is deliberate: consecutive frames five seconds apart are
nearly identical, so a random per-frame split puts near-duplicates on both
sides of it and reports a val accuracy that has nothing to do with tomorrow
night. Read `cmd_dataset`'s docstring before changing any of it.

Also do not train on `weak_state`. It is the monitor's own asleep/awake
decision, derived from the very thresholds a model would be used to check.

### 4. Train

```bash
python train/train_presence.py \
    --manifest dataset/manifest.csv \
    --frames-root . \
    --epochs 20 --width 1.0 \
    --out models/presence.onnx
```

MobileNetV2 on ImageNet weights, one sigmoid output. Defaults: `--input
160x128`, `--lr 1e-4` AdamW with cosine decay, `--batch 32`, early stopping on
val loss with `--patience 5`, best epoch kept. `--device auto` picks MPS when
it is there.

- `--frames-root` is what the manifest's `file` column is relative to. The
  exporter writes paths relative to the repo root, so `.` is usually right.
- `--width 0.5` is about 4x cheaper and has **no pretrained weights** in
  torchvision, so it trains from random init and needs far more labeled data.
  Use 1.0 unless the Pi latency actually demands otherwise, which at a 5 second
  sample interval it does not.
- Empty frames will outnumber dog frames. Handled by class-balanced sampling
  rather than by discarding the majority class, because every one of those
  frames is real evidence of what an empty pen looks like at that hour.
- Augmentation is train-only: random resized crop 0.85-1.0 keeping aspect,
  small translation, brightness and contrast jitter, light gaussian noise,
  horizontal flip. Eval runs the plain pipeline untouched, because what the
  model sees at eval has to match what cv2.dnn feeds it on the Pi.

It reports accuracy, per-class precision and recall, a confusion matrix and a
histogram of p for **both** val and test. Read the histogram, not just the
accuracy: a model that is right but clustered around 0.5 will abstain on every
sample once the deadband is applied, and a model whose mass sits at the two ends
is one that will actually vote. `frac_in_deadband` is that number directly.

The suggested threshold in the sidecar is picked on val for best balanced
accuracy. It is a suggestion; `presence_model_threshold` in the Pi's config is
what the runtime uses.

### 5. Verify

```bash
python train/verify_onnx.py --model models/presence.onnx \
    --frame .local/pi/archive/2026-08-21T03-11-02_0.0000.jpg
```

Checks three things, each of which has been a real way to ship a broken model:

1. `train/preprocess.py` and `monitor.model_blob` produce a bit-identical
   tensor. They are separate implementations by necessity -- the Pi has no
   torch and `monitor.py` can never import from `train/` -- so nothing else
   holds them in lockstep, and a one-pixel difference in the crop moves p by
   more than the deadband.
2. onnxruntime and cv2.dnn agree to 1e-4. Old cv2 can load a graph, run it,
   and return a confidently different number.
3. Latency, printed rather than asserted.

This runs against the Mac's newer cv2, so it settles the numerical questions
but **not** "does the Pi's cv2 4.6 support these ops". Only the Pi answers
that.

### 6. Prove the Pi can load it

Before deploying, and after any change to the export. Copy the model and one
frame to the Pi's `/tmp`, and run the graph there with the system python:

```bash
scp models/presence.onnx models/presence.json josue@PI:/tmp/
scp .local/pi/archive/SOME-FRAME.jpg josue@PI:/tmp/frame.jpg
ssh josue@PI 'python3 - <<PY
import cv2, json, time, numpy as np
size = tuple(json.load(open("/tmp/presence.json"))["input"])
roi  = json.load(open("/tmp/presence.json"))["roi"]
raw  = cv2.imread("/tmp/frame.jpg")
h, w = raw.shape[:2]
x, y = int(round(roi[0]*w)), int(round(roi[1]*h))
rw, rh = int(round(roi[2]*w)), int(round(roi[3]*h))
g = cv2.cvtColor(raw[y:y+rh, x:x+rw], cv2.COLOR_BGR2GRAY)
g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
blob = np.ascontiguousarray(np.repeat(g[None, None], 3, axis=1))
net = cv2.dnn.readNetFromONNX("/tmp/presence.onnx")
net.setInput(blob); net.forward()
t = time.perf_counter()
for _ in range(20):
    net.setInput(blob); out = net.forward()
print("cv2", cv2.__version__, "p", float(out.reshape(-1)[0]),
      "mean ms", 1000*(time.perf_counter()-t)/20)
PY'
ssh josue@PI 'rm -f /tmp/presence.onnx /tmp/presence.json /tmp/frame.jpg'
```

The preprocessing in that snippet is `monitor.model_blob` written out by hand,
which is the point: it proves the file runs under 4.6 with nothing but the
stdlib and cv2.

**If `readNetFromONNX` throws, the export is wrong, not the Pi.** cv2.dnn 4.6
is from 2022 and implements a subset of ONNX. Two things about this export
exist only because of it, both measured against the real machine:

- **Opset 10, not 12.** At opset 11 and up, torch exports ReLU6 as a
  three-input `Clip` with min and max as tensors. 4.6 only knows the opset-6
  form where they are attributes and rejects the file outright. Opset 10 emits
  the attribute form, and nothing else in a MobileNetV2 needs opset 11
  semantics.
- **No Identity-aliased initializers.** torch deduplicates identical
  initializers and inserts an `Identity` per alias. 4.6 cannot import an
  `Identity` whose input is an initializer -- it asserts on `inputs.size()` and
  fails the whole file -- so `strip_identity_initializers` deletes them and
  gives each consumer its own copy of the constant.

The same reasoning is why ImageNet normalization is a frozen 1x1 grouped `Conv`
in the graph rather than the obvious `(x - mean) / std`: written as arithmetic
it exports as `Sub`/`Div` against constants, which is how the `Identity`
problem appears in the first place. `export_onnx`'s docstring has the full list.

Batch is fixed at 1 with no dynamic axes, which is where old cv2 gives up
anyway and costs nothing since the watch loop passes one frame per forward
pass.

## Deploying is a human step

**Nothing in this directory touches the Pi.** `pull.sh` only reads. Deploy is
done by hand, deliberately, because it changes what a running monitor believes
about an empty pen:

```bash
scp models/presence.onnx models/presence.json josue@PI:~/dog-sleep-monitor/models/
```

Then edit the Pi's `config.json`:

```json
"presence_model": "models/presence.onnx",
"presence_model_input": [160, 128],
"presence_model_threshold": 0.5,
"presence_model_deadband": 0.15,
"presence_model_log": "presence_model.csv",
"presence_model_shadow": true
```

Start with `presence_model_shadow: true`. The model loads, predicts and logs
every sample, the journal shows `p` and its vote beside the reference diff,
and `/v1/state` carries `p_dog` with `shadow: true`, but the references keep
driving the machine. Nothing rests on the model until you flip shadow off, and
the section after this one says when.

`presence_model_input` must match the sidecar. If it does not, the monitor says
so loudly and uses the sidecar's numbers, because feeding the wrong shape does
not crash -- cv2 resizes to it and returns confident nonsense.

```bash
sudo systemctl restart dog-monitor-watch
journalctl -u dog-monitor-watch -f
```

The startup banner names the model, its input size and the two vote edges. The
per-sample journal line then carries `p 0.97 vote occ` next to the reference
diff's own numbers.

A restart is not strictly required: the model hot-reloads on mtime, so scp-ing
a new ONNX over the old one takes effect on the next sample. That is the path
for swapping a model mid-run, which is exactly when you most want to.

### Then leave it alone for a night

`presence_model.csv` logs one row per sample: `timestamp, p, vote,
ref_presence, ref_corr, ir`. That file is the entire argument that the model
beat the references rather than merely replaced them, and it is what
`presence_model_threshold` and `presence_model_deadband` get tuned against. The
reference diff keeps being computed while the model drives, precisely so the
two sit side by side in there.

Read a night of it before trusting the thing, especially the `ir` rows. Night
is where references failed, so night is where this has to be checked. When the
votes agree with what you see in the frames across a day and a night, set
`presence_model_shadow` to `false` and restart (or wait: config is read at
start, so this one does need the restart). From then on the model drives and
the references are the fallback.

### Backing out

Set `presence_model` back to `null` and restart. Reference-diff presence is
untouched and still running underneath: the diff is computed every sample even
under a model, the auto-refresh gate still keeps the references fresh, and the
only thing that changes is which layer's verdict reaches the state machine.

A model that fails to load does this by itself. One loud line, `model_error`
in the heartbeat and on `/v1/state`, and reference-diff presence carries on.
The monitor does not stop watching a dog because a file is corrupt.

## Files

| file | what |
| --- | --- |
| `pull.sh` | rsync the Pi's archive and csvs into `.local/pi/`. Read-only. |
| `preprocess.py` | the training side of the preprocessing contract |
| `train_presence.py` | train, evaluate, export ONNX + sidecar |
| `verify_onnx.py` | preprocessing lockstep, runtime agreement, latency |
| `requirements-train.txt` | Mac-only deps |

The preprocessing contract, which both sides implement separately and
`verify_onnx.py` enforces:

1. crop the ROI from the raw BGR frame, by fraction
2. greyscale
3. resize to `[w, h]`, `INTER_AREA`
4. scale to 0..1 float32
5. replicate to 3 identical channels, NCHW `[1, 3, h, w]`

ImageNet mean/std is **not** in that list. It is baked into the graph, so
neither side knows the constants and neither can drift from them. Change any
step and you change both copies and run `verify_onnx.py`.
