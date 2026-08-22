# models/

Trained presence classifiers. **Nothing in here is tracked** except this file.

The weights are derived from thousands of pictures of the inside of a home, and
this repo is public. They live on the laptop that trained them and on the Pi
that runs them, and nowhere else. `.gitignore` excludes `*.onnx`, `*.json`,
`*.pt` and `*.pth` from this directory by extension rather than by name, so a
renamed copy cannot escape either.

## What lands here

| file | written by | read by |
| --- | --- | --- |
| `presence.onnx` | `train/train_presence.py` | `monitor.py` via `cv2.dnn` on the Pi |
| `presence.json` | `train/train_presence.py` | `monitor.py`, for the input size |

The sidecar is not optional bookkeeping. `PresenceModel` reads the input size
out of it and **overrides** `presence_model_input` from `config.json` when the
two disagree, printing a loud line when it does. Feeding a graph the wrong
input shape does not raise: cv2 resizes to whatever it is given and returns a
confident number, so a stale config key would degrade the model silently and
look like a bad model rather than a bad deploy.

It also carries the record of what produced the file: architecture and width,
the metrics on val and test, the span of labeled data behind it, per-class
counts per split, library versions, the training date, and the repo's git sha.
That is the only provenance a model file has once it is sitting on a Pi.

## Getting one

See `train/README.md`. Short version: pull frames off the Pi, label windows
with `label-presence`, export a manifest with `dataset --labeled-only`, train,
verify, then copy the `.onnx` and `.json` here and to the Pi by hand.

## Turning it off

Set `"presence_model": null` in `config.json`. Presence falls back to
reference-differencing against `references/`, which is why the auto-refresh
keeps running even while a model is driving: the fallback stays warm.
