#!/usr/bin/env python3
"""Prove the exported graph is the same model the Pi will run.

    python train/verify_onnx.py --model models/presence.onnx \\
        --frame archive/2026-08-21T03-11-02_0.0000.jpg

Three things get checked, and each one has already been a real way to ship a
broken model:

  1. train/preprocess.py and monitor.model_blob produce the SAME tensor. They
     are separate implementations by necessity (the Pi has no torch, and
     monitor.py cannot import from train/), so nothing but this check keeps
     them in lockstep. A one-pixel difference in the crop or a different resize
     interpolation moves p by more than the deadband.
  2. onnxruntime and cv2.dnn agree on the output. cv2 4.6 is from 2022 and
     implements a subset of ONNX; it can load a graph, run it, and return a
     confidently different number. Agreement to 1e-4 is what says the Pi is
     running the model that was trained, not an approximation of it.
  3. it is fast enough. Printed, not asserted -- the budget belongs to whoever
     is looking at the sample interval.

This runs on the Mac, where cv2 is newer than the Pi's. It catches the
numerical questions but NOT "does cv2 4.6 support these ops". Only the Pi
answers that; see train/README.md for the standalone timing script that does.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from preprocess import preprocess  # noqa: E402
import monitor  # noqa: E402  -- the runtime's own copy of the preprocessing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(REPO, "models", "presence.onnx"))
    ap.add_argument("--frame", required=True, help="an archived JPEG to run")
    ap.add_argument("--config", default=os.path.join(REPO, "config.json"),
                    help="read the ROI fractions from here")
    ap.add_argument("--input", help="model input WxH; default: read the sidecar")
    ap.add_argument("--runs", type=int, default=20, help="timed forward passes")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    model = args.model if os.path.isabs(args.model) else os.path.join(REPO, args.model)
    frame_path = (args.frame if os.path.isabs(args.frame)
                  else os.path.join(REPO, args.frame))

    roi = None
    if os.path.exists(args.config):
        with open(args.config) as fh:
            roi = json.load(fh).get("roi")

    size = None
    if args.input:
        w, h = args.input.lower().split("x")
        size = (int(w), int(h))
    else:
        side = os.path.splitext(model)[0] + ".json"
        try:
            with open(side) as fh:
                size = tuple(int(v) for v in json.load(fh)["input"])
            print(f"  input {size[0]}x{size[1]} from {os.path.basename(side)}")
        except (OSError, ValueError, KeyError):
            raise SystemExit("No --input and no readable sidecar json next to "
                             "the model. Pass --input WxH.")

    raw = cv2.imread(frame_path)
    if raw is None:
        raise SystemExit(f"cannot read {frame_path}")
    print(f"  frame {os.path.relpath(frame_path, REPO)}  "
          f"{raw.shape[1]}x{raw.shape[0]}  roi {roi}")

    # 1. the two preprocessing implementations
    a = preprocess(raw, roi, size)
    b = monitor.model_blob(raw, roi, size)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise SystemExit(f"PREPROCESSING MISMATCH: train {a.shape}/{a.dtype} vs "
                         f"monitor {b.shape}/{b.dtype}")
    gap = float(np.abs(a - b).max())
    if gap != 0.0:
        raise SystemExit(f"PREPROCESSING MISMATCH: max abs difference {gap:g}. "
                         f"train/preprocess.py and monitor.model_blob have "
                         f"drifted apart; fix both before deploying.")
    print(f"  preprocessing   train/preprocess.py == monitor.model_blob  "
          f"{a.shape} float32 in [{a.min():.3f}, {a.max():.3f}]")

    # 2. the two runtimes
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    p_ort = float(np.asarray(sess.run([out_name], {in_name: a})[0]).reshape(-1)[0])

    net = cv2.dnn.readNetFromONNX(model)
    net.setInput(b)
    p_cv = float(np.asarray(net.forward()).reshape(-1)[0])

    print(f"  onnxruntime     p = {p_ort:.6f}   (input '{in_name}', "
          f"output '{out_name}')")
    print(f"  cv2.dnn {cv2.__version__:<8}p = {p_cv:.6f}")
    delta = abs(p_ort - p_cv)
    if delta > args.tolerance:
        raise SystemExit(f"RUNTIME MISMATCH: {delta:g} > {args.tolerance:g}. "
                         f"cv2.dnn is not computing the same graph "
                         f"onnxruntime is.")
    print(f"  agreement       {delta:.3g} <= {args.tolerance:g}  OK")

    # 3. latency, through the path the monitor actually uses
    net.setInput(monitor.model_blob(raw, roi, size))
    net.forward()  # warmup: the first pass allocates every layer's buffers
    t0 = time.perf_counter()
    for _ in range(args.runs):
        net.setInput(monitor.model_blob(raw, roi, size))
        net.forward()
    ms = 1000.0 * (time.perf_counter() - t0) / max(1, args.runs)
    print(f"  latency         {ms:.1f} ms/frame over {args.runs} runs "
          f"(preprocess + forward, this Mac)")
    print("\n  Verified. This does NOT prove cv2 4.6 on the Pi can load the "
          "graph;\n  run the timing script from train/README.md there before "
          "deploying.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
