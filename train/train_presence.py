#!/usr/bin/env python3
"""Train the dog/no-dog presence classifier and export it as ONNX for the Pi.

Runs on the Mac, in train/.venv. The Pi never sees torch: it loads the exported
graph through cv2.dnn and nothing else.

    python train/train_presence.py --manifest dataset/manifest.csv

What this is replacing, and why it is worth a model at all: reference
differencing answers "does this frame differ from a stored empty pen", which is
not the question. A toy dragged three inches produces the same contiguous blob
a curled-up dog does. At night the numbers invert outright -- an empty but
rearranged pen measured 0.045, a sleeping dog 0.023 -- so every cutoff that
calls the first empty calls the second empty too. There is no threshold that
fixes that, because the measurement is answering a different question from the
one being asked.

Three things this script refuses to do, each because getting it wrong produces
an impressive number and a useless model:

  - it never re-splits the data. The splits come from the manifest's `split`
    column, which `dataset` assigned per contiguous time block. Frames five
    seconds apart are near-duplicates; a per-frame shuffle would put the same
    moment in train and test and report 99% accuracy on a model that memorized
    a blanket.
  - it never trains on `weak_state`. That is the monitor's own decision, made
    by the thresholds the model exists to check.
  - it never balances classes by throwing data away. Empty frames dominate any
    real archive (she is out of the pen for hours at a time); a sampler
    reweights them instead, so every labeled frame is still seen.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import crop_roi, resize_to, to_chw3, to_gray, to_unit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Applied inside the exported graph, not in the data pipeline. See preprocess.py.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- data --------------------------------------------------------------------

def read_manifest(path):
    """Labeled rows only, grouped by split.

    A blank label_dog_present is not a zero. It means nobody has said, and
    guessing "no dog" for every unlabeled frame is how you train a model that
    reports an empty pen all night.
    """
    by_split = {"train": [], "val": [], "test": []}
    skipped = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            lab = (r.get("label_dog_present") or "").strip()
            if lab not in ("0", "1"):
                skipped += 1
                continue
            split = (r.get("split") or "train").strip()
            if split not in by_split:
                split = "train"
            by_split[split].append({
                "file": r["file"],
                "y": float(lab),
                "timestamp": r.get("timestamp") or "",
                "dark": (r.get("dark") or "0").strip() == "1",
            })
    return by_split, skipped


class Frames(Dataset):
    """Archived JPEGs, cropped to the ROI and preprocessed on the fly.

    Decoding a 640x360 JPEG and area-resizing it to 160x128 is a fraction of a
    millisecond, and the whole labeled set is a few thousand frames, so there
    is nothing here worth a cache. Reading from disk every epoch also means
    the augmentation is genuinely fresh each time.
    """

    def __init__(self, rows, roi, size, frames_root, augment, seed=0):
        self.rows = rows
        self.roi = roi
        self.size = size
        self.root = frames_root
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def _geometry(self, gray):
        """Random resized crop of the ROI, keeping aspect, plus a small shift.

        The camera is bolted in place, so the model would otherwise be free to
        learn "a dog is whatever is at these coordinates". A crop of 85-100% of
        the ROI with a random offset forces it to look at the contents. The
        range is deliberately narrow: crop harder and the dog leaves the frame
        in the very shots where she is curled against an edge.
        """
        h, w = gray.shape[:2]
        scale = float(self.rng.uniform(0.85, 1.0))
        ch, cw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
        y = int(self.rng.integers(0, h - ch + 1))
        x = int(self.rng.integers(0, w - cw + 1))
        return gray[y:y + ch, x:x + cw]

    def _photometry(self, unit):
        """Brightness, contrast, noise, flip: the ways this camera lies.

        Every one of these is a real drift mode. The Tapo re-tunes its own
        exposure and gain without telling anyone, IR night frames carry visible
        sensor grain, and the horizontal flip is free symmetry -- a dog facing
        left is the same dog.
        """
        unit = unit * float(self.rng.uniform(0.85, 1.15))          # contrast
        unit = unit + float(self.rng.uniform(-0.10, 0.10))         # brightness
        unit = unit + self.rng.normal(0.0, 0.02, unit.shape).astype(np.float32)
        if self.rng.random() < 0.5:
            unit = unit[:, ::-1]
        return np.clip(unit, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, i):
        row = self.rows[i]
        path = row["file"]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"cannot read frame {path}")
        gray = to_gray(crop_roi(img, self.roi))
        if self.augment:
            gray = self._geometry(gray)
        unit = to_unit(resize_to(gray, self.size))
        if self.augment:
            unit = self._photometry(unit)
        return torch.from_numpy(to_chw3(unit)), torch.tensor([row["y"]])


# --- model -------------------------------------------------------------------

class PresenceNet(nn.Module):
    """Backbone plus the normalization, as one exportable graph.

    Normalization is a layer here rather than a data-loading step so that the
    runtime never has to know the ImageNet constants. monitor.py feeds plain
    0..1 greyscale-x3 and the graph does the rest; there is no second place for
    the numbers to be wrong.

    It is a frozen 1x1 depthwise convolution rather than the obvious
    `(x - mean) / std`, and that is not an aesthetic choice. Written as
    arithmetic, torch exports it as Sub and Div against constant initializers,
    and the exporter's initializer deduplication then wraps those constants in
    Identity nodes. cv2.dnn 4.6 -- what the Pi has, and there will be no pip
    install to change that -- cannot import an Identity whose input is an
    initializer: it asserts on inputs.size() and refuses the whole file. A
    Conv with groups=3 computes exactly the same per-channel affine
    (weight = 1/std, bias = -mean/std) using the one op every ONNX importer
    ever written supports.

    The weights are frozen. They encode a fixed preprocessing contract, and a
    gradient step that nudged them would silently desynchronize the graph from
    what train/preprocess.py and monitor.model_blob feed it.

    forward() returns the probability, because that is what the Pi wants to
    read out of one output tensor. Training calls logits() and uses
    BCEWithLogitsLoss, which is the numerically stable pairing.
    """

    def __init__(self, backbone):
        super().__init__()
        self.norm = nn.Conv2d(3, 3, kernel_size=1, groups=3, bias=True)
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor(
                [[[[1.0 / s]]] for s in IMAGENET_STD]))
            self.norm.bias.copy_(torch.tensor(
                [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]))
        self.norm.weight.requires_grad_(False)
        self.norm.bias.requires_grad_(False)
        self.backbone = backbone

    def logits(self, x):
        return self.backbone(self.norm(x))

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def build_model(arch, width):
    if arch != "mobilenet_v2":
        raise SystemExit(f"unknown --arch {arch}")
    # torchvision ships ImageNet weights for width_mult 1.0 only. At 0.5 there
    # is nothing to load and the shapes do not line up, so that width trains
    # from scratch. Say so rather than silently producing a much worse model.
    if abs(width - 1.0) < 1e-6:
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        net = torchvision.models.mobilenet_v2(weights=weights)
    else:
        print(f"  NOTE: torchvision has no ImageNet weights at width {width}; "
              f"training from random init.\n"
              f"  Expect to need far more labeled data. Use --width 1.0 unless "
              f"the Pi latency demands otherwise.")
        net = torchvision.models.mobilenet_v2(width_mult=width)
    net.classifier[1] = nn.Linear(net.last_channel, 1)
    return PresenceNet(net)


# --- metrics -----------------------------------------------------------------

def evaluate(net, loader, device):
    """Returns (probs, targets) for a whole split."""
    net.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            p = net(x.to(device)).detach().cpu().numpy().reshape(-1)
            ps.append(p)
            ys.append(y.numpy().reshape(-1))
    if not ps:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return np.concatenate(ps), np.concatenate(ys)


def confusion(probs, targets, threshold):
    pred = (probs >= threshold).astype(np.int32)
    true = targets.astype(np.int32)
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    return tp, tn, fp, fn


def metrics(probs, targets, threshold):
    tp, tn, fp, fn = confusion(probs, targets, threshold)
    n = max(1, tp + tn + fp + fn)
    def ratio(a, b):
        return (a / b) if b else float("nan")
    return {
        "n": tp + tn + fp + fn,
        "accuracy": (tp + tn) / n,
        "dog_precision": ratio(tp, tp + fp),
        "dog_recall": ratio(tp, tp + fn),
        "empty_precision": ratio(tn, tn + fn),
        "empty_recall": ratio(tn, tn + fp),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "threshold": threshold,
    }


def bimodality(probs):
    """How far the probabilities sit from the middle, and how many are stuck in it.

    The single number that decides whether the deadband is usable. A model
    whose output piles up around 0.5 will abstain on every sample and the
    presence layer will hold state forever; one that lives at the ends can
    afford a wide deadband and still vote on almost everything.
    """
    if probs.size == 0:
        return {}
    edges = np.linspace(0.0, 1.0, 11)
    hist = np.histogram(probs, bins=edges)[0].astype(int).tolist()
    return {
        "histogram_10": hist,
        "frac_below_0.35": float((probs <= 0.35).mean()),
        "frac_above_0.65": float((probs >= 0.65).mean()),
        "frac_in_deadband": float(((probs > 0.35) & (probs < 0.65)).mean()),
        "mean_distance_from_half": float(np.abs(probs - 0.5).mean()),
    }


def suggest_threshold(probs, targets):
    """The cutoff with the best balanced accuracy, or 0.5 if it cannot be read.

    Balanced rather than plain accuracy because the classes are lopsided: with
    80% empty frames, "always empty" scores 0.80 and any threshold that beats
    it on raw accuracy may still never detect the dog.
    """
    if probs.size == 0 or len(set(targets.tolist())) < 2:
        return 0.5
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        tp, tn, fp, fn = confusion(probs, targets, float(t))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if (sens + spec) / 2.0 > best:
            best, best_t = (sens + spec) / 2.0, float(t)
    return round(best_t, 2)


def report(name, probs, targets, threshold):
    m = metrics(probs, targets, threshold)
    b = bimodality(probs)
    c = m["confusion"]
    print(f"\n  {name}  ({m['n']} frames, threshold {threshold:.2f})")
    print(f"    accuracy          {m['accuracy']:.4f}")
    print(f"    dog    precision  {m['dog_precision']:.4f}   "
          f"recall {m['dog_recall']:.4f}")
    print(f"    empty  precision  {m['empty_precision']:.4f}   "
          f"recall {m['empty_recall']:.4f}")
    print(f"    confusion         predicted dog   predicted empty")
    print(f"      actual dog      {c['tp']:>13}   {c['fn']:>15}")
    print(f"      actual empty    {c['fp']:>13}   {c['tn']:>15}")
    if b:
        print(f"    p histogram       {b['histogram_10']}  (0.0 .. 1.0 in tenths)")
        print(f"    below 0.35 {b['frac_below_0.35']:.3f}   "
              f"above 0.65 {b['frac_above_0.65']:.3f}   "
              f"in the 0.35-0.65 deadband {b['frac_in_deadband']:.3f}")
        if b["frac_in_deadband"] > 0.10:
            print(f"    WARNING: {100 * b['frac_in_deadband']:.0f}% of samples "
                  f"land in the default deadband and would abstain.\n"
                  f"    Either the model is undertrained or the deadband needs "
                  f"narrowing.")
    return {**m, **b}


# --- export ------------------------------------------------------------------

def strip_identity_initializers(model):
    """Remove Identity nodes that merely alias a constant. Returns the count.

    torch's exporter deduplicates identical initializers -- every all-zero
    convolution bias of the same shape becomes one tensor -- and then inserts
    an Identity node per alias to give each consumer the name it expects.
    Every ONNX runtime written this decade handles that. cv2.dnn 4.6 does not:
    it tries to build a layer for the Identity, finds its input is an
    initializer rather than a tensor, and fails the assertion at layer.cpp:246,
    which aborts the import of the entire file.

    So each such Identity is deleted and its output name re-registered as its
    own copy of the constant. Semantically identical, and it costs a few
    kilobytes of duplicated zeros.

    An Identity that produces a graph output is left alone: deleting it would
    leave the output with nothing producing it.
    """
    init = {i.name: i for i in model.graph.initializer}
    outputs = {o.name for o in model.graph.output}
    keep, removed = [], 0
    for node in model.graph.node:
        if (node.op_type == "Identity" and node.input[0] in init
                and node.output[0] not in outputs):
            dup = onnx.TensorProto()
            dup.CopyFrom(init[node.input[0]])
            dup.name = node.output[0]
            model.graph.initializer.append(dup)
            init[dup.name] = dup
            removed += 1
            continue
        keep.append(node)
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return removed


def export_onnx(net, out_path, size, opset):
    """Fixed batch 1, input `input`, output `prob`, and an old opset on purpose.

    Every constraint here comes from the far end. The Pi runs cv2.dnn 4.6,
    which is from 2022 and implements a subset of ONNX, and there will never be
    a pip install on that machine to change it. Measured against it directly:

      - opset 11 and up export ReLU6 as a three-input Clip, with min and max as
        tensors. 4.6's importer only knows the opset-6 form where they are
        attributes, and rejects the file outright. Opset 10 emits the
        attribute form, and nothing else in a MobileNetV2 needs opset 11
        semantics, so 10 costs nothing.
      - Identity-aliased initializers have to go. See above.
      - dynamic axes are where old cv2 gives up, and the watch loop passes
        exactly one frame per forward pass, so batch 1 is free.

    The alternative to the low opset is swapping every ReLU6 for a ReLU, which
    removes Clip from the graph entirely and lets opset 12 through. It also
    changes the function the ImageNet weights were trained to compute, so it is
    the second choice, not the first.

    do_constant_folding stays on. It is what collapses each BatchNorm into the
    convolution ahead of it, and cv2's own BatchNorm handling is one more thing
    that does not have to be right if the graph never asks for it.
    """
    net = net.cpu().eval()
    dummy = torch.zeros(1, 3, int(size[1]), int(size[0]), dtype=torch.float32)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    kwargs = dict(input_names=["input"], output_names=["prob"],
                  opset_version=opset, do_constant_folding=True)
    try:
        # Force the legacy tracing exporter. Newer torch defaults to the dynamo
        # path, which emits a correct graph full of ops cv2 4.6 has never heard
        # of, and which does not honour opset_version the same way.
        torch.onnx.export(net, (dummy,), out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(net, (dummy,), out_path, **kwargs)

    model = onnx.load(out_path)
    removed = strip_identity_initializers(model)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"\n  exported opset {opset}, stripped {removed} aliasing Identity "
          f"node(s)")
    print(f"  graph ops: {', '.join(ops)}")
    return ops


def json_safe(value):
    """Replace non-finite floats with null, recursively, for the sidecar.

    A precision or recall with an empty denominator is a nan, and json.dump
    writes that as a bare `NaN`. Python reads it back happily, so monitor.py
    never notices, but `NaN` is not JSON: node, Go and Swift all reject the
    whole file. The sidecar is the only provenance a model has once it is
    sitting on a Pi, and a record nothing but Python can open is a worse
    record. null is the honest encoding anyway -- the metric was undefined,
    not zero.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- main --------------------------------------------------------------------

def parse_input(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise SystemExit(f"--input wants WxH, e.g. 160x128, not {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(REPO, "dataset", "manifest.csv"),
                    help="dataset/manifest.csv from `monitor.py dataset`")
    ap.add_argument("--frames-root", default=REPO,
                    help="directory the manifest's `file` column is relative to "
                         "(default: the repo root, which is what dataset writes)")
    ap.add_argument("--config", default=os.path.join(REPO, "config.json"),
                    help="read the ROI fractions from here")
    ap.add_argument("--input", default="160x128", help="model input WxH")
    ap.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2"])
    ap.add_argument("--width", type=float, default=1.0, choices=[1.0, 0.5],
                    help="mobilenet width multiplier; 0.5 is ~4x cheaper and "
                         "has no pretrained weights")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs with no val improvement")
    ap.add_argument("--out", default=os.path.join(REPO, "models", "presence.onnx"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opset", type=int, default=10,
                    help="ONNX opset. 10 by default, not 12: at 11+ ReLU6 "
                         "exports as a three-input Clip that the Pi's cv2 4.6 "
                         "cannot import. See export_onnx.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = parse_input(args.input)

    roi = None
    if os.path.exists(args.config):
        with open(args.config) as fh:
            roi = json.load(fh).get("roi")
    print(f"  ROI {roi}  input {size[0]}x{size[1]}  arch {args.arch} "
          f"width {args.width}")
    if roi is None:
        print("  WARNING: no roi in the config, training on whole frames. That "
              "is almost\n  certainly not what the monitor will feed it.")

    by_split, skipped = read_manifest(args.manifest)
    counts = {k: Counter(int(r["y"]) for r in v) for k, v in by_split.items()}
    print(f"  manifest {os.path.relpath(args.manifest, REPO)}: "
          f"{sum(len(v) for v in by_split.values())} labeled, "
          f"{skipped} unlabeled rows ignored")
    for k in ("train", "val", "test"):
        print(f"    {k:<6}{len(by_split[k]):>6} frames   "
              f"dog {counts[k].get(1, 0)}  empty {counts[k].get(0, 0)}")
    if not by_split["train"]:
        raise SystemExit("No labeled training frames. Run `label-presence` and "
                         "re-export the manifest with --labeled-only.")

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device {device}")

    train_ds = Frames(by_split["train"], roi, size, args.frames_root,
                      augment=True, seed=args.seed)
    # Class-balanced sampling rather than discarding the majority class. An
    # archive is mostly empty pen, and every one of those frames is real
    # evidence of what an empty pen looks like at that hour.
    ny = counts["train"]
    weights = [1.0 / max(1, ny.get(int(r["y"]), 1)) for r in by_split["train"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, drop_last=False)
    loaders = {}
    for k in ("val", "test"):
        if by_split[k]:
            loaders[k] = DataLoader(
                Frames(by_split[k], roi, size, args.frames_root, augment=False),
                batch_size=args.batch, shuffle=False, num_workers=args.workers)

    net = build_model(args.arch, args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    epoch = 0  # read below; --epochs 0 would leave it unbound
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    started = time.time()
    print()
    for epoch in range(1, args.epochs + 1):
        net.train()
        total, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(net.logits(x), y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        sched.step()
        train_loss = total / max(1, seen)

        line = f"  epoch {epoch:>3}/{args.epochs}  train {train_loss:.4f}"
        if "val" in loaders:
            p, t = evaluate(net, loaders["val"], device)
            # Same loss as training, recomputed from the probabilities the
            # exported graph will actually emit.
            eps = 1e-7
            pc = np.clip(p, eps, 1 - eps)
            val_loss = float(-(t * np.log(pc) + (1 - t) * np.log(1 - pc)).mean())
            acc = float(((p >= 0.5).astype(np.float32) == t).mean())
            line += f"  val {val_loss:.4f}  acc {acc:.4f}"
            # Early stopping keeps the best weights, not the last: with a few
            # hundred frames the last epoch is usually overfit and the val
            # curve turns up well before the loop ends.
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                line += "  *"
            else:
                stale += 1
        print(line)
        if "val" in loaders and stale >= args.patience:
            print(f"  no val improvement for {args.patience} epochs, stopping.")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\n  restored the best epoch ({best_epoch}, val loss "
              f"{best_loss:.4f})")
    print(f"  trained in {time.time() - started:.0f}s")

    results, threshold = {}, 0.5
    if "val" in loaders:
        pv, tv = evaluate(net, loaders["val"], device)
        threshold = suggest_threshold(pv, tv)
        print(f"\n  suggested threshold from val: {threshold:.2f} "
              f"(best balanced accuracy)")
        results["val"] = report("val", pv, tv, threshold)
    if "test" in loaders:
        pt, tt = evaluate(net, loaders["test"], device)
        results["test"] = report("test", pt, tt, threshold)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    graph_ops = export_onnx(net, out_path, size, args.opset)
    stamps = sorted(r["timestamp"] for v in by_split.values() for r in v
                    if r["timestamp"])
    sidecar = {
        "input": [size[0], size[1]],
        "threshold": threshold,
        "opset": args.opset,
        "graph_ops": graph_ops,
        "deadband": 0.15,
        "arch": args.arch,
        "width": args.width,
        "epochs_run": epoch,
        "best_epoch": best_epoch or epoch,
        "lr": args.lr,
        "batch": args.batch,
        "seed": args.seed,
        "roi": roi,
        "metrics": results,
        "data_span": {"first": stamps[0] if stamps else None,
                      "last": stamps[-1] if stamps else None},
        "counts": {k: {"dog": counts[k].get(1, 0), "empty": counts[k].get(0, 0)}
                   for k in ("train", "val", "test")},
        "manifest": os.path.relpath(args.manifest, REPO),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        # Nothing in train/ is version-pinned, so the record of what produced a
        # given model lives here, next to the model.
        "versions": {"torch": torch.__version__,
                     "torchvision": torchvision.__version__,
                     "opencv": cv2.__version__,
                     "python": sys.version.split()[0]},
    }
    side_path = os.path.splitext(out_path)[0] + ".json"
    with open(side_path, "w") as fh:
        json.dump(json_safe(sidecar), fh, indent=2)

    mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Wrote {os.path.relpath(out_path, REPO)} ({mb:.1f} MB)")
    print(f"  Wrote {os.path.relpath(side_path, REPO)}")
    print(f"\n  Next: python train/verify_onnx.py --model "
          f"{os.path.relpath(out_path, REPO)} --frame <some archived frame>")
    print("  Deploying to the Pi is a human step. See train/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
