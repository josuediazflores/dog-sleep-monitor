#!/usr/bin/env python3
"""Camera-free checks on the scoring math.

Builds a synthetic 720p room with a playpen and a dog-sized blob, then confirms:
  - identical frames score ~0
  - a global brightness or gain shift still scores ~0 (exposure drift immunity)
  - sensor noise stays under the quiet threshold
  - real movement clears the active threshold once the ROI is cropped
  - a day/night IR switch trips the scene-change guard instead of faking a wake
  - the fractional ROI gives the same answer on stream1 and stream2 resolutions

Run: python test_math.py
"""

import cv2
import numpy as np

from monitor import DEFAULTS, correlation, motion_score, prepare, roi_pixels

RNG = np.random.default_rng(1234)

# The playpen as fractions of the frame, which is how config.json stores it.
PLAYPEN_ROI = [0.2344, 0.5833, 0.5469, 0.3889]

QUIET = DEFAULTS["quiet_score"]
ACTIVE = DEFAULTS["active_score"]
SCENE = DEFAULTS["scene_change_score"]


def scene(dog_at=None, brightness=0.0, gain=1.0, noise=0.0, ir=False):
    """A 720p room: floor gradient, a wall, a playpen mat, maybe a dog."""
    h, w = 720, 1280
    img = np.tile(np.linspace(40, 110, h, dtype=np.float32)[:, None], (1, w))
    img[:220, :] = 75.0                                  # wall
    img[420:700, 300:1000] = 130.0                       # playpen mat
    img[430:690, 310:320] = 45.0                         # pen bar
    if dog_at is not None:
        cx, cy = dog_at
        img[cy - 60:cy + 60, cx - 110:cx + 110] = 195.0  # the dog
    if ir:
        # Night mode: IR cut filter drops out and the IR LEDs light the scene,
        # so surface brightness no longer tracks visible reflectance at all.
        img = 255.0 - img
    img = img * gain + brightness
    if noise:
        img = img + RNG.normal(0.0, noise, img.shape)
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8)] * 3)


DOG = (650, 560)
CASES = [
    # name,                                 frame A,     frame B,                           expect
    ("identical frames",                    scene(DOG),  scene(DOG),                        "still"),
    ("brightness +40 (exposure drift)",     scene(DOG),  scene(DOG, brightness=40),         "still"),
    ("gain x1.3 (auto-gain kick)",          scene(DOG),  scene(DOG, gain=1.3),              "still"),
    ("sensor noise sigma=6 (night grain)",  scene(DOG),  scene(DOG, noise=6.0),             "still"),
    ("twitch 20px",                         scene(DOG),  scene((670, 560)),                 "moving"),
    ("shift 40px + brightness +25",         scene(DOG),  scene((690, 560), brightness=25),  "moving"),
    ("dog crosses the pen",                 scene(DOG),  scene((430, 520)),                 "moving"),
    ("day -> night IR switch",              scene(DOG),  scene(DOG, ir=True),               "scene"),
]


def score_with(roi, a, b):
    cfg = dict(DEFAULTS)
    cfg["roi"] = roi
    return motion_score(prepare(a, cfg), prepare(b, cfg), cfg["pixel_threshold"])


def verdict(score, expect):
    if expect == "still":
        return score < QUIET, f"< quiet {QUIET}"
    if expect == "moving":
        return ACTIVE < score <= SCENE, f"in ({ACTIVE}, {SCENE}]"
    return score > SCENE, f"> scene {SCENE}"


def main():
    print(f"thresholds: quiet < {QUIET}   active > {ACTIVE}   scene > {SCENE}\n")
    print(f"{'case':<38}{'ROI':>9}{'full':>9}   result")

    passed = 0
    for name, a, b, expect in CASES:
        roi_score = score_with(PLAYPEN_ROI, a, b)
        raw_score = score_with(None, a, b)
        ok, want = verdict(roi_score, expect)
        passed += ok
        print(f"{name:<38}{roi_score:>9.4f}{raw_score:>9.4f}   "
              f"{'PASS' if ok else 'FAIL'}  {expect} ({want})")

    # A fractional ROI has to mean the same box whether the camera is serving
    # stream1 (1280x720) or stream2 (640x360). Pixel coordinates would not.
    small = [cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
             for f in (scene(DOG), scene((690, 560), brightness=25))]
    hi = score_with(PLAYPEN_ROI, scene(DOG), scene((690, 560), brightness=25))
    lo = score_with(PLAYPEN_ROI, *small)
    res_ok = abs(hi - lo) < 0.02
    passed += res_ok
    print(f"\n{'resolution independence (720p vs 360p)':<38}{hi:>9.4f}{lo:>9.4f}   "
          f"{'PASS' if res_ok else 'FAIL'}  gap {abs(hi - lo):.4f} (< 0.02)")

    # Correlation is what separates a global re-lighting from a large local
    # change. Magnitude cannot: a dog close to the camera measured 0.82 live,
    # above any plausible magnitude-only scene threshold.
    cfg = dict(DEFAULTS); cfg["roi"] = PLAYPEN_ROI
    p_ = lambda f: prepare(f, cfg)
    ir_corr = correlation(p_(scene(DOG)), p_(scene(DOG, ir=True)))
    big_dog = scene(DOG)
    huge = scene((650, 560))
    huge[300:700, 200:1100] = 210          # dog filling most of the ROI
    dog_corr = correlation(p_(big_dog), p_(huge))
    dog_score = motion_score(p_(big_dog), p_(huge), cfg["pixel_threshold"])
    limit = DEFAULTS["scene_corr_max"]
    ir_ok = ir_corr < limit
    dog_ok = dog_corr >= limit
    passed += ir_ok + dog_ok
    print(f"\n{'IR switch correlation':<38}{ir_corr:>9.3f}{'':>9}   "
          f"{'PASS' if ir_ok else 'FAIL'}  scene (< {limit})")
    print(f"{'frame-filling dog correlation':<38}{dog_corr:>9.3f}{'':>9}   "
          f"{'PASS' if dog_ok else 'FAIL'}  movement (>= {limit}), "
          f"despite score {dog_score:.3f}")

    box_ok = roi_pixels(PLAYPEN_ROI, 640, 360) == (150, 210, 350, 140)
    passed += box_ok
    print(f"{'roi_pixels maps onto 640x360':<38}{'':>18}   "
          f"{'PASS' if box_ok else 'FAIL'}  {roi_pixels(PLAYPEN_ROI, 640, 360)}")

    total = len(CASES) + 4
    print(f"\n{passed}/{total} passed (ROI column is what the monitor uses)")

    weak = [n for n, a, b, e in CASES
            if e == "moving" and score_with(None, a, b) <= ACTIVE]
    if weak:
        print("\nWithout an ROI these movements fall into the deadband and would be "
              "missed:\n  " + "\n  ".join(weak) +
              "\nThat is why `preview --pick-roi` is a required setup step, not "
              "a nicety.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
