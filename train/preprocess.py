"""Canonical preprocessing for the presence classifier, training side.

THIS MUST STAY IN LOCKSTEP WITH `model_blob()` IN ../monitor.py.

The two are separate implementations on purpose. monitor.py runs on the Pi with
stdlib, cv2 and numpy and nothing else, and it can never import from here
because here needs torch. So the pipeline is written out twice and the copies
are held together by train/verify_onnx.py, which imports both and asserts they
produce bit-identical tensors for the same frame. If you change one, change the
other, and run verify_onnx.py before deploying anything.

The pipeline, in order:

  1. crop the ROI from the raw BGR frame, by fraction of the frame
  2. greyscale                    -- the camera is IR-lit and greyscale for
                                     half of every day, so colour is a feature
                                     that only exists in daylight and would
                                     teach the model to use the time of day
  3. resize to [w, h], INTER_AREA -- area-averaging, the same downsampler
                                     `prepare` uses, which is what makes the
                                     archive's JPEG grain harmless
  4. scale to 0..1 float32
  5. replicate to 3 identical channels, NCHW [1, 3, h, w]

ImageNet mean/std normalization is NOT here. It is baked into the exported ONNX
graph as its first two ops, so neither side has to know the constants and
neither can drift from them. Both sides feed the graph plain 0..1 greyscale-x3.

Training augmentation is deliberately kept out of this file: it slots in
between steps 3 and 4 (geometry) and after step 4 (photometry), and eval must
run this path untouched so that what the model sees in training matches what
cv2.dnn feeds it on the Pi.
"""

import cv2
import numpy as np


def roi_box(roi, width, height):
    """Fractional ROI [x, y, w, h] in 0..1 -> pixel box, clamped to the frame.

    Deliberately a copy of monitor.roi_pixels, rounding included. The rounding
    is not incidental: the ROI is stored as fractions so the same config crops
    a 1280x720 live frame and a 640x360 archived one to the same view, and a
    half-pixel disagreement between training and inference shifts the crop by a
    whole pixel at one of the two resolutions.
    """
    fx, fy, fw, fh = (float(v) for v in roi)
    x = max(0, min(width - 1, int(round(fx * width))))
    y = max(0, min(height - 1, int(round(fy * height))))
    w = max(1, min(width - x, int(round(fw * width))))
    h = max(1, min(height - y, int(round(fh * height))))
    return x, y, w, h


def crop_roi(bgr, roi):
    """The ROI as a BGR sub-image. Whole frame if roi is None."""
    if not roi:
        return bgr
    h, w = bgr.shape[:2]
    x, y, rw, rh = roi_box(roi, w, h)
    return bgr[y:y + rh, x:x + rw]


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def resize_to(gray, size):
    """size is (width, height), matching config and the sidecar."""
    return cv2.resize(gray, (int(size[0]), int(size[1])),
                      interpolation=cv2.INTER_AREA)


def to_unit(gray):
    return gray.astype(np.float32) / 255.0


def to_chw3(unit):
    """(h, w) float32 -> (3, h, w), three identical channels.

    The backbone is ImageNet-pretrained and expects 3 channels. Replicating one
    is not a waste: the pretrained filters were learned on colour images but
    the useful early ones are edges and texture, which survive intact.
    """
    return np.ascontiguousarray(np.repeat(unit[None, :, :], 3, axis=0))


def preprocess(bgr, roi, size):
    """The full eval-time pipeline: raw BGR frame -> blob [1, 3, h, w], 0..1.

    Byte-for-byte what monitor.model_blob produces. verify_onnx.py checks it.
    """
    gray = resize_to(to_gray(crop_roi(bgr, roi)), size)
    return to_chw3(to_unit(gray))[None, ...]
