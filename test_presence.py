#!/usr/bin/env python3
"""Checks on the learned presence layer, without a model file or a camera.

The classifier itself is a black box that lives in an ONNX file and is not
checked in. What IS checkable, and what these pin, is everything around it: how
hand-labeled windows are read, how a probability becomes a vote, how a vote
reaches the state machine, and what the machine does with a stream of them.

Those are the parts that decide whether a night gets recorded honestly. The
model only decides how often they are handed the right answer.

Run: python test_presence.py
"""

import os
import tempfile

import numpy as np

from monitor import (DEFAULTS, SleepState, carry_vote, model_blob, model_vote,
                     presence_label_at, presence_source, read_presence_labels,
                     vote_to_presence)

CFG = {**DEFAULTS, "quiet_score": 0.010, "active_score": 0.040,
       "presence_samples_to_flip": 2, "quiet_samples_to_sleep": 12,
       "active_samples_to_wake": 2}

STILL, MOVE = 0.000, 0.100

LABELS_CSV = """start,end,label,notes
2026-08-20T22:00:00,2026-08-21T06:00:00,dog,overnight in the pen
2026-08-21T08:00:00,2026-08-21T09:30:00,empty,breakfast walk
2026-08-21T00:10:00,2026-08-21T00:40:00,empty,carried upstairs, pen empty
not-a-time,2026-08-21T10:00:00,dog,malformed start
2026-08-21T12:00:00,2026-08-21T11:00:00,dog,end before start
2026-08-21T13:00:00,2026-08-21T14:00:00,maybe,not a valid label
"""


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} {got!r}"
          + ("" if ok else f"  wanted {want!r}"))
    return ok


def at(text):
    from datetime import datetime
    return datetime.fromisoformat(text)


def sample(vote, score=STILL):
    """One watch-loop sample as the machine sees it under a model.

    Mirrors cmd_watch exactly: the vote is encoded through vote_to_presence,
    and ref_corr is None because the model does not consult the references and
    their staleness says nothing about whether it can see a dog.
    """
    pres, thr = vote_to_presence(vote)
    return (score, 1.0, pres, thr, None)


def run(votes, start="unknown"):
    m = SleepState(CFG, state=start)
    out = [m.update(*s) for s in votes]
    return m, out


def main():
    results = []

    # --- label windows ---
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as fh:
        fh.write(LABELS_CSV)
    try:
        windows = read_presence_labels(path)
        results.append(check("malformed and inverted rows are dropped",
                             len(windows), 3))
        results.append(check("windows come back sorted by start",
                             [w[0].isoformat() for w in windows],
                             ["2026-08-20T22:00:00", "2026-08-21T00:10:00",
                              "2026-08-21T08:00:00"]))
        results.append(check("notes survive the round trip",
                             windows[2][3], "breakfast walk"))

        results.append(check("inside a dog window -> dog",
                             presence_label_at(at("2026-08-20T23:30:00"),
                                               windows), "dog"))
        results.append(check("inside an empty window -> empty",
                             presence_label_at(at("2026-08-21T08:45:00"),
                                               windows), "empty"))
        # The vet-trip case: a narrow correction written after a broad claim.
        # The later window is the one made with the frames in front of you.
        results.append(check("overlap: the later window wins",
                             presence_label_at(at("2026-08-21T00:20:00"),
                                               windows), "empty"))
        results.append(check("...and outside the correction, the broad claim holds",
                             presence_label_at(at("2026-08-21T01:00:00"),
                                               windows), "dog"))
        results.append(check("between windows -> None",
                             presence_label_at(at("2026-08-21T07:00:00"),
                                               windows), None))
        results.append(check("before every window -> None",
                             presence_label_at(at("2026-08-19T12:00:00"),
                                               windows), None))
        # Boundaries are inclusive on both ends, so a frame at the exact
        # second a window opens is labeled rather than silently dropped.
        results.append(check("the first instant of a window is inside",
                             presence_label_at(at("2026-08-20T22:00:00"),
                                               windows), "dog"))
        results.append(check("the last instant of a window is inside",
                             presence_label_at(at("2026-08-21T09:30:00"),
                                               windows), "empty"))
    finally:
        os.remove(path)

    results.append(check("a missing label file is empty, not fatal",
                         read_presence_labels("/nonexistent/labels.csv"), []))
    results.append(check("no labels means no verdict",
                         presence_label_at(at("2026-08-21T08:45:00"), []), None))

    # --- p -> vote ---
    T, D = 0.5, 0.15
    results.append(check("confident dog -> occupied",
                         model_vote(0.97, T, D), "occupied"))
    results.append(check("confident empty -> empty",
                         model_vote(0.02, T, D), "empty"))
    results.append(check("dead centre -> abstain", model_vote(0.50, T, D), None))
    results.append(check("just inside the upper band -> abstain",
                         model_vote(0.6499, T, D), None))
    results.append(check("exactly on the upper edge -> occupied",
                         model_vote(0.65, T, D), "occupied"))
    results.append(check("exactly on the lower edge -> empty",
                         model_vote(0.35, T, D), "empty"))
    results.append(check("just inside the lower band -> abstain",
                         model_vote(0.3501, T, D), None))
    # A zero deadband must not create a hole at the threshold itself.
    results.append(check("zero deadband still decides at the threshold",
                         model_vote(0.5, 0.5, 0.0), "occupied"))
    results.append(check("p = 0 is empty at any threshold",
                         model_vote(0.0, 0.5, 0.15), "empty"))
    results.append(check("p = 1 is occupied at any threshold",
                         model_vote(1.0, 0.5, 0.15), "occupied"))

    # --- vote -> what the machine takes ---
    # The machine compares a score against a cutoff. The verdict is encoded as
    # a score that lands unambiguously on the right side of a fixed one, so
    # the machine needs no second input shape.
    results.append(check("occupied maps above its cutoff",
                         vote_to_presence("occupied"), (1.0, 0.5)))
    results.append(check("empty maps below its cutoff",
                         vote_to_presence("empty"), (0.0, 0.5)))
    results.append(check("abstain maps to 'not measured'",
                         vote_to_presence(None), (None, None)))
    occ, thr = vote_to_presence("occupied")
    results.append(check("...and the occupied encoding clears the cutoff",
                         occ > thr, True))
    emp, thr = vote_to_presence("empty")
    results.append(check("...and the empty encoding does not",
                         emp > thr, False))

    # --- which layer is answering ---
    results.append(check("a loaded model is the source",
                         presence_source(True, True), "model"))
    results.append(check("no model, references present -> reference",
                         presence_source(False, True), "reference"))
    results.append(check("neither -> none, which is a real answer",
                         presence_source(False, False), "none"))

    # --- model votes driving the state machine, ref_corr=None throughout ---
    _m, r = run([sample("occupied")] * 12)
    results.append(check("occupied + still 12x -> asleep", r[-1][0], "asleep"))
    _m, r = run([sample("occupied", MOVE)] * 2, start="asleep")
    results.append(check("occupied + moving 2x -> awake", r[-1][0], "awake"))
    _m, r = run([sample("empty")] * 2, start="asleep")
    results.append(check("empty votes 2x -> away", r[-1][0], "away"))
    _m, r = run([sample("empty")], start="asleep")
    results.append(check("1 empty vote does not flip to away", r[-1][0], "asleep"))

    # The whole reason this layer exists: an empty pen is perfectly still, so
    # motion alone reports it as sleep. 30 samples is 2.5 minutes.
    _m, r = run([sample("empty")] * 30)
    results.append(check("empty pen never reports asleep", r[-1][0], "away"))

    # ref_corr is never fed, so reference trust must stay untouched. If it
    # collapsed, `away` would be disabled and stillness would report unknown
    # -- the model would be doing its job and the machine would ignore it.
    m, _r = run([sample("empty")] * 30)
    results.append(check("reference trust is untouched by model votes",
                         m.presence_reliable, True))

    # The interlock survives the encoding: flipping to empty still requires
    # the frame to be BOTH unlike-a-dog AND still, so a wrong "empty" while
    # someone is reaching into the pen cannot delete real activity.
    _m, r = run([sample("empty", MOVE)] * 10, start="awake")
    results.append(check("empty vote + motion holds state", r[-1][0], "awake"))
    results.append(check("...and is tagged for review", r[-1][1], "empty+motion"))

    # Abstain is "not measured", so the presence branch is skipped entirely
    # and the sample is judged on motion alone.
    _m, r = run([sample("occupied")] * 12 + [sample(None)] * 5)
    results.append(check("abstaining does not end a sleep", r[-1][0], "asleep"))
    _m, r = run([sample("occupied", MOVE)] * 2 + [sample(None, MOVE)] * 3,
                start="asleep")
    results.append(check("abstaining does not undo a wake", r[-1][0], "awake"))
    # The machine alone cannot hold an away claim through an abstain: None is
    # "not measured", and a pen last voted empty falls straight into the
    # stillness logic, where twelve quiet abstains would manufacture a sleep.
    # This pins the raw machine behaviour so the carry below is visibly what
    # prevents it, not an accident of some other branch.
    _m, r = run([sample("empty")] * 2 + [sample(None)])
    results.append(check("raw machine: an abstain drops away to unknown",
                         r[-1][0], "unknown"))
    _m, r = run([sample("empty")] * 2 + [sample(None)] * 12)
    results.append(check("raw machine: 12 abstains on an empty pen WOULD sleep",
                         r[-1][0], "asleep"))

    # ...which is why cmd_watch never feeds the machine a bare abstain once a
    # decisive vote exists: carry_vote repeats the last real vote.
    results.append(check("carry: a decisive vote passes through",
                         carry_vote("occupied", None), ("occupied", "occupied")))
    results.append(check("carry: an abstain repeats the last vote",
                         carry_vote(None, "empty"), ("empty", "empty")))
    results.append(check("carry: nothing to carry before the first vote",
                         carry_vote(None, None), (None, None)))

    def run_carried(votes, start="unknown"):
        """The watch loop's real behaviour: abstains carry the last vote."""
        m = SleepState(CFG, state=start)
        last, out = None, []
        for vote, score in votes:
            acted, last = carry_vote(vote, last)
            out.append(m.update(*sample(acted, score)))
        return m, out

    _m, r = run_carried([("empty", STILL)] * 2 + [(None, STILL)] * 30)
    results.append(check("carried: 30 abstains on an empty pen stay away",
                         r[-1][0], "away"))
    results.append(check("...and never sleep", all(s != "asleep" for s, _t, _c in r), True))
    _m, r = run_carried([("empty", STILL)] * 2 + [(None, MOVE)] * 3)
    results.append(check("carried: motion under a carried empty is held+tagged",
                         r[-1][1], "empty+motion"))
    _m, r = run_carried([("empty", STILL)] * 2 + [(None, STILL)] * 5
                        + [("occupied", STILL)] * 13)
    results.append(check("carried: a real occupied vote still ends away",
                         r[-1][0], "asleep"))
    _m, r = run_carried([("occupied", STILL)] * 13 + [(None, STILL)] * 30)
    results.append(check("carried: abstains on an occupied pen keep sleeping",
                         r[-1][0], "asleep"))

    # Coming back must not inherit the quiet run built up while she was away.
    # 13, not 12: the first sample back is spent confirming presence.
    _m, r = run([sample("empty")] * 12 + [sample("occupied")] * 12)
    results.append(check("12 still after returning is not yet asleep",
                         r[-1][0], "unknown"))
    _m, r = run([sample("empty")] * 12 + [sample("occupied")] * 13)
    results.append(check("13 still after returning -> asleep", r[-1][0], "asleep"))

    # --- the preprocessing contract ---
    # Duplicated in train/preprocess.py and checked for equality by
    # train/verify_onnx.py. These pin the shape and range the ONNX graph is
    # entitled to assume, which is the half that can be checked without torch.
    frame = np.zeros((360, 640, 3), np.uint8)
    frame[:, :] = 128
    blob = model_blob(frame, None, (160, 128))
    results.append(check("blob is NCHW [1, 3, h, w]", blob.shape, (1, 3, 128, 160)))
    results.append(check("blob is float32", blob.dtype.name, "float32"))
    results.append(check("blob is scaled to 0..1",
                         round(float(blob.max()), 4), round(128 / 255, 4)))
    results.append(check("the three channels are identical",
                         bool(np.array_equal(blob[0, 0], blob[0, 2])), True))
    results.append(check("the blob is contiguous for cv2.dnn",
                         blob.flags["C_CONTIGUOUS"], True))
    # The ROI is a fraction of the frame, so the same config crops a 1280x720
    # live frame and a 640x360 archived one to the same view. The blob comes
    # out at the model's input size either way.
    roi = [0.0625, 0.0667, 0.7266, 0.9264]
    big = np.zeros((720, 1280, 3), np.uint8)
    results.append(check("a fractional ROI still yields the model input size",
                         model_blob(big, roi, (160, 128)).shape,
                         (1, 3, 128, 160)))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
