#!/usr/bin/env python3
"""Checks on the state machine, independent of any camera or image data.

SleepState is the layer that decides asleep/awake from a stream of scores.
These pin the behaviour that the thresholds are tuned against, so a refactor
cannot quietly change what `tune` is optimizing.

Run: python test_policy.py
"""

from monitor import DEFAULTS, SleepState

CFG = {**DEFAULTS, "quiet_score": 0.010, "active_score": 0.040,
       "scene_change_score": 0.60, "scene_corr_max": 0.50,
       "presence_threshold": 0.030, "presence_samples_to_flip": 2,
       "quiet_samples_to_sleep": 12, "active_samples_to_wake": 2}

STILL, MOVE, DEAD = 0.000, 0.100, 0.025
SCENE = (0.900, -0.80)   # huge and uncorrelated: the room re-lit
BIG_DOG = (0.820, 0.93)  # huge but still correlated: a dog filling the frame


def run(scores, start="unknown"):
    """Scores may be plain floats, or (score, corr) or (score, corr, presence)."""
    m = SleepState(CFG, state=start)
    return [m.update(*s) if isinstance(s, tuple) else m.update(s) for s in scores]


# Three-state inputs: (motion_score, correlation, presence_score).
# presence above 0.030 = a dog is in the pen; 0.0000 = matches an empty
# reference exactly, which is what the real measurements looked like.
IN_STILL = (0.000, 1.0, 0.125)
IN_MOVING = (0.100, 1.0, 0.150)
GONE = (0.000, 1.0, 0.000)
GONE_BUT_MOTION = (0.100, 1.0, 0.000)   # stale reference, or a person

# Five-value inputs add ref_corr, the correlation between the frame and the
# reference the presence score came from. The numbers are the real 2026-08-14
# failure: the crate was rearranged, an EMPTY pen scored 0.84 "occupied"
# against the old reference at corr +0.05, and the monitor reported 12 hours
# of it. A matched reference measures corr 0.77-1.00 whether or not the dog
# is in frame.
IN_STILL_MATCHED = (0.000, 1.0, 0.125, None, 0.90)
GONE_MATCHED = (0.000, 1.0, 0.000, None, 0.99)
STALE_STILL = (0.000, 1.0, 0.840, None, 0.05)   # empty pen, moved furniture
STALE_MOVING = (0.100, 1.0, 0.840, None, 0.05)  # ...with someone in frame


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} {got!r}"
          + ("" if ok else f"  wanted {want!r}"))
    return ok


def main():
    results = []

    # Sleeping takes a full run of 12; 11 is not enough.
    r = run([STILL] * 11)
    results.append(check("11 still samples do not sleep", r[-1][0], "unknown"))
    r = run([STILL] * 12)
    results.append(check("12 still samples sleep", r[-1][0], "asleep"))
    results.append(check("...and the 12th is the transition", r[11][2], True))

    # Waking is fast and asymmetric.
    r = run([MOVE], start="asleep")
    results.append(check("1 moving sample does not wake", r[-1][0], "asleep"))
    r = run([MOVE, MOVE], start="asleep")
    results.append(check("2 moving samples wake", r[-1][0], "awake"))

    # The deadband is the anti-flap mechanism: it must advance nothing.
    r = run([STILL] * 6 + [DEAD] * 20 + [STILL] * 6)
    results.append(check("deadband never sleeps on its own", r[25][0], "unknown"))
    results.append(check("deadband preserves the quiet run across it",
                         r[-1][0], "asleep"))
    r = run([DEAD] * 50, start="awake")
    results.append(check("deadband alone never flips state", r[-1][0], "awake"))

    # A single still sample must not undo a wake, and vice versa.
    r = run([MOVE, MOVE] + [STILL] * 11, start="asleep")
    results.append(check("11 still after waking stays awake", r[-1][0], "awake"))

    # Scene change resets both counters instead of counting as movement.
    r = run([STILL] * 11 + [SCENE] + [STILL] * 11, start="awake")
    results.append(check("scene change resets the quiet run", r[-1][0], "awake"))
    r = run([MOVE, SCENE, MOVE], start="asleep")
    results.append(check("scene change resets the active run", r[-1][0], "asleep"))
    results.append(check("scene change is tagged, not scored", r[1][1], "scene"))

    # The live failure this guard originally caused: a dog close to the camera
    # scored 0.82, tripped the magnitude-only scene test, and stayed "asleep"
    # while visibly moving.
    r = run([BIG_DOG, BIG_DOG], start="asleep")
    results.append(check("huge but correlated change wakes", r[-1][0], "awake"))
    results.append(check("...and is tagged as movement", r[0][1], "MOVING"))

    # --- three-state composition: away / asleep / awake ---
    r = run([IN_STILL] * 12)
    results.append(check("occupied + still 12x -> asleep", r[-1][0], "asleep"))
    r = run([IN_MOVING] * 2, start="asleep")
    results.append(check("occupied + moving 2x -> awake", r[-1][0], "awake"))

    r = run([GONE] * 2, start="asleep")
    results.append(check("empty 2x -> away", r[-1][0], "away"))
    r = run([GONE], start="asleep")
    results.append(check("1 empty sample does not flip to away", r[-1][0], "asleep"))

    # The failure that matters: an empty pen is still, so without presence it
    # accumulates a quiet run and reports sleep. 30 samples is 2.5 minutes.
    r = run([GONE] * 30, start="unknown")
    results.append(check("empty pen never reports asleep", r[-1][0], "away"))

    # A stale reference plus a moving person must not read as "the dog left".
    r = run([GONE_BUT_MOTION] * 10, start="awake")
    results.append(check("empty-looking but moving holds state", r[-1][0], "awake"))
    results.append(check("...and is tagged for review", r[-1][1], "empty+motion"))

    # Coming back must not inherit the quiet run built up while she was away.
    r = run([GONE] * 12 + [IN_STILL] * 11)
    results.append(check("returning does not inherit the away quiet run",
                         r[-1][0], "unknown"))
    # 13, not 12: the first sample back is spent confirming presence before
    # the quiet run can start. A 5-second cost on re-entry.
    r = run([GONE] * 12 + [IN_STILL] * 12)
    results.append(check("12 still after returning is not yet asleep",
                         r[-1][0], "unknown"))
    r = run([GONE] * 12 + [IN_STILL] * 13)
    results.append(check("13 still after returning -> asleep", r[-1][0], "asleep"))

    # --- stale-reference honesty ---
    # A matched reference changes nothing about the composed behaviour.
    r = run([IN_STILL_MATCHED] * 12)
    results.append(check("matched ref: occupied + still -> asleep",
                         r[-1][0], "asleep"))
    r = run([GONE_MATCHED] * 2, start="asleep")
    results.append(check("matched ref: empty 2x -> away", r[-1][0], "away"))

    # The 2026-08-14 failure: empty pen, rearranged furniture, huge presence
    # score against a reference that no longer describes the room. The old
    # machine called this asleep for 12 hours. It must be "unknown" -- with a
    # stale reference, stillness proves nothing.
    r = run([STALE_STILL] * 30, start="unknown")
    results.append(check("stale ref never reports asleep", r[-1][0], "unknown"))
    results.append(check("...and the sample is tagged", r[-1][1], "ref-stale"))
    r = run([STALE_STILL] * 30, start="asleep")
    results.append(check("stale ref ends an in-progress sleep claim",
                         r[-1][0], "unknown"))

    # One low-correlation sample is an occlusion, not a scene change.
    r = run([IN_STILL_MATCHED] * 11 + [STALE_STILL] + [IN_STILL_MATCHED])
    results.append(check("1 low-corr sample does not break trust",
                         r[-1][0], "asleep"))

    # Movement stands on its own: motion needs no reference to mean activity.
    r = run([STALE_MOVING] * 2, start="unknown")
    results.append(check("stale ref still allows waking on motion",
                         r[-1][0], "awake"))

    # Recapturing a reference (frames match again) restores the full machine:
    # trust flips back after the same 2-sample hysteresis, and an empty pen
    # can read away again.
    r = run([STALE_STILL] * 15 + [GONE_MATCHED] * 4)
    results.append(check("fresh ref restores away detection", r[-1][0], "away"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
