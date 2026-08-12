#!/usr/bin/env python3
"""Checks on the state machine, independent of any camera or image data.

SleepState is the layer that decides asleep/awake from a stream of scores.
These pin the behaviour that the thresholds are tuned against, so a refactor
cannot quietly change what `tune` is optimizing.

Run: python test_policy.py
"""

from monitor import DEFAULTS, SleepState

CFG = {**DEFAULTS, "quiet_score": 0.010, "active_score": 0.040,
       "scene_change_score": 0.60, "quiet_samples_to_sleep": 12,
       "active_samples_to_wake": 2}

STILL, MOVE, DEAD, SCENE = 0.000, 0.100, 0.025, 0.900


def run(scores, start="unknown"):
    m = SleepState(CFG, state=start)
    return [m.update(s) for s in scores]


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

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
