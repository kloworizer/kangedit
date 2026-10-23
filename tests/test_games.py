"""Gaming moments (Auto Edit → Prioritize: Gaming): kills, multi-kills, a gacha reveal, measured against what
is usual in the recording, so a game that is loud all the time doesn't count as one long moment."""

import numpy as np

from kangedit import games
from kangedit.analysis import SAMPLE_RATE, Analysis
from kangedit.autoedit import STYLE_LABELS, AutoEditSettings, hook_suggestions, select
from kangedit.models import Footage

KILL, TRIPLE, GACHA, DECOY = 20.0, (50.0, 52.0, 54.0), 90.0, (100.0, 115.0)


def gameplay(dur=130.0, seed=3) -> Analysis:
    """A busy, loud game recording with one kill, a triple kill, a gacha pull after a quiet spin and a long
    stretch of steady loud music (not a moment)."""
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    audio = -24.0 + rng.normal(0, 1.5, n)
    act = 6.0 + rng.normal(0, 0.8, n)
    bright = 0.40 + rng.normal(0, 0.02, n)
    color = 30.0 + rng.normal(0, 3.0, n)

    def burst(at, length=0.75, db=12.0, a=6.0, b=0.0, c=0.0):
        k = (t >= at) & (t < at + length)
        audio[k] += db
        act[k] += a
        bright[k] += b
        color[k] += c

    burst(KILL)
    for at in TRIPLE:
        burst(at)
    spin = (t >= GACHA - 8) & (t < GACHA)                        # the quiet spin before the reveal
    audio[spin] -= 6.0
    act[spin] -= 3.0
    burst(GACHA, 1.5, db=8.0, a=8.0, b=0.35, c=45.0)             # golden flash, colour flood, sting
    loud = (t >= DECOY[0]) & (t < DECOY[1])
    audio[loud] += 10.0                                           # steady loud music: no burst in it
    z = np.zeros(n, np.float32)
    return Analysis(times=t, sharp=np.full(n, 5.0, np.float32), activity=act.astype(np.float32), motion=z + 30,
                    jerk=z + 0.5, brightness=bright.astype(np.float32), clipped=z, colorful=color.astype(np.float32),
                    faces=z, audio=audio.astype(np.float32), cuts=np.array([GACHA], np.float32),
                    embed=rng.random((n, 144), dtype=np.float32), thumbs=np.zeros((0, 4, 4, 3), np.uint8),
                    thumb_times=np.zeros(0, np.float32), wave=np.zeros(10, np.float32))


def at(score, a, sec):
    return float(score[int(round(sec * SAMPLE_RATE))])


def test_kills_multi_kills_and_a_reveal_stand_out():
    a = gameplay()
    m = games.moments(a)
    kill, triple, gacha = at(m, a, KILL + 0.3), at(m, a, TRIPLE[1] + 0.3), at(m, a, GACHA + 0.5)
    usual = float(np.median(m))
    decoy = at(m, a, (DECOY[0] + DECOY[1]) / 2)
    assert usual < 0.1 and decoy < 0.15                         # loud all the time is not a moment
    assert kill > 0.4 and gacha > 0.5 and triple > kill         # a triple kill counts more than one kill
    # the moment's score covers its lead-up and aftermath, not the burst alone
    assert at(m, a, KILL - 1.0) > 0.3 and at(m, a, KILL + 2.5) > 0.3 and at(m, a, KILL - 4.0) < 0.15


def test_gaming_edit_picks_the_plays():
    a = gameplay()
    f = Footage("game.mp4", 130.0, 1920, 1080, 60.0, True)
    s = AutoEditSettings(target_duration=15.0, clip_min=3.0, clip_max=5.0, style="gaming", order="chronological")
    clips = select({f.id: f}, {f.id: a}, s)

    def has(c, sec):
        return c.src_in - 0.01 <= sec <= c.src_out

    assert any(has(c, KILL) for c in clips)
    assert any(any(has(c, x) for x in TRIPLE) for c in clips)
    assert any(has(c, GACHA) for c in clips)                     # even with a "scene change" at the reveal
    assert not any(c.src_in > DECOY[0] + 2 and c.src_out < DECOY[1] - 2 for c in clips)
    first = [c for c in clips if has(c, KILL)][0]
    assert first.src_in < KILL                                   # it starts before the kill
    hook = hook_suggestions({f.id: f}, {f.id: a}, s, [], 1, 4.0)[0]
    assert any(has(hook, x) for x in TRIPLE + (GACHA,))          # the hook is one of the best plays
    assert "gaming" in STYLE_LABELS


def test_short_or_silent_recordings():
    a = gameplay(dur=1.5)
    assert games.moments(a).shape == (len(a.times),)
    b = gameplay()
    m = games.moments(b, has_audio=False)                        # no sound: the picture still shows the reveal
    assert at(m, b, GACHA + 0.5) > 0.3
