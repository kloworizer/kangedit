"""Story order and the opening / closing moments of an edit.

Footage is put in story order by recording time (the oldest of the file's creation tag, creation date and
modification date, see scenes.recorded_time), then by natural file-name order.
The opening is looked for in footage named like an intro (intro, opening, pembukaan, …) and in the
first recorded files; the closing in outro-named and the last recorded files. Within those, speech is
searched for greeting / sign-off phrases (English and Indonesian) and the whole utterance around the
first greeting / last sign-off becomes the opening / closing clip. Without speech, a file named like an
intro or outro still gives its start / end.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, Optional

from .models import Footage, TimelineClip
from .scenes import recorded_time

OPENING_PHRASES = [
    "hello everyone", "hello guys", "hi everyone", "hi guys", "hey guys", "hey everyone", "welcome back",
    "welcome to", "welcome", "good morning", "good afternoon", "good evening", "today we", "in this video",
    "let's get started", "let's start", "hello", "opening",
    "assalamualaikum", "assalamu alaikum", "assalamu'alaikum", "halo semuanya", "halo guys", "hai semuanya",
    "hai guys", "selamat datang", "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
    "kembali lagi", "di video kali ini", "pada video kali ini", "hari ini kita", "kita mulai", "bismillah",
    "pembukaan", "halo", "hai", "hi", "hey", "salam", "assalamualaikum warahmatullahi", "asalamualaikum",
    "assalamu", "salam sejahtera", "apa kabar", "how are you", "what's up", "whats up", "yo guys",
]
CLOSING_PHRASES = [
    "thanks for watching", "thank you for watching", "see you next time", "see you in the next", "see you",
    "goodbye", "bye bye", "subscribe", "that's it for", "that's all for", "thank you", "thanks", "bye", "closing",
    "terima kasih sudah menonton", "terima kasih", "makasih", "sampai jumpa", "sampai ketemu",
    "wassalamualaikum", "wassalamu alaikum", "wassalamu'alaikum", "jangan lupa subscribe", "jangan lupa",
    "sekian", "demikian", "dadah", "penutup",
]
_INTRO_NAME = re.compile(r"(^|[^a-z])(intro|opening|opener|pembuka|pembukaan|prolog|prologue)([^a-z]|$)")
_OUTRO_NAME = re.compile(r"(^|[^a-z])(outro|closing|ending|penutup|penutupan|epilog|epilogue|credits?)([^a-z]|$)")
SPOT_SECONDS = 120.0     # how much of a file's start (opening) or end (closing) is listened to
GAP = 1.5                # silence that ends an utterance (pauses between sentences are shorter)


def recorded_at(f: Footage) -> Optional[float]:
    """When the footage was recorded: the oldest of its creation tag, file creation and modification dates."""
    return recorded_time(f)


def _natural(name: str) -> list:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name.lower())]


def story_order(footage: dict[str, Footage], ids: list[str]) -> list[str]:
    """Recording time first (files without one keep their place after timed ones), then natural name order."""
    def key(fid):
        t = recorded_at(footage[fid])
        return (t is None, t or 0.0, _natural(footage[fid].name))
    return sorted(ids, key=key)


def _tokens(words: list[dict]) -> list[str]:
    return [re.sub(r"[^\w']+", "", w["w"].lower()) for w in words]


def _find(words: list[dict], phrases: list[str], lo: float, hi: float, last: bool) -> Optional[tuple[int, int]]:
    """(first, last) word index of the earliest (or latest) phrase spoken within [lo, hi]."""
    toks = _tokens(words)
    hits = []
    for ph in phrases:
        pt = ph.split()
        for i in range(len(toks) - len(pt) + 1):
            if toks[i:i + len(pt)] == pt and lo <= words[i]["s"] <= hi:
                hits.append((i, i + len(pt) - 1))
    if not hits:
        return None
    return max(hits) if last else min(hits)


def _utterance(words: list[dict], i: int, j: int, max_len: float, forward: bool) -> tuple[float, float]:
    """Source span of the utterance holding words i..j: back to where speech started (within a few
    seconds), forward to the next pause (or the reverse for a closing), at most max_len long."""
    a, b = i, j
    while a > 0 and words[a]["s"] - words[a - 1]["e"] < GAP and words[b]["e"] - words[a - 1]["s"] <= max_len \
            and (forward is False or words[i]["s"] - words[a - 1]["s"] < 3.0):
        a -= 1
    while b + 1 < len(words) and words[b + 1]["s"] - words[b]["e"] < GAP and words[b + 1]["e"] - words[a]["s"] <= max_len:
        b += 1
    return words[a]["s"] - 0.25, words[b]["e"] + 0.4


def _clip(f: Footage, a: float, b: float, min_len: float, label: str) -> Optional[TimelineClip]:
    a, b = max(0.0, a), min(f.duration, b)
    if b - a < min_len:
        b = min(f.duration, a + min_len)
        a = max(0.0, b - min_len)
    if b - a < 0.5:
        return None
    return TimelineClip(footage_id=f.id, src_in=round(a, 3), src_out=round(b, 3), label=label)


def find_bookends(footage: dict[str, Footage], order: list[str], max_len: float, min_len: float,
                  listen: Optional[Callable[[Footage, float, float], list[dict]]] = None,
                  progress: Optional[Callable[[str], None]] = None,
                  cancel: Optional[threading.Event] = None) -> tuple[Optional[TimelineClip], Optional[TimelineClip], list[str]]:
    """Opening and closing clips (either may be None) and notes on how they were found.

    listen(footage, start, duration) returns words (source times) for that part of the file; it is only
    called for candidate files without a transcript."""
    notes: list[str] = []
    names = {fid: footage[fid].name.lower() for fid in order}
    intro_c = [fid for fid in order if _INTRO_NAME.search(names[fid])] + order[:2]
    outro_c = [fid for fid in reversed(order) if _OUTRO_NAME.search(names[fid])] + list(reversed(order[-2:]))
    intro_c = list(dict.fromkeys(intro_c))
    outro_c = list(dict.fromkeys(outro_c))

    def words_of(f: Footage, closing: bool) -> list[dict]:
        if f.transcript is not None:
            return f.transcript
        if not f.has_audio or listen is None:
            return []
        start = max(0.0, f.duration - SPOT_SECONDS) if closing else 0.0
        if progress:
            progress(f"Listening for the {'closing' if closing else 'opening'} in {f.name}")
        return listen(f, start, min(SPOT_SECONDS, f.duration))

    def search(cands: list[str], phrases: list[str], closing: bool, label: str) -> Optional[TimelineClip]:
        for fid in cands:
            if cancel is not None and cancel.is_set():
                return None
            f = footage[fid]
            words = words_of(f, closing)
            lo, hi = (max(0.0, f.duration - SPOT_SECONDS), f.duration) if closing else (0.0, SPOT_SECONDS)
            hit = _find(words, phrases, lo, hi, last=closing) if words else None
            if hit:
                a, b = _utterance(words, hit[0], hit[1], max_len, forward=not closing)
                said = " ".join(w["w"] for w in words[hit[0]:hit[1] + 1])
                notes.append(f"{label.capitalize()}: \"{said}\" in {f.name} at {words[hit[0]]['s']:.1f}s")
                return _clip(f, a, b, min_len, label)
        # No speech cue: a file named like an intro / outro still gives its start / end.
        rx = _OUTRO_NAME if closing else _INTRO_NAME
        for fid in cands:
            f = footage[fid]
            if rx.search(names[fid]):
                notes.append(f"{label.capitalize()}: from the file name {f.name}")
                if closing:
                    return _clip(f, f.duration - 0.6 - max_len, f.duration - 0.6, min_len, label)
                return _clip(f, 0.6, 0.6 + max_len, min_len, label)
        return None

    intro = search(intro_c, OPENING_PHRASES, False, "opening")
    outro = search(outro_c, CLOSING_PHRASES, True, "closing")
    if intro and outro and intro.footage_id == outro.footage_id and intro.src_out > outro.src_in:
        outro = None   # one short file holding both: keep the opening
    return intro, outro, notes
