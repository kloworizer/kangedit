"""Auto edit from a script: each line of the script is found in the footage's transcripts (the word timings the
captions use) by text similarity, and the clips are laid out in the script's order.

Matching is fuzzy so speech-recognition slips ("their" / "there", a dropped word) and small ad-libs still
match: the score is a character-level similarity of the normalised texts (0..1), and only matches at or above
a threshold are used. A word-overlap prefilter keeps it fast on hours of footage: only windows of the
transcript that share enough words with the line are scored in full.

When a line was said in several takes, the best-scoring one is used, or the last good take (within a little
of the best), as an editor usually keeps the last attempt. Clips are cut on word boundaries with padding,
never into the neighbouring words, and consecutive lines found one after the other in the same take become
one clip (no jump cut).
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import Footage, TimelineClip

DEFAULT_THRESHOLD = 0.6
LAST_TAKE_MARGIN = 0.05      # "last good take": a later take within this of the best score
MERGE_GAP = 0.6              # seconds: consecutive lines in the same take closer than this become one clip
_LABEL = re.compile(r"^\s*[A-Z][A-Z0-9 .'\-]{0,30}:\s+")      # "ANNA:" speaker labels
_DIRECTIONS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")    # [stage directions] (asides)


@dataclass
class Match:
    """Where one script line was found (footage_id "" when it wasn't)."""
    index: int
    line: str
    score: float = 0.0
    footage_id: str = ""
    w0: int = 0                  # first and last matched word (indices into the footage's transcript)
    w1: int = 0
    src_in: float = 0.0
    src_out: float = 0.0
    heard: str = ""              # what the transcript says there
    takes: int = 0               # how many takes reached the threshold

    @property
    def found(self) -> bool:
        return bool(self.footage_id)


@dataclass
class ScriptEdit:
    matches: list[Match] = field(default_factory=list)
    clips: list[TimelineClip] = field(default_factory=list)

    @property
    def missing(self) -> list[Match]:
        return [m for m in self.matches if not m.found]


def split_script(text: str) -> list[str]:
    """The script's lines: one per line of text, long lines split into sentences; speaker labels and
    [directions] removed, empty lines dropped."""
    out = []
    for raw in text.splitlines():
        line = _DIRECTIONS.sub(" ", _LABEL.sub("", raw)).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?…])\s+(?=\S)", line)
        out += [p.strip() for p in parts if normalize(p)]
    return out


def normalize(text: str) -> str:
    """Lower case, accents and punctuation off, single spaces: what is compared."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^\w\s']|_", " ", text).replace("'", "")
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    """0..1 similarity of two normalised texts (characters, in order)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


class _Index:
    """One footage's transcript prepared for matching: normalised words, their ids, and times."""

    def __init__(self, footage: Footage):
        self.footage = footage
        words = footage.transcript or []
        self.raw = words
        self.norm = [normalize(w["w"]) for w in words]
        keep = [i for i, w in enumerate(self.norm) if w]
        self.idx = np.array(keep, dtype=np.int64)          # transcript index of each kept word
        self.text = [self.norm[i] for i in keep]
        vocab: dict[str, int] = {}
        self.ids = np.array([vocab.setdefault(w, len(vocab)) for w in self.text], dtype=np.int64)
        self.vocab = vocab

    def candidates(self, line_words: list[str], top: int = 12) -> list[int]:
        """Start positions (into self.text) of windows sharing the most words with the line."""
        n = len(line_words)
        if n == 0 or len(self.ids) == 0:
            return []
        want = {self.vocab[w] for w in line_words if w in self.vocab}
        if not want:
            return []
        hit = np.isin(self.ids, list(want)).astype(np.int32)
        win = max(1, n)
        csum = np.concatenate([[0], np.cumsum(hit)])
        starts = np.arange(max(1, len(hit) - win + 1))
        overlap = csum[np.minimum(starts + win, len(hit))] - csum[starts]
        need = max(1, int(np.ceil(0.35 * len(set(line_words)))))
        good = np.flatnonzero(overlap >= need)
        if good.size == 0:
            return []
        order = good[np.argsort(-overlap[good], kind="stable")]
        picked: list[int] = []
        for s in order:                      # spread out: one candidate per neighbourhood
            if all(abs(int(s) - p) > win // 2 for p in picked):
                picked.append(int(s))
            if len(picked) >= top:
                break
        return picked

    def best_at(self, line: str, n: int, start: int) -> tuple[float, int, int]:
        """The best window near `start`: (score, first word, last word) in self.text positions. Windows are
        ranked by a quick word-level comparison; only the best few get the exact character-level score."""
        lo_len, hi_len = max(1, int(n * 0.7)), int(np.ceil(n * 1.35)) + 1
        words = difflib.SequenceMatcher(None, autojunk=False)
        words.set_seq2(line.split())
        ranked = []
        for s in range(max(0, start - 3), min(len(self.text), start + 4)):
            for k in range(lo_len, hi_len + 1):
                e = s + k
                if e > len(self.text):
                    break
                words.set_seq1(self.text[s:e])
                ranked.append((words.ratio(), s, e))
        ranked.sort(key=lambda r: -r[0])
        best = (0.0, start, start)
        chars = difflib.SequenceMatcher(None, autojunk=False)
        chars.set_seq2(line)
        for _wr, s, e in ranked[:3]:
            chars.set_seq1(" ".join(self.text[s:e]))
            sc = chars.ratio()
            if sc > best[0]:
                best = (sc, s, e - 1)
        return best


def match_script(script: str | list[str], footage: list[Footage], threshold: float = DEFAULT_THRESHOLD,
                 prefer: str = "best") -> list[Match]:
    """Find every script line in the footage's transcripts. prefer: "best" (highest score) or "last" (the last
    take within LAST_TAKE_MARGIN of the best; footage order then time)."""
    lines = split_script(script) if isinstance(script, str) else list(script)
    indexes = [_Index(f) for f in footage if f.transcript]
    out = []
    for i, line in enumerate(lines):
        norm = normalize(line)
        words = norm.split()
        m = Match(i, line)
        takes = []          # (score, order, index, w0, w1)
        for order, ix in enumerate(indexes):
            for s in ix.candidates(words):
                sc, a, b = ix.best_at(norm, len(words), s)
                if sc >= threshold:
                    takes.append((sc, order, ix, a, b))
        best_score = max((t[0] for t in takes), default=0.0)
        if not takes:
            # Not found: keep the closest for the report.
            for ix in indexes:
                for s in ix.candidates(words, top=3):
                    sc, a, b = ix.best_at(norm, len(words), s)
                    if sc > m.score:
                        m.score, m.heard = sc, " ".join(ix.text[a:b + 1])
            out.append(m)
            continue
        # The same take found from nearby starts counts once: overlapping word ranges are one take.
        distinct: list = []
        for t in sorted(takes, key=lambda t: -t[0]):
            if not any(t[1] == d[1] and t[3] <= d[4] and d[3] <= t[4] for d in distinct):
                distinct.append(t)
        takes = distinct
        if prefer == "last":
            ok = [t for t in takes if t[0] >= best_score - LAST_TAKE_MARGIN]
            pick = max(ok, key=lambda t: (t[1], t[2].raw[t[2].idx[t[3]]]["s"]))
        else:
            pick = max(takes, key=lambda t: t[0])
        sc, _order, ix, a, b = pick
        w0, w1 = int(ix.idx[a]), int(ix.idx[b])
        m.score, m.footage_id, m.w0, m.w1 = sc, ix.footage.id, w0, w1
        m.src_in, m.src_out = ix.raw[w0]["s"], ix.raw[w1]["e"]
        m.heard = " ".join(ix.text[a:b + 1])
        m.takes = len(takes)
        out.append(m)
    return out


def build_clips(matches: list[Match], footage: dict[str, Footage], pad_in: float = 0.15, pad_out: float = 0.25,
                merge: bool = True) -> list[TimelineClip]:
    """Clips for the found lines in script order: padded, never into the neighbouring words; consecutive
    lines from the same take, one after the other, joined into one clip."""
    clips: list[TimelineClip] = []
    last: Optional[Match] = None
    for m in matches:
        if not m.found:
            continue
        f = footage[m.footage_id]
        words = f.transcript or []
        prev_e = words[m.w0 - 1]["e"] if m.w0 > 0 else 0.0
        next_s = words[m.w1 + 1]["s"] if m.w1 + 1 < len(words) else f.duration
        a = max(0.0, m.src_in - pad_in, min(m.src_in, prev_e + 0.02))
        b = min(f.duration, m.src_out + pad_out, max(m.src_out, next_s - 0.02))
        if (merge and clips and last is not None and last.footage_id == m.footage_id
                and m.w0 > last.w1 and m.src_in - last.src_out <= MERGE_GAP):
            clips[-1].src_out = round(max(clips[-1].src_out, b), 3)
        else:
            clips.append(TimelineClip(f.id, round(a, 3), round(max(b, a + 0.1), 3), label="script"))
        last = m
    return clips


def script_edit(script: str, footage: list[Footage], threshold: float = DEFAULT_THRESHOLD, prefer: str = "best",
                pad_in: float = 0.15, pad_out: float = 0.25, merge: bool = True) -> ScriptEdit:
    matches = match_script(script, footage, threshold, prefer)
    return ScriptEdit(matches, build_clips(matches, {f.id: f for f in footage}, pad_in, pad_out, merge))
