"""Sound effects: the local library, quick import from MyInstants, and audio helpers for the SFX layer.

MyInstants (www.myinstants.com) hosts short user-uploaded sound clips. Searching reads its public search
page; a chosen sound is downloaded once into the library. The sounds are uploaded by its users, so check
that you may use one before publishing a video with it.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import sfx_dir
from .models import LOOP_CROSSFADE, SoundClip
from .ffmpeg import ffmpeg_bin, ffprobe_bin, run

AUDIO_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".opus", ".wma", ".aif", ".aiff")
MYINSTANTS = "https://www.myinstants.com"
_UA = {"User-Agent": "Mozilla/5.0 (KangEdit video editor)"}
ENVELOPE_RATE = 50   # waveform points per second for drawing


@dataclass
class OnlineSound:
    name: str
    url: str          # direct audio URL
    page: str         # the sound's page on the site


def library() -> list[tuple[str, str]]:
    """(name, path) of every sound in the library, sorted by name."""
    out = []
    for e in os.scandir(sfx_dir()):
        if e.is_file() and e.name.lower().endswith(AUDIO_EXTENSIONS):
            out.append((os.path.splitext(e.name)[0], e.path))
    return sorted(out, key=lambda it: it[0].lower())


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-. ()]+", "_", name, flags=re.UNICODE).strip(" ._") or "sound"
    return name[:80]


def import_file(src: str) -> str:
    """Copy an audio file into the library (an identical copy is reused). Returns the library path."""
    base, ext = os.path.splitext(os.path.basename(src))
    dst = os.path.join(sfx_dir(), _safe_name(base) + ext.lower())
    k = 2
    while os.path.exists(dst):
        if os.path.getsize(dst) == os.path.getsize(src):
            return dst
        dst = os.path.join(sfx_dir(), f"{_safe_name(base)} ({k}){ext.lower()}")
        k += 1
    shutil.copy2(src, dst)
    return dst


def search_myinstants(query: str, page: int = 1, timeout: float = 20.0) -> list[OnlineSound]:
    """Sounds matching `query` on MyInstants (about 36 per page)."""
    q = urllib.parse.urlencode({"name": query, "page": page} if page > 1 else {"name": query})
    req = urllib.request.Request(f"{MYINSTANTS}/en/search/?{q}", headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")
    out = []
    for block in text.split('<div class="instant"')[1:]:
        m = re.search(r"play\('([^']+)'", block)
        n = re.search(r'<a href="([^"]+)"[^>]*class="instant-link[^"]*"[^>]*>([^<]+)</a>', block)
        if m and n:
            out.append(OnlineSound(html.unescape(n.group(2)).strip(), urllib.parse.urljoin(MYINSTANTS, m.group(1)),
                                   urllib.parse.urljoin(MYINSTANTS, n.group(1))))
    return out


def download(sound: OnlineSound, timeout: float = 30.0) -> str:
    """Save an online sound into the library (once). Returns the library path."""
    ext = os.path.splitext(urllib.parse.urlparse(sound.url).path)[1].lower() or ".mp3"
    if ext not in AUDIO_EXTENSIONS:
        ext = ".mp3"
    dst = os.path.join(sfx_dir(), _safe_name(sound.name) + ext)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    req = urllib.request.Request(sound.url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) < 256:
        raise OSError("the download is empty")
    tmp = dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dst)
    return dst


FIT_CROSSFADE = LOOP_CROSSFADE   # seconds of crossfade between the repeats of a sound fit to the timeline


def fit_to_timeline(x: SoundClip, file_len: float, total: float) -> list[SoundClip]:
    """The sound laid over the whole timeline (0..total), from its in point to the end of its file, as one
    clip on one row: looping with crossfades between repeats when it is shorter than the timeline, cut with a
    fade out when it is longer. Keeps the sound's id. (A list, for callers that replace sounds.)"""
    src_in = min(max(0.0, x.src_in), max(0.0, file_len - 0.05))
    unit = file_len - src_in
    if unit <= 0.05 or total <= 0.05:
        return [x]
    fade_out = round(max(x.fade_out, min(FIT_CROSSFADE, total / 4)), 3)
    y = SoundClip(x.path, 0.0, round(src_in, 3), round(src_in + min(unit, total), 3), x.volume, x.name,
                  x.fade_in, fade_out, x.duck, id=x.id)
    if unit < total - 0.01:
        if unit <= 2 * FIT_CROSSFADE + 0.05:   # too short to crossfade: one clip that ends with its file
            return [y]
        y.loop = round(unit, 3)
        y.src_out = round(src_in + total, 3)
    return [y]


def duration(path: str) -> float:
    """Length of an audio file in seconds (0 if unknown)."""
    res = run([ffprobe_bin(), "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    try:
        return float(json.loads(res.stdout.decode("utf-8", "replace"))["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0


_env_cache: dict[str, np.ndarray] = {}
_env_lock = threading.Lock()


def envelope(path: str) -> Optional[np.ndarray]:
    """Peak envelope 0..1 at ENVELOPE_RATE points/s for drawing (cached; None if it can't be decoded)."""
    with _env_lock:
        if path in _env_cache:
            return _env_cache[path]
    res = run([ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"])
    env = None
    if res.returncode == 0 and res.stdout:
        a = np.abs(np.frombuffer(res.stdout, np.float32))
        hop = 8000 // ENVELOPE_RATE
        n = len(a) // hop
        if n:
            env = a[:n * hop].reshape(n, hop).max(axis=1)
            env = np.clip(env / max(1e-4, float(env.max())), 0, 1).astype(np.float32)
    with _env_lock:
        _env_cache[path] = env
    return env
