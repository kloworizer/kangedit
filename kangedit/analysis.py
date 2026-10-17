"""Per-footage content analysis: the raw signals used to find the best moments.

Speed: ffmpeg decodes at the 4 fps sample rate and skips non-reference frames (they are never sampled).
Long clips are split into segments, and the segments of every queued clip are spread over several CPU
lanes plus one GPU lane (hardware decoding on NVIDIA, AMD or Intel) when one is available.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from . import smart
from .faces import detect as face_detect
from .config import cache_dir, file_key
from .ffmpeg import HW_OUTPUT_FORMAT, Cancelled, ProgressCb, ffmpeg_bin, ffmpeg_matrix, hw_decoder, popen
from .models import Footage

log = logging.getLogger("kangedit.analysis")

SAMPLE_RATE = 4.0       # analysis samples per second
THUMB_EVERY = 4         # keep one thumbnail per second
THUMB_W = 128
ANALYSIS_W = 320
WAVE_RATE = 20          # waveform envelope points per second
CACHE_VERSION = 2

SEGMENT_MIN = 20.0      # seconds; clips are split into segments at least this long
OVERLAP = 2             # samples decoded before a segment to seed motion and shake

HW_CODECS = {"h264", "hevc", "vp9", "av1", "mpeg2video", "vc1"}
# Source pixel format -> format of the frames downloaded from the GPU (4:2:0 only; others decode on the CPU).
_HW_FORMATS = {"yuv420p": "nv12", "yuvj420p": "nv12", "nv12": "nv12", "yuv420p10le": "p010le", "p010le": "p010le"}
# Rotation (counter-clockwise degrees) applied by hand when ffmpeg's autorotate is off (GPU path).
_ROTATE = {90: ",transpose=cclock", 180: ",hflip,vflip", 270: ",transpose=clock"}

_hw_failed: set[tuple[str, str]] = set()   # (codec, pix_fmt) the GPU could not decode this session
_hw_misses: dict[tuple[str, str], int] = {}   # files of a format that failed on the GPU (none of them worked)
_hw_worked: set[tuple[str, str]] = set()     # formats that decoded on the GPU this session
HW_GIVE_UP = 3          # failed files (and no success) before a format stops being tried on the GPU


def _gpu_failed(j) -> None:
    """A file failed to decode on the GPU: it goes to the CPU, and its format too after a few such files (one
    damaged file doesn't take the GPU away from all the others)."""
    key = (j.footage.codec, j.footage.pix_fmt)
    j.hw_format = None
    _hw_misses[key] = _hw_misses.get(key, 0) + 1
    log.info("analysis: GPU decoding failed for %s (%s %s)", j.footage.path, *key)
    if key not in _hw_worked and _hw_misses[key] >= HW_GIVE_UP:
        _hw_failed.add(key)
        log.info("analysis: %s %s is decoded on the CPU from now on", *key)


def _gpu_worked(j) -> None:
    _hw_worked.add((j.footage.codec, j.footage.pix_fmt))


@dataclass
class Analysis:
    times: np.ndarray        # (N,) sample times
    sharp: np.ndarray        # log Laplacian variance
    activity: np.ndarray     # mean abs frame difference
    motion: np.ndarray       # global camera translation per sample (px @ 320w)
    jerk: np.ndarray         # change of translation (shake)
    brightness: np.ndarray   # mean luma 0..1
    clipped: np.ndarray      # fraction of crushed/clipped pixels
    colorful: np.ndarray     # Hasler-Suesstrunk colorfulness
    faces: np.ndarray        # face area fraction
    audio: np.ndarray        # RMS loudness (dBFS) per sample
    cuts: np.ndarray         # times of detected scene changes
    embed: np.ndarray        # (N, 144) tiny grey frames for similarity
    thumbs: np.ndarray       # (M, h, w, 3) RGB thumbnails, 1 per second
    thumb_times: np.ndarray  # (M,)
    wave: np.ndarray         # (K,) waveform envelope 0..1 at WAVE_RATE
    # Smarter analysis (smart.py), empty without it: what each thumbnail shows, and what is heard.
    sem: np.ndarray = field(default_factory=lambda: np.zeros((0, smart.EMBED_DIM), np.float16))   # (M, 1280)
    sound: np.ndarray = field(default_factory=lambda: np.zeros((0, len(smart.GROUP_NAMES)), np.uint8))
    quick: bool = False      # a quick first look (keyframes only, see quick_look); the full analysis follows

    def save(self, path: str) -> None:
        np.savez_compressed(path, version=CACHE_VERSION,
                            **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @staticmethod
    def load(path: str) -> Optional["Analysis"]:
        try:
            with np.load(path) as d:
                if int(d.get("version", 0)) != CACHE_VERSION:
                    return None
                kw = {k: d[k] for k in Analysis.__dataclass_fields__ if k in d.files}
                if "quick" in kw:
                    kw["quick"] = bool(kw["quick"])
                return Analysis(**kw)     # older caches have no smarter analysis: its fields stay empty
        except Exception:
            return None

    def sem_at(self, times) -> Optional[np.ndarray]:
        """What the picture shows at these times (nearest thumbnail), or None without smarter analysis."""
        if not len(self.sem) or not len(self.thumb_times):
            return None
        tt = np.asarray(self.thumb_times, np.float32)
        k = np.clip(np.searchsorted(tt, np.asarray(times, np.float32)), 0, len(tt) - 1)
        prev = np.clip(k - 1, 0, len(tt) - 1)
        k = np.where(np.abs(tt[prev] - times) < np.abs(tt[k] - times), prev, k)
        return np.asarray(self.sem, np.float32)[np.clip(k, 0, len(self.sem) - 1)]


def analysis_path_for(footage: Footage) -> str:
    return str(cache_dir("analysis") / f"{file_key(footage.path)}_v{CACHE_VERSION}.npz")


def _colorfulness(img: np.ndarray) -> float:
    r, g, b = (img[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


AUDIO_SR = 16000        # the sound is read at this rate (smart.py hears it at 16 kHz)


def _read_audio(path: str, cancel: threading.Event) -> np.ndarray:
    exe = ffmpeg_bin()
    proc = popen([exe, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                  "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks = []
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            buf = proc.stdout.read(AUDIO_SR * 2 * 4)
            if not buf:
                break
            chunks.append(buf)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0


_SERIES = ("sharp", "activity", "motion", "jerk", "bright", "clipped", "colorful", "faces", "hists", "embeds")
SEM_BATCH = 16          # pictures described at once (smarter analysis)


@dataclass
class _Job:
    """One footage being analyzed, possibly as several segments decoded in parallel."""
    footage: Footage
    src: str
    w: int
    h: int
    matrix: str
    rotation: int            # only honoured on the GPU path; the CPU path lets ffmpeg autorotate
    hw_format: Optional[str]  # download format when the GPU may decode this file
    expected: int
    parts: dict = field(default_factory=dict)
    remaining: int = 0
    audio: Optional[Future] = None
    failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def create(footage: Footage, hw: Optional[str]) -> "_Job":
        src = footage.proxy_path if footage.has_proxy() else footage.path
        original = src == footage.path
        w = ANALYSIS_W
        h = max(2, int(round(ANALYSIS_W * footage.height / max(1, footage.width) / 2)) * 2)
        fmt = _HW_FORMATS.get(footage.pix_fmt)
        gpu = (hw is not None and original and footage.codec in HW_CODECS and fmt is not None
               and (footage.codec, footage.pix_fmt) not in _hw_failed)
        return _Job(footage, src, w, h,
                    matrix=ffmpeg_matrix(footage.color_space, min(footage.width, footage.height)) if original else "bt709",
                    rotation=footage.rotation % 360 if original else 0,
                    hw_format=fmt if gpu else None,
                    expected=max(1, int(footage.duration * SAMPLE_RATE)))

    def split(self, lanes: int) -> list["_Unit"]:
        k = max(1, min(lanes, int(self.footage.duration // SEGMENT_MIN)))
        bounds = [round(i * self.expected / k) for i in range(k + 1)]
        self.remaining = k
        return [_Unit(self, i, bounds[i], bounds[i + 1] - bounds[i] if i < k - 1 else None) for i in range(k)]

    def gpu_ok(self) -> bool:
        return self.hw_format is not None and (self.footage.codec, self.footage.pix_fmt) not in _hw_failed

    def put(self, index: int, data: dict) -> bool:
        """Store a finished segment; True when it was the last one."""
        with self.lock:
            self.parts[index] = data
            self.remaining -= 1
            return self.remaining == 0


@dataclass
class _Unit:
    job: _Job
    index: int
    first: int               # first sample index kept
    count: Optional[int]     # samples to keep, None = until the end of the file


def _scan(u: _Unit, hw: Optional[str], threads: int, cancel: threading.Event,
          tick: Callable[[], None]) -> Optional[dict]:
    """Decode one segment and extract per-sample signals. Returns None if GPU decoding failed."""
    j = u.job
    start = max(0, u.first - OVERLAP)
    skip = u.first - start
    w, h = j.w, j.h
    scale = f"scale={{}}:{{}}:flags=area:in_color_matrix={j.matrix}:in_range=auto:out_range=pc"
    args = [ffmpeg_bin(), "-v", "error", "-nostdin", "-skip_frame", "noref"]
    if hw:
        # -xerror: a GPU that can't decode a stream (e.g. high-bitrate 4K H.265 on older cards) otherwise
        # yields garbage frames with exit code 0; failing makes this segment rerun on the CPU.
        args += ["-xerror", "-hwaccel", hw, "-hwaccel_output_format", HW_OUTPUT_FORMAT[hw], "-noautorotate"]
        sw, sh = (h, w) if j.rotation in (90, 270) else (w, h)
        vf = f"fps={SAMPLE_RATE},hwdownload,format={j.hw_format}," + scale.format(sw, sh) + _ROTATE.get(j.rotation, "")
    else:
        if threads:
            args += ["-threads", str(threads)]
        vf = f"fps={SAMPLE_RATE}," + scale.format(w, h)
    if start:
        args += ["-ss", f"{start / SAMPLE_RATE:.3f}"]
    args += ["-i", j.src, "-an", "-sn", "-dn", "-vf", vf]
    if u.count is not None:
        args += ["-frames:v", str(u.count + skip)]
    args += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = w * h * 3
    th = int(round(THUMB_W * h / w / 2)) * 2
    out = {k: [] for k in _SERIES}
    sharp, activity, motion, jerk, bright, clipped, colorful, faces, hists, embeds = (out[k] for k in _SERIES)
    thumbs, thumb_t, sem, sem_imgs = [], [], [], []
    describe = smart.enabled()
    prev_gray = None
    prev_shift = np.zeros(2)
    hann = None
    i = start
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sharp.append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())))
            bright.append(float(gray.mean()) / 255.0)
            clipped.append(float(np.mean((gray > 250) | (gray < 5))))
            colorful.append(_colorfulness(img[::2, ::2]))
            g32 = gray.astype(np.float32)
            if prev_gray is not None:
                activity.append(float(np.mean(np.abs(g32 - prev_gray))))
                if hann is None:
                    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(prev_gray, g32, hann)
                shift = np.array([dx, dy])
                motion.append(float(np.hypot(dx, dy)))
                jerk.append(float(np.linalg.norm(shift - prev_shift)))
                prev_shift = shift
            else:
                activity.append(0.0)
                motion.append(0.0)
                jerk.append(0.0)
            prev_gray = g32
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
            hists.append(hist / max(1.0, hist.sum()))
            embeds.append(cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).flatten().astype(np.float32))
            area = 0.0
            if i % 2 == 0:
                area = float(sum(fw * fh for (_, _, fw, fh, _) in face_detect(img, retry=False)))
            elif faces:
                area = faces[-1]
            faces.append(area)
            if i >= u.first:
                if i % THUMB_EVERY == 0:
                    thumbs.append(cv2.resize(img, (THUMB_W, th), interpolation=cv2.INTER_AREA))
                    thumb_t.append(i / SAMPLE_RATE)
                    if describe:
                        sem_imgs.append(img.copy())
                        if len(sem_imgs) >= SEM_BATCH:
                            sem.append(smart.describe(sem_imgs))
                            sem_imgs = []
                tick()
            i += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    if hw and (proc.returncode != 0 or i == start):
        return None
    for k in _SERIES:
        out[k] = out[k][skip:]
    if describe and sem_imgs:
        sem.append(smart.describe(sem_imgs))
    out["thumbs"], out["thumb_t"] = thumbs, thumb_t
    out["sem"] = [row for block in sem for row in block] if describe else []
    return out


# Camera RAW video FFmpeg can't decode (extension or codec name -> what it is).
_RAW_FORMATS = {".braw": "Blackmagic RAW", ".r3d": "REDCODE RAW", ".ari": "ARRIRAW", ".crm": "Canon Cinema RAW Light",
                ".nev": "Nikon N-RAW"}


def undecodable_message(footage) -> str:
    """Why no frames came out of a file, and what to do about it. Camera RAW codecs (Sony X-OCN in MXF,
    Blackmagic RAW, REDCODE, ARRIRAW...) are proprietary: FFmpeg, and so KangEdit, can't decode them."""
    codec = (getattr(footage, "codec", "") or "").lower()
    ext = os.path.splitext(footage.path)[1].lower()
    raw = _RAW_FORMATS.get(ext)
    if raw is None and ext == ".mxf" and codec in ("", "none", "unknown"):
        raw = "Sony X-OCN (or another camera RAW codec)"
    if raw is None and codec in ("", "none", "unknown"):
        raw = "a codec FFmpeg doesn't know"
    if raw is None:
        return f"Could not decode frames from {footage.name}"
    return (f"{footage.name} is {raw}, which FFmpeg (and so KangEdit) can't decode. Convert it to ProRes, "
            f"DNxHR or H.264 / H.265 first (DaVinci Resolve, or the camera maker's app such as Sony Catalyst "
            f"Browse / RAW Viewer), then import the converted file.")


def _finish(j: _Job) -> Analysis:
    """Join the segments of a job, then add scene cuts and audio."""
    data = {k: [] for k in _SERIES + ("thumbs", "thumb_t", "sem")}
    for idx in sorted(j.parts):
        for k, v in j.parts[idx].items():
            data[k] += v
    footage = j.footage
    n = len(data["sharp"])
    if n == 0:
        raise RuntimeError(undecodable_message(footage))
    times = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    # Scene cuts: big jumps in colour histogram.
    hists_a = np.array(data["hists"])
    cuts = []
    if n > 2:
        d = np.r_[0.0, 0.5 * np.abs(np.diff(hists_a, axis=0)).sum(axis=1)]
        thr = max(0.45, float(np.median(d) + 6 * np.std(d)))
        cuts = [float(times[k]) for k in range(1, n) if d[k] > thr]

    # Audio loudness per analysis sample + waveform envelope.
    audio_db = np.full(n, -60.0, dtype=np.float32)
    wave = np.zeros(max(1, int(footage.duration * WAVE_RATE)), dtype=np.float32)
    sound = np.zeros((0, len(smart.GROUP_NAMES)), np.uint8)
    if j.audio is not None:
        pcm = j.audio.result()
        audio_db, wave = _loudness(pcm, n, footage.duration)
        if pcm.size and smart.enabled():
            sound = smart.hear(pcm)
    sem = np.array(data["sem"], np.float16).reshape(-1, smart.EMBED_DIM)
    if len(sem) != len(data["thumbs"]):
        sem = np.zeros((0, smart.EMBED_DIM), np.float16)

    return Analysis(
        times=times,
        sharp=np.array(data["sharp"], np.float32),
        activity=np.array(data["activity"], np.float32),
        motion=np.array(data["motion"], np.float32),
        jerk=np.array(data["jerk"], np.float32),
        brightness=np.array(data["bright"], np.float32),
        clipped=np.array(data["clipped"], np.float32),
        colorful=np.array(data["colorful"], np.float32),
        faces=np.array(data["faces"], np.float32),
        audio=audio_db,
        cuts=np.array(cuts, np.float32),
        embed=np.array(data["embeds"], np.float32),
        thumbs=np.array(data["thumbs"], np.uint8),
        thumb_times=np.array(data["thumb_t"], np.float32),
        wave=wave,
        sem=sem,
        sound=sound,
    )


def _loudness(pcm: np.ndarray, n: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Loudness (dBFS) per analysis sample and the waveform envelope, from the sound at AUDIO_SR."""
    audio_db = np.full(n, -60.0, dtype=np.float32)
    wave = np.zeros(max(1, int(duration * WAVE_RATE)), dtype=np.float32)
    if pcm.size:
        win = int(AUDIO_SR / SAMPLE_RATE)
        m = min(n, pcm.size // win)
        if m:
            rms = np.sqrt(np.mean(pcm[: m * win].reshape(m, win) ** 2, axis=1) + 1e-10)
            audio_db[:m] = 20 * np.log10(rms)
        ww = int(AUDIO_SR / WAVE_RATE)
        k = pcm.size // ww
        if k:
            env = np.abs(pcm[: k * ww]).reshape(k, ww).max(axis=1)
            wave = np.clip(env / max(1e-3, float(np.percentile(env, 99))), 0, 1).astype(np.float32)
    return audio_db, wave


def make_smart(footage: Footage, a: Analysis, cancel: Optional[threading.Event] = None) -> bool:
    """Add the smarter analysis to an analysis made without it (an older cache, or Smarter analysis turned on
    since): what each thumbnail shows, and the sound heard again. The video isn't decoded again. Returns
    whether it changed."""
    if not smart.enabled() or footage.is_image:
        return False
    changed = False
    if not len(a.sem) and len(a.thumbs):
        a.sem = smart.describe(list(a.thumbs))
        changed = True
    if not len(a.sound) and footage.has_audio:
        pcm = _read_audio(footage.proxy_path if footage.has_proxy() else footage.path, cancel or threading.Event())
        if pcm.size:
            a.sound = smart.hear(pcm, cancel)
            changed = True
    return changed


def quick_path_for(footage: Footage) -> str:
    return str(cache_dir("analysis") / f"{file_key(footage.path)}_v{CACHE_VERSION}_quick.npz")


def quick_look(footage: Footage, cancel: Optional[threading.Event] = None) -> Analysis:
    """A first look at a file from its keyframes only (often one a second): many times faster than the full
    analysis, so new footage can be browsed and edited at once while that runs. Every signal is measured on
    the keyframes and spread over the usual 4-a-second samples; movement and shake are estimated between
    keyframes. The sound is analysed in full, as it is quick to read."""
    import av
    cancel = cancel or threading.Event()
    src = footage.proxy_path if footage.has_proxy() else footage.path
    w = ANALYSIS_W
    h = max(2, int(round(ANALYSIS_W * footage.height / max(1, footage.width) / 2)) * 2)
    rot = footage.rotation % 360 if src == footage.path else 0
    rw, rh = (h, w) if rot in (90, 270) else (w, h)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-audio")
    audio = pool.submit(_read_audio, src, cancel) if footage.has_audio else None
    pool.shutdown(wait=False)
    keys: list[tuple[float, np.ndarray]] = []
    after: dict[int, list[tuple[float, np.ndarray]]] = {}     # keyframe index -> the frames decoded right after it
    with av.open(src) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        t0 = c.start_time / 1_000_000 if c.start_time is not None else 0.0
        grab = 0

        def take(frames):
            for frame in frames:
                if frame.time is None:
                    continue
                f = frame.reformat(width=rw, height=rh, format="rgb24", dst_color_range="JPEG")
                img = f.to_ndarray()
                if rot:
                    img = np.ascontiguousarray(np.rot90(img, rot // 90))
                t = max(0.0, float(frame.time) - t0)
                if frame.key_frame or not keys:
                    keys.append((t, img))
                elif t > keys[-1][0]:
                    after.setdefault(len(keys) - 1, []).append((t, img))

        # Each keyframe and the two packets after it (frames that depend on it alone, a fraction of a second
        # later: enough to measure movement and shake); every other packet is never decoded.
        for packet in c.demux(st):
            if cancel.is_set():
                raise Cancelled()
            if packet.is_keyframe:
                grab = 3
            if grab > 0:
                grab -= 1
                try:
                    take(st.codec_context.decode(packet))
                except av.error.InvalidDataError:
                    pass
        take(st.codec_context.decode(None))
    kt = np.array([t for t, _ in keys], np.float32)
    n = max(1, int(footage.duration * SAMPLE_RATE))
    times = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    stats = {k: [] for k in ("sharp", "bright", "clipped", "colorful", "faces", "activity", "motion", "jerk")}
    hists, embeds = [], []
    hann = None

    def shift_between(a, b):
        nonlocal hann
        if hann is None:
            hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        return np.array(cv2.phaseCorrelate(a, b, hann)[0])

    def grey(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    step = 1.0 / SAMPLE_RATE
    for k, (t, img) in enumerate(keys):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        stats["sharp"].append(float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())))
        stats["bright"].append(float(gray.mean()) / 255.0)
        stats["clipped"].append(float(np.mean((gray > 250) | (gray < 5))))
        stats["colorful"].append(_colorfulness(img[::2, ::2]))
        stats["faces"].append(float(sum(fw * fh for (_, _, fw, fh, _) in face_detect(img, retry=False))))
        # Movement as the full analysis measures it (between samples a quarter of a second apart), from the
        # frames just after the keyframe, scaled to that spacing.
        act = mot = jerk = 0.0
        nxt = [(tt, im) for tt, im in after.get(k, []) if 0.0 < tt - t < 0.6]
        if nxt:
            g0 = gray.astype(np.float32)
            t1, im1 = nxt[0]
            g1 = grey(im1)
            r1 = step / (t1 - t)
            s1 = shift_between(g0, g1) * r1
            act = float(np.mean(np.abs(g1 - g0))) * r1
            mot = float(np.hypot(*s1))
            if len(nxt) > 1:
                t2, im2 = nxt[1]
                s2 = shift_between(g1, grey(im2)) * (step / (t2 - t1))
                jerk = float(np.linalg.norm(s2 - s1))
        stats["activity"].append(act)
        stats["motion"].append(mot)
        stats["jerk"].append(jerk)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        hists.append(hist / max(1.0, hist.sum()))
        embeds.append(cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).flatten().astype(np.float32))
    near = np.clip(np.searchsorted(kt, times, side="right") - 1, 0, len(kt) - 1)      # last keyframe at or before

    def spread(name):
        return np.interp(times, kt, np.array(stats[name], np.float32)).astype(np.float32)

    hists_a = np.array(hists)
    cuts = []
    if len(kt) > 2:
        d = np.r_[0.0, 0.5 * np.abs(np.diff(hists_a, axis=0)).sum(axis=1)]
        thr = max(0.45, float(np.median(d) + 6 * np.std(d)))
        cuts = [float(kt[k]) for k in range(1, len(kt)) if d[k] > thr]
    thumb_times = np.arange(0, max(1, int(np.ceil(footage.duration))), dtype=np.float32)
    tk = np.clip(np.searchsorted(kt, thumb_times, side="right") - 1, 0, len(kt) - 1)
    th = int(round(THUMB_W * h / w / 2)) * 2
    thumbs = np.array([cv2.resize(keys[i][1], (THUMB_W, th), interpolation=cv2.INTER_AREA) for i in tk], np.uint8)
    sem = np.zeros((0, smart.EMBED_DIM), np.float16)
    if smart.enabled():
        uniq = sorted(set(int(i) for i in tk))
        desc = smart.describe([keys[i][1] for i in uniq])
        pos = {i: r for r, i in enumerate(uniq)}
        sem = desc[[pos[int(i)] for i in tk]]
    pcm = audio.result() if audio is not None else np.zeros(0, np.float32)
    audio_db, wave = _loudness(pcm, n, footage.duration)
    sound = smart.hear(pcm, cancel) if pcm.size and smart.enabled() else np.zeros((0, len(smart.GROUP_NAMES)), np.uint8)
    return Analysis(
        times=times, sharp=spread("sharp"), activity=spread("activity"), motion=spread("motion"),
        jerk=spread("jerk"), brightness=spread("bright"), clipped=spread("clipped"), colorful=spread("colorful"),
        faces=spread("faces"), audio=audio_db, cuts=np.array(cuts, np.float32),
        embed=np.array(embeds, np.float32)[near], thumbs=thumbs, thumb_times=thumb_times, wave=wave,
        sem=sem, sound=sound, quick=True)


def load_plain(footage: Footage) -> Optional[Analysis]:
    """The cached analysis as it is (the full one, else the quick first look), or None. Quick: no upgrade."""
    for path in (analysis_path_for(footage), quick_path_for(footage)):
        if os.path.exists(path):
            a = Analysis.load(path)
            if a is not None:
                footage.analysis_path = path
                return a
    return None


def needs_smart(footage: Footage, a: Analysis) -> bool:
    """Whether make_smart would add something to this analysis."""
    return (smart.enabled() and not footage.is_image
            and ((not len(a.sem) and len(a.thumbs)) or (not len(a.sound) and footage.has_audio)))


def load_cached(footage: Footage, cancel: Optional[threading.Event] = None,
                quick_ok: bool = True) -> Optional[Analysis]:
    """The cached analysis of a file: the full one, else (quick_ok) the quick first look; with the smarter
    analysis added when it's on and the cache was made without it. None when there is none."""
    a, path = None, analysis_path_for(footage)
    if os.path.exists(path):
        a = Analysis.load(path)
    if a is None and quick_ok:
        path = quick_path_for(footage)
        a = Analysis.load(path) if os.path.exists(path) else None
    if a is None:
        return None
    try:
        if make_smart(footage, a, cancel):
            a.save(path)
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 - the plain analysis still works
        traceback.print_exc()
    footage.analysis_path = path
    return a


def lane_plan(use_gpu: bool = True) -> tuple[Optional[str], int]:
    """(hardware decode method or None, number of CPU decode lanes)."""
    cores = os.cpu_count() or 4
    return (hw_decoder() if use_gpu else None), max(1, min(4, cores // 4))


def gpu_lanes() -> int:
    """GPU decode lanes next to the CPU ones: each lane also does CPU work per frame (scaling, faces, movement,
    the smarter analysis), so with 8 threads or more two lanes keep the graphics card's decoder busy (an AMD
    RX 480 or any card decodes several streams at once)."""
    return 2 if (os.cpu_count() or 4) >= 8 else 1


def analyze_many(footages: list[Footage], progress: Optional[ProgressCb] = None,
                 cancel: Optional[threading.Event] = None,
                 on_result: Optional[Callable[[Footage, Analysis], None]] = None,
                 on_error: Optional[Callable[[Footage, str], None]] = None,
                 use_gpu: bool = True) -> dict[str, Analysis]:
    """Analyze several footage files in parallel and cache the results.

    `on_result` / `on_error` are called from worker threads as each file finishes.
    Returns {footage id: Analysis} for the files that succeeded; raises Cancelled if cancelled.
    """
    cancel = cancel or threading.Event()
    if not footages:
        return {}
    hw, cpu_lanes = lane_plan(use_gpu)
    n_gpu = gpu_lanes() if hw else 0
    lanes = cpu_lanes + n_gpu
    cores = os.cpu_count() or 4
    jobs = [_Job.create(f, hw) for f in footages]
    pending: list[_Unit] = [u for j in jobs for u in j.split(lanes)]
    total = sum(j.expected for j in jobs)
    lock = threading.Lock()
    state = {"done": 0, "finished": 0, "cpu_busy": 0}
    results: dict[str, Analysis] = {}
    audio_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis-audio")
    on_gpu = sum(1 for j in jobs if j.gpu_ok())
    if not hw:
        lanes_label = f"{cpu_lanes} CPU"
    elif on_gpu == 0:       # e.g. 4:2:2 or 4:4:4 footage: no graphics card decodes it, every lane uses the CPU
        lanes_label = f"{lanes} CPU, the GPU can't decode this footage"
    elif on_gpu < len(jobs):
        lanes_label = f"{n_gpu} GPU + {cpu_lanes} CPU, {len(jobs) - on_gpu} file(s) on the CPU only"
    else:
        lanes_label = f"{n_gpu} GPU + {cpu_lanes} CPU"
    log.info("analysis: %d file(s), %d GPU-decodable, %s", len(jobs), on_gpu, lanes_label)

    def label() -> str:
        if len(jobs) == 1:
            return f"Analyzing {jobs[0].footage.name} ({lanes_label})"
        return f"Analyzing {state['finished']}/{len(jobs)} clips ({lanes_label})"

    def tick():
        with lock:
            state["done"] += 1
            done = state["done"]
        if progress and done % 8 == 0:
            progress(min(0.99, done / total), label())

    def take(gpu_lane: bool) -> Optional[tuple[_Unit, bool, int]]:
        """The next segment for a lane: (unit, decode it on the GPU, CPU decoding threads). A GPU lane takes what
        the GPU can decode first, a CPU lane what it can't; then either takes anything left, so no lane sits idle
        while work waits (footage the GPU can't decode, such as 4:2:2, keeps every lane busy on the CPU)."""
        with lock:
            if not pending:
                return None
            k = next((k for k, u in enumerate(pending) if u.job.gpu_ok() == gpu_lane), 0) if n_gpu else 0
            u = pending.pop(k)
            on_gpu = gpu_lane and u.job.gpu_ok()
            threads = 0
            if not on_gpu:
                state["cpu_busy"] += 1
                threads = max(2, cores // max(1, min(lanes, state["cpu_busy"] + len(pending))))
            if u.job.audio is None and u.job.footage.has_audio:
                u.job.audio = audio_pool.submit(_read_audio, u.job.src, cancel)
            return u, on_gpu, threads

    def fail(j: _Job, msg: str):
        with lock:
            if j.failed:
                return
            j.failed = True
            pending[:] = [p for p in pending if p.job is not j]
        if on_error:
            on_error(j.footage, msg)

    def lane(gpu_lane: bool):
        while not cancel.is_set():
            got = take(gpu_lane)
            if got is None:
                return
            u, on_gpu, threads = got
            j = u.job
            try:
                data = None
                if on_gpu:
                    data = _scan(u, hw, 0, cancel, tick)
                    if data is None:   # this file won't decode on the GPU: the CPU decodes it
                        _gpu_failed(j)
                        with lock:
                            state["cpu_busy"] += 1
                            on_gpu = False
                            threads = max(2, cores // max(1, min(lanes, state["cpu_busy"] + len(pending))))
                    else:
                        _gpu_worked(j)
                if data is None:
                    data = _scan(u, None, threads, cancel, tick)
                if j.put(u.index, data) and not j.failed:
                    a = _finish(j)
                    path = analysis_path_for(j.footage)
                    a.save(path)
                    j.footage.analysis_path = path
                    try:
                        os.remove(quick_path_for(j.footage))     # the quick first look isn't needed any more
                    except OSError:
                        pass
                    with lock:
                        results[j.footage.id] = a
                        state["finished"] += 1
                    if on_result:
                        on_result(j.footage, a)
            except Cancelled:
                return
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                fail(j, str(e))
            finally:
                if not on_gpu:
                    with lock:
                        state["cpu_busy"] -= 1

    threads = [threading.Thread(target=lane, args=(True,), daemon=True, name=f"analysis-gpu{k}")
               for k in range(n_gpu)]
    threads += [threading.Thread(target=lane, args=(False,), daemon=True, name=f"analysis-cpu{k}")
                for k in range(cpu_lanes)]
    if progress:
        progress(0.0, label())
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    audio_pool.shutdown(wait=True, cancel_futures=True)
    if cancel.is_set():
        raise Cancelled()
    if progress:
        progress(1.0, f"Analyzed {len(results)} clip(s)")
    return results


def analyze(footage: Footage, progress: Optional[ProgressCb] = None,
            cancel: Optional[threading.Event] = None, use_gpu: bool = True) -> Analysis:
    errors: list[str] = []
    res = analyze_many([footage], progress, cancel, on_error=lambda f, msg: errors.append(msg), use_gpu=use_gpu)
    if footage.id not in res:
        raise RuntimeError(errors[0] if errors else f"Could not analyze {footage.name}")
    return res[footage.id]


def load_or_analyze(footage: Footage, progress: Optional[ProgressCb] = None,
                    cancel: Optional[threading.Event] = None, use_gpu: bool = True,
                    quick_ok: bool = True) -> Analysis:
    """The cached analysis (the quick first look will do when `quick_ok`), else the full analysis now."""
    a = load_cached(footage, cancel, quick_ok)
    return a if a is not None else analyze(footage, progress, cancel, use_gpu)
