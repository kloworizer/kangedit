"""Per-footage content analysis: the raw signals used to find the best moments.

Speed: ffmpeg decodes at the 4 fps sample rate and skips non-reference frames (they are never sampled).
Long clips are split into segments, and the segments of every queued clip are spread over several CPU
lanes plus one GPU lane (hardware decoding on NVIDIA, AMD or Intel) when one is available.
"""

from __future__ import annotations

import os
import subprocess
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .faces import detect as face_detect
from .config import cache_dir, file_key
from .ffmpeg import HW_OUTPUT_FORMAT, Cancelled, ProgressCb, ffmpeg_bin, ffmpeg_matrix, hw_decoder, popen
from .models import Footage

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

    def save(self, path: str) -> None:
        np.savez_compressed(path, version=CACHE_VERSION, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @staticmethod
    def load(path: str) -> Optional["Analysis"]:
        try:
            with np.load(path) as d:
                if int(d.get("version", 0)) != CACHE_VERSION:
                    return None
                return Analysis(**{k: d[k] for k in Analysis.__dataclass_fields__})
        except Exception:
            return None


def analysis_path_for(footage: Footage) -> str:
    return str(cache_dir("analysis") / f"{file_key(footage.path)}_v{CACHE_VERSION}.npz")


def _colorfulness(img: np.ndarray) -> float:
    r, g, b = (img[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _read_audio(path: str, cancel: threading.Event) -> np.ndarray:
    exe = ffmpeg_bin()
    proc = popen([exe, "-v", "error", "-nostdin", "-i", path, "-vn", "-ac", "1", "-ar", "8000",
                  "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks = []
    try:
        while True:
            if cancel.is_set():
                raise Cancelled()
            buf = proc.stdout.read(8000 * 2 * 4)
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
    thumbs, thumb_t = [], []
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
    out["thumbs"], out["thumb_t"] = thumbs, thumb_t
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
    data = {k: [] for k in _SERIES + ("thumbs", "thumb_t")}
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
    if j.audio is not None:
        pcm = j.audio.result()
        if pcm.size:
            win = int(8000 / SAMPLE_RATE)
            m = min(n, pcm.size // win)
            if m:
                rms = np.sqrt(np.mean(pcm[: m * win].reshape(m, win) ** 2, axis=1) + 1e-10)
                audio_db[:m] = 20 * np.log10(rms)
            ww = int(8000 / WAVE_RATE)
            k = pcm.size // ww
            if k:
                env = np.abs(pcm[: k * ww]).reshape(k, ww).max(axis=1)
                wave = np.clip(env / max(1e-3, float(np.percentile(env, 99))), 0, 1).astype(np.float32)

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
    )


def lane_plan(use_gpu: bool = True) -> tuple[Optional[str], int]:
    """(hardware decode method or None, number of CPU decode lanes)."""
    cores = os.cpu_count() or 4
    return (hw_decoder() if use_gpu else None), max(1, min(4, cores // 4))


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
    lanes = cpu_lanes + (1 if hw else 0)
    cores = os.cpu_count() or 4
    jobs = [_Job.create(f, hw) for f in footages]
    pending: list[_Unit] = [u for j in jobs for u in j.split(lanes)]
    total = sum(j.expected for j in jobs)
    lock = threading.Lock()
    state = {"done": 0, "finished": 0, "cpu_busy": 0}
    results: dict[str, Analysis] = {}
    audio_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="analysis-audio")
    lanes_label = f"GPU + {cpu_lanes} CPU" if hw else f"{cpu_lanes} CPU"

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

    def take(gpu_lane: bool) -> Optional[tuple[_Unit, int]]:
        with lock:
            for k, u in enumerate(pending):
                if gpu_lane and not u.job.gpu_ok():
                    continue
                pending.pop(k)
                threads = 0
                if not gpu_lane:
                    state["cpu_busy"] += 1
                    threads = max(2, cores // max(1, min(cpu_lanes, state["cpu_busy"] + len(pending))))
                if u.job.audio is None and u.job.footage.has_audio:
                    u.job.audio = audio_pool.submit(_read_audio, u.job.src, cancel)
                return u, threads
            return None

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
            u, threads = got
            j = u.job
            try:
                data = None
                if gpu_lane and j.gpu_ok():
                    data = _scan(u, hw, 0, cancel, tick)
                    if data is None:   # unsupported by this GPU: decode on the CPU from now on
                        _hw_failed.add((j.footage.codec, j.footage.pix_fmt))
                        j.hw_format = None
                if data is None:
                    data = _scan(u, None, threads, cancel, tick)
                if j.put(u.index, data) and not j.failed:
                    a = _finish(j)
                    path = analysis_path_for(j.footage)
                    a.save(path)
                    j.footage.analysis_path = path
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
                if not gpu_lane:
                    with lock:
                        state["cpu_busy"] -= 1

    threads = [threading.Thread(target=lane, args=(True,), daemon=True, name="analysis-gpu")] if hw else []
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
                    cancel: Optional[threading.Event] = None, use_gpu: bool = True) -> Analysis:
    path = analysis_path_for(footage)
    if os.path.exists(path):
        a = Analysis.load(path)
        if a is not None:
            footage.analysis_path = path
            return a
    return analyze(footage, progress, cancel, use_gpu)
