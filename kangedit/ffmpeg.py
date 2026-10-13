"""FFmpeg / FFprobe helpers: discovery, probing and running with progress."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .ffmpeg_setup import local_binary

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class Cancelled(Exception):
    pass


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> Optional[str]:
    """The FFmpeg downloaded next to the app (see ffmpeg_setup), else the one on PATH."""
    return local_binary("ffmpeg") or shutil.which("ffmpeg")


def ffprobe_bin() -> Optional[str]:
    return local_binary("ffprobe") or shutil.which("ffprobe")


def _check_program(args: list[str]) -> None:
    """A clear error when the program (ffmpeg / ffprobe from ffmpeg_bin()) isn't installed."""
    if not args or not args[0]:
        raise FFmpegError("FFmpeg was not found: install it, or start KangEdit again to download it")


def popen(args: list[str], **kw) -> subprocess.Popen:
    _check_program(args)
    return subprocess.Popen(args, creationflags=CREATE_NO_WINDOW, **kw)


def run(args: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    _check_program(args)
    return subprocess.run(args, capture_output=True, cwd=cwd, creationflags=CREATE_NO_WINDOW)


@dataclass
class ProbeInfo:
    duration: float
    width: int           # display width (after rotation)
    height: int          # display height (after rotation)
    fps: float
    has_audio: bool
    codec: str
    pix_fmt: str
    bit_depth: int
    color_space: str
    color_range: str
    color_transfer: str
    color_primaries: str
    rotation: int        # display-matrix rotation in degrees (counter-clockwise)
    audio_channels: int
    creation_time: str
    # Container + stream metadata tags (lower-case keys), used to identify the camera.
    tags: dict = field(default_factory=dict)


def _parse_rate(rate: str) -> float:
    try:
        if "/" in rate:
            n, d = rate.split("/")
            return float(n) / float(d) if float(d) else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str) -> ProbeInfo:
    exe = ffprobe_bin()
    if not exe:
        raise FFmpegError("ffprobe was not found on PATH")
    res = run([exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path])
    if res.returncode != 0:
        raise FFmpegError(res.stderr.decode("utf-8", "replace").strip() or "ffprobe failed")
    data = json.loads(res.stdout.decode("utf-8", "replace"))
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise FFmpegError("No video stream found")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and "rotate" in (v.get("tags") or {}):
        rotation = -int(v["tags"]["rotate"])  # legacy tag is clockwise
    rotation %= 360

    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    if rotation in (90, 270):
        w, h = h, w

    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
    fps = _parse_rate(v.get("avg_frame_rate", "0/0")) or _parse_rate(v.get("r_frame_rate", "0/0")) or 30.0
    pix_fmt = v.get("pix_fmt", "")
    bit_depth = int(v.get("bits_per_raw_sample") or 0)
    if not bit_depth:
        bit_depth = 10 if ("10" in pix_fmt) else 12 if "12" in pix_fmt else 8
    tags = data.get("format", {}).get("tags", {}) or {}
    all_tags: dict[str, str] = {}
    for src in [data.get("format", {})] + streams:
        for k, val in (src.get("tags") or {}).items():
            k = k.lower()
            if k in all_tags and str(val) not in all_tags[k]:
                all_tags[k] += f" | {val}"
            else:
                all_tags.setdefault(k, str(val))
    return ProbeInfo(
        duration=duration,
        width=w,
        height=h,
        fps=fps,
        has_audio=a is not None,
        codec=v.get("codec_name", ""),
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        color_space=v.get("color_space", ""),
        color_range=v.get("color_range", ""),
        color_transfer=v.get("color_transfer", ""),
        color_primaries=v.get("color_primaries", ""),
        rotation=rotation,
        audio_channels=int(a.get("channels", 0)) if a else 0,
        creation_time=tags.get("creation_time", ""),
        tags=all_tags,
    )


def ffmpeg_matrix(color_space: str, height: int) -> str:
    """Matrix name for the scale filter's in_color_matrix."""
    cs = (color_space or "").lower()
    if cs in ("bt709",):
        return "bt709"
    if cs.startswith("bt2020"):
        return "bt2020"
    if cs in ("smpte170m", "bt470bg", "bt601"):
        return "bt601"
    return "bt709" if height >= 576 else "bt601"


ProgressCb = Callable[[float, str], None]


def run_ffmpeg(
    args: list[str],
    duration: float,
    progress: Optional[ProgressCb] = None,
    cancel: Optional[threading.Event] = None,
    cwd: Optional[str] = None,
    label: str = "",
    stall: Optional[float] = None,
) -> None:
    """Run ffmpeg with -progress parsing. `args` excludes the ffmpeg executable. With `stall`, an ffmpeg whose
    output time hasn't moved for that many seconds (a GPU driver that hangs, for one) is stopped and
    FFmpegError raised, so the caller can try another way. A cancel stops ffmpeg at once, even a silent one."""
    exe = ffmpeg_bin()
    if not exe:
        raise FFmpegError("ffmpeg was not found on PATH")
    cmd = [exe, "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats"] + args
    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    err_tail: deque[str] = deque(maxlen=30)

    def _drain():
        for line in iter(proc.stderr.readline, b""):
            err_tail.append(line.decode("utf-8", "replace").rstrip())

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    moved = [time.monotonic(), -1]      # when the output time last advanced, and to what
    stalled = threading.Event()

    def _watch():
        # Reading ffmpeg's progress blocks while it prints nothing: cancel and stall are watched from here.
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.kill()
                return
            if stall and time.monotonic() - moved[0] > stall:
                stalled.set()
                proc.kill()
                return
            time.sleep(0.2)

    w = threading.Thread(target=_watch, daemon=True)
    w.start()
    try:
        for raw in iter(proc.stdout.readline, b""):
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise Cancelled()
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if us > moved[1]:
                    moved[:] = [time.monotonic(), us]
                if progress and duration > 0:
                    progress(min(1.0, us / 1e6 / duration), label)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        t.join(timeout=2)
        w.join(timeout=1)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    if stalled.is_set():
        raise FFmpegError(f"ffmpeg stopped making progress for {stall:.0f} s" + (f" ({label})" if label else ""))
    if proc.returncode != 0:
        tail = "\n".join(l for l in err_tail if l)
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}):\n{tail}")


# Hardware decoders worth trying per platform, best first. d3d11va covers NVIDIA, AMD and Intel on Windows.
_HW_CANDIDATES = {
    "win32": ["d3d11va"],
    "darwin": ["videotoolbox"],
    "linux": ["cuda", "vaapi"],
}
# -hwaccel name -> pixel format of frames kept on the GPU.
HW_OUTPUT_FORMAT = {"d3d11va": "d3d11", "cuda": "cuda", "vaapi": "vaapi", "videotoolbox": "videotoolbox"}

_hw_lock = threading.Lock()
_hw_checked = False
_hw_method: Optional[str] = None


def hw_decoder() -> Optional[str]:
    """First hardware decode method that ffmpeg can actually open on this machine (cached), or None."""
    global _hw_checked, _hw_method
    with _hw_lock:
        if _hw_checked:
            return _hw_method
        _hw_checked = True
        exe = ffmpeg_bin()
        if not exe:
            return None
        plat = "linux" if sys.platform.startswith("linux") else sys.platform
        listed = set(run([exe, "-hide_banner", "-hwaccels"]).stdout.decode("utf-8", "replace").split())
        for method in _HW_CANDIDATES.get(plat, []):
            if method not in listed:
                continue
            res = run([exe, "-v", "error", "-init_hw_device", method, "-f", "lavfi",
                       "-i", "nullsrc=s=64x64:d=0.04", "-f", "null", "-"])
            if res.returncode == 0:
                _hw_method = method
                break
        return _hw_method


def available_encoders() -> set[str]:
    exe = ffmpeg_bin()
    if not exe:
        return set()
    res = run([exe, "-hide_banner", "-encoders"])
    out = res.stdout.decode("utf-8", "replace")
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names


# Hardware video encoders per platform and output codec, best first.
_HW_ENCODERS = {
    "win32": {"h264": ["h264_nvenc", "h264_amf", "h264_qsv"], "h265": ["hevc_nvenc", "hevc_amf", "hevc_qsv"]},
    "darwin": {"h264": ["h264_videotoolbox"], "h265": ["hevc_videotoolbox"]},
    "linux": {"h264": ["h264_nvenc", "h264_qsv"], "h265": ["hevc_nvenc", "hevc_qsv"]},
}
HW_ENCODER_NAMES = {"nvenc": "NVIDIA NVENC", "amf": "AMD AMF", "qsv": "Intel Quick Sync", "videotoolbox": "VideoToolbox"}

_enc_lock = threading.Lock()
_enc_found: dict[str, Optional[str]] = {}


def hw_encoder_args(encoder: str, q: int) -> list[str]:
    """Quality-targeted rate control for a hardware encoder; q is on the x264/x265 CRF scale."""
    if encoder.endswith("_nvenc"):
        return ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(q + 1), "-b:v", "0", "-spatial-aq", "1"]
    if encoder.endswith("_amf"):
        # At constant QP the preset mostly trades file size for speed; "quality" H.264 is slow on older cards.
        preset = "balanced" if encoder.startswith("h264") else "quality"
        return ["-quality", preset, "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q + 2)]
    if encoder.endswith("_qsv"):
        return ["-preset", "slower", "-global_quality", str(q)]
    if encoder.endswith("_videotoolbox"):
        return ["-q:v", str(int(max(30, min(85, 118 - q * 2.7))))]
    return []


def hw_encoder(codec: str) -> Optional[str]:
    """First hardware encoder for `codec` ("h264" | "h265") that really encodes on this machine (cached)."""
    with _enc_lock:
        if codec in _enc_found:
            return _enc_found[codec]
        found = None
        exe = ffmpeg_bin()
        plat = "linux" if sys.platform.startswith("linux") else sys.platform
        listed = available_encoders() if exe else set()
        for enc in _HW_ENCODERS.get(plat, {}).get(codec, []):
            if enc not in listed:
                continue
            # Listed only means ffmpeg was built with it; a short test encode proves the GPU and driver work.
            res = run([exe, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=0.2",
                       "-pix_fmt", "nv12", "-c:v", enc] + hw_encoder_args(enc, 20) + ["-f", "null", "-"])
            if res.returncode == 0 and not res.stderr.strip():
                found = enc
                break
        _enc_found[codec] = found
        return found


def hw_encoder_label(encoder: str) -> str:
    return HW_ENCODER_NAMES.get(encoder.split("_", 1)[-1], encoder)
