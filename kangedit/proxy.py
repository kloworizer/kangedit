"""Proxy generation: small, short-GOP H.264 copies for responsive editing.

With use_gpu, the graphics card decodes the original (the slow part for 4K H.265) and encodes the proxy,
using the same decoder as analysis and the same encoder as export. A failed GPU attempt (unsupported
stream, driver hiccup) is retried with CPU decoding, then fully on the CPU.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from .config import cache_dir, file_key
from .ffmpeg import (FFmpegError, ProgressCb, ffmpeg_matrix, hw_decoder, hw_encoder, hw_encoder_args, hw_encoder_label,
                     run_ffmpeg)
from .models import Footage

GOP = 12          # short GOP: any frame decodes quickly when scrubbing
QUALITY = 22      # x264 CRF scale
STALL = 60.0      # seconds without progress before a (GPU) attempt is given up for the next one

log = logging.getLogger(__name__)
# GPU decode / encode pairs that hung on this machine: later proxies skip them (for this session).
_stalled: set[tuple] = set()


def proxy_path_for(footage: Footage, height: int) -> str:
    return str(cache_dir("proxies") / f"{file_key(footage.path)}_{height}p.mp4")


def proxy_args(footage: Footage, height: int, out: str, hw: Optional[str] = None,
               encoder: Optional[str] = None) -> list[str]:
    """ffmpeg arguments: hw = hardware decode method (frames come back to memory for scaling), encoder =
    hardware H.264 encoder; None for the CPU."""
    w, h = footage.width, footage.height
    short = min(w, h)
    if short > height:
        s = height / short
        w, h = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    matrix = ffmpeg_matrix(footage.color_space, min(footage.width, footage.height))
    vf = (f"scale={w}:{h}:flags=bicubic:in_color_matrix={matrix}:out_color_matrix=bt709"
          f":in_range=auto:out_range=tv,format={'nv12' if encoder else 'yuv420p'}")
    # -xerror: a GPU that can't decode a stream otherwise yields garbage frames with exit code 0.
    args = ["-xerror", "-hwaccel", hw] if hw else []
    args += ["-i", footage.path, "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf]
    if encoder:
        args += ["-c:v", encoder] + hw_encoder_args(encoder, QUALITY)
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(QUALITY)]
    args += ["-g", str(GOP), "-bf", "0",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
             "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", out]
    return args


def generate_proxy(footage: Footage, height: int = 540, progress: Optional[ProgressCb] = None,
                   cancel: Optional[threading.Event] = None, use_gpu: bool = True) -> str:
    out = proxy_path_for(footage, height)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    hw, enc = (hw_decoder(), hw_encoder("h264")) if use_gpu else (None, None)
    attempts = list(dict.fromkeys([(hw, enc), (None, enc), (None, None)]))   # best first, no repeats
    attempts = [a for a in attempts if a not in _stalled] or [(None, None)]
    tmp = out + ".part.mp4"
    try:
        for k, (dec, encoder) in enumerate(attempts):
            gpu = " + ".join(p for p in (f"{dec} decode" if dec else "", hw_encoder_label(encoder) if encoder else "")
                             if p)
            label = f"Proxy: {footage.name}" + (f" (GPU: {gpu})" if gpu else " (CPU)" if use_gpu else "")
            if progress:
                progress(0.0, label)
            try:
                # A GPU attempt that hangs (some driver / encoder combinations never start) is stopped after
                # STALL seconds and the next way is tried; the CPU is only ever slow, never stopped.
                run_ffmpeg(proxy_args(footage, height, tmp, dec, encoder), footage.duration, progress, cancel,
                           label=label, stall=STALL if (dec or encoder) else None)
                break
            except FFmpegError as e:
                if "stopped making progress" in str(e):
                    _stalled.add((dec, encoder))
                log.warning("Proxy attempt %s failed for %s: %s", gpu or "CPU", footage.name, str(e)[-300:])
                if k == len(attempts) - 1:
                    raise
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out
