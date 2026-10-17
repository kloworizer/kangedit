"""Smarter analysis with two small neural networks, run on the CPU (onnxruntime, which faster-whisper already
brings): optional, as they take a little longer than the plain measurements.

* What a picture shows: MobileNetV2 (ImageNet, from the ONNX Model Zoo), cut before its classifier, gives a
  1280-number description of each analysed second. Two shots of the same view (filmed again, a little to the
  side, zoomed or darker) come out alike, different views don't, where the plain 12×12 grey picture used
  otherwise mixes them up (measured: the same view at least 0.76 alike, different views at most 0.64).
* What is heard: YAMNet (AudioSet, from MediaPipe) scores 521 kinds of sound every half second, grouped here
  into SOUND_GROUPS: speech, laughter, shouting / cheering, applause, crowd, music, wind, nature, vehicles and
  silence.

Without the models (or with Smarter analysis off) everything works as before on the plain measurements.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
EMBED_MODEL = os.path.join(ASSETS, "mobilenetv2_embed.onnx")
SOUND_MODEL = os.path.join(ASSETS, "yamnet.onnx")
EMBED_SIZE = 224
EMBED_DIM = 1280
SAME_VIEW = 0.75          # descriptions at least this alike: the same view (see the module docstring)
SOUND_RATE = 2.0          # sound scores per second
SOUND_WIN = 15600         # YAMNet window: 0.975 s at 16 kHz
SOUND_SR = 16000
# YAMNet class indices (AudioSet ontology, yamnet_class_map.csv) per group.
SOUND_GROUPS = {
    "speech": [0, 1, 2, 3, 4, 12],
    "laughter": [13, 14, 15, 16, 17, 18],
    "excited": [6, 7, 8, 9, 10, 11],          # shouting, whooping, yelling, screaming
    "applause": [56, 57, 58, 61, 62],         # clapping, cheering, applause
    "crowd": [63, 64, 65, 66],
    "music": list(range(24, 30)) + list(range(132, 277)),
    "wind": [277, 278, 279],
    "nature": [106, 107, 108, 109, 110, 111, 112, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289],
    "vehicle": list(range(294, 345)),
    "silence": [494],
}
GROUP_NAMES = tuple(SOUND_GROUPS)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_lock = threading.Lock()
_sessions: dict[str, object] = {}
_enabled = True


def set_enabled(on: bool) -> None:
    """Smarter analysis on or off (the app setting)."""
    global _enabled
    _enabled = bool(on)


def available() -> bool:
    """Whether the models can run here (onnxruntime and the model files)."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return os.path.exists(EMBED_MODEL) and os.path.exists(SOUND_MODEL)


def enabled() -> bool:
    return _enabled and available()


def _session(path: str):
    with _lock:
        s = _sessions.get(path)
        if s is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            # Analysis runs several files at once: two threads each keeps every core busy without
            # oversubscribing them.
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            s = _sessions[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return s


def _prep(img: np.ndarray) -> np.ndarray:
    import cv2
    x = cv2.resize(img, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - _MEAN) / _STD).transpose(2, 0, 1)


def describe(images: list[np.ndarray], batch: int = 16) -> np.ndarray:
    """(N, EMBED_DIM) float16 unit-length descriptions of RGB uint8 pictures (any size)."""
    if not images:
        return np.zeros((0, EMBED_DIM), np.float16)
    s = _session(EMBED_MODEL)
    out = []
    for k in range(0, len(images), batch):
        x = np.stack([_prep(i) for i in images[k:k + batch]])
        e = s.run(None, {"input": x})[0]
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-6))
    return np.concatenate(out).astype(np.float16)


def hear(pcm16k: np.ndarray, cancel: Optional[threading.Event] = None) -> np.ndarray:
    """(K, len(SOUND_GROUPS)) uint8 scores (0..255) of each sound group, SOUND_RATE times a second, from 16 kHz
    mono audio: the k-th row covers the window around k / SOUND_RATE seconds."""
    hop = int(SOUND_SR / SOUND_RATE)
    n = int(np.ceil(len(pcm16k) / hop)) if len(pcm16k) else 0
    out = np.zeros((n, len(GROUP_NAMES)), np.uint8)
    if n == 0:
        return out
    s = _session(SOUND_MODEL)
    pad = np.pad(pcm16k.astype(np.float32), (SOUND_WIN // 2, SOUND_WIN))
    idx = [np.array(v) for v in SOUND_GROUPS.values()]
    for k in range(n):
        if cancel is not None and k % 64 == 0 and cancel.is_set():
            break
        x = pad[k * hop:k * hop + SOUND_WIN]
        p = s.run(None, {"waveform_binary": x})[0][0]
        out[k] = np.clip(np.array([p[i].max() for i in idx]) * 255.0, 0, 255).astype(np.uint8)
    return out


def group(scores: np.ndarray, name: str) -> np.ndarray:
    """One group's scores (0..1) from hear()'s output."""
    if scores is None or not len(scores):
        return np.zeros(0, np.float32)
    return scores[:, GROUP_NAMES.index(name)].astype(np.float32) / 255.0


def at_samples(scores: np.ndarray, name: str, times: np.ndarray) -> Optional[np.ndarray]:
    """A group's scores (0..1) at the analysis sample times, or None without sound scores."""
    g = group(scores, name)
    if not g.size:
        return None
    k = np.clip(np.round(np.asarray(times) * SOUND_RATE).astype(int), 0, g.size - 1)
    return g[k]
