"""Face detection for Zoom to face.

Uses OpenCV's YuNet face detector (a small neural network, kangedit/assets/face_detection_yunet_2023mar.onnx,
MIT licence, from github.com/opencv/opencv_zoo). It finds faces turned partly away, tilted, small or badly lit
that the older Haar cascades miss; those (frontal and both profiles) remain the fallback when the model can't
be loaded. Flat log footage is contrast-stretched for a second look when nothing is found as recorded.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "face_detection_yunet_2023mar.onnx")
MIN_SCORE = 0.6          # YuNet confidence
DETECT_LONG = 1280       # pictures are searched at most this big (long side): small faces stay findable

_local = threading.local()   # detectors are not safe to share between threads

Face = tuple[float, float, float, float, float]   # centre x, centre y, width, height (parts of the frame), score


def _yunet():
    det = getattr(_local, "yunet", None)
    if det is None:
        det = False
        if hasattr(cv2, "FaceDetectorYN") and os.path.exists(MODEL):
            try:
                det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), MIN_SCORE, 0.3, 5000)
            except cv2.error:
                det = False
        _local.yunet = det
    return det


def _cascades() -> list:
    cs = getattr(_local, "cascades", None)
    if cs is None:
        cs = []
        for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
            try:   # OpenCV 5 moved the cascades out of the main package
                c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
                if not c.empty():
                    cs.append(c)
            except Exception:  # noqa: BLE001
                pass
        _local.cascades = cs
    return cs


def _stretch(rgb: np.ndarray) -> np.ndarray:
    """Local contrast boost (CLAHE on the lightness): makes flat log or dark footage readable."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _detect_yunet(det, rgb: np.ndarray) -> list[Face]:
    h, w = rgb.shape[:2]
    det.setInputSize((w, h))
    _, rows = det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    out = []
    for r in rows if rows is not None else []:
        x, y, fw, fh, score = float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[-1])
        out.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, score))
    return out


def _detect_haar(rgb: np.ndarray) -> list[Face]:
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    h, w = gray.shape
    out = []
    for k, c in enumerate(_cascades()):
        for flip in ((False, True) if k == 1 else (False,)):   # the profile cascade only knows left profiles
            img = cv2.flip(gray, 1) if flip else gray
            for x, y, fw, fh in c.detectMultiScale(img, scaleFactor=1.08, minNeighbors=4,
                                                   minSize=(max(14, h // 40),) * 2):
                cx = (w - x - fw / 2) if flip else (x + fw / 2)
                out.append((cx / w, (y + fh / 2) / h, fw / w, fh / h, 0.7 if k == 0 else 0.6))
    return out


def detect(rgb: np.ndarray, retry: bool = True) -> list[Face]:
    """Faces in an RGB picture, most confident first. retry: look again at a contrast-stretched copy when
    nothing is found (flat log or dark footage)."""
    h, w = rgb.shape[:2]
    k = min(1.0, DETECT_LONG / float(max(h, w)))
    if k < 1.0:
        rgb = cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(rgb)
    det = _yunet()
    find = (lambda img: _detect_yunet(det, img)) if det is not False else _detect_haar
    faces = find(rgb)
    if not faces and retry:
        faces = find(_stretch(rgb))
    return sorted(faces, key=lambda f: -f[4])


def main_face(frames: list[np.ndarray]) -> Optional[tuple[float, float, float]]:
    """The clip's main face over several frames: (centre x, centre y, height), as parts of the frame.

    Each face found is a candidate; the main face is the one seen in the most frames (big, confident and
    central faces break ties), so a passer-by in one frame doesn't win over the person the clip is about.
    Its position is the median over the frames it is in."""
    per_frame = [detect(img) for img in frames]
    every = [f for fs in per_frame for f in fs]
    if not every:
        return None

    def same(a: Face, b: Face) -> bool:
        return abs(a[0] - b[0]) < max(a[2], b[2]) * 1.2 and abs(a[1] - b[1]) < max(a[3], b[3]) * 1.2

    best, best_key = None, None
    for cand in every:
        matches = [next((f for f in fs if same(cand, f)), None) for fs in per_frame]
        matches = [m for m in matches if m is not None]
        weight = sum(m[4] * np.sqrt(m[2] * m[3]) * (1.2 - abs(m[0] - 0.5)) for m in matches)
        key = (len(matches), weight)
        if best_key is None or key > best_key:
            best, best_key = matches, key
    arr = np.asarray(best)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 3]))
