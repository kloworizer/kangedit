"""Automatic captions.

Each footage is transcribed once (word-level timestamps in *source* time). Timeline
captions are then derived from the clips' in/out points, so they follow every trim,
slip and reorder without re-transcribing. Editing a caption rewrites the underlying
words, so manual fixes survive later edits too.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import asr_qwen
from .config import frame_size
from .ffmpeg import Cancelled, ProgressCb, ffmpeg_bin, run
from .models import CaptionStyle, Footage, Project

log = logging.getLogger("kangedit.captions")

WHISPER_MODELS = {
    "auto": "Auto — follows the computer: Qwen3-ASR on a strong CPU (6+ cores, 12+ GB memory), else large-v3 on "
            "an NVIDIA GPU or large-v3-turbo",
    "qwen3-asr": "Qwen3-ASR 1.7B — most accurate in many languages (Indonesian, English, Chinese…), slower; "
                 "2.5 GB download",
    "large-v3": "large-v3 — most accurate (slowest)",
    "large-v3-turbo": "large-v3-turbo — nearly as accurate, much faster",
    "medium": "medium",
    "small": "small — fast",
    "base": "base — faster, less accurate",
    "tiny": "tiny — fastest, least accurate",
}

# Word-by-word caption animations (preview and burned-in export draw them the same way).
ANIMATIONS = {
    "none": "None (whole caption at once)",
    "highlight": "Highlight the spoken word (karaoke)",
    "box": "Highlight box behind the spoken word",
    "pop": "Pop: words appear as spoken, current word bounces",
    "reveal": "Reveal: words appear as spoken",
    "word": "One word at a time (big)",
}
POP_MS = 140          # pop-in duration
POP_SCALE = 1.25      # starting scale of a popping word
WORD_SCALE = 1.5      # size of the single word in "word" mode

LANGUAGES = [
    ("", "Auto detect"), ("en", "English"), ("id", "Indonesian"), ("es", "Spanish"), ("fr", "French"),
    ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("ja", "Japanese"),
    ("ko", "Korean"), ("zh", "Chinese"), ("hi", "Hindi"), ("ar", "Arabic"), ("ru", "Russian"),
    ("tr", "Turkish"), ("vi", "Vietnamese"), ("th", "Thai"), ("ms", "Malay"), ("tl", "Tagalog"),
]

LISTEN_TIMEOUT = 180.0   # seconds before listen() gives up on a part of a file

_model_lock = threading.Lock()
_models: "OrderedDict[str, object]" = OrderedDict()   # loaded speech models, most recently used last
MAX_MODELS = 2           # the captions model and the small one used to listen for the opening / closing


def model_ready(name: str) -> bool:
    """Whether the speech model is loaded already (using it starts at once)."""
    if resolve_model(name) == asr_qwen.MODEL:
        return asr_qwen.ready()
    return resolve_model(name) in _models


_loading: dict[str, threading.Event] = {}


_cuda: Optional[bool] = None


def _device() -> str:
    """"cuda" with an NVIDIA GPU CTranslate2 can use (the speech recognition runs there), else "cpu"."""
    global _cuda
    if _cuda is None:
        try:
            import ctranslate2
            _cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001
            _cuda = False
    return "cuda" if _cuda else "cpu"


def total_ram_gb() -> float:
    """The computer's memory, GB (0 when it can't be read)."""
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = Status()
            st.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (OSError, ValueError, AttributeError):
        return 0.0


QWEN_MIN_CORES = 6       # Qwen3-ASR on the CPU: about real time on 4 cores, well under on 6 or more
QWEN_MIN_RAM = 12.0      # GB: the 1.7B model takes about 3 GB while it works, next to the editor


def qwen_suits() -> bool:
    """Whether this computer runs Qwen3-ASR well enough for Auto to pick it: sherpa-onnx installed, a CPU of
    QWEN_MIN_CORES physical cores or more and QWEN_MIN_RAM GB of memory."""
    return (not _qwen_failed and asr_qwen.available() and cpu_threads() >= QWEN_MIN_CORES
            and total_ram_gb() >= QWEN_MIN_RAM)


_qwen_failed = ""        # why Qwen3-ASR couldn't be had under Auto (no internet for its download): Whisper then


def resolve_model(name: str) -> str:
    """The speech model to use for a setting. "auto" follows the computer: Qwen3-ASR (the most accurate) on a
    strong CPU with plenty of memory (qwen_suits), else large-v3 on an NVIDIA GPU and large-v3-turbo on the CPU
    (CTranslate2 has no AMD or Intel GPU support: there, turbo's 4 decoder layers instead of 32 make it several
    times faster at nearly the same accuracy)."""
    if name in ("", "auto"):
        if qwen_suits():
            return asr_qwen.MODEL
        return "large-v3" if _device() == "cuda" else "large-v3-turbo"
    return name


def cpu_threads() -> int:
    """Threads for speech recognition on the CPU: one per physical core (a 6-core / 12-thread Ryzen: 6), as
    hyper-threads only slow its matrix work down."""
    n = os.cpu_count() or 4
    return max(1, n // 2 if n >= 8 else n)


def _get_model(name: str):
    """The speech model, loaded (and on first use downloaded) once: it stays loaded for every later file, next
    to one other model (the auto edit's quick listening model doesn't push out the captions one). The loading
    itself runs outside the lock, so a slow download of one model doesn't hold up another."""
    name = resolve_model(name)
    while True:
        with _model_lock:
            if name in _models:
                _models.move_to_end(name)
                return _models[name]
            busy = _loading.get(name)
            if busy is None:
                busy = _loading[name] = threading.Event()
                break
        busy.wait()                       # someone else is loading it: use theirs (or try again if it failed)
    try:
        from faster_whisper import WhisperModel
        device = _device()
        model = WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else "int8",
                             cpu_threads=0 if device == "cuda" else cpu_threads())
        with _model_lock:
            _models[name] = model
            while len(_models) > MAX_MODELS:
                _models.popitem(last=False)
        return model
    finally:
        with _model_lock:
            _loading.pop(name, None)
        busy.set()


_batch_pipes: dict[int, object] = {}


def _batched(model):
    pipe = _batch_pipes.get(id(model))
    if pipe is None:
        from faster_whisper import BatchedInferencePipeline
        pipe = _batch_pipes[id(model)] = BatchedInferencePipeline(model)
    return pipe


def load_model(name: str, progress: Optional[ProgressCb] = None, cancel: Optional[threading.Event] = None) -> None:
    """Load the speech model now (before a batch of files), saying so only when it isn't loaded yet."""
    global _qwen_failed
    if resolve_model(name) == asr_qwen.MODEL:
        if not asr_qwen.available():
            raise RuntimeError("Qwen3-ASR needs sherpa-onnx: run the installer again (or pip install sherpa-onnx)")
        try:
            asr_qwen.ensure_model(progress, cancel)
        except RuntimeError as e:
            if name not in ("", "auto"):
                raise                           # chosen by name: said as is
            # Auto: Whisper for the rest of the session (it worked offline before Auto took Qwen3-ASR)
            _qwen_failed = str(e)
            log.warning("Qwen3-ASR unavailable, Auto uses Whisper: %s", e)
            return load_model(name, progress, cancel)
        _get_model(asr_qwen.TIMING_MODEL)       # the Whisper model that times its words
        return
    if not model_ready(name):
        if progress:
            progress(0.0, f"Loading speech model '{name}' (first run downloads it)…")
        _get_model(name)


CLEANUPS = {
    "auto": "Auto: clean up loud background noise (road, wind, crowd)",
    "on": "Always clean up the sound first",
    "off": "Never (the sound as recorded)",
}
NOISY_FLOOR = -45.0       # dBFS: background noise this loud (the quiet moments' level)...
NOISY_GAP = 30.0          # ...with the voice less than this far above it: cleaned up before transcribing


def noise_profile(audio: np.ndarray, rate: int = 16000) -> tuple[float, float]:
    """(background noise level, voice level) of 16 kHz audio, dBFS: its quietest and loudest moments (50 ms)."""
    hop = rate // 20
    n = audio.size // hop
    if n < 10:
        return -90.0, -90.0
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop).astype(np.float32) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    return float(np.percentile(db, 10)), float(np.percentile(db, 95))


def needs_cleanup(audio: np.ndarray, mode: str = "auto") -> Optional[float]:
    """The background noise level to clean the sound at before transcribing (see cleanup_filter), or None to
    transcribe it as recorded: always with "on", never with "off", with "auto" when the noise is loud and the
    voice doesn't stand far above it (a car, a road, wind, a crowd)."""
    if mode == "off":
        return None
    floor, voice = noise_profile(audio)
    if mode == "on" or (floor > NOISY_FLOOR and voice - floor < NOISY_GAP):
        return float(np.clip(floor, -80.0, -20.0))
    return None


def cleanup_filter(floor: float) -> str:
    """FFmpeg filters that bring a voice out of steady noise for speech recognition: the rumble below and the
    hiss above the voice cut, the noise at its measured level taken down, the voice evened out."""
    return (f"highpass=f=100:p=2,lowpass=f=7600,afftdn=nr=18:nf={floor:.1f}:tn=1,"
            "dynaudnorm=f=150:g=15:p=0.9")


def _load_audio_16k(path: str, start: float = 0.0, duration: Optional[float] = None,
                    filters: str = "") -> np.ndarray:
    """Decode audio (or the part from `start`) to 16 kHz mono float32 with ffmpeg (through `filters`)."""
    rng = (["-ss", f"{start:.3f}"] if start > 0 else []) + (["-t", f"{duration:.3f}"] if duration else [])
    af = ["-af", filters] if filters else []
    res = run([ffmpeg_bin(), "-v", "error", "-nostdin"] + rng + ["-i", path, "-vn"] + af + ["-ac", "1", "-ar",
               "16000", "-f", "f32le", "-"])
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", "replace").strip() or "audio decode failed")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


# Phrases Whisper is known to write over noise or silence (from the subtitles it learned on), dropped when it
# isn't sure there is speech.
_INVENTED = {"terima kasih", "terima kasih sudah menonton", "thank you", "thanks for watching",
             "thank you for watching", "subscribe", "please subscribe", "you", "bye", "terima kasih telah menonton",
             "sampai jumpa", "sous-titrage st' 501", "untertitel im auftrag des zdf"}


def _invented(text: str) -> bool:
    return re.sub(r"[^\w\s']", "", text.lower()).strip() in _INVENTED


def transcribe_audio(audio: np.ndarray, model_name: str, language: str = "",
                     progress=None, cancel: Optional[threading.Event] = None, offset: float = 0.0,
                     quick: bool = False, vad: bool = True) -> tuple[list[dict], str]:
    """Words [{"s", "e", "w"}] (times + offset) and the detected language, decoded for accuracy:
    beam search with temperature fallback, voice-activity filtering, and hallucination guards.
    quick decodes greedily without fallback, enough to spot a phrase. vad=False hears all of the sound: voice
    activity detection finds no voice in loud noise (a car, wind), where the speech is still clear enough to
    recognise. Qwen3-ASR (asr_qwen.MODEL) writes the words and a quick Whisper pass times them (quick: Whisper
    small alone)."""
    if resolve_model(model_name) == asr_qwen.MODEL:
        if quick:
            model_name = "small"
        else:
            return asr_qwen.transcribe(
                audio, language, progress, cancel, offset,
                timing=lambda a, lang: transcribe_audio(a, asr_qwen.TIMING_MODEL, lang, None, cancel, quick=True,
                                                        vad=vad), vad=vad)
    model = _get_model(model_name)
    opts = dict(
        language=language or None,
        word_timestamps=True,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        beam_size=1 if quick else 5,
        best_of=1 if quick else 5,
        temperature=0.0 if quick else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,       # no runaway repetition from earlier mistakes
        hallucination_silence_threshold=2.0,    # drop invented text over long silences
    )
    segments = info = None
    if not quick:
        # Batched: the stretches of speech are decoded several at a time, much faster on a GPU and still
        # quicker on a multi-core CPU. The plain decoder is the fallback.
        try:
            segments, info = _batched(model).transcribe(audio, batch_size=16 if _device() == "cuda" else 8, **opts)
        except Exception:  # noqa: BLE001
            segments = None
    if segments is None:
        segments, info = model.transcribe(audio, **opts)
    words: list[dict] = []
    dur = info.duration or 1.0
    for seg in segments:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if seg.no_speech_prob > 0.75 and seg.avg_logprob < -0.8:
            continue    # almost certainly noise transcribed as words
        if _invented(seg.text) and (seg.no_speech_prob > 0.4 or not vad and seg.end > dur - 3.0):
            continue    # a phrase Whisper is known to invent over noise or silence (in loud noise its
            #             no-speech score stays 0: the sign-off it adds at the end of the file is dropped)
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"s": round(float(w.start) + offset, 3), "e": round(float(w.end) + offset, 3), "w": text})
        if progress:
            progress(min(0.99, seg.end / dur))
    return words, info.language or language


def listen(footage: Footage, start: float, duration: float, model_name: str = "small",
           language: str = "", cancel: Optional[threading.Event] = None) -> list[dict]:
    """Quick transcript of part of a file (source times), e.g. to spot an opening or closing phrase.

    Whisper runs in a helper thread so cancel takes effect at once even while it is inside native code
    (model load or a long decode). It gives up and returns no words LISTEN_TIMEOUT after the model is ready:
    the first call loads (or downloads) the model, which must not use up the time for listening."""
    out: dict = {}
    stop = threading.Event()          # ends the helper's transcription when we give up on it or are cancelled

    def work():
        try:
            _get_model(model_name)
            out["ready"] = True
            if stop.is_set():
                return
            audio = _load_audio_16k(footage.path, start, duration)
            out["words"] = transcribe_audio(audio, model_name, language, None, stop, offset=start,
                                            quick=True)[0] if audio.size and not stop.is_set() else []
        except Cancelled:
            pass
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    waited = 0.0
    while t.is_alive():
        t.join(0.1)
        if out.get("ready"):
            waited += 0.1
        if cancel is not None and cancel.is_set():
            stop.set()
            raise Cancelled()
        if waited > LISTEN_TIMEOUT:
            stop.set()
            return []
    if "error" in out:
        raise out["error"]
    return out.get("words", [])


MIN_SPEECH = 0.8         # seconds of speech below which a file isn't worth transcribing (views, B-roll)
TRANSCRIPT_VERSION = 2   # 2: noisy sound heard whole (version 1 left some speech in loud noise untranscribed)


def speech_seconds(audio: np.ndarray) -> float:
    """How much speech 16 kHz audio holds (Silero voice activity detection, as Whisper's own filter uses)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        spans = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=100))
    except Exception:  # noqa: BLE001 - can't tell: transcribe it
        return float("inf")
    return sum(sp["end"] - sp["start"] for sp in spans) / 16000.0


def refine_word_times(words: list[dict], audio: np.ndarray, offset: float = 0.0, rate: int = 16000) -> list[dict]:
    """Words with their start and end moved onto the sound: Whisper's word times are often 0.1-0.3 s early or
    late, and cuts between words, on whole sentences and around fillers land where these say. A start moves to
    where the voice rises out of a pause from 0.15 s before to 0.3 s after it; an end to where it falls silent
    from 0.3 s before to 0.25 s after. A word whose edge has no clear pause next to it (wind, music,
    words run together) keeps its time; no word crosses its neighbours."""
    if not words or audio.size < rate // 10:
        return words
    hop = rate // 100                                         # 10 ms frames
    n = audio.size // hop
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop).astype(np.float32) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    floor = float(np.percentile(db, 10))
    on = max(floor + 10.0, -50.0)                              # voice: this far above the quietest tenth
    loud = db >= on
    fr = lambda t: int(round((t - offset) * 100))  # noqa: E731
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        lo_lim = out[i - 1]["e"] + 0.01 if i else offset
        hi_lim = words[i + 1]["s"] - 0.01 if i + 1 < len(words) else offset + n / 100
        a, b = fr(w["s"] - 0.15), fr(min(w["s"] + 0.30, w["e"] - 0.05))
        if 0 <= a < b < n and not loud[a] and loud[a:b + 1].any():
            s_new = offset + (a + int(np.argmax(loud[a:b + 1]))) / 100 - 0.02
            w["s"] = round(max(lo_lim, s_new), 3)
        a, b = fr(max(w["e"] - 0.30, w["s"] + 0.05)), fr(w["e"] + 0.25)
        if 0 <= a < b < n and not loud[b] and loud[a:b + 1].any():
            last = a + int(np.flatnonzero(loud[a:b + 1])[-1])
            w["e"] = round(min(hi_lim, offset + (last + 1) / 100 + 0.03), 3)
        if w["e"] - w["s"] < 0.05:                            # something went wrong: keep Whisper's
            w["s"], w["e"] = words[i]["s"], words[i]["e"]
    return out


def model_note(model_name: str) -> str:
    """" with Qwen3-ASR" when that model transcribes (the status says which is at work), else ""."""
    return " with Qwen3-ASR" if resolve_model(model_name) == asr_qwen.MODEL else ""


def transcript_path(footage: Footage, model_name: str, language: str) -> str:
    """Where a file's transcript is kept for every project (by the file, the model and the language)."""
    from .config import cache_dir, file_key
    return str(cache_dir("transcripts") / f"{file_key(footage.path)}_{resolve_model(model_name)}_"
                                          f"{language or 'auto'}.json")


def transcript_key(model_name: str, language: str) -> str:
    """The speech model (as resolved: Auto's choice) and language setting a transcript is made with."""
    return f"{resolve_model(model_name)}|{language or 'auto'}"


def transcribed_with(footage: Footage, model_name: str, language: str) -> bool:
    """Whether the file's transcript was made with this speech model and language (a project from before
    transcripts recorded it: whether the transcript cache has this model's)."""
    if footage.transcript is None:
        return False
    if footage.transcript_model:
        return footage.transcript_model == transcript_key(model_name, language)
    return cached_transcript(footage, model_name, language) is not None


def cached_transcript(footage: Footage, model_name: str, language: str,
                      any_model: bool = False) -> Optional[tuple[list[dict], str]]:
    """(words, language) of this file transcribed before (in any project), or None. `any_model`: another
    model's transcript will do when there is none from this one (the most accurate first)."""
    import json
    paths = [transcript_path(footage, model_name, language)]
    if any_model:
        paths += [transcript_path(footage, m, language) for m in WHISPER_MODELS if m != "auto"]
    for p in dict.fromkeys(paths):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            # An empty transcript from version 1 may be speech it missed in loud noise: transcribed again.
            if d.get("version") == TRANSCRIPT_VERSION or (d.get("version") == 1 and d.get("words")):
                return d["words"], d.get("language", language)
        except (OSError, ValueError, KeyError):
            continue
    return None


def save_transcript(footage: Footage, model_name: str, language: str, words: list[dict], lang: str) -> None:
    import json
    path = transcript_path(footage, model_name, language)
    try:
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": TRANSCRIPT_VERSION, "language": lang, "words": words}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass                          # the project keeps its transcript anyway


def transcribe(footage: Footage, model_name: str = "auto", language: str = "",
               progress: Optional[ProgressCb] = None, cancel: Optional[threading.Event] = None,
               use_cache: bool = True, cleanup: str = "auto") -> list[dict]:
    """Transcribe a file (words with times in source seconds): from the transcript cache when it was
    transcribed before (unless not `use_cache`), as nothing when it holds no speech (a view, B-roll), else
    with Whisper (or Qwen3-ASR), its word times then set on the sound (refine_word_times). cleanup (CLEANUPS):
    whether loud background noise is cleaned out of the sound first."""
    if not footage.has_audio:
        footage.transcript = []
        return []
    if use_cache:
        hit = cached_transcript(footage, model_name, language)
        if hit is not None:
            footage.transcript, footage.transcript_lang = hit[0], hit[1]
            footage.transcript_model = transcript_key(model_name, language)
            return hit[0]
    if progress:
        progress(0.0, f"Reading the sound of {footage.name}…")
    audio = _load_audio_16k(footage.path)   # the original's audio: proxies may be re-encoded
    heard = audio
    floor = needs_cleanup(audio, cleanup) if audio.size else None
    # In loud noise (a car, wind, a crowd) voice activity detection finds no voice even where the speech is
    # clear enough to recognise: such sound is heard whole, never skipped as "nobody speaks".
    noisy = audio.size > 0 and needs_cleanup(audio, "auto") is not None
    if floor is not None:
        if progress:
            progress(0.0, f"Cleaning the background noise out of {footage.name}…")
        heard = _load_audio_16k(footage.path, filters=cleanup_filter(floor))
    if audio.size == 0 or (not noisy and speech_seconds(heard) < MIN_SPEECH):
        words, lang = [], language
    else:
        load_model(model_name, progress)
        words, lang = transcribe_audio(heard, model_name, language,
                                       (lambda p: progress(p, f"Transcribing {footage.name}{model_note(model_name)}"))
                                       if progress else None,
                                       cancel, vad=not noisy)
        words = refine_word_times(words, heard)       # (on the cleaned sound the voice's edges stand out)
    save_transcript(footage, model_name, language, words, lang)
    footage.transcript = words
    footage.transcript_lang = lang
    footage.transcript_model = transcript_key(model_name, language)
    return words


@dataclass
class Caption:
    start: float
    end: float
    text: str
    footage_id: str
    word_idx: list[int] = field(default_factory=list)
    clip_id: str = ""
    words: list[tuple[float, float, str]] = field(default_factory=list)   # timeline times

    def active_word(self, t: float) -> int:
        """Index of the word being spoken at timeline time t (the last one started), -1 before the first."""
        k = -1
        for i, (s, _, _) in enumerate(self.words):
            if s <= t + 1e-6:
                k = i
            else:
                break
        return k


@dataclass
class CaptionLayout:
    size_pct: float      # font height, % of frame height
    anchor: str          # "bottom" | "center" | "top": which edge of the text block sits at y
    y: float             # 0..1 of frame height
    max_chars: int       # per line
    max_lines: int
    max_duration: float  # seconds per caption
    margin_x: float      # fraction of the width kept clear on each side


def caption_layout(style: CaptionStyle, width: int, height: int, video_format: str = "") -> CaptionLayout:
    """Where and how big captions are. With auto layout this follows the frame ratio and the format:
    vertical social video keeps them in the lower middle, clear of the app overlays at the bottom and
    side; landscape long-form uses classic bottom subtitles."""
    if not style.auto_layout:
        y = {"bottom": 0.92, "middle": 0.5, "top": 0.08}.get(style.position, 0.92)
        anchor = {"bottom": "bottom", "middle": "center", "top": "top"}.get(style.position, "bottom")
        return CaptionLayout(style.size_pct, anchor, y, style.max_chars, style.max_lines, style.max_duration, 0.06)
    ratio = width / max(1, height)
    short = video_format != "long"
    if ratio < 0.75:          # 9:16 Reels / TikTok / Shorts
        return CaptionLayout(3.9, "center", 0.66, 16, 2, 1.6 if short else 2.5, 0.10)
    if ratio <= 1.4:          # 1:1, 4:5, 4:3
        return CaptionLayout(5.0, "bottom", 0.85, 22, 2, 2.0 if short else 3.5, 0.08)
    if short:                 # landscape social: big and punchy
        return CaptionLayout(6.0, "bottom", 0.88, 26, 2, 2.0, 0.08)
    return CaptionLayout(4.8, "bottom", 0.92, 42, 2, 5.0, 0.06)   # landscape long-form subtitles


def project_layout(project: Project, aspect: Optional[str] = None, style: Optional[CaptionStyle] = None) -> CaptionLayout:
    w, h = frame_size(aspect or project.settings.aspect, "1080p")
    return caption_layout(style or project.settings.captions, w, h, project.settings.video_format)


_SENT_END = re.compile(r"[.!?…。！？]$")


def build_captions(project: Project, style: Optional[CaptionStyle] = None, aspect: Optional[str] = None) -> list[Caption]:
    style = style or project.settings.captions
    lay = project_layout(project, aspect, style)
    max_chars_total = lay.max_chars * max(1, lay.max_lines)
    out: list[Caption] = []
    t0 = 0.0
    for clip in project.clips:
        f = project.footage.get(clip.footage_id)
        words = f.transcript if f and clip.audio_mode != "mute" else None
        # The words are heard where the clip's sound plays them (at its speed, or at normal speed).
        heard_end = clip.sound_src_at(clip.duration)
        if words:
            cur: Optional[Caption] = None
            last_end = None
            for idx, w in enumerate(words):
                mid = (w["s"] + w["e"]) / 2
                if mid < clip.src_in or mid >= heard_end:
                    continue
                ts = t0 + clip.sound_time_at(max(w["s"], clip.src_in))
                te = t0 + clip.sound_time_at(min(w["e"], heard_end))
                text = w["w"]
                if cur is not None:
                    new_len = len(cur.text) + 1 + len(text)
                    gap = ts - last_end if last_end is not None else 0
                    if (new_len > max_chars_total or te - cur.start > lay.max_duration or gap > 0.7
                            or (_SENT_END.search(cur.text) and len(cur.word_idx) >= 3)):
                        out.append(cur)
                        cur = None
                if cur is None:
                    cur = Caption(ts, te, text, clip.footage_id, [idx], clip.id, [(ts, te, text)])
                else:
                    cur.text += " " + text
                    cur.end = te
                    cur.word_idx.append(idx)
                    cur.words.append((ts, te, text))
                last_end = te
            if cur is not None:
                out.append(cur)
        t0 += clip.duration
    # Hold each caption a little after the last word, without overlapping the next one.
    for a, b in zip(out, out[1:]):
        if b.start > a.end:
            a.end = min(a.end + 0.25, b.start)
    if out:
        out[-1].end = min(out[-1].end + 0.25, project.duration)
    if style.uppercase:
        for c in out:
            c.text = c.text.upper()
            c.words = [(a, b, w.upper()) for a, b, w in c.words]
    return out


def wrap_words(words: list[str], max_chars: int, max_lines: int) -> list[list[int]]:
    """Word indices per line, balanced like wrap()."""
    if not words:
        return []
    text = " ".join(words)
    if len(text) <= max_chars or max_lines <= 1:
        return [list(range(len(words)))]
    lines, cur, cur_len = [], [], 0
    target = max(max_chars // 2, len(text) // max_lines + 1)
    for i, w in enumerate(words):
        if cur and cur_len + 1 + len(w) > target and len(lines) < max_lines - 1:
            lines.append(cur)
            cur, cur_len = [i], len(w)
        else:
            cur_len = cur_len + 1 + len(w) if cur else len(w)
            cur.append(i)
    lines.append(cur)
    return lines


def wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(text) <= max_chars or max_lines <= 1:
        return [text]
    # Balanced split into at most max_lines lines.
    lines, cur = [], ""
    target = max(max_chars // 2, len(text) // max_lines + 1)
    for w in words:
        if cur and len(cur) + 1 + len(w) > target and len(lines) < max_lines - 1:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return lines


def edit_caption(project: Project, cap: Caption, new_text: str) -> None:
    f = project.footage.get(cap.footage_id)
    if not f or not f.transcript or not cap.word_idx:
        return
    idx = sorted(cap.word_idx)
    old = [f.transcript[k] for k in idx]
    new_words = new_text.split()
    if len(new_words) == len(old):
        # Same word count: keep every word's timing (needed by the word-by-word animations).
        new = [{"s": o["s"], "e": o["e"], "w": w} for o, w in zip(old, new_words)]
    else:
        # Spread the new words over the old span in proportion to their length.
        s0, e0 = old[0]["s"], old[-1]["e"]
        total = sum(len(w) + 1 for w in new_words) or 1
        new, t = [], s0
        for w in new_words:
            d = (e0 - s0) * (len(w) + 1) / total
            new.append({"s": round(t, 3), "e": round(t + d, 3), "w": w})
            t += d
    for k in reversed(idx):
        del f.transcript[k]
    f.transcript[idx[0]:idx[0]] = new


def delete_captions(project: Project, caps: list[Caption]) -> int:
    """Remove these captions' words from the transcripts (so they stay gone after re-trims). Returns how
    many words were removed. "Generate captions" can transcribe the footage again."""
    by_footage: dict[str, set[int]] = {}
    for cap in caps:
        by_footage.setdefault(cap.footage_id, set()).update(cap.word_idx)
    removed = 0
    for fid, idx in by_footage.items():
        f = project.footage.get(fid)
        if not f or not f.transcript:
            continue
        for k in sorted(idx, reverse=True):
            if 0 <= k < len(f.transcript):
                del f.transcript[k]
                removed += 1
    return removed


def _ts(t: float, sep: str = ",") -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(path: str, caps: list[Caption], style: CaptionStyle, layout: Optional[CaptionLayout] = None) -> None:
    max_chars = layout.max_chars if layout else style.max_chars
    max_lines = layout.max_lines if layout else style.max_lines
    with open(path, "w", encoding="utf-8") as fh:
        for i, c in enumerate(caps, 1):
            lines = wrap(c.text, max_chars, max_lines)
            fh.write(f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n" + "\n".join(lines) + "\n\n")


def _ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_text(t: str) -> str:
    return t.replace("\\", "/").replace("{", "(").replace("}", ")")


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(path: str, caps: list[Caption], style: CaptionStyle, width: int, height: int,
              layout: Optional[CaptionLayout] = None) -> None:
    lay = layout or caption_layout(style, width, height)
    anim = style.animation if style.animation in ANIMATIONS else "none"
    size = max(10, int(round(height * lay.size_pct / 100.0)))
    box = style.box and anim != "box"          # the box animation draws its own highlight
    border = 3 if box else 1
    outline = max(1, size // 14) if style.outline or box else 0
    if box:
        outline = max(4, size // 5)
    back = "&H80000000"
    out_col = "&H90000000" if box else "&H00000000"
    an = {"bottom": 2, "center": 5, "top": 8}[lay.anchor]
    pos = f"\\an{an}\\pos({width // 2},{int(round(lay.y * height))})"
    hl = _ass_color(style.highlight)
    base = _ass_color(style.color)
    hl_bord = max(4, size // 4)

    def word_tag(i: int, w: str, active: int, t_rel_ms: int) -> str:
        w = _ass_text(w)
        if anim == "highlight" and i == active:
            return f"{{\\c{hl}}}{w}{{\\c{base}}}"
        if anim == "box" and i == active:
            return f"{{\\3c{hl}\\bord{hl_bord}\\shad0}}{w}{{\\3c&H000000&\\bord{outline}}}"
        if anim in ("pop", "reveal") and i > active:
            return f"{{\\alpha&HFF&}}{w}{{\\alpha&H00&}}"
        if anim == "pop" and i == active:
            s0 = int(POP_SCALE * 100)
            return (f"{{\\fscx{s0}\\fscy{s0}\\t(0,{POP_MS},\\fscx100\\fscy100)}}{w}"
                    f"{{\\fscx100\\fscy100}}")
        return w

    def render(c: Caption, active: int) -> str:
        words = [w for _, _, w in c.words] or c.text.split()
        lines = wrap_words(words, lay.max_chars, lay.max_lines)
        return "\\N".join(" ".join(word_tag(i, words[i], active, 0) for i in line) for line in lines)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n")
        fh.write(f"PlayResX: {width}\nPlayResY: {height}\n\n")
        fh.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                 "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
                 "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        fh.write(f"Style: Default,{style.font},{size},{base},&H000000FF,{out_col},{back},"
                 f"-1,0,0,0,100,100,0,0,{border},{outline},{0 if box else 1},{an},"
                 f"{int(width * lay.margin_x)},{int(width * lay.margin_x)},0,1\n\n")
        fh.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        def event(t0: float, t1: float, text: str):
            if t1 - t0 >= 0.01 and text:
                fh.write(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Default,,0,0,0,,{{{pos}}}{text}\n")

        for c in caps:
            if anim == "none" or not c.words:
                event(c.start, c.end, render(c, -1) if anim in ("none", "highlight", "box") else _ass_text(c.text))
                continue
            starts = [max(c.start, s) for s, _, _ in c.words]
            if anim in ("highlight", "box") and starts[0] > c.start:
                event(c.start, starts[0], render(c, -1))
            for i, (t0, (_, _, w)) in enumerate(zip(starts, c.words)):
                t1 = starts[i + 1] if i + 1 < len(starts) else c.end
                if anim == "word":
                    fs = int(size * WORD_SCALE)
                    pop = f"\\fscx{int(POP_SCALE * 100)}\\fscy{int(POP_SCALE * 100)}\\t(0,{POP_MS},\\fscx100\\fscy100)"
                    event(t0, t1, f"{{\\fs{fs}{pop}}}{_ass_text(w)}")
                else:
                    event(t0, t1, render(c, i))


def merged_transcript(footage: list[Footage], headers: bool = True) -> str:
    """The transcripts of these files as one text in recording order (the oldest date each file carries, see
    scenes.recorded_time), a sentence per line; with `headers`, each file's part starts with its recording
    time and name. Files without a transcript (or with no words) are left out."""
    from datetime import datetime

    from .polish import sentence_spans
    from .scenes import recorded_time

    def key(f: Footage):
        t = recorded_time(f)
        return (t is None, t or 0.0, f.name.lower())

    parts = []
    for f in sorted((f for f in footage if f.transcript), key=key):
        lines = []
        for i, j in sentence_spans(f.transcript):
            line = " ".join(t for t in (str(w.get("w", "")).strip() for w in f.transcript[i:j + 1]) if t)
            if line:
                lines.append(line)
        if not lines:
            continue
        if headers:
            t = recorded_time(f)
            when = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t is not None else "no date"
            lines.insert(0, f"── {when} · {f.name}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)

