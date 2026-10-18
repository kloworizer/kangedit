"""Qwen3-ASR 1.7B speech recognition, offline: more accurate than Whisper large-v3 in many languages (Indonesian,
English, Chinese and others; Qwen3-ASR technical report, Apache-2.0), run with sherpa-onnx on the CPU.

It writes the words; their times come from a quick Whisper pass over the same sound, the two lined up letter by
letter (Qwen3-ASR's own aligner covers only 11 languages, Indonesian not among them). The model runs in a
helper process: sherpa-onnx brings its own ONNX Runtime, kept apart from the one the smarter analysis uses, and
the 2.5 GB model is let go of when the transcription is done. The model is downloaded once from Hugging Face."""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

MODEL = "qwen3-asr"                        # the speech model setting that picks it (captions.WHISPER_MODELS)
# Qwen3-ASR-1.7B in sherpa-onnx's layout, int8 (the same files in both; the second if the first is gone).
REPOS = ("thieunv-asilla/sherpa-onnx-qwen3-asr-1.7B-int8", "ilmina/qwen3-asr-1.7b-sherpa-onnx")
DOWNLOAD_GB = 2.4
CHUNK = 20.0                               # seconds of speech decoded at once (stretches of speech, joined)
MAX_TOTAL_LEN = 512                        # the model's limit (prompt + audio + text tokens): about 20 s at once
MAX_NEW_TOKENS = 256
STALL = 180.0           # seconds without a byte downloaded before the download is given up
TIMING_MODEL = "base"                      # the Whisper model that times the words
RATE = 16000
WORKER_FLAG = "--asr-worker"

# Language names Qwen3-ASR answers with, as Whisper's codes (to time the words in the same language).
LANG_CODES = {"english": "en", "indonesian": "id", "malay": "ms", "chinese": "zh", "cantonese": "yue",
              "japanese": "ja", "korean": "ko", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
              "portuguese": "pt", "dutch": "nl", "russian": "ru", "arabic": "ar", "hindi": "hi", "thai": "th",
              "vietnamese": "vi", "turkish": "tr", "polish": "pl", "filipino": "tl"}
_TAG = re.compile(r"^\s*language\s+([A-Za-z]+)\s*<asr_text>", re.I)
_SPECIAL = re.compile(r"<\|[^|>]*\|>|<asr_text>|</?asr_text>")
_CJK = r"぀-ヿ㐀-鿿가-힯"


def available() -> bool:
    """sherpa-onnx is installed (it runs the model)."""
    import importlib.util
    return importlib.util.find_spec("sherpa_onnx") is not None


def model_dir() -> Path:
    from .config import cache_dir
    return cache_dir("models") / "qwen3-asr-1.7b"


def find_files(folder: Path) -> Optional[dict]:
    """The model's files in `folder`: conv_frontend / encoder / decoder .onnx and the tokenizer's folder, or None
    when some are missing (not downloaded, or an unfinished download)."""
    if not folder.is_dir():
        return None

    def pick(prefix: str) -> Optional[str]:
        got = sorted(p for p in folder.rglob(f"{prefix}*.onnx"))
        int8 = [p for p in got if "int8" in p.name]
        return str((int8 or got)[0]) if got else None

    vocab = next(iter(sorted(folder.rglob("vocab.json"))), None)
    files = dict(conv_frontend=pick("conv_frontend"), encoder=pick("encoder"), decoder=pick("decoder"),
                 tokenizer=str(vocab.parent) if vocab is not None else None)
    return files if all(files.values()) else None


def ready() -> bool:
    return find_files(model_dir()) is not None


PATTERNS = ["*.onnx", "*.data", "*.onnx_data", "*.json", "*.txt", "*.model", "*.tiktoken"]


def _folder_bytes(folder: Path) -> int:
    """Bytes in the folder so far (finished files and the ones still downloading)."""
    total = 0
    for root, _, names in os.walk(folder):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def ensure_model(progress: Optional[Callable[[float, str], None]] = None,
                 cancel: Optional[threading.Event] = None) -> dict:
    """The model's files, downloaded first when they aren't here (saying how much is done; a cancelled or
    broken download carries on where it stopped next time)."""
    from fnmatch import fnmatch

    from .ffmpeg import Cancelled
    files = find_files(model_dir())
    if files is not None:
        return files
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()             # (bars printed to a console the app may not have: the app shows its own)
    repo, total, failed = None, 0, None
    for rid in REPOS:
        try:
            info = HfApi().model_info(rid, files_metadata=True, timeout=30)
        except Exception as e:  # noqa: BLE001 - gone or unreachable: the next one
            failed = e
            continue
        repo = rid
        total = sum(s.size or 0 for s in info.siblings if any(fnmatch(s.rfilename, p) for p in PATTERNS))
        break
    if repo is None:
        raise RuntimeError(f"Can't download the Qwen3-ASR model ({failed}). Check the internet connection, or "
                           "choose a Whisper model.") from failed
    folder = model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    def work():
        try:
            snapshot_download(repo, local_dir=str(folder), allow_patterns=PATTERNS)
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=work, daemon=True, name="qwen3-asr download")
    t.start()
    seen, since = _folder_bytes(folder), time.monotonic()
    while t.is_alive():
        t.join(0.5)
        if cancel is not None and cancel.is_set():
            raise Cancelled()              # (the download goes on in the background, or resumes next time)
        now = _folder_bytes(folder)
        if now != seen:
            seen, since = now, time.monotonic()
        elif time.monotonic() - since > STALL:
            raise RuntimeError(f"The Qwen3-ASR model download stopped: nothing arrived for {STALL // 60:g} "
                               "minutes. Check the internet connection and try again (it carries on where it "
                               "stopped), or choose a Whisper model.")
        if progress and total:
            done = min(now, total)
            progress(done / total, f"Downloading the Qwen3-ASR speech model: {done / 1e9:.1f} of "
                                   f"{total / 1e9:.1f} GB (first use only)")
    if "error" in out:
        raise RuntimeError(f"The Qwen3-ASR model download failed ({out['error']}). Try again, or choose a "
                           "Whisper model.") from out["error"]
    files = find_files(folder)
    if files is None:
        raise RuntimeError(f"The Qwen3-ASR model downloaded to {folder} is incomplete")
    return files


# ----------------------------------------------------------------------------- the words
def clean(text: str) -> tuple[str, str]:
    """(transcript, language code) from Qwen3-ASR's answer ("language Indonesian<asr_text>…")."""
    lang = ""
    m = _TAG.match(text or "")
    if m:
        lang = LANG_CODES.get(m.group(1).lower(), "")
        text = text[m.end():]
    return collapse_repeats(re.sub(r"\s+", " ", _SPECIAL.sub(" ", text or "")).strip()), lang


def collapse_repeats(text: str, keep: int = 1) -> str:
    """A phrase said over and over (a language model stuck in a loop over noise: "Ibu siapa? Ibu siapa? ...")
    kept `keep` times: runs of 3 or more of the same 1-40 words (a whole sentence over and over too)."""
    words = text.split()
    norm = [_norm(w) for w in words]
    for n in range(1, 41):
        i = 0
        while i + 3 * n <= len(words):
            unit = norm[i:i + n]
            if not any(unit):
                i += 1
                continue
            k = 1
            while norm[i + k * n:i + (k + 1) * n] == unit:
                k += 1
            if k >= 3:
                del words[i + keep * n:i + k * n]
                del norm[i + keep * n:i + k * n]
            i += 1
    return " ".join(words)


def split_words(text: str) -> list[str]:
    """Words of a transcript: by spaces, and each Chinese / Japanese / Korean character on its own."""
    return re.findall(rf"[{_CJK}]|[^\s{_CJK}]+", text)


def _norm(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower())


def fixed_chunks(audio: np.ndarray, chunk: float = CHUNK, a: float = 0.0,
                 dur: Optional[float] = None) -> list[tuple[float, float]]:
    """The sound (from a to dur seconds: all of it) in pieces of up to `chunk` seconds, each cut at the quietest
    moment of its last 4 s (less likely in the middle of a word)."""
    dur = audio.size / RATE if dur is None else dur
    out = []
    while dur - a > 0.3:
        b = min(dur, a + chunk)
        if b < dur:
            lo, hop = int((b - 4.0) * RATE), RATE // 20
            seg = audio[lo:int(b * RATE)]
            n = seg.size // hop
            if n > 2:
                rms = np.sqrt(np.mean(seg[: n * hop].reshape(n, hop) ** 2, axis=1))
                b = (lo + int(np.argmin(rms)) * hop + hop // 2) / RATE
        out.append((round(a, 3), round(b, 3)))
        a = b
    return out


def speech_chunks(audio: np.ndarray, chunk: float = CHUNK) -> list[tuple[float, float]]:
    """(start, end) seconds of the stretches of speech (voice activity), joined into pieces of up to `chunk`
    seconds (a longer stretch is cut in pieces)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        spans = [(sp["start"] / RATE, sp["end"] / RATE) for sp in
                 get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))]
    except Exception:  # noqa: BLE001 - no voice activity detection: the whole sound
        spans = [(0.0, audio.size / RATE)]
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if b - a > chunk:                       # a long stretch without a pause: cut where it is quietest
            pieces = fixed_chunks(audio, chunk, a, b)
            out += pieces[:-1]
            a = pieces[-1][0] if pieces else a
        if out and b - out[-1][0] <= chunk and a - out[-1][1] < 2.0:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def align(text: str, a: float, b: float, ref: list[dict]) -> list[dict]:
    """Words of `text` (said between a and b seconds) with times taken from `ref`, timed words of the same
    sound (Whisper's), lined up letter by letter. Words Whisper didn't hear share the time between their
    neighbours (or the stretch, by length)."""
    hyp = split_words(text)
    if not hyp:
        return []
    ref = [w for w in ref if (w["s"] + w["e"]) / 2 >= a - 0.3 and (w["s"] + w["e"]) / 2 <= b + 0.3]
    # each reference letter's time: spread over its word
    r_chars, r_times = [], []
    for w in ref:
        n = _norm(w["w"])
        for j, ch in enumerate(n):
            r_chars.append(ch)
            r_times.append((w["s"] + (w["e"] - w["s"]) * j / len(n), w["s"] + (w["e"] - w["s"]) * (j + 1) / len(n)))
    h_chars, owner = [], []
    for k, w in enumerate(hyp):
        for ch in _norm(w):
            h_chars.append(ch)
            owner.append(k)
    times: list[Optional[list[float]]] = [None] * len(hyp)
    if r_chars and h_chars:
        sm = difflib.SequenceMatcher(None, "".join(h_chars), "".join(r_chars), autojunk=False)
        for blk in sm.get_matching_blocks():
            for d in range(blk.size):
                k = owner[blk.a + d]
                s, e = r_times[blk.b + d]
                if times[k] is None:
                    times[k] = [s, e]
                else:
                    times[k][0], times[k][1] = min(times[k][0], s), max(times[k][1], e)
    # words without a time: share the gap between the timed neighbours, by length
    k = 0
    while k < len(hyp):
        if times[k] is not None:
            k += 1
            continue
        j = k
        while j < len(hyp) and times[j] is None:
            j += 1
        lo = times[k - 1][1] if k > 0 else a
        hi = times[j][0] if j < len(hyp) else b
        if hi - lo < 0.05 * (j - k):
            hi = lo + 0.05 * (j - k)
        lens = [max(1, len(hyp[i])) for i in range(k, j)]
        t = lo
        for i, n in zip(range(k, j), lens):
            d = (hi - lo) * n / sum(lens)
            times[i] = [t, t + d]
            t += d
        k = j
    out, last = [], a
    for w, (s, e) in zip(hyp, times):
        s = max(s, last)
        e = max(e, s + 0.05)
        out.append({"s": round(s, 3), "e": round(e, 3), "w": w})
        last = e
    return out


# ----------------------------------------------------------------------------- the helper process
def _worker_cmd() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, WORKER_FLAG]
    main = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main), WORKER_FLAG]


def run_model(audio: np.ndarray, chunks: list[tuple[float, float]], files: dict, threads: int,
              progress: Optional[Callable[[float], None]] = None,
              cancel: Optional[threading.Event] = None) -> list[str]:
    """Qwen3-ASR's answer for each chunk of the sound, from the helper process."""
    from .ffmpeg import CREATE_NO_WINDOW, Cancelled
    with tempfile.TemporaryDirectory(prefix="asr_") as tmp:
        wav = os.path.join(tmp, "audio.f32")
        audio.astype(np.float32).tofile(wav)
        job = os.path.join(tmp, "job.json")
        with open(job, "w", encoding="utf-8") as fh:
            json.dump({"files": files, "audio": wav, "chunks": chunks, "threads": threads,
                       "parent": os.getpid()}, fh)
        # The helper answers in a file, not on its console: a packaged app without a console window may
        # have no standard output at all.
        out_path = os.path.join(tmp, "out.jsonl")
        open(out_path, "w").close()
        with open(os.path.join(tmp, "err.txt"), "w", encoding="utf-8") as err:
            proc = subprocess.Popen(_worker_cmd() + [job, out_path], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=err, creationflags=CREATE_NO_WINDOW)
            texts = [""] * len(chunks)
            error, pos, buf = "", 0, ""
            with open(out_path, encoding="utf-8") as fh:
                while True:
                    alive = proc.poll() is None
                    fh.seek(pos)
                    buf += fh.read()
                    pos = fh.tell()
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        try:
                            msg = json.loads(line)
                        except ValueError:
                            continue
                        if "error" in msg:
                            error = msg["error"]
                        elif "k" in msg:
                            texts[msg["k"]] = msg.get("text", "")
                            if progress:
                                progress((msg["k"] + 1) / max(1, len(chunks)))
                    if not alive:
                        break
                    if cancel is not None and cancel.is_set():
                        proc.kill()
                        proc.wait()
                        raise Cancelled()
                    time.sleep(0.2)
        if proc.returncode != 0 or error:
            with open(os.path.join(tmp, "err.txt"), encoding="utf-8", errors="replace") as fh:
                tail = error or fh.read().strip()[-500:]
            raise RuntimeError(f"Qwen3-ASR failed: {tail or f'exit code {proc.returncode}'}")
        return texts


def parent_alive(pid: int) -> bool:
    """Whether the app that started the helper still runs (it may have closed or crashed mid-transcription)."""
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return not ok or code.value == 259               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        pass
    return True


def _watch_parent(pid: int) -> None:
    """The helper ends itself when the app is gone: nothing else stops a windowless process using every core
    and a few GB of memory."""
    def watch():
        while parent_alive(pid):
            time.sleep(2.0)
        os._exit(3)
    threading.Thread(target=watch, daemon=True, name="parent watch").start()


def worker_main(args: list[str]) -> int:
    """The helper process (job file, answer file): load the model, decode each chunk, answer a JSON line per
    chunk in the answer file."""
    out = open(args[1], "a", encoding="utf-8") if len(args) > 1 else sys.stdout

    def say(msg: dict) -> None:
        out.write(json.dumps(msg, ensure_ascii=False) + "\n")
        out.flush()

    try:
        with open(args[0], encoding="utf-8") as fh:
            job = json.load(fh)
        if job.get("parent"):
            _watch_parent(int(job["parent"]))
        import sherpa_onnx
        f = job["files"]
        rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=f["conv_frontend"], encoder=f["encoder"], decoder=f["decoder"], tokenizer=f["tokenizer"],
            num_threads=int(job.get("threads", 4)), max_total_len=MAX_TOTAL_LEN, max_new_tokens=MAX_NEW_TOKENS)
        audio = np.fromfile(job["audio"], dtype=np.float32)
        for k, (a, b) in enumerate(job["chunks"]):
            st = rec.create_stream()
            st.accept_waveform(RATE, audio[int(a * RATE):int(b * RATE)])
            rec.decode_stream(st)
            say({"k": k, "text": st.result.text})
        return 0
    except Exception as e:  # noqa: BLE001 - said to the app, which shows it
        say({"error": f"{type(e).__name__}: {e}"})
        return 1


_CJK_RUN = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]+")


def drop_odd_script(texts: list[str]) -> list[str]:
    """When the speech is mostly in Latin letters, Chinese / Japanese / Korean in it is noise Qwen3-ASR heard as
    words (a road's roar before anyone speaks: "宝宝好，再见。"), not speech: taken out."""
    cjk = sum(len(m) for t in texts for m in _CJK_RUN.findall(t))
    latin = sum(len(re.findall(r"[A-Za-z]", t)) for t in texts)
    if latin <= cjk:
        return texts
    return [re.sub(r"\s+", " ", _CJK_RUN.sub(" ", t)).strip(" ,.") for t in texts]


def transcribe(audio: np.ndarray, language: str = "", progress: Optional[Callable[[float], None]] = None,
               cancel: Optional[threading.Event] = None, offset: float = 0.0,
               timing: Optional[Callable[[np.ndarray, str], tuple[list[dict], str]]] = None,
               vad: bool = True) -> tuple[list[dict], str]:
    """Words [{"s", "e", "w"}] (times + offset) and the language. timing(audio, language) -> Whisper's timed
    words of the same sound and its language (captions.transcribe_audio with TIMING_MODEL): the language when
    neither the setting nor Qwen3-ASR names one."""
    from .captions import cpu_threads
    files = ensure_model(cancel=cancel)
    chunks = speech_chunks(audio) if vad else fixed_chunks(audio)   # (noise: the whole sound, see captions)
    if not chunks:
        return [], language
    answers = run_model(audio, chunks, files, cpu_threads(),
                        (lambda p: progress(0.85 * p)) if progress else None, cancel)
    texts, langs = [], []
    for ans in answers:
        t, lang = clean(ans)
        texts.append(t)
        if lang:
            langs.append(lang)
    texts = drop_odd_script(texts)
    # (timed in the language asked for, else Whisper's own guess: Qwen3-ASR may call Indonesian Malay)
    ref, ref_lang = timing(audio, language) if timing is not None else ([], "")
    lang = language or (max(set(langs), key=langs.count) if langs else "") or ref_lang
    if progress:
        progress(0.99)
    words: list[dict] = []
    for (a, b), t in zip(chunks, texts):
        words += align(t, a, b, ref)
    for w in words:
        w["s"], w["e"] = round(w["s"] + offset, 3), round(w["e"] + offset, 3)
    return words, lang


if __name__ == "__main__":
    sys.exit(worker_main(sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG else sys.argv[1:]))
