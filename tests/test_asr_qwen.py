"""Qwen3-ASR: its answers cleaned, the sound cut into chunks, its words timed against Whisper's, and the helper
process (here with a stand-in for sherpa-onnx: the real model can't be downloaded in the tests)."""

import sys
import textwrap

import numpy as np
import pytest

from kangedit import asr_qwen, captions


def test_answers_are_cleaned():
    assert asr_qwen.clean("language Indonesian<asr_text>Halo semua, apa kabar?") == ("Halo semua, apa kabar?", "id")
    assert asr_qwen.clean("  just text <|endoftext|> ") == ("just text", "")
    assert asr_qwen.clean("language English<asr_text>") == ("", "en")
    assert asr_qwen.split_words("Hello  world!") == ["Hello", "world!"]
    assert asr_qwen.split_words("我们 go") == ["我", "们", "go"]


def test_speech_is_cut_into_chunks(monkeypatch):
    import faster_whisper.vad as vad
    spans = [(0.5, 3.0), (3.6, 10.0), (14.0, 70.0), (71.0, 72.0)]
    monkeypatch.setattr(vad, "get_speech_timestamps",
                        lambda audio, opts: [{"start": int(a * 16000), "end": int(b * 16000)} for a, b in spans])
    got = asr_qwen.speech_chunks(np.zeros(16000 * 80, np.float32))
    assert got[0] == (0.5, 10.0)                                 # close stretches joined
    assert all(b - a <= asr_qwen.CHUNK + 0.3 for a, b in got)    # a long one cut in pieces
    assert got[1][0] == 14.0 and got[-1][1] == 72.0
    assert all(x[1] == y[0] for x, y in zip(got[1:-1], got[2:-1]))  # with nothing left out


def test_long_speech_is_cut_where_it_is_quiet(monkeypatch):
    import faster_whisper.vad as vad
    monkeypatch.setattr(vad, "get_speech_timestamps", lambda audio, opts: [{"start": 0, "end": 16000 * 50}])
    t = np.arange(16000 * 50) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio[int(17.5 * 16000):int(17.6 * 16000)] = 0.0             # a breath, not long enough to count as a pause
    got = asr_qwen.speech_chunks(audio)
    assert 17.45 <= got[0][1] <= 17.65                           # cut in the breath, not in a word at 20 s


def test_helper_ends_when_the_app_is_gone():
    import os
    import subprocess
    assert asr_qwen.parent_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not asr_qwen.parent_alive(p.pid)


def test_words_take_whispers_times():
    ref = [{"s": 1.0, "e": 1.4, "w": "Halo"}, {"s": 1.5, "e": 2.0, "w": "semua,"},
           {"s": 2.2, "e": 2.5, "w": "apa"}, {"s": 2.6, "e": 3.1, "w": "kabar?"}]
    # Qwen heard "semuanya" where Whisper heard "semua", and a word Whisper missed
    words = asr_qwen.align("Halo semuanya, apa ya kabar?", 0.8, 3.3, ref)
    assert [w["w"] for w in words] == ["Halo", "semuanya,", "apa", "ya", "kabar?"]
    assert words[0]["s"] == pytest.approx(1.0) and words[0]["e"] == pytest.approx(1.4)
    assert words[2]["s"] == pytest.approx(2.2) and words[4]["e"] == pytest.approx(3.1)
    assert words[2]["e"] <= words[3]["s"] <= words[3]["e"] <= words[4]["s"] + 1e-6     # "ya": in the gap
    assert all(b["s"] >= a["e"] - 1e-6 for a, b in zip(words, words[1:]))
    # nothing from Whisper there: spread over the stretch by length
    spread = asr_qwen.align("satu dua tiga", 10.0, 13.0, ref)
    assert spread[0]["s"] == 10.0 and spread[-1]["e"] == pytest.approx(13.0) and spread[1]["s"] == spread[0]["e"]


FAKE_SHERPA = textwrap.dedent('''
    class _Result:
        def __init__(self, text):
            self.text = text

    class _Stream:
        def accept_waveform(self, rate, samples):
            self.n = len(samples)

    class OfflineRecognizer:
        @classmethod
        def from_qwen3_asr(cls, conv_frontend, encoder, decoder, tokenizer, num_threads=1, max_total_len=512,
                           max_new_tokens=128, **kw):
            assert conv_frontend.endswith("conv_frontend.onnx") and max_total_len == 512
            return cls()

        def create_stream(self):
            return _Stream()

        def decode_stream(self, st):
            st.result = _Result("language Indonesian<asr_text>halo " + str(st.n // 16000) + " detik")
''')


@pytest.fixture
def fake_sherpa(tmp_path, monkeypatch):
    pkg = tmp_path / "fake" / "sherpa_onnx"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_SHERPA)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake"))
    model = tmp_path / "model"
    (model / "tok").mkdir(parents=True)
    for name in ("conv_frontend.onnx", "encoder.onnx", "decoder.int8.onnx", "decoder.onnx"):
        (model / name).write_bytes(b"x")
    (model / "tok" / "vocab.json").write_text("{}")
    monkeypatch.setattr(asr_qwen, "model_dir", lambda: model)
    return model


def test_model_files_are_found(fake_sherpa):
    files = asr_qwen.find_files(fake_sherpa)
    assert files["decoder"].endswith("decoder.int8.onnx") and files["tokenizer"].endswith("tok")
    assert asr_qwen.ready()


def test_helper_process_transcribes_and_words_are_timed(fake_sherpa, monkeypatch):
    monkeypatch.setattr(asr_qwen, "speech_chunks", lambda audio, chunk=asr_qwen.CHUNK: [(0.0, 2.0), (3.0, 7.0)])
    seen = []

    def timing(audio, lang):
        seen.append(lang)
        return [{"s": 0.2, "e": 0.6, "w": "halo"}, {"s": 3.2, "e": 3.5, "w": "halo"}], "ms"

    progress = []
    words, lang = asr_qwen.transcribe(np.zeros(16000 * 8, np.float32), "", progress.append, None, offset=10.0,
                                      timing=timing)
    assert lang == "id" and seen == [""]                        # timed in the language asked for (or detected)
    assert [w["w"] for w in words] == ["halo", "2", "detik", "halo", "4", "detik"]
    assert words[0]["s"] == pytest.approx(10.2) and words[3]["s"] == pytest.approx(13.2)
    assert words[-1]["e"] == pytest.approx(17.0) and progress and progress[-1] > 0.9


def test_helper_failure_is_reported(fake_sherpa, monkeypatch):
    (fake_sherpa / "conv_frontend.onnx").rename(fake_sherpa / "x_conv.onnx")
    (fake_sherpa / "conv_frontend_fp16.onnx").write_bytes(b"x")     # found, but the fake insists on the name
    monkeypatch.setattr(asr_qwen, "speech_chunks", lambda audio, chunk=asr_qwen.CHUNK: [(0.0, 1.0)])
    with pytest.raises(RuntimeError, match="Qwen3-ASR failed"):
        asr_qwen.transcribe(np.zeros(16000, np.float32), "", timing=lambda a, lang: ([], ""))


def test_captions_use_it_when_chosen(monkeypatch):
    called = {}

    def fake(audio, language, progress, cancel, offset, timing, vad=True):
        called["timing"] = timing
        return [{"s": 0.0, "e": 0.5, "w": "hi"}], "en"

    monkeypatch.setattr(asr_qwen, "transcribe", fake)
    words, lang = captions.transcribe_audio(np.zeros(1600, np.float32), asr_qwen.MODEL)
    assert words[0]["w"] == "hi" and lang == "en" and callable(called["timing"])
    assert asr_qwen.MODEL in captions.WHISPER_MODELS
    assert captions.resolve_model(asr_qwen.MODEL) == asr_qwen.MODEL
    assert captions.transcript_path.__name__                    # (transcripts are kept per model)


def test_captions_panel_chooses_the_model(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        panel = w.caps_panel
        assert panel.cb_model.currentData() == w.project.settings.whisper_model
        panel.cb_model.setCurrentIndex(panel.cb_model.findData(asr_qwen.MODEL))
        assert w.project.settings.whisper_model == asr_qwen.MODEL
        w.project.settings.whisper_model = "small"
        w._sync_project_controls()
        assert panel.cb_model.currentData() == "small"
    finally:
        w.project.dirty = False
        w.close()


def test_model_download_says_how_far_it_is(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    import huggingface_hub
    model = tmp_path / "m"
    monkeypatch.setattr(asr_qwen, "model_dir", lambda: model)
    names = ["conv_frontend.onnx", "encoder.onnx", "decoder.int8.onnx", "vocab.json"]

    class Api:
        def model_info(self, repo, files_metadata=False, **kw):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=1000) for n in names]
                                   + [SimpleNamespace(rfilename="README.md", size=10 ** 9)])   # not downloaded

    def download(repo, local_dir, allow_patterns):
        from huggingface_hub.utils import tqdm
        for n in tqdm(names):                        # as snapshot_download: a progress bar on stderr
            time.sleep(0.3)
            (model / n).write_bytes(b"x" * 1000)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(sys, "stderr", None)             # the app without a console (pythonw.exe)
    said = []
    files = asr_qwen.ensure_model(lambda p, m: said.append((p, m)))
    assert files["decoder"].endswith("decoder.int8.onnx")
    assert said and "of 0.0 GB" in said[-1][1] and all(0 <= p <= 1 for p, _ in said)
    assert max(p for p, _ in said) > 0.2                                        # it moved

    def offline(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(asr_qwen, "model_dir", lambda: tmp_path / "other")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=offline))
    with pytest.raises(RuntimeError, match="internet connection"):
        asr_qwen.ensure_model()

    def stuck(repo, local_dir, allow_patterns):
        time.sleep(3)                                    # connected, but nothing arrives

    monkeypatch.setattr(asr_qwen, "STALL", 1.0)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", stuck)
    t = time.monotonic()
    with pytest.raises(RuntimeError, match="nothing arrived"):
        asr_qwen.ensure_model()
    assert time.monotonic() - t < 2.5                     # said, not waited out


def test_auto_picks_qwen_on_a_strong_computer(monkeypatch):
    monkeypatch.setattr(asr_qwen, "available", lambda: True)
    monkeypatch.setattr(captions, "_device", lambda: "cpu")
    monkeypatch.setattr(captions, "cpu_threads", lambda: 8)
    monkeypatch.setattr(captions, "total_ram_gb", lambda: 32.0)
    assert captions.resolve_model("auto") == asr_qwen.MODEL
    assert captions.model_note("auto") == " with Qwen3-ASR"
    monkeypatch.setattr(captions, "total_ram_gb", lambda: 8.0)                 # little memory
    assert captions.resolve_model("auto") == "large-v3-turbo"
    monkeypatch.setattr(captions, "_device", lambda: "cuda")
    assert captions.resolve_model("auto") == "large-v3"
    monkeypatch.setattr(captions, "total_ram_gb", lambda: 32.0)
    monkeypatch.setattr(captions, "cpu_threads", lambda: 4)                     # few cores
    assert captions.resolve_model("auto") == "large-v3"
    monkeypatch.setattr(asr_qwen, "available", lambda: False)
    monkeypatch.setattr(captions, "cpu_threads", lambda: 8)
    assert captions.resolve_model("auto") == "large-v3"                       # not installed
    assert captions.resolve_model("small") == "small"
    assert captions.total_ram_gb.__name__ and captions.total_ram_gb() >= 0


def test_loops_are_collapsed():
    loop = "Liburan ke mana? " + "Ia di dalam tanah. Terlihat, ada matahari. " * 5 + "Terus."
    assert asr_qwen.collapse_repeats(loop) == "Liburan ke mana? Ia di dalam tanah. Terlihat, ada matahari. Terus."
    assert asr_qwen.collapse_repeats("Ibu siapa? " * 30 + "Ibu si") == "Ibu siapa? Ibu si"
    assert asr_qwen.collapse_repeats("terus terus ada mobil") == "terus terus ada mobil"      # twice is speech
    assert asr_qwen.clean("language Malay<asr_text>ke mana? ke mana? ke mana? ke mana?") == ("ke mana?", "ms")


def test_noise_heard_as_chinese_is_taken_out_of_latin_speech():
    got = asr_qwen.drop_odd_script(["宝宝好，再见。好，再见", "Hello. Kita sekarang mau ke mana Pak?", "nggak tahu, ng 。"])
    assert got == ["", "Hello. Kita sekarang mau ke mana Pak?", "nggak tahu, ng"]
    assert asr_qwen.drop_odd_script(["你好，我们去哪里", "OK"]) == ["你好，我们去哪里", "OK"]   # Chinese speech stays


def test_language_is_whispers_when_qwen_names_none(monkeypatch):
    monkeypatch.setattr(asr_qwen, "ensure_model", lambda cancel=None: {})
    monkeypatch.setattr(asr_qwen, "speech_chunks", lambda audio, chunk=asr_qwen.CHUNK: [(0.0, 2.0)])
    monkeypatch.setattr(asr_qwen, "run_model", lambda *a, **k: ["kita mau ke mana"])     # no "language X" tag
    words, lang = asr_qwen.transcribe(np.zeros(16000 * 2, np.float32), "", timing=lambda a, lang: ([], "id"))
    assert lang == "id" and len(words) == 4


def test_auto_uses_whisper_when_qwen_cant_be_downloaded(monkeypatch):
    monkeypatch.setattr(captions, "_qwen_failed", "")
    monkeypatch.setattr(asr_qwen, "available", lambda: True)
    monkeypatch.setattr(captions, "cpu_threads", lambda: 8)
    monkeypatch.setattr(captions, "total_ram_gb", lambda: 32.0)
    monkeypatch.setattr(captions, "_cuda", False)

    def offline(progress=None, cancel=None):
        raise RuntimeError("Can't download the Qwen3-ASR model (no network)")
    monkeypatch.setattr(asr_qwen, "ensure_model", offline)
    loaded = []
    monkeypatch.setattr(captions, "_get_model", loaded.append)
    monkeypatch.setattr(captions, "model_ready", lambda name: False)
    assert captions.resolve_model("auto") == asr_qwen.MODEL
    captions.load_model("auto")
    assert loaded == ["auto"] and captions.resolve_model("auto") == "large-v3-turbo"    # Whisper, loaded
    with pytest.raises(RuntimeError, match="no network"):
        captions.load_model(asr_qwen.MODEL)                      # chosen by name: said
