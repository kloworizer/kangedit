"""Transcribing several files loads the speech model once, then transcribes them one after the other."""

import sys
import threading
import types

import pytest

from kangedit import captions
from kangedit.models import Footage, Project


@pytest.fixture
def fake_whisper(monkeypatch):
    """faster_whisper.WhisperModel that only counts how often a model is loaded."""
    loads = []

    class WhisperModel:
        def __init__(self, name, **kw):
            loads.append(name)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setattr(captions, "_models", captions.OrderedDict())
    return loads


def test_model_loaded_once_and_kept_next_to_another(fake_whisper):
    said = []
    captions.load_model("large-v3", lambda p, m: said.append(m))
    captions.load_model("large-v3", lambda p, m: said.append(m))
    assert fake_whisper == ["large-v3"] and len(said) == 1 and "Loading speech model" in said[0]
    captions._get_model("small")                     # the auto edit listening for the opening
    captions._get_model("large-v3")                  # back to the captions model: still loaded
    assert fake_whisper == ["large-v3", "small"]
    captions._get_model("medium")                    # a third one pushes out the least recently used
    assert not captions.model_ready("small") and captions.model_ready("large-v3")


def test_files_transcribed_in_one_task(qapp, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p = Project()
        files = [Footage(f"clip{k}.mp4", 30.0, 1920, 1080, 30.0, True) for k in range(3)]
        p.footage = {f.id: f for f in files}
        w.set_project(p)
        loads, seen, said = [], [], []
        monkeypatch.setattr("kangedit.ui.main_window.load_model", lambda name, progress=None, cancel=None: loads.append(name))

        def fake_transcribe(f, model, lang, progress, cancel, use_cache=True, cleanup="auto"):
            seen.append(f.name)
            progress(0.5, f"Transcribing {f.name}")
            if f.name == "clip1.mp4":
                raise RuntimeError("no sound stream")
            return [{"s": 0.0, "e": 0.5, "w": "hello"}]
        monkeypatch.setattr("kangedit.ui.main_window.transcribe", fake_transcribe)
        tasks = []

        def submit(name, job, done, err=None):
            tasks.append(name)
            done(job(lambda frac, msg="": said.append(msg), threading.Event()))
        monkeypatch.setattr(w.tasks, "submit", submit)
        warned = []
        monkeypatch.setattr("kangedit.ui.main_window.QMessageBox.warning", lambda *a: warned.append(a[2]))
        w.queue_transcription([f.id for f in files])
        assert tasks == ["Transcribing 3 files"] and loads == [p.settings.whisper_model]
        assert seen == ["clip0.mp4", "clip1.mp4", "clip2.mp4"]           # one failing doesn't stop the rest
        assert files[0].transcript and files[2].transcript and files[1].transcript is None
        assert any(m.endswith("(3/3)") for m in said) and "clip1.mp4: no sound stream" in warned[0]
        w.queue_transcription([f.id for f in files])                       # only what is still missing
        assert tasks[-1] == "Transcribing clip1.mp4"
    finally:
        w.project.dirty = False
        w.close()


def test_cancelled_batch_keeps_the_finished_files(qapp, monkeypatch):
    from kangedit.ffmpeg import Cancelled
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p = Project()
        files = [Footage(f"c{k}.mp4", 30.0, 1920, 1080, 30.0, True) for k in range(3)]
        p.footage = {f.id: f for f in files}
        w.set_project(p)
        p.dirty = False
        monkeypatch.setattr("kangedit.ui.main_window.load_model", lambda name, progress=None, cancel=None: None)

        def fake_transcribe(f, model, lang, progress, cancel, use_cache=True, cleanup="auto"):
            if f.name == "c1.mp4":
                raise Cancelled()                           # the user pressed Cancel during the second file
            return [{"s": 0.0, "e": 0.5, "w": "hi"}]
        monkeypatch.setattr("kangedit.ui.main_window.transcribe", fake_transcribe)
        monkeypatch.setattr(w.tasks, "submit",
                            lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
        w.queue_transcription([f.id for f in files])
        assert files[0].transcript and files[2].transcript is None and w.project.dirty
        assert "cancelled: 1 of 3" in w.lbl_status.text()
    finally:
        w.project.dirty = False
        w.close()


def test_media_panel_transcribes_what_the_set_model_has_not(qapp, monkeypatch, tmp_path):
    from kangedit.ui.main_window import MainWindow
    monkeypatch.setattr(captions, "cache_dir", lambda kind: tmp_path, raising=False)
    monkeypatch.setattr("kangedit.config.cache_dir", lambda kind: tmp_path)
    w = MainWindow()
    try:
        p = Project()
        p.settings.whisper_model, p.settings.whisper_language = "small", "id"
        files = [Footage(str(tmp_path / f"m{k}.mp4"), 30.0, 1920, 1080, 30.0, True) for k in range(4)]
        files[0].transcript = [{"s": 0.0, "e": 0.5, "w": "kept"}]                 # with the model set: kept
        files[0].transcript_model = captions.transcript_key("small", "id")
        files[1].transcript = [{"s": 0.0, "e": 0.5, "w": "other"}]                # another model: again
        files[1].transcript_model = captions.transcript_key("medium", "id")
        files[2].transcript = []                                                 # an older project: no record,
        img = Footage("photo.jpg", 5.0, 1920, 1080, 30.0, False)                 # nor this model's cache
        img.kind = "image"
        p.footage = {f.id: f for f in files + [img]}
        w.set_project(p)
        queued = []
        monkeypatch.setattr(w, "queue_transcription",
                            lambda ids, force=False, use_cache=None: queued.append((list(ids), force, use_cache)))
        asked = []
        monkeypatch.setattr("kangedit.ui.main_window.QMessageBox.question", lambda *a: asked.append(a) or None)
        w.bin_action("transcribe", [f.id for f in files] + [img.id])
        assert queued == [([files[1].id, files[2].id, files[3].id], True, True)] and not asked
        assert "1 already transcribed with it, kept" in w.lbl_status.text()
        w.bin_action("transcribe", [files[0].id])                                # nothing to do: said
        assert len(queued) == 1 and "already transcribed with small" in w.lbl_status.text()
        p.settings.whisper_language = "en"                                       # another language: again
        w.bin_action("transcribe", [files[0].id])
        assert queued[-1][0] == [files[0].id]
    finally:
        w.project.dirty = False
        w.close()


def test_transcript_remembers_its_model(tmp_path, monkeypatch):
    monkeypatch.setattr("kangedit.config.cache_dir", lambda kind: tmp_path)
    f = Footage(str(tmp_path / "x.mp4"), 6.0, 160, 90, 30.0, True)
    monkeypatch.setattr(captions, "cached_transcript", lambda *a, **k: ([{"s": 0, "e": 1, "w": "hi"}], "id"))
    captions.transcribe(f, "small", "id")
    assert f.transcript_model == "small|id" and captions.transcribed_with(f, "small", "id")
    assert not captions.transcribed_with(f, "medium", "id") and not captions.transcribed_with(f, "small", "")
    assert Footage.from_dict(f.to_dict()).transcript_model == "small|id"            # saved with the project
