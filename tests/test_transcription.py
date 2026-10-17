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
        monkeypatch.setattr("kangedit.ui.main_window.load_model", lambda name, progress=None: loads.append(name))

        def fake_transcribe(f, model, lang, progress, cancel, use_cache=True):
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
        monkeypatch.setattr("kangedit.ui.main_window.load_model", lambda name, progress=None: None)

        def fake_transcribe(f, model, lang, progress, cancel, use_cache=True):
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


def test_media_panel_transcribes_only_what_is_missing(qapp, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p = Project()
        files = [Footage(f"m{k}.mp4", 30.0, 1920, 1080, 30.0, True) for k in range(3)]
        files[0].transcript = [{"s": 0.0, "e": 0.5, "w": "kept"}]
        files[2].transcript = []                                         # transcribed: no speech in it
        p.footage = {f.id: f for f in files}
        w.set_project(p)
        queued = []
        monkeypatch.setattr(w, "queue_transcription", lambda ids, force=False: queued.append((list(ids), force)))
        asked = []
        monkeypatch.setattr("kangedit.ui.main_window.QMessageBox.question",
                            lambda *a: asked.append(a[2]) or QMessageBox.No)
        from kangedit.ui.main_window import QMessageBox
        w.bin_action("transcribe", [f.id for f in files])                 # all selected, right-click
        assert queued == [([files[1].id], False)] and not asked
        assert "2 already transcribed, kept" in w.lbl_status.text()
        files[1].transcript = []
        w.bin_action("transcribe", [f.id for f in files])                 # all done: ask first
        assert len(asked) == 1 and "already transcribed" in asked[0] and len(queued) == 1
        monkeypatch.setattr("kangedit.ui.main_window.QMessageBox.question", lambda *a: QMessageBox.Yes)
        w.bin_action("transcribe", [files[0].id])
        assert queued[-1] == ([files[0].id], True)
    finally:
        w.project.dirty = False
        w.close()
