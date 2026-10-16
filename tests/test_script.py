"""Auto edit from a script: splitting, fuzzy matching, takes, clip boundaries, and the dialog / app flow."""

import threading

import pytest

from kangedit import scriptedit as S
from kangedit.models import Footage, Project


def _take(text, t0, gap=0.1):
    words, t = [], t0
    for w in text.split():
        words.append({"s": round(t, 2), "e": round(t + 0.3, 2), "w": w})
        t += 0.3 + gap
    return words


@pytest.fixture
def footage():
    a = Footage("a.mp4", 120.0, 1920, 1080, 30.0, True)
    b = Footage("b.mp4", 120.0, 1920, 1080, 30.0, True)
    a.transcript = (_take("um okay so welcome to the channel today we're looking at there new camera. and honestly "
                          "it surprised me um", 2.0)
                    + _take("wait let me do that again welcome to the channel today we are looking at their new "
                            "camera", 30.0))
    b.transcript = _take("the battery lasted all day long which is great. let's talk about the price", 5.0)
    return a, b


SCRIPT = """HOST: Welcome to the channel! Today we're looking at their new camera.
And honestly, it surprised me. [beat]

The battery lasts all day, which is great.
Let's talk about the price. Something nobody ever said in this footage at all."""


def test_split_and_normalize():
    lines = S.split_script(SCRIPT)
    assert lines[0] == "Welcome to the channel!" and lines[1] == "Today we're looking at their new camera."
    assert "[beat]" not in lines[2] and len(lines) == 6
    assert S.normalize("Héllo, WORLD!  It's") == "hello world its"
    assert S.split_script("ANNA:\nHello there friend.") == ["Hello there friend."]     # a label on its own line
    assert S.similarity("their new camera", "there new camera") > 0.85


def test_lines_found_despite_recognition_slips(footage):
    a, b = footage
    ms = S.match_script(SCRIPT, [a, b], 0.6)
    found = [m for m in ms if m.found]
    assert len(found) == 5 and ms[5].found is False and ms[5].score < 0.6
    assert ms[3].footage_id == b.id and ms[3].score > 0.85              # "lasts" heard as "lasted ... long"
    assert ms[1].takes == 2                                              # said twice in file a
    best = ms[0]
    assert best.footage_id == a.id and best.src_in == pytest.approx(3.2)   # the first take matches as well
    last = S.match_script(SCRIPT, [a, b], 0.6, prefer="last")[0]
    assert last.src_in > 30.0                                            # the retake


def test_threshold_decides(footage):
    a, b = footage
    lenient = sum(m.found for m in S.match_script("the battery lasts for a whole day", [a, b], 0.5))
    strict = sum(m.found for m in S.match_script("the battery lasts for a whole day", [a, b], 0.95))
    assert lenient == 1 and strict == 0


def test_clips_cut_on_words_and_join_one_take(footage):
    a, b = footage
    r = S.script_edit(SCRIPT, [a, b], 0.6, prefer="last")
    assert [c.footage_id for c in r.clips] == [a.id, a.id, b.id]          # lines 1+2 one take; 4+5 joined
    first = r.clips[0]
    words = a.transcript
    w_before = next(w for w in words if w["s"] < 32.4 and w["e"] > 31.5)  # "again" before "welcome"
    assert first.src_in >= w_before["e"]                                 # never cuts into the previous word
    assert first.src_in == pytest.approx(32.4 - 0.08, abs=0.03)
    assert r.clips[2].src_in < 5.0 and r.clips[2].src_out > 9.0           # "battery ... price" in one clip
    assert len(r.missing) == 1


@pytest.fixture
def window(qapp):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def test_script_dialog_preview_and_app_flow(window, footage, monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from kangedit.ui.script_dialog import ScriptEditDialog
    a, b = footage
    c = Footage(str(tmp_path / "c.mp4"), 60.0, 1920, 1080, 30.0, True)   # not transcribed yet
    p = Project()
    p.footage = {f.id: f for f in (a, b, c)}
    window.set_project(p)
    d = ScriptEditDialog(p, [], window)
    d.ed_script.setPlainText(SCRIPT)
    for i in range(d.lst_footage.count()):                               # c: unticked
        it = d.lst_footage.item(i)
        if it.data(Qt.UserRole) == c.id:
            it.setCheckState(Qt.Unchecked)
    d.update_preview()
    assert d.tree.topLevelItemCount() == 6 and "5 of 6 lines found" in d.lbl_summary.text()
    d.sl_threshold.setValue(99)
    d.update_preview()
    assert "of 6 lines found" in d.lbl_summary.text()

    # the app: c needs a transcript first; the queue transcribes it, then matches and lays out the clips
    calls = []

    def fake_transcribe(f, *args, **kw):
        calls.append(f.name)
        f.transcript = _take("nothing from the script here", 1.0)
        return f.transcript
    monkeypatch.setattr("kangedit.ui.main_window.transcribe", fake_transcribe)
    monkeypatch.setattr("kangedit.ui.main_window.load_model", lambda name, progress=None: None)
    monkeypatch.setattr(window.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *x: None, threading.Event())))
    st = {"script": SCRIPT, "footage": [a.id, b.id, c.id], "threshold": 0.6, "prefer": "last", "pad_in": 0.15,
          "pad_out": 0.25, "merge": True, "replace": True}
    monkeypatch.setattr(ScriptEditDialog, "exec", lambda self: ScriptEditDialog.Accepted)
    monkeypatch.setattr(ScriptEditDialog, "settings", lambda self: st)
    shown = []
    monkeypatch.setattr("kangedit.ui.main_window.QMessageBox.exec", lambda self: shown.append(self.text()))
    window.script_edit_dialog()
    assert calls == ["c.mp4"] and len(window.project.clips) == 3
    assert "5 of 6 lines found" in window.lbl_status.text() and "1 of 6 lines were not found" in shown[0]
    assert window.project.settings.script["prefer"] == "last"
    window.undo()
    assert window.project.clips == []
