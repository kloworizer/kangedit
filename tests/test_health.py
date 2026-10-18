"""The health check before an export: issues, their fixes, and the dialog."""

from kangedit import health
from kangedit.models import Footage, Project, SoundClip, TextClip, TimelineClip


def _project(tmp_path, has_audio=True):
    path = tmp_path / "a.mp4"
    path.write_bytes(b"x")
    f = Footage(str(path), 30.0, 1920, 1080, 30.0, has_audio)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 5.0), TimelineClip(f.id, 10.0, 15.0)]
    return p, f


def test_a_healthy_project_has_nothing_to_say(tmp_path):
    p, _ = _project(tmp_path)
    assert health.check(p) == []
    assert [i.level for i in health.check(Project())] == [health.ERROR]          # an empty timeline


def test_issues_and_their_fixes(tmp_path):
    p, f = _project(tmp_path)
    f.transcript = None
    p.sounds = [SoundClip(str(tmp_path / "gone.mp3"), 0.0, 0.0, 5.0, name="Song")]
    p.texts = [TextClip("Late", 50.0)]
    p.clips[0].motion, p.clips[1].stabilize = "face", 0.5
    other = Footage(str(tmp_path / "moved.mp4"), 30.0, 1920, 1080, 30.0, True)
    p.footage[other.id] = other
    p.clips.append(TimelineClip(other.id, 0.0, 3.0))
    got = health.check(p, captions="burn", use_proxies=True)
    by_fix = {i.fix.split(":")[0]: i for i in got}
    assert got[0].level == health.ERROR and by_fix["locate"].fix == f"locate:{other.id}"
    assert set(by_fix) == {"locate", "drop_sounds", "transcribe", "measure", "faces", "drop_texts", "originals"}
    assert "Song" in by_fix["drop_sounds"].text
    rank = [[health.ERROR, health.WARNING, health.INFO].index(i.level) for i in got]
    assert rank == sorted(rank)                                                    # errors first, then warnings
    assert not any(i.fix == "transcribe" for i in health.check(p, captions="none"))


def test_no_sound_at_all(tmp_path):
    p, f = _project(tmp_path, has_audio=False)
    assert any("no sound" in i.text for i in health.check(p))
    p.sounds = [SoundClip(str(tmp_path / "a.mp4"), 0.0, 0.0, 5.0)]                # music: fine
    assert not any("no sound" in i.text for i in health.check(p))
    p, f = _project(tmp_path)
    p.clips[0].volume = 0.0
    assert [i.text for i in health.check(p)] == ["1 clip(s) with sound are muted (volume 0)."]


def test_dialog_fixes_then_lets_the_export_go(qapp, tmp_path):
    from kangedit.ui.health_dialog import HealthDialog
    p, f = _project(tmp_path)
    missing = f.path + ".gone"
    f.path = missing
    p.texts = [TextClip("Late", 50.0)]
    done = []

    def fix(key):
        done.append(key)
        if key.startswith("locate:"):
            f.path = missing[:-5]
        elif key == "drop_texts":
            p.texts = []
        return "fixed"

    d = HealthDialog(lambda: health.check(p), fix)
    assert not d.btn_export.isEnabled() and "stop the export" in d.head.text()
    d.apply(d.issues[0].fix)                              # the file found: the export may go
    assert d.btn_export.isEnabled() and d.btn_export.text() == "Export anyway" and d.lbl.text() == "fixed"
    d.apply("drop_texts")
    assert d.issues == [] and d.btn_export.text() == "Export" and done[0].startswith("locate:")


def test_export_runs_the_check(qapp, tmp_path, monkeypatch):
    from kangedit.export import ExportSettings
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, f = _project(tmp_path)
        p.texts = [TextClip("Late", 50.0)]
        w.set_project(p)
        es = ExportSettings(str(tmp_path / "o.mp4"), p.settings.aspect, "720p", 30.0, use_proxies=True)
        from kangedit.ui import health_dialog
        seen = {}

        def run(dlg):
            seen["issues"] = [i.fix for i in dlg.issues]
            dlg.apply("originals")
            dlg.apply("drop_texts")
            seen["after"] = list(dlg.issues)
            return health_dialog.HealthDialog.Accepted

        monkeypatch.setattr(health_dialog.HealthDialog, "exec", run)
        assert w.health_check(es)
        assert set(seen["issues"]) == {"drop_texts", "originals"} and seen["after"] == []
        assert es.use_proxies is False and p.texts == []
        w.undo()
        assert len(w.project.texts) == 1
    finally:
        w.project.dirty = False
        w.close()
