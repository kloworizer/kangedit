"""The update check at startup and the Windows restart helper."""

import time

import pytest

from kangedit import updater


def test_windows_helper_runs_hidden_and_waits_without_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr(updater.os, "getpid", lambda: 28604)
    cmd = updater._helper_frozen(tmp_path / "new" / "KangEdit", tmp_path / "KangEdit", str(tmp_path))
    script = (tmp_path / "install_update.bat").read_text(encoding="utf-8")
    assert cmd[:2] == ["cmd", "/c"]
    assert "timeout" not in script                  # needs an interactive console: it failed and spun
    assert "ping -n 2 127.0.0.1" in script and 'find "28604"' in script and "GEQ 120" in script
    assert script.index(":copy") < script.index("robocopy") < script.index('start ""')

    seen = {}
    monkeypatch.setattr(updater.subprocess, "Popen", lambda command, **kw: seen.update(kw))
    updater.launch(cmd)
    flags = seen["creationflags"]
    assert flags & updater.WIN_NO_WINDOW and not flags & 0x00000008   # a hidden console, not none at all


@pytest.fixture
def window(qapp, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.project.dirty = False
    w.close()


def test_startup_check_runs_at_every_start(window, monkeypatch):
    calls = []
    monkeypatch.setattr(updater, "latest_release", lambda *a, **k: calls.append(1) or updater.Release(
        "0.0.1", "v0.0.1", "", "", "", ""))
    window.qs.setValue("update_last_check", time.time())      # a manual check just now (the old throttle)
    window.qs.setValue("update_last_auto", time.time() - 3600)
    window.check_updates(manual=False)
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert calls == [1]                                        # an hour later: checked, whatever the manual check
    window.check_updates(manual=False)                         # restarted a moment later: not again
    time.sleep(0.2)
    assert calls == [1]
