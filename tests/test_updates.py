"""The update check at startup and the Windows restart helper."""

import os
import shutil
import subprocess
import sys
import time

import pytest

from kangedit import updater


def test_windows_helper_runs_hidden_and_waits_without_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.sys, "platform", "win32")
    monkeypatch.setattr(updater.os, "getpid", lambda: 28604)
    odd = tmp_path / "Jösé & (50%)"                                # an accented user name, & ( ) and %
    odd.mkdir()
    cmd, env = updater._helper_frozen(odd / "new" / "KangEdit", odd / "KangEdit", str(odd), odd / "update.log")
    script = (odd / "install_update.bat").read_bytes().decode("ascii")   # plain ASCII: no code page trouble
    assert "/IS /IT" in script and "%KE_LOG%" in script           # copies every file, and says what it did
    assert "Jösé" not in script and env["KE_DST"] == str(odd / "KangEdit") and env["KE_LOG"].endswith("update.log")
    assert cmd == 'cmd /d /c call "%KE_BAT%"' and env["KE_BAT"].endswith("install_update.bat")
    assert "timeout" not in script                  # needs an interactive console: it failed and spun
    assert "ping -n 2 127.0.0.1" in script and 'find "28604"' in script and "GEQ 120" in script
    assert script.index(":copy") < script.index("robocopy") < script.index('start ""')

    seen = {}
    monkeypatch.setattr(updater.subprocess, "Popen", lambda command, **kw: seen.update(kw))
    updater.launch(cmd, env)
    assert seen["env"]["KE_SRC"] == env["KE_SRC"] and "PATH" in {k.upper() for k in seen["env"]}
    flags = seen["creationflags"]
    assert flags & updater.WIN_NO_WINDOW and not flags & 0x00000008   # a hidden console, not none at all


@pytest.mark.skipif(sys.platform != "win32", reason="runs the real Windows helper")
def test_windows_helper_really_waits_copies_and_restarts(tmp_path, monkeypatch):
    """The helper exactly as an update runs it (hidden console): it waits while the app is still running,
    then copies the new build over the old one, starts it and removes its folder, all without hanging."""
    tmp_path = tmp_path / "Jösé & (50%)"                     # paths cmd used to get wrong
    work = tmp_path / "update"
    staging = work / "new" / "KangEdit"
    staging.mkdir(parents=True)
    whoami = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "whoami.exe")
    shutil.copy(whoami, staging / "KangEdit.exe")              # "the new app": starts and exits at once
    (staging / "data.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "app"
    target.mkdir(parents=True)
    (target / "KangEdit.exe").write_bytes(b"old")
    (target / "data.txt").write_text("old", encoding="utf-8")
    app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"])   # "the running app"
    monkeypatch.setattr(updater.os, "getpid", lambda: app.pid)
    log = tmp_path / "update.log"
    cmd, env = updater._helper_frozen(staging, target, str(work), log)
    t0 = time.monotonic()
    updater.launch(cmd, env)
    time.sleep(2)
    assert (target / "data.txt").read_text(encoding="utf-8") == "old"    # still waiting for the app to close
    app.wait()
    while work.exists() and time.monotonic() - t0 < 60:
        time.sleep(0.25)
    said = log.read_text(errors="replace") if log.exists() else "(no log)"
    assert not work.exists(), "the helper didn't finish:\n" + said
    assert (target / "data.txt").read_text(encoding="utf-8") == "new", said
    assert "robocopy result" in said
    assert (target / "KangEdit.exe").read_bytes() == open(whoami, "rb").read()
    assert time.monotonic() - t0 < 30                                    # no two-minute fallback wait


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
