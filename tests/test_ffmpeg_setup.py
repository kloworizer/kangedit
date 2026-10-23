"""Finding FFmpeg on the computer (new enough, with FFprobe) and downloading it, from another server when one
fails."""

import io
import os
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from kangedit import ffmpeg_setup as setup


@pytest.fixture(autouse=True)
def _fresh_lookup(monkeypatch):
    """Each test looks for FFmpeg anew; the real lookup comes back after it (other tests use it)."""
    monkeypatch.setattr(setup, "_installed", None)


@pytest.mark.parametrize("out, want", [
    ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023", (6, 1)),
    ("ffmpeg version n7.1 Copyright", (7, 1)),
    ("ffmpeg version 8.0-essentials_build-www.gyan.dev", (8, 0)),
    ("ffmpeg version N-117000-gabcdef Copyright", (99, 0)),          # a build from the development tree
    ("", None),                                                      # doesn't run
])
def test_version(monkeypatch, out, want):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert setup.version_of("ffmpeg") == want


def _programs(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / setup.exe(n)).write_bytes(b"")
    return folder


def test_an_old_or_partial_ffmpeg_is_said_not_used(tmp_path, monkeypatch):
    old = _programs(tmp_path / "old", "ffmpeg", "ffprobe")
    monkeypatch.setenv("PATH", str(old))
    monkeypatch.setattr(setup, "_windows_dirs", lambda: [])
    monkeypatch.setattr(setup, "version_of", lambda p: (6, 1))
    found = setup.installed(refresh=True)
    assert found.ffmpeg and not found.usable and setup.installed_binary("ffmpeg") is None
    assert "FFmpeg 6.1 is installed" in setup.missing_note() and "7 or newer" in setup.missing_note()

    alone = _programs(tmp_path / "alone", "ffmpeg")                 # FFmpeg without FFprobe
    monkeypatch.setenv("PATH", str(alone))
    monkeypatch.setattr(setup, "version_of", lambda p: (7, 1))
    setup.installed(refresh=True)
    assert "not FFprobe" in setup.missing_note()

    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    setup.installed(refresh=True)
    assert "was not found" in setup.missing_note()


def test_a_new_ffmpeg_installed_off_path_is_used(tmp_path, monkeypatch):
    old = _programs(tmp_path / "old", "ffmpeg", "ffprobe")
    winget = _programs(tmp_path / "winget", "ffmpeg", "ffprobe")   # installed, its folder not on PATH yet
    monkeypatch.setenv("PATH", str(old))
    monkeypatch.setattr(setup, "_windows_dirs", lambda: [winget])
    monkeypatch.setattr(setup, "version_of", lambda p: (6, 0) if str(old) in p else (8, 0))
    found = setup.installed(refresh=True)
    assert found.usable and found.ffmpeg == str(winget / setup.exe("ffmpeg"))
    assert setup.installed_binary("ffprobe") == str(winget / setup.exe("ffprobe"))


def test_download_tries_the_next_server(tmp_path, monkeypatch):
    dest = tmp_path / "ffmpeg" / "bin"
    monkeypatch.setattr(setup, "bin_dir", lambda: dest)
    monkeypatch.setattr(setup, "mirrors", lambda: [["https://github.com/a.zip"], ["https://www.gyan.dev/b.zip"]])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in ("ffmpeg", "ffprobe"):
            zf.writestr(f"ffmpeg-8.0/bin/{setup.exe(n)}", b"x")

    def download(url, progress, label, cancel=None):
        if "github" in url:
            raise TimeoutError("timed out")                          # blocked, or not answering
        return buf.getvalue()

    monkeypatch.setattr(setup, "_download", download)
    said = []
    assert setup.install(lambda p, m: said.append(m)) == dest
    assert sorted(os.listdir(dest)) == sorted(setup.exe(n) for n in ("ffmpeg", "ffprobe"))
    assert any("github.com didn't work" in m and "trying www.gyan.dev" in m for m in said)
    assert [p.name for p in dest.parent.iterdir()] == ["bin"]          # nothing left over

    monkeypatch.setattr(setup, "_download", lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
    with pytest.raises(RuntimeError, match="github.com: no network"):
        setup.install()
    assert (dest / setup.exe("ffmpeg")).is_file()                    # the earlier download is kept


def test_every_platform_has_a_download():
    assert setup.mirrors() and setup.sources() == setup.mirrors()[0]
