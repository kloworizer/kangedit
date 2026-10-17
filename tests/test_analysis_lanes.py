"""How the full analysis shares out its work: every lane busy whatever the GPU can decode, GPU segments on the
GPU lanes, one damaged file not taking the GPU away from the rest, and the smarter-analysis upgrade in parallel."""

import threading
import time

import pytest

from kangedit import analysis as an
from kangedit.models import Footage


class _Done:
    def save(self, path):
        pass


@pytest.fixture
def lanes(monkeypatch):
    """analyze_many on a 12-thread computer with a GPU (2 GPU + 3 CPU lanes) and a fake decoder that records
    which lane decoded what, and how."""
    monkeypatch.setattr(an, "lane_plan", lambda use_gpu=True: ("d3d11va", 3))
    monkeypatch.setattr(an, "gpu_lanes", lambda: 2)
    monkeypatch.setattr(an.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(an, "_finish", lambda j: _Done())
    monkeypatch.setattr(an, "analysis_path_for", lambda f: "unused")
    monkeypatch.setattr(an, "quick_path_for", lambda f: "/nonexistent/quick.npz")
    for name in ("_hw_failed", "_hw_worked"):
        monkeypatch.setattr(an, name, set())
    monkeypatch.setattr(an, "_hw_misses", {})
    rec = {"active": 0, "peak": 0, "calls": [], "fail": set()}
    lock = threading.Lock()

    def scan(u, hw, threads, cancel, tick):
        with lock:
            rec["active"] += 1
            rec["peak"] = max(rec["peak"], rec["active"])
            rec["calls"].append((threading.current_thread().name, u.job.footage.path, hw, threads))
        time.sleep(0.03)
        with lock:
            rec["active"] -= 1
        if hw and u.job.footage.path in rec["fail"]:
            return None
        return {}

    monkeypatch.setattr(an, "_scan", scan)
    return rec


def _clips(n, pix_fmt, start=0):
    return [Footage(f"/f/{start + i}.mp4", 10.0, 3840, 2160, 25.0, False, codec="h264", pix_fmt=pix_fmt)
            for i in range(n)]


def test_footage_the_gpu_cant_decode_keeps_every_lane_busy(lanes):
    msgs = []
    an.analyze_many(_clips(20, "yuv422p10le"), progress=lambda p, m="": msgs.append(m))
    assert lanes["peak"] == 5                               # was 3: the GPU lanes sat idle
    assert any(name.startswith("analysis-gpu") for name, *_ in lanes["calls"])
    assert all(hw is None and threads >= 2 for _, _, hw, threads in lanes["calls"])
    assert "the GPU can't decode this footage" in msgs[0]


def test_gpu_lanes_take_what_the_gpu_decodes(lanes):
    fs = _clips(6, "yuv420p") + _clips(6, "yuv422p10le", start=6)
    an.analyze_many(fs)
    gpu = {p for name, p, hw, _ in lanes["calls"] if hw}
    assert gpu and gpu <= {f.path for f in fs[:6]}
    cpu_first = [p for name, p, hw, _ in lanes["calls"] if name.startswith("analysis-cpu")][:3]
    assert set(cpu_first) <= {f.path for f in fs[6:]}       # the CPU lanes start on what only they can do
    assert lanes["peak"] == 5


def test_one_damaged_file_does_not_turn_the_gpu_off(lanes):
    fs = _clips(8, "yuv420p")
    lanes["fail"] = {fs[0].path}
    an.analyze_many(fs)
    assert ("h264", "yuv420p") not in an._hw_failed
    assert any(hw for _, p, hw, _ in lanes["calls"] if p != fs[0].path)
    assert [hw for _, p, hw, _ in lanes["calls"] if p == fs[0].path][-1] is None   # redone on the CPU


def test_a_format_that_never_decodes_on_the_gpu_goes_to_the_cpu(lanes):
    fs = _clips(30, "yuv420p10le")
    lanes["fail"] = {f.path for f in fs}
    an.analyze_many(fs)
    assert ("h264", "yuv420p10le") in an._hw_failed
    assert sum(1 for _, _, hw, _ in lanes["calls"] if hw) <= an.HW_GIVE_UP + 1   # tried a few times, not 30


def test_smarter_analysis_upgrade_runs_in_parallel(qapp, monkeypatch):
    from kangedit.ui import main_window as mw
    w = mw.MainWindow()
    try:
        active, peak, lock = [0], [0], threading.Lock()

        def fake(f, a, cancel=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return False

        monkeypatch.setattr(mw, "make_smart", fake)
        monkeypatch.setattr(mw.os, "cpu_count", lambda: 12)
        items = [(f, None) for f in _clips(12, "yuv420p")]
        t = time.perf_counter()
        assert w._upgrade_smart(items, threading.Event())
        assert peak[0] >= 4 and time.perf_counter() - t < 0.4     # one at a time would take 0.6 s
    finally:
        w.project.dirty = False
        w.close()
