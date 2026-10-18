"""New hook, timeline fades and cancelling a stuck background task."""
import os
import time

import numpy as np

from kangedit.analysis import SAMPLE_RATE, Analysis
from kangedit.autoedit import AutoEditSettings, find_hook, select
from kangedit.export import ExportSettings, build_export
from kangedit.models import Footage, Project, ProjectSettings, TimelineClip
from kangedit.preview import Look


def fake_analysis(dur: float, peak: float, seed: int) -> Analysis:
    """Flat footage with one lively moment around `peak` seconds."""
    n = int(dur * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    bump = np.exp(-((t - peak) ** 2) / 2.0).astype(np.float32)
    return Analysis(
        times=t, sharp=np.full(n, 5.0, np.float32) + rng.random(n, dtype=np.float32) * 0.1,
        activity=bump * 10, motion=np.ones(n, np.float32), jerk=np.zeros(n, np.float32),
        brightness=np.full(n, 0.5, np.float32), clipped=np.zeros(n, np.float32),
        colorful=np.full(n, 20.0, np.float32), faces=bump * 0.05, audio=-40 + bump * 30,
        cuts=np.zeros(0, np.float32), embed=rng.random((n, 144), dtype=np.float32),
        thumbs=np.zeros((0, 4, 4, 3), np.uint8), thumb_times=np.zeros(0, np.float32), wave=np.zeros(10, np.float32))


def two_files():
    fa = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    fb = Footage("b.mp4", 30.0, 1920, 1080, 30.0, True)
    footage = {fa.id: fa, fb.id: fb}
    analyses = {fa.id: fake_analysis(30.0, 8.0, 1), fb.id: fake_analysis(30.0, 20.0, 2)}
    return footage, analyses


def test_hook_first_edit_labels_its_hook():
    footage, analyses = two_files()
    clips = select(footage, analyses, AutoEditSettings(target_duration=9.0, order="hook"))
    assert clips[0].label == "hook"
    assert all(c.label == "" for c in clips[1:])


def test_find_hook_avoids_the_timeline_and_tried_hooks():
    footage, analyses = two_files()
    s = AutoEditSettings(target_duration=9.0)
    first = find_hook(footage, analyses, s, 3.0, [])
    assert first is not None and abs(first.duration - 3.0) < 0.05
    second = find_hook(footage, analyses, s, 3.0, [first])
    assert second is not None
    assert not (second.footage_id == first.footage_id
                and second.src_in < first.src_out + s.min_gap and first.src_in < second.src_out + s.min_gap)


def test_fades_fit_the_timeline_and_follow_the_switches():
    st = ProjectSettings(fade_in=4.0, fade_out=4.0)
    assert st.fades(10.0) == (4.0, 4.0)
    assert st.fades(4.0) == (2.0, 2.0)          # shortened to fit
    st.fade_sound = False
    assert st.fades(10.0, sound=True) == (0.0, 0.0)
    assert ProjectSettings().fades(10.0) == (0.0, 0.0)


def test_preview_fade_gain():
    look = Look(False, "fill", 64, 36, 36, True, fade_in=1.0, fade_out=2.0, end=10.0)
    assert look.fade_gain(0.0) == 0.0
    assert abs(look.fade_gain(0.5) - 0.5) < 1e-6
    assert look.fade_gain(5.0) == 1.0
    assert abs(look.fade_gain(9.0) - 0.5) < 1e-6


def test_export_graph_fades_picture_and_sound(tmp_path):
    p = Project()
    f = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    p.footage[f.id] = f
    p.clips = [TimelineClip(f.id, 1.0, 4.0), TimelineClip(f.id, 10.0, 13.0)]
    p.settings.fade_in, p.settings.fade_out = 1.0, 1.5

    def graph():
        build_export(p, ExportSettings(str(tmp_path / "out.mp4"), p.settings.aspect, "1080p", 30.0), str(tmp_path))
        return (tmp_path / "graph.txt").read_text(encoding="utf-8")

    g = graph()
    assert "fade=t=in:st=0:d=1.0000" in g and "fade=t=out:st=4.5000:d=1.5000" in g
    assert "afade=t=in:st=0:d=1.0000" in g and "afade=t=out:st=4.5000:d=1.5000" in g
    p.settings.fade_picture = False
    g = graph()
    assert "[vfaded]" not in g and "[afaded]" in g


def test_cancel_answers_at_once_while_a_task_is_stuck(qapp):
    from kangedit.tasks import TaskQueue

    q = TaskQueue()
    got = []
    q.submit("stuck", lambda progress, cancel: time.sleep(5), lambda r: got.append("done"),
             lambda msg: got.append("cancelled"))
    time.sleep(0.2)
    t0 = time.perf_counter()
    q.cancel_all()
    while not got and time.perf_counter() - t0 < 3:
        qapp.processEvents()
        time.sleep(0.02)
    assert got == ["cancelled"]
    assert time.perf_counter() - t0 < 1.5
    q.shutdown()


def test_long_take_of_one_scene_gives_several_moments():
    """A long recording where every moment looks alike (one location) used to give a single clip: the
    no-repeat rule rejected everything else. Far-apart moments are distinct and the file gets its share."""
    long_f = Footage("long.mp4", 300.0, 1920, 1080, 30.0, True)
    short_f = Footage("short.mp4", 10.0, 1920, 1080, 30.0, True)
    a = fake_analysis(300.0, 100.0, 3)
    base = np.random.default_rng(9).random(144, dtype=np.float32)
    a.embed = base + np.random.default_rng(4).random(a.embed.shape, dtype=np.float32) * 0.05
    footage = {long_f.id: long_f, short_f.id: short_f}
    analyses = {long_f.id: a, short_f.id: fake_analysis(10.0, 5.0, 5)}
    s = AutoEditSettings(target_duration=60.0, clip_min=4.0, clip_max=10.0, min_gap=8.0, order="chronological")
    clips = select(footage, analyses, s)
    from_long = sorted(c.src_in for c in clips if c.footage_id == long_f.id)
    assert len(from_long) >= 5
    assert from_long[-1] - from_long[0] > 100          # spread over the recording, not bunched
    assert sum(c.duration for c in clips) > 0.9 * s.target_duration


def test_delete_some_and_all_captions_and_undo():
    from kangedit.captions import build_captions, delete_captions

    p = Project()
    f = Footage("a.mp4", 30.0, 1920, 1080, 30.0, True)
    f.transcript = [{"s": 1.0 + 2 * k, "e": 1.5 + 2 * k, "w": f"w{k}."} for k in range(8)]   # one caption each
    p.footage[f.id] = f
    p.clips = [TimelineClip(f.id, 0.0, 20.0)]
    caps = build_captions(p)
    assert len(caps) == 8
    snap = p._snapshot()
    delete_captions(p, [caps[1], caps[4]])
    assert [c.text for c in build_captions(p)] == ["w0.", "w2.", "w3.", "w5.", "w6.", "w7."]
    delete_captions(p, build_captions(p))
    assert build_captions(p) == []
    p._restore(snap)
    assert len(build_captions(p)) == 8


def test_font_files_are_found_by_name():
    from kangedit import fonts

    fams = fonts.system_fonts()
    if not fams:
        return   # a machine without fonts (minimal CI image)
    name, files = next(iter(fams.items()))
    assert all(os.path.exists(x) for x in files)
    assert fonts.font_files(name.upper()) == files
    assert name in fonts.families_in_file(files[0])
    assert fonts.font_files("No Such Font 123") == []


def test_hook_clips_setting():
    from kangedit.autoedit import hook_plan
    assert hook_plan(AutoEditSettings()) == (1, 3.0)                              # short-form: one quick clip
    assert hook_plan(AutoEditSettings(video_format="long", clip_max=10.0)) == (3, 2.0)   # long: a teaser
    assert hook_plan(AutoEditSettings(hook_clips=4)) == (4, 2.0)
    assert hook_plan(AutoEditSettings(video_format="long", clip_max=10.0, hook_clips=1)) == (1, 4.0)
    assert hook_plan(AutoEditSettings(hook_clips=9))[0] == 5
    footage, analyses = two_files()
    clips = select(footage, analyses, AutoEditSettings(target_duration=12.0, order="hook", hook_clips=2))
    assert [c.label for c in clips[:3]] == ["hook", "hook", ""]
    assert all(abs(c.duration - 2.0) < 0.05 for c in clips[:2])


def test_auto_edit_dialog_sets_the_hook_clips(qapp):
    from kangedit.models import Project
    from kangedit.ui.dialogs import AutoEditDialog
    d = AutoEditDialog(Project(), [], AutoEditSettings(order="hook", hook_clips=3))
    assert d.cb_hook.currentData() == 3 and d.cb_hook.isEnabled()
    d.cb_hook.setCurrentIndex(d.cb_hook.findData(5))
    assert d.settings().hook_clips == 5
    d.cb_order.setCurrentIndex(d.cb_order.findData("chronological"))
    assert not d.cb_hook.isEnabled()                       # no hook in story order
