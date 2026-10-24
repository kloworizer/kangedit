"""Export presets per platform (frame, format, loudness target) and batch export in several aspect ratios."""

import os
import re
import time

import pytest

from kangedit.export import (CUSTOM, PRESETS, ExportSettings, apply_preset, batch_settings, build_export, export)
from kangedit.ffmpeg import FFmpegError, ffmpeg_bin, probe, run
from kangedit.models import Footage, Project, TimelineClip


def _project(tmp_path, dur=10.0, audio=True):
    f = Footage(str(tmp_path / "x.mp4"), dur, 1920, 1080, 30.0, audio)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, min(dur, 4.0))]
    return p


def _graph(tmp_path, p, s):
    args = build_export(p, s, str(tmp_path))
    return args, open(tmp_path / args[args.index("-/filter_complex") + 1]).read()


def test_preset_fills_in_the_platform_settings():
    s = ExportSettings("out/myvideo.mov", "4:3 (Classic)", "720p", 25.0, codec="prores", quality="Small file")
    yt = apply_preset(s, "youtube")
    assert (yt.aspect, yt.resolution, yt.codec, yt.quality, yt.loudness) == \
        ("16:9 (Landscape)", "1080p", "h264", "High", -14.0)
    assert yt.fps == 25.0 and yt.output == "out/myvideo.mp4"     # the project's frame rate kept; .mp4 for H.264
    assert apply_preset(s, "youtube_4k").resolution == "4K (2160p)"
    for key in ("shorts", "reels", "tiktok"):
        v = apply_preset(s, key)
        assert (v.aspect, v.resolution, v.codec, v.loudness) == ("9:16 (Vertical)", "1080p", "h264", -14.0)
    assert apply_preset(s, "instagram_feed").aspect == "4:5 (Portrait)"
    assert PRESETS["tiktok"].overlay == "tiktok" and PRESETS["reels"].overlay == "reels"
    assert apply_preset(s, CUSTOM) is s


def test_loudness_target_adds_loudnorm_after_the_mix_and_fades(tmp_path):
    p = _project(tmp_path)
    p.settings.fade_in = p.settings.fade_out = 0.5
    s = ExportSettings(str(tmp_path / "out.mp4"), "16:9 (Landscape)", "720p", 30.0, use_gpu=False)
    args, graph = _graph(tmp_path, p, s)
    assert "loudnorm" not in graph                                  # Off: the mix as it is
    s.loudness = -14.0
    args, graph = _graph(tmp_path, p, s)
    assert "loudnorm=I=-14.0:TP=-1.5:LRA=11" in graph
    assert "[afaded]loudnorm" in graph                               # after the final fades
    assert args[args.index("-map", args.index("-map") + 1) + 1] == "[aloud]"   # what is encoded


def test_batch_settings_add_files_with_the_ratio_as_suffix():
    s = ExportSettings("/v/myvideo.mp4", "16:9 (Landscape)", "1080p", 30.0, loudness=-14.0)
    out = batch_settings(s, ["9:16 (Vertical)", "16:9 (Landscape)", "1:1 (Square)", "9:16 (Vertical)"])
    assert [(x.aspect, x.output) for x in out] == [
        ("16:9 (Landscape)", "/v/myvideo.mp4"), ("9:16 (Vertical)", "/v/myvideo_9x16.mp4"),
        ("1:1 (Square)", "/v/myvideo_1x1.mp4")]
    assert all(x.loudness == -14.0 and x.resolution == "1080p" for x in out)


def test_export_dialog_presets_custom_and_batch(qapp, tmp_path):
    from kangedit.config import settings
    from kangedit.ui.dialogs import ExportDialog
    settings().remove("export_preset")
    p = _project(tmp_path)
    p.path = str(tmp_path / "myvideo.kgproj")
    d = ExportDialog(p, False)
    assert d.cb_preset.currentData() == CUSTOM and d.settings().loudness is None
    d.cb_preset.setCurrentIndex(d.cb_preset.findData("tiktok"))
    s = d.settings()
    assert (s.aspect, s.resolution, s.codec, s.loudness, s.fps) == ("9:16 (Vertical)", "1080p", "h264", -14.0, 30.0)
    assert d.cb_preset.currentData() == "tiktok"                     # filling it in keeps the preset
    assert not d.chk_overlay.isHidden() and d.platform_overlay() == "tiktok"
    d.cb_preset.setCurrentIndex(d.cb_preset.findData("youtube"))
    assert d.chk_overlay.isHidden() and d.platform_overlay() is None
    d.cb_loud.setCurrentIndex(d.cb_loud.findData(-23.0))             # changed by hand: Custom
    assert d.cb_preset.currentData() == CUSTOM and d.settings().loudness == -23.0
    assert d.settings().aspect == "16:9 (Landscape)"                 # the rest stays as the preset left it

    d.chk_batch["16:9 (Landscape)"].setChecked(True)                 # the main ratio: not a second file
    d.chk_batch["9:16 (Vertical)"].setChecked(True)
    d.chk_batch["1:1 (Square)"].setChecked(True)
    assert not d.chk_batch["16:9 (Landscape)"].isEnabled()
    out = d.all_settings()
    base = str(tmp_path / "myvideo")
    assert [(x.aspect, x.output) for x in out] == [("16:9 (Landscape)", base + ".mp4"),
                                                   ("9:16 (Vertical)", base + "_9x16.mp4"),
                                                   ("1:1 (Square)", base + "_1x1.mp4")]
    assert all(x.loudness == -23.0 for x in out)

    d.cb_preset.setCurrentIndex(d.cb_preset.findData("reels"))
    d._accept()                                                      # the last preset is remembered
    d2 = ExportDialog(p, False)
    assert d2.cb_preset.currentData() == "reels" and d2.settings().aspect == "9:16 (Vertical)"
    settings().remove("export_preset")


def _talk(tmp_path, dur=6, volume=0.05):
    """A test clip with a steady 440 Hz tone at a known (quiet) level."""
    src = str(tmp_path / "talk.mp4")
    try:
        run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={dur}",
             "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur},volume={volume}",
             "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", src])
    except Exception as e:   # noqa: BLE001
        pytest.skip(f"FFmpeg can't make the test clip: {e}")
    i = probe(src)
    return Footage(src, i.duration, i.width, i.height, i.fps, i.has_audio)


def _integrated_lufs(path) -> float:
    res = run([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", path, "-map", "0:a", "-af", "ebur128", "-f", "null", "-"])
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", res.stderr.decode("utf-8", "replace"))
    assert found, res.stderr[-500:]
    return float(found[-1])


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_exported_loudness_meets_the_target(tmp_path):
    f = _talk(tmp_path)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 5.5)]
    out = str(tmp_path / "loud.mp4")
    before = str(tmp_path / "plain.mp4")
    try:
        export(p, ExportSettings(before, "16:9 (Landscape)", "720p", 30.0, use_gpu=False))
        export(p, ExportSettings(out, "16:9 (Landscape)", "720p", 30.0, use_gpu=False, loudness=-14.0))
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    assert _integrated_lufs(before) < -25                       # the tone as recorded: quiet
    assert abs(_integrated_lufs(out) + 14.0) < 1.5
    assert abs(probe(out).duration - 5.5) < 0.2                 # loudnorm keeps the length


@pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")
def test_batch_export_makes_a_file_per_aspect_ratio(tmp_path):
    f = _talk(tmp_path, dur=2)
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 1.5)]
    main = ExportSettings(str(tmp_path / "myvideo.mp4"), "16:9 (Landscape)", "720p", 30.0, use_gpu=False)
    batch = batch_settings(main, ["9:16 (Vertical)"])
    try:
        for s in batch:
            export(p, s)
    except FFmpegError as e:
        pytest.skip(f"export needs a newer FFmpeg: {e}")
    sizes = {os.path.basename(s.output): (probe(s.output).width, probe(s.output).height) for s in batch}
    assert sizes == {"myvideo.mp4": (1280, 720), "myvideo_9x16.mp4": (720, 1280)}


def test_batch_queue_goes_on_after_a_failure_and_sums_up(qapp, tmp_path, monkeypatch, no_blocking_message_boxes):
    """Each file is its own task, named with its place in the batch; one that fails doesn't stop the next, and
    one summary lists them all at the end."""
    from kangedit.ui import main_window as mw
    made = []

    def fake_export(project, s, progress=None, cancel=None):
        if s.aspect.startswith("9:16"):
            raise RuntimeError("disk full")
        made.append(s.output)
        return s.output

    monkeypatch.setattr(mw, "export", fake_export)
    w = mw.MainWindow()
    names = []
    w.tasks.started.connect(names.append)
    try:
        p = _project(tmp_path)
        main = ExportSettings(str(tmp_path / "v.mp4"), "16:9 (Landscape)", "720p", 30.0)
        w._export_batch(p, batch_settings(main, ["9:16 (Vertical)", "1:1 (Square)"]))
        end = time.time() + 10
        while w.tasks.busy and time.time() < end:
            qapp.processEvents()
            time.sleep(0.01)
        qapp.processEvents()
    finally:
        w.project.dirty = False
        w.close()
    assert names == ["Exporting v.mp4 (1 of 3)", "Exporting v_9x16.mp4 (2 of 3)", "Exporting v_1x1.mp4 (3 of 3)"]
    assert made == [str(tmp_path / "v.mp4"), str(tmp_path / "v_1x1.mp4")]
    boxes = [b for b in no_blocking_message_boxes if b[0] in ("warning", "information")]
    assert len(boxes) == 1 and boxes[0][0] == "warning"
    text = boxes[0][1][1]
    assert "Exported 2 of 3" in text and "v_9x16.mp4: disk full" in text and "v_1x1.mp4" in text
