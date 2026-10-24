"""Caption tools: auto-sync to the speech, style templates, and speaker labels and colours."""

import threading
import wave

import numpy as np
import pytest

from kangedit import caption_templates, captions, speakers
from kangedit.captions import (ANIMATIONS, Caption, apply_sync, build_captions, caption_color, edit_caption,
                               estimate_sync_offset, measure_sync, set_caption_speaker, sync_jobs, write_ass)
from kangedit.ffmpeg import ffmpeg_bin, run
from kangedit.models import SPEAKER_COLORS, CaptionStyle, Footage, Project, TimelineClip

RATE = 16000
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_bin(), reason="needs FFmpeg")


def _write_wav(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(path)


def _bursts(seconds=16.0, seed=3):
    """Tone bursts (syllables) at irregular known times, over faint noise: (audio, [(start, end)])."""
    rng = np.random.default_rng(seed)
    spans, t = [], 0.6
    while t < seconds - 1.0:
        d = rng.uniform(0.2, 0.5)
        spans.append((round(t, 3), round(t + d, 3)))
        t += d + rng.uniform(0.15, 0.8)
    a = rng.normal(0, 0.002, int(seconds * RATE)).astype(np.float32)
    for s, e in spans:
        i, j = int(s * RATE), int(e * RATE)
        a[i:j] += 0.3 * np.sin(2 * np.pi * 330 * np.arange(j - i) / RATE).astype(np.float32)
    return a, spans


def _words(spans, off=0.0):
    """A transcript of the bursts, `off` seconds late (negative: early); a sentence every 4 words."""
    return [{"s": round(s + off, 3), "e": round(e + off, 3), "w": f"w{i}" + ("." if i % 4 == 3 else "")}
            for i, (s, e) in enumerate(spans)]


# ------------------------------------------------------------------------------------------- auto-sync
@pytest.mark.parametrize("off", [0.35, -0.5, 1.2])
def test_offset_is_measured(off):
    audio, spans = _bursts()
    got = estimate_sync_offset(_words(spans, -off), audio, 0.0)
    assert got == pytest.approx(off, abs=0.05)


def test_aligned_words_and_silence():
    audio, spans = _bursts()
    assert estimate_sync_offset(_words(spans), audio) == 0.0
    assert estimate_sync_offset(_words(spans, 0.02), audio) == 0.0          # too small to bother
    assert estimate_sync_offset(_words(spans), np.zeros(RATE * 10, np.float32)) is None   # no speech heard


def _mp4(tmp_path, audio):
    wav = _write_wav(tmp_path / "talk.wav", audio)
    src = str(tmp_path / "talk.mp4")
    dur = audio.size / RATE
    run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", f"color=c=gray:size=160x90:rate=30:duration={dur}",
         "-i", wav, "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-y", src])
    return src


def _sync_project(src, words):
    f = Footage(src, 16.0, 160, 90, 30.0, True)
    f.transcript = [{"s": 0.1, "e": 0.3, "w": "Before"}] + words + [{"s": 15.8, "e": 15.95, "w": "after"}]
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.4, 15.0)]
    return p, f


@needs_ffmpeg
def test_auto_sync_recovers_an_offset_from_the_file(tmp_path):
    audio, spans = _bursts()
    src = _mp4(tmp_path, audio)
    inside = [sp for sp in spans if 0.8 < sp[0] < 14.0]
    p, f = _sync_project(src, _words(inside, -0.35))                 # the words 0.35 s early
    jobs = sync_jobs(p)
    assert len(jobs) == 1 and jobs[0].clip_id == p.clips[0].id
    off = measure_sync(jobs[0])
    assert off == pytest.approx(0.35, abs=0.05)
    assert apply_sync(p, jobs[0], off) == pytest.approx(off)
    for w, (s, _e) in zip(f.transcript[1:-1], inside):
        assert w["s"] == pytest.approx(s, abs=0.05)
    assert f.transcript[0]["s"] == 0.1 and f.transcript[-1]["s"] == 15.8   # outside the clip: untouched
    # in sync now: nothing more to move
    assert measure_sync(sync_jobs(p)[0]) == 0.0


@needs_ffmpeg
def test_auto_sync_leaves_aligned_captions(tmp_path):
    audio, spans = _bursts()
    p, f = _sync_project(_mp4(tmp_path, audio), _words([sp for sp in spans if 0.8 < sp[0] < 14.0]))
    before = [dict(w) for w in f.transcript]
    assert measure_sync(sync_jobs(p)[0]) == 0.0
    assert f.transcript == before


def test_sync_stops_at_the_words_around_and_skips_a_changed_transcript():
    p, f = _sync_project("talk.mp4", [{"s": 1.0, "e": 1.4, "w": "a"}, {"s": 2.0, "e": 2.4, "w": "b"}])
    job = sync_jobs(p)[0]
    assert apply_sync(p, job, -2.0) == pytest.approx(-(1.0 - 0.3))     # up to "Before" (outside the clip)
    words = [w["s"] for w in f.transcript]
    assert words == sorted(words)
    assert apply_sync(p, job, 0.5) == 0.0                              # moved since the job was made


def _sync_window(qapp, monkeypatch, tmp_path, off):
    from kangedit.ui.main_window import MainWindow
    audio, spans = _bursts()
    src = _mp4(tmp_path, audio)
    w = MainWindow()
    p, f = _sync_project(src, _words([sp for sp in spans if 0.8 < sp[0] < 14.0], off))
    w.set_project(p)
    w.refresh_all()
    monkeypatch.setattr(w.tasks, "submit",
                        lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
    return w, p, f


@needs_ffmpeg
def test_auto_sync_in_the_window(qapp, monkeypatch, tmp_path):
    w, p, f = _sync_window(qapp, monkeypatch, tmp_path, -0.35)
    try:
        start = w.captions[0].start
        w.tl.context_action.emit("caption_sync", "0")                 # right-click a caption → Auto-sync
        assert w.captions[0].start == pytest.approx(start + 0.35, abs=0.05)
        assert "Captions moved 0.3" in w.lbl_status.text() and "later" in w.lbl_status.text()
        w.undo()
        assert w.captions[0].start == pytest.approx(start)
        w.caps_panel.b_sync.click()                                   # Captions panel: every clip
        assert w.captions[0].start == pytest.approx(start + 0.35, abs=0.05)
        w.caps_panel.b_sync.click()
        assert "already in sync" in w.lbl_status.text()
    finally:
        w.project.dirty = False
        w.close()


# ------------------------------------------------------------------------------------------- templates
def test_builtin_templates_are_drawable():
    assert {"Clean subtitles", "Bold pop", "Karaoke", "Boxed", "Minimal lower"} <= set(caption_templates.BUILTIN)
    for name, values in caption_templates.BUILTIN.items():
        assert set(values) <= set(caption_templates.TEMPLATE_FIELDS), name
        assert values.get("animation", "none") in ANIMATIONS, name


def test_applying_a_template_changes_only_the_look():
    st = CaptionStyle(enabled=False, speakers="2")
    caption_templates.apply_template(st, caption_templates.BUILTIN["Bold pop"])
    assert st.uppercase and st.animation == "pop" and st.highlight == "#FFD400" and st.size_pct == 7.5
    assert st.enabled is False and st.speakers == "2"                 # not part of a look
    caption_templates.apply_template(st, caption_templates.BUILTIN["Boxed"])
    assert st.box and not st.outline and st.animation == "none" and not st.uppercase


def test_user_templates_round_trip_through_the_settings():
    st = CaptionStyle(font="Verdana", color="#123456", animation="reveal", size_pct=9.0, speakers="3")
    try:
        assert caption_templates.save_user_template("  My look ", st) == "My look"
        saved = caption_templates.user_templates()["My look"]          # read back from the settings file
        assert saved["font"] == "Verdana" and saved["color"] == "#123456" and saved["animation"] == "reveal"
        assert "speakers" not in saved and "enabled" not in saved
        assert "My look" in caption_templates.all_templates()
        other = CaptionStyle()
        caption_templates.apply_template(other, caption_templates.all_templates()["My look"])
        assert other.font == "Verdana" and other.size_pct == 9.0 and other.speakers == "off"
        with pytest.raises(ValueError):
            caption_templates.save_user_template("Karaoke", st)        # a built-in name
        assert caption_templates.delete_user_template("My look")
        assert "My look" not in caption_templates.user_templates()
        assert not caption_templates.delete_user_template("Karaoke")   # built-ins stay
    finally:
        caption_templates.delete_user_template("My look")


def _talk_project():
    f = Footage("talk.mp4", 60.0, 1920, 1080, 30.0, True)
    f.transcript = [{"s": 1.0 + i, "e": 1.5 + i, "w": f"w{i}" + ("." if i % 3 == 2 else "")} for i in range(12)]
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, 20.0)]
    return p, f


def test_templates_in_the_window_and_the_dialog(qapp, monkeypatch):
    from kangedit.ui.dialogs import CaptionStyleDialog
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, _f = _talk_project()
        w.set_project(p)
        w.refresh_all()
        assert w.caps_panel.cb_template.findData("Karaoke") > 0
        w.caps_panel.cb_template.setCurrentIndex(w.caps_panel.cb_template.findData("Karaoke"))
        w.caps_panel._template_activated(0)                             # chosen from the Look list
        st = p.settings.captions
        assert st.animation == "highlight" and "Caption look: Karaoke" in w.lbl_status.text()
        assert w.caps_panel.cb_template.currentIndex() == 0
        w.undo()
        assert p.settings.captions.animation == "none"
        dlg = CaptionStyleDialog(p, w)
        dlg.cb_template.setCurrentIndex(dlg.cb_template.findData("Bold pop"))
        dlg._template_chosen()
        assert dlg.chk_upper.isChecked() and dlg.cb_anim.currentData() == "pop" and not dlg.chk_auto.isChecked()
        from PySide6.QtWidgets import QInputDialog
        monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Loud", True)))
        dlg._save_template()
        assert caption_templates.user_templates()["Loud"]["animation"] == "pop"
        assert dlg.cb_template.currentData() == "Loud" and dlg.b_tpl_delete.isEnabled()
        dlg.apply()
        assert p.settings.captions.uppercase and p.settings.captions.animation == "pop"
        dlg._delete_template()
        assert "Loud" not in caption_templates.user_templates()
        dlg.close()
    finally:
        caption_templates.delete_user_template("Loud")
        w.project.dirty = False
        w.close()


# ------------------------------------------------------------------------------------------- speakers
def _voice(f0, seconds, bright, rng):
    """A harmonic "voice" at pitch f0 (with vibrato and syllable gaps), `bright`: how strong its overtones."""
    t = np.arange(int(seconds * RATE)) / RATE
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))) / RATE
    x = sum((bright ** h) * np.sin(h * ph) for h in range(1, 15))
    return (0.07 * x * (np.sin(2 * np.pi * 3 * t) > -0.3) + rng.normal(0, 0.002, t.size)).astype(np.float32)


def _podcast(tmp_path, n=8):
    """Two voices (low and high) taking turns, a sentence each: the file and its project."""
    rng = np.random.default_rng(5)
    audio = np.zeros(int((n * 2.5 + 1) * RATE), np.float32)
    words = []
    for k in range(n):
        s = 0.5 + k * 2.5
        v = _voice(110, 1.6, 0.6, rng) if k % 2 == 0 else _voice(235, 1.6, 0.8, rng)
        audio[int(s * RATE):int(s * RATE) + v.size] += v
        words += [{"s": round(s + 0.55 * i, 3), "e": round(s + 0.55 * i + 0.5, 3),
                   "w": f"k{k}w{i}" + ("." if i == 2 else "")} for i in range(3)]
    path = _write_wav(tmp_path / "podcast.wav", audio)
    f = Footage(path, audio.size / RATE, 1920, 1080, 30.0, True)
    f.transcript = words
    p = Project()
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 0.0, audio.size / RATE)]
    return p, f


def test_voice_features_tell_pitch():
    rng = np.random.default_rng(0)
    lo = speakers.voice_features(_voice(110, 1.5, 0.6, rng))
    hi = speakers.voice_features(_voice(235, 1.5, 0.8, rng))
    assert np.exp(lo[0]) == pytest.approx(110, rel=0.08) and np.exp(hi[0]) == pytest.approx(235, rel=0.08)
    assert speakers.voice_features(np.zeros(100, np.float32)) is None
    same = np.array([speakers.voice_features(_voice(150, 1.5, 0.7, rng)) for _ in range(6)])
    assert set(speakers.cluster(same, "auto")) == {0}                   # one voice: one speaker


@needs_ffmpeg
def test_two_voices_are_told_apart(tmp_path):
    p, f = _podcast(tmp_path)
    units = speakers.speaker_units(p)
    assert len(units) == 8
    for mode in ("auto", "2"):
        found = speakers.detect(units, mode)
        assert found == [0, 1] * 4, mode
    assert speakers.apply_speakers(p, units, found) == 2
    assert all("p" in w for w in f.transcript)
    p.settings.captions.speakers = "auto"
    caps = build_captions(p)
    assert [c.speaker for c in caps] == [0, 1] * 4
    cols = [caption_color(p.settings.captions, c) for c in caps]
    assert cols[:2] == [SPEAKER_COLORS[0], SPEAKER_COLORS[1]]
    # the speakers stay through a trim (they are on the words) and a save
    p.clips[0].src_in = 3.0
    assert [c.speaker for c in build_captions(p)][:2] == [1, 0]
    d = f.to_dict() if hasattr(f, "to_dict") else None
    if d is not None:
        assert Footage.from_dict(d).transcript[0]["p"] == 0


def test_speaker_colours_in_the_export(tmp_path):
    p, f = _talk_project()
    st = p.settings.captions
    plain = str(tmp_path / "plain.ass")
    write_ass(plain, build_captions(p), st, 1920, 1080)
    st.speakers = "auto"                                                # on, but no speakers known: as before
    same = str(tmp_path / "same.ass")
    write_ass(same, build_captions(p), st, 1920, 1080)
    assert open(plain, encoding="utf-8").read() == open(same, encoding="utf-8").read()
    caps = build_captions(p)
    assert all(c.speaker == -1 for c in caps) and caption_color(st, caps[0]) == st.color
    set_caption_speaker(p, [caps[1]], 1)
    caps = build_captions(p)
    assert caps[1].speaker == 1 and caps[0].speaker == -1
    out = str(tmp_path / "spk.ass")
    write_ass(out, caps, st, 1920, 1080)
    cyan = captions._ass_color(SPEAKER_COLORS[1])
    lines = [ln for ln in open(out, encoding="utf-8").read().splitlines() if ln.startswith("Dialogue")]
    assert cyan not in lines[0] and f"\\c{cyan}" in lines[1]
    st.animation = "highlight"                                          # the word highlight goes back to it
    write_ass(out, build_captions(p), st, 1920, 1080)
    hl = f"{{\\c{captions._ass_color(st.highlight)}}}"
    lines = [ln for ln in open(out, encoding="utf-8").read().splitlines() if ln.startswith("Dialogue")]
    second = [ln for ln in lines if "w3" in ln and hl in ln]
    assert second and all(ln.count(f"{{\\c{cyan}}}") == 2 for ln in second)   # its colour, and after the word
    st.speakers = "off"                                                 # off: one colour again
    write_ass(out, build_captions(p), st, 1920, 1080)
    assert cyan not in open(out, encoding="utf-8").read()


def test_edit_keeps_the_speaker():
    p, f = _talk_project()
    caps = build_captions(p)
    set_caption_speaker(p, [caps[0]], 2)
    edit_caption(p, build_captions(p)[0], "one two three four")
    assert build_captions(p)[0].speaker == 2
    set_caption_speaker(p, [build_captions(p)[0]], -1)
    assert all("p" not in w for w in f.transcript)


def test_speaker_colour_in_the_preview(qapp):
    from PySide6.QtGui import QImage, QColor
    from kangedit.ui.viewer import Canvas
    cv = Canvas()
    cv.resize(640, 360)
    img = QImage(640, 360, QImage.Format_RGB32)
    img.fill(QColor("#000000"))
    cv.image = img
    st = CaptionStyle(outline=False, auto_layout=True, speakers="2")
    cv.caption_style = st
    cv.layout = captions.caption_layout(st, 1920, 1080, "long")
    cv.caption = Caption(0.0, 2.0, "HELLO THERE", "f", [0, 1], "c", [(0.0, 1.0, "HELLO"), (1.0, 2.0, "THERE")],
                         speaker=1)
    cv.caption_t = 1.5

    def cyan_pixels():
        shot = cv.grab().toImage()
        n = 0
        for y in range(0, shot.height(), 2):
            for x in range(0, shot.width(), 2):
                c = shot.pixelColor(x, y)
                n += c.blue() > 200 and c.green() > 180 and c.red() < 140
        return n

    assert cyan_pixels() > 20
    st.speakers = "off"
    assert cyan_pixels() == 0


def test_speakers_in_the_window(qapp, monkeypatch, tmp_path):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, f = _talk_project()
        w.set_project(p)
        w.refresh_all()
        assert p.settings.captions.speakers == "off" and all(c.speaker == -1 for c in w.captions)
        w.tl.context_action.emit("caption_speaker", "1:2")             # right-click → Speaker → 3
        assert w.captions[1].speaker == 2 and p.settings.captions.speakers == "auto"
        assert w.caps_panel.cb_speakers.currentData() == "auto"
        w.tl.resize(1200, 300)
        w.tl.grab()                                                     # the lane shows who speaks
        w.undo()
        assert w.captions[1].speaker == -1 and p.settings.captions.speakers == "off"
        assert all("p" not in x for x in f.transcript)
        w.caps_panel.cb_speakers.setCurrentIndex(w.caps_panel.cb_speakers.findData("2"))
        assert p.settings.captions.speakers == "2"
        # Detect speakers (the voices measured by a stand-in): written onto the words, undoable
        monkeypatch.setattr(w.tasks, "submit",
                            lambda name, job, done, err=None: done(job(lambda *a: None, threading.Event())))
        monkeypatch.setattr(speakers, "detect", lambda units, mode, progress=None, cancel=None:
                            [i % 2 for i in range(len(units))])
        w.caps_panel.b_detect.click()
        assert [c.speaker for c in w.captions] == [i % 2 for i in range(len(w.captions))] and "2 speaker(s) found" in w.lbl_status.text()
        w.undo()
        assert all(c.speaker == -1 for c in w.captions)
    finally:
        w.project.dirty = False
        w.close()
