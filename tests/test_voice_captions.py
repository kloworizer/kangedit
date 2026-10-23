"""Captions from a voice-over on the sound layer (SoundClip.caption), next to the clips' own speech."""

from kangedit import captions
from kangedit.captions import build_captions, delete_captions, edit_caption
from kangedit.models import Footage, Project, SoundClip, TimelineClip


def _project(tmp_path):
    p = Project()
    f = Footage(str(tmp_path / "game.mp4"), 60.0, 1920, 1080, 30.0, True)
    f.transcript = [{"s": 1.0, "e": 1.4, "w": "nice"}, {"s": 1.5, "e": 2.0, "w": "shot!"},       # at 0-1 s
                    {"s": 9.0, "e": 9.5, "w": "again"}]                                       # at 8-8.5 s
    p.footage = {f.id: f}
    p.clips = [TimelineClip(f.id, 1.0, 11.0)]                                                 # 10 s
    vo = SoundClip(str(tmp_path / "voice.wav"), start=4.0, src_in=2.0, src_out=6.0, caption=True)
    p.sounds = [vo, SoundClip(str(tmp_path / "music.mp3"), start=0.0, src_in=0.0, src_out=10.0)]
    v = p.voice_for(vo.path)
    v.transcript = [{"s": 0.5, "e": 1.0, "w": "cut"},                                         # before src_in
                    {"s": 2.5, "e": 3.0, "w": "Halo"}, {"s": 3.1, "e": 3.6, "w": "semua."},     # at 4.5-5.6 s
                    {"s": 5.0, "e": 5.4, "w": "Gas!"}]                                        # at 7-7.4 s
    return p, f, vo, v


def test_voice_over_is_captioned_where_it_plays(tmp_path):
    p, f, vo, v = _project(tmp_path)
    caps = build_captions(p)
    texts = [c.text for c in caps]
    assert "Halo semua." in texts and "Gas!" in texts and "cut" not in " ".join(texts)
    halo = caps[texts.index("Halo semua.")]
    assert abs(halo.start - 4.5) < 1e-6 and halo.footage_id == v.id and halo.clip_id == vo.id
    assert texts[0] == "nice shot!" and texts[-1] == "again"                # the clip's own speech elsewhere
    assert [c.start for c in caps] == sorted(c.start for c in caps)
    vo.caption = False                                                       # not a voice-over: not captioned
    assert [c.text for c in build_captions(p)] == ["nice shot!", "again"]


def test_voice_over_wins_where_both_speak(tmp_path):
    p, f, vo, v = _project(tmp_path)
    f.transcript.insert(2, {"s": 5.6, "e": 6.2, "w": "talking"})              # at 4.6 s, under the voice-over
    texts = [c.text for c in build_captions(p)]
    assert "talking" not in " ".join(texts) and "Halo semua." in texts


def test_voice_captions_are_edited_saved_and_undone(tmp_path):
    p, f, vo, v = _project(tmp_path)
    cap = next(c for c in build_captions(p) if c.text == "Gas!")
    p.checkpoint()
    edit_caption(p, cap, "Gaskeun!")
    assert v.transcript[-1]["w"] == "Gaskeun!" and "Gaskeun!" in [c.text for c in build_captions(p)]
    p.undo()
    assert v.transcript[-1]["w"] == "Gas!"
    assert delete_captions(p, [cap]) == 1 and "Gas!" not in [c.text for c in build_captions(p)]
    path = str(tmp_path / "p.kgproj")
    p.save(path)
    q = Project.load(path)
    assert q.sounds[0].caption and q.voice_for(vo.path).transcript == v.transcript
    assert "Halo semua." in [c.text for c in build_captions(q)]


def test_voice_file_is_transcribed_like_footage(tmp_path, monkeypatch):
    p, f, vo, v = _project(tmp_path)
    (tmp_path / "voice.wav").write_bytes(b"x")
    v.transcript = None
    monkeypatch.setattr(captions, "_load_audio_16k", lambda *a, **k: __import__("numpy").ones(16000 * 4, "float32"))
    monkeypatch.setattr(captions, "needs_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(captions, "speech_seconds", lambda a: 3.0)
    monkeypatch.setattr(captions, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(captions, "save_transcript", lambda *a, **k: None)
    monkeypatch.setattr(captions, "refine_word_times", lambda w, a: w)
    monkeypatch.setattr(captions, "transcribe_audio",
                        lambda *a, **k: ([{"s": 2.5, "e": 3.0, "w": "Halo"}], "id"))
    assert captions.transcribe(v, "small", "", use_cache=False)[0]["w"] == "Halo" and v.transcript_lang == "id"


def test_sound_menu_captions_a_voice_over(qapp, tmp_path, monkeypatch):
    from kangedit.ui.main_window import MainWindow
    w = MainWindow()
    try:
        p, f, vo, v = _project(tmp_path)
        vo.caption = False
        v.transcript = None
        w.set_project(p)
        queued = []
        monkeypatch.setattr(w, "queue_transcription", lambda ids, force=False, use_cache=None: queued.append(ids))
        w.clip_action("sound_caption:1", vo.id)
        assert vo.caption and queued == [[v.id]]
        v.transcript = [{"s": 2.5, "e": 3.0, "w": "Halo"}]
        w.clip_action("sound_caption:1", vo.id)                                # transcribed: shown at once
        assert len(queued) == 1 and "Halo" in [c.text for c in w.captions]
        w.undo()
        w.undo()                                                               # (each toggle: one step)
        assert not w.project.sounds[0].caption
    finally:
        w.project.dirty = False
        w.close()
