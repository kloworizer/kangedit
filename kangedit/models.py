"""Project data model: footage, timeline clips, settings, undo history and persistence."""

from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .color import Grade
from .speed import timemap

LOOP_CROSSFADE = 2.0   # seconds of crossfade between the repeats of a looping sound


def new_id() -> str:
    return uuid.uuid4().hex[:10]


@dataclass
class Footage:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    color_space: str = ""
    color_range: str = ""
    color_transfer: str = ""
    rotation: int = 0
    creation_time: str = ""
    gps: Optional[list] = None     # where it was recorded: [latitude, longitude]; [] = no location, None = not read
    face: Optional[list] = None    # main face (centre x, y, height as parts of the picture), for framing crops
    log_profile: str = "none"      # built-in camera profile key, "lut:<path>" or "none"
    tonemap: str = "filmic"
    camera: str = ""               # identified camera make/model
    log_note: str = ""             # how log_profile was chosen (shown in tooltips)
    log_pending: bool = False      # waiting for the picture check after analysis (see camera.py)
    log_guess: str = ""            # the camera's usual log profile, applied if the picture looks log
    proxy_path: Optional[str] = None
    analysis_path: Optional[str] = None
    # Word-level transcript in source time: [{"s": start, "e": end, "w": text}, ...]
    transcript: Optional[list] = None
    transcript_lang: str = ""
    transcript_model: str = ""     # what made it: "speech model|language" (see captions.transcript_key)
    kind: str = "video"            # "video" | "image" (a still, or a looping animated GIF) | "audio" (a voice-over)
    bin: str = ""                  # the Media panel bin it is in ("" = none; see bins.py)
    projection: str = ""           # 360° footage: "equirect" | "dfisheye" ("" = a normal picture; see spherical.py)
    lens_fov: float = 195.0        # dual fisheye: each lens's field of view, degrees
    id: str = field(default_factory=new_id)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    def has_proxy(self) -> bool:
        return bool(self.proxy_path) and os.path.exists(self.proxy_path)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Footage":
        known = {k: v for k, v in d.items() if k in Footage.__dataclass_fields__}
        return Footage(**known)


@dataclass
class TimelineClip:
    footage_id: str
    src_in: float
    src_out: float
    grade: Grade = field(default_factory=Grade)
    volume: float = 1.0
    zoom: float = 1.0
    offset_x: float = 0.0   # -1..1 reframe pan
    offset_y: float = 0.0
    score: Optional[float] = None
    # Edit from the previous clip into this one (see edits.py). transition: "" (cut) or an edits.TRANSITIONS
    # key, centered on the cut. audio_split: where the audio cut sits relative to the picture cut
    # (< 0: J-cut, this clip's sound starts early; > 0: L-cut, the previous clip's sound runs on).
    transition: str = ""
    transition_dur: float = 0.5
    audio_split: float = 0.0
    label: str = ""                # "opening" | "closing" bookend, or "hook", when auto edit placed it
    rotation: float = 0.0          # degrees clockwise (see reframe.split_rotation)
    # Box (multicam layouts): box_w > 0 shows the clip in a box_w x box_h part of the frame (centre box_x,
    # box_y, all parts of the frame), cropped to fill it (zoom / offsets choose the crop), black around it.
    box_x: float = 0.5
    box_y: float = 0.5
    box_w: float = 0.0
    box_h: float = 0.0
    speed: float = 1.0             # 0.5 = half speed (slow motion), 2 = double (see speed.py)
    ramp: str = ""                 # speed.RAMPS key: speed ramp template along the clip ("" = constant)
    audio_mode: str = "sync"       # speed.SOUND_MODES key: the sound at another speed (synced, pitch, normal, mute)
    view_yaw: float = 0.0          # 360° footage: where the view looks (degrees: right, up, tilt) and how wide
    view_pitch: float = 0.0
    view_roll: float = 0.0
    view_fov: float = 100.0
    view_tiny: bool = False        # the whole sphere as a "tiny planet"
    # Movement inside the framed picture (see motion.py): a dynamic zoom (motion.MOTIONS key, how much it
    # zooms, motion.EASES key) and stabilization strength (0 = off, up to 1).
    motion: str = ""
    motion_amount: float = 0.12
    motion_ease: str = "both"
    motion_face: Optional[list] = None   # where "zoom on the face" zooms: [x, y] in parts of the clip's picture
    stabilize: float = 0.0
    voice_clean: float = 0.0       # take noise and wind out of the speech: 0 = off, up to 1 (see voice.py)
    voice_floor: float = 0.0       # its background noise level, dBFS (0 = not known)
    id: str = field(default_factory=new_id)

    @property
    def src_dur(self) -> float:
        """Length of the source range used."""
        return max(0.0, self.src_out - self.src_in)

    @property
    def duration(self) -> float:
        """Length on the timeline (the source range at the clip's speed)."""
        return self.timemap().duration

    def timemap(self):
        return timemap(self.src_dur, self.speed, self.ramp)

    def src_at(self, t: float) -> float:
        """Source time shown `t` seconds into the clip on the timeline."""
        return self.src_in + self.timemap().src(t)

    def time_at(self, src: float) -> float:
        """Seconds into the clip (timeline) where source time `src` shows."""
        return self.timemap().time(src - self.src_in)

    def sound_src_at(self, t: float) -> float:
        """Source time heard `t` seconds into the clip: the picture's, unless the sound plays at normal speed."""
        return self.src_in + t if self.audio_mode == "normal" else self.src_at(t)

    def sound_time_at(self, src: float) -> float:
        """Seconds into the clip where the sound of source time `src` is heard."""
        return src - self.src_in if self.audio_mode == "normal" else self.time_at(src)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grade"] = self.grade.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "TimelineClip":
        d = dict(d)
        g = Grade.from_dict(d.pop("grade", None))
        known = {k: v for k, v in d.items() if k in TimelineClip.__dataclass_fields__}
        return TimelineClip(grade=g, **known)


@dataclass
class OverlayClip:
    """A video or image on a layer above the main track (picture in picture, logo, B-roll): placed at a
    timeline time like a sound, scaled, rotated and positioned anywhere in the frame."""
    footage_id: str
    start: float                   # timeline position, seconds
    src_in: float = 0.0
    src_out: float = 5.0
    track: int = 0                 # layer: 0 sits right above the main track, higher layers on top of it
    x: float = 0.5                 # centre, 0..1 of the frame
    y: float = 0.5
    scale: float = 1.0             # 1 = fits the whole frame (letterboxed), 0.5 = half of that
    rotation: float = 0.0          # degrees clockwise
    opacity: float = 1.0           # 0..1
    fade_in: float = 0.0           # seconds of fade from / to transparent
    fade_out: float = 0.0
    volume: float = 1.0            # the overlay video's own sound
    grade: Grade = field(default_factory=Grade)
    # Box: > 0 crops the picture to fill a box this part of the frame's width / height (times `scale`) instead
    # of showing it whole; pan_x / pan_y (-1..1) choose which part of the picture fills it.
    box_w: float = 0.0
    box_h: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    angle: int = 0                 # multicam: camera angle shown (1 = the main angle, 2...), 0 = not multicam
    link: str = ""                 # multicam: id of the main-track clip it follows
    linked: bool = True            # False: unlinked by the user, edited on its own (no longer kept in sync)
    speed: float = 1.0             # see TimelineClip.speed
    ramp: str = ""
    audio_mode: str = "sync"
    mix_in: float = 0.0            # multicam: fades for the main track's transitions (set by multicam, kept apart
    mix_out: float = 0.0           # from the user's fade_in / fade_out)
    piece: int = 0                 # multicam switching: which shot of its main clip
    # Chroma key (keying.py): "#rrggbb" of a green / blue screen made see-through, "" = off
    key_color: str = ""
    key_similarity: float = 0.12
    key_blend: float = 0.08
    key_despill: bool = True
    view_yaw: float = 0.0          # 360° footage: where the view looks (degrees: right, up, tilt) and how wide
    view_pitch: float = 0.0
    view_roll: float = 0.0
    view_fov: float = 100.0
    view_tiny: bool = False        # the whole sphere as a "tiny planet"
    motion: str = ""               # dynamic zoom and stabilization inside the layer's picture (see TimelineClip)
    motion_amount: float = 0.12
    motion_ease: str = "both"
    motion_face: Optional[list] = None
    stabilize: float = 0.0
    voice_clean: float = 0.0
    voice_floor: float = 0.0
    id: str = field(default_factory=new_id)

    @property
    def src_dur(self) -> float:
        return max(0.0, self.src_out - self.src_in)

    @property
    def duration(self) -> float:
        return self.timemap().duration

    def timemap(self):
        return timemap(self.src_dur, self.speed, self.ramp)

    def src_at(self, t: float) -> float:
        return self.src_in + self.timemap().src(t)

    def time_at(self, src: float) -> float:
        return self.timemap().time(src - self.src_in)

    def sound_src_at(self, t: float) -> float:
        return self.src_in + t if self.audio_mode == "normal" else self.src_at(t)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def fades(self) -> tuple[float, float]:
        d = self.duration
        fi, fo = max(0.0, self.fade_in, self.mix_in), max(0.0, self.fade_out, self.mix_out)
        if fi + fo > d:
            k = d / (fi + fo)
            fi, fo = fi * k, fo * k
        return fi, fo

    def alpha_at(self, t: float) -> float:
        """Opacity at timeline time t (fades included); 0 outside the clip."""
        if not (self.start <= t < self.end):
            return 0.0
        fi, fo = self.fades()
        a = max(0.0, min(1.0, self.opacity))
        if fi > 0:
            a *= min(1.0, (t - self.start) / fi)
        if fo > 0:
            a *= min(1.0, (self.end - t) / fo)
        return max(0.0, a)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grade"] = self.grade.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "OverlayClip":
        d = dict(d)
        g = Grade.from_dict(d.pop("grade", None))
        return OverlayClip(grade=g, **{k: v for k, v in d.items() if k in OverlayClip.__dataclass_fields__})


@dataclass
class SoundClip:
    """A sound effect on the SFX layer: placed at a timeline time, independent of the video clips."""
    path: str
    start: float                   # timeline position, seconds
    src_in: float = 0.0
    src_out: float = 0.0
    volume: float = 1.0
    name: str = ""
    fade_in: float = 0.0           # seconds
    fade_out: float = 0.0
    duck: bool = False             # music: turned down automatically while the clips have speech / sound
    # Looping (fit to timeline): > 0 repeats the source from src_in for this many seconds, again and again with
    # a LOOP_CROSSFADE between repeats, for as long as the clip lasts (src_out may then run past the file's end).
    loop: float = 0.0
    caption: bool = False          # a voice-over: its speech goes into the captions (see captions.build_captions)
    id: str = field(default_factory=new_id)

    @property
    def duration(self) -> float:
        return max(0.0, self.src_out - self.src_in)

    def fades(self) -> tuple[float, float]:
        """Fade in / out lengths, shortened together when they don't both fit in the sound."""
        d, fi, fo = self.duration, max(0.0, self.fade_in), max(0.0, self.fade_out)
        if fi + fo > d:
            k = d / (fi + fo)
            fi, fo = fi * k, fo * k
        return fi, fo

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def looping(self) -> bool:
        return self.loop > LOOP_CROSSFADE * 2 and self.duration > self.loop + 1e-3

    def loop_step(self) -> float:
        """Timeline seconds between the starts of two repeats of a looping sound."""
        return self.loop - LOOP_CROSSFADE

    def repeats(self) -> int:
        """How many times the source plays (1 unless looping)."""
        if not self.looping:
            return 1
        return int(np.ceil((self.duration - self.loop) / self.loop_step() - 1e-9)) + 1

    def source_time(self, t: np.ndarray) -> np.ndarray:
        """Source time heard at clip time t (seconds from the clip's start; for drawing its waveform)."""
        if not self.looping:
            return self.src_in + t
        step = self.loop_step()
        return self.src_in + np.mod(t, step)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SoundClip":
        return SoundClip(**{k: v for k, v in d.items() if k in SoundClip.__dataclass_fields__})


@dataclass
class TextClip:
    """An animated title / text overlay on the text track (see titles.py)."""
    text: str
    start: float                   # timeline position, seconds
    duration: float = 3.0
    preset: str = ""               # titles.TEXT_PRESETS key it was made from
    font: str = "Arial"
    size_pct: float = 8.0          # font height, % of frame height
    color: str = "#FFFFFF"
    bold: bool = True
    italic: bool = False
    outline: bool = True           # dark outline around the letters (ignored with a box)
    box: bool = False              # background box behind each line
    box_color: str = "#000000"
    box_opacity: float = 60.0      # %
    align: str = "center"          # left | center | right (also which side of the block sits at x)
    x: float = 0.5                 # anchor position, 0..1 of the frame (vertical: block centre)
    y: float = 0.5
    anim_in: str = "fade"          # titles.ANIM_IN key
    anim_out: str = "fade"         # titles.ANIM_OUT key
    anim_dur: float = 0.5          # seconds of each animation
    id: str = field(default_factory=new_id)

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TextClip":
        return TextClip(**{k: v for k, v in d.items() if k in TextClip.__dataclass_fields__})


@dataclass
class CaptionStyle:
    enabled: bool = True
    font: str = "Arial"
    size_pct: float = 5.5          # font height as % of frame height
    position: str = "bottom"       # bottom | middle | top
    uppercase: bool = False
    max_chars: int = 32            # characters per line
    max_lines: int = 2
    max_duration: float = 4.0      # seconds per caption
    color: str = "#FFFFFF"
    outline: bool = True
    box: bool = False
    animation: str = "none"        # see captions.ANIMATIONS
    highlight: str = "#FFD400"     # active-word color for the animations
    auto_layout: bool = True       # size / position / line length chosen from the frame ratio and format


@dataclass
class ProjectSettings:
    aspect: str = "16:9 (Landscape)"
    resolution: str = "1080p"
    fps: float = 30.0
    fit_mode: str = "fill"         # fill | fit | blur
    use_proxies: bool = True
    proxy_height: int = 540
    whisper_model: str = "auto"         # captions.resolve_model: large-v3 on an NVIDIA GPU, else turbo
    whisper_language: str = ""     # "" = auto detect
    speech_cleanup: str = "auto"   # captions.CLEANUPS key: clean road / wind noise out before transcribing
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    video_format: str = "short"    # short | long | custom (auto-edit preset)
    # Fade from / to black and silence at the start / end of the timeline (seconds, 0 = off).
    fade_in: float = 0.0
    fade_out: float = 0.0
    fade_picture: bool = True
    fade_sound: bool = True
    # Auto ducking: sounds marked `duck` (music) drop by this many dB under the clips' own sound.
    duck_db: float = 12.0
    overlay_tracks: int = 1        # video layers shown above the main track
    # Multicam (see multicam.py): {"angles": {footage id: angle number}, "template": key, "audio": choice}
    multicam: dict = field(default_factory=dict)
    bins: list = field(default_factory=list)   # Media panel bins made by hand (kept even when empty)
    script: dict = field(default_factory=dict)  # the last auto edit from a script (see ScriptEditDialog.settings)

    def to_dict(self) -> dict:
        return asdict(self)

    def fades(self, duration: float, sound: bool = False) -> tuple[float, float]:
        """(fade in, fade out) lengths for the picture (or the sound), shortened to fit the timeline."""
        if not (self.fade_sound if sound else self.fade_picture) or duration <= 0:
            return 0.0, 0.0
        fi, fo = max(0.0, self.fade_in), max(0.0, self.fade_out)
        if fi + fo > duration:
            k = duration / (fi + fo)
            fi, fo = fi * k, fo * k
        return fi, fo

    @staticmethod
    def from_dict(d: dict) -> "ProjectSettings":
        d = dict(d or {})
        cap = CaptionStyle(**{k: v for k, v in (d.pop("captions", None) or {}).items()
                              if k in CaptionStyle.__dataclass_fields__})
        known = {k: v for k, v in d.items() if k in ProjectSettings.__dataclass_fields__}
        return ProjectSettings(captions=cap, **known)


class Project:
    FILE_EXT = ".kgproj"

    def __init__(self):
        self.footage: dict[str, Footage] = {}
        self.clips: list[TimelineClip] = []
        self.sounds: list[SoundClip] = []
        self.texts: list[TextClip] = []
        self.overlays: list[OverlayClip] = []
        # Sound files on the sound layer captioned as voice-over: {id: Footage holding the file's transcript}
        # (not media: not in the Media panel; see voice_for).
        self.voices: dict[str, Footage] = {}
        self.settings = ProjectSettings()
        self.path: Optional[str] = None
        self.dirty = False
        self._undo: list[dict] = []
        self._redo: list[dict] = []

    # ------------------------------------------------------------------ timeline helpers
    @property
    def duration(self) -> float:
        return sum(c.duration for c in self.clips)

    def usage(self) -> dict[str, int]:
        """{footage id: how many times the timeline uses it} (main track and overlays)."""
        out: dict[str, int] = {}
        for c in list(self.clips) + list(self.overlays):
            out[c.footage_id] = out.get(c.footage_id, 0) + 1
        return out

    def voice_for(self, path: str) -> Footage:
        """The transcript holder of a sound file captioned as voice-over (made the first time)."""
        for f in self.voices.values():
            if f.path == path:
                return f
        f = Footage(path, 0.0, 0, 0, 0.0, True, kind="audio")
        self.voices[f.id] = f
        return f

    def transcript_source(self, fid: str) -> Optional[Footage]:
        """The footage, or the voice-over sound file, whose transcript has this id."""
        return self.footage.get(fid) or self.voices.get(fid)

    def voice_sounds(self) -> list["SoundClip"]:
        """The sound layer's clips captioned as voice-over."""
        return [x for x in self.sounds if x.caption and x.duration > 0.01]

    def clip_starts(self) -> list[float]:
        starts, t = [], 0.0
        for c in self.clips:
            starts.append(t)
            t += c.duration
        return starts

    def clip_at(self, t: float) -> tuple[Optional[int], float]:
        """Index of the clip under timeline time t and the offset into it."""
        acc = 0.0
        for i, c in enumerate(self.clips):
            if t < acc + c.duration or i == len(self.clips) - 1:
                return i, min(max(0.0, t - acc), c.duration)
            acc += c.duration
        return None, 0.0

    def index_of(self, clip_id: str) -> int:
        for i, c in enumerate(self.clips):
            if c.id == clip_id:
                return i
        return -1

    def conflicts(self, min_gap: float = 0.0) -> set[str]:
        """Clip ids that overlap (or sit closer than min_gap to) another clip of the same footage. Pieces of one
        take cut straight together (one right after the other on the timeline, going on in the source, such as
        what's left after fillers are taken out) are not counted."""
        bad: set[str] = set()
        by_src: dict[str, list[TimelineClip]] = {}
        for c in self.clips:
            by_src.setdefault(c.footage_id, []).append(c)
        pos = {id(c): k for k, c in enumerate(self.clips)}
        for items in by_src.values():
            items = sorted(items, key=lambda c: c.src_in)
            for a, b in zip(items, items[1:]):
                if abs(b.src_in - a.src_out) < 0.02:
                    continue  # contiguous pieces (e.g. after a split) are not a repeat
                if pos[id(b)] == pos[id(a)] + 1 and b.src_in >= a.src_out - 0.02:
                    continue  # a jump cut (fillers / a pause taken out): the take goes on, nothing is repeated
                if b.src_in < a.src_out + min_gap - 1e-3:
                    bad.add(a.id)
                    bad.add(b.id)
        return bad

    # ------------------------------------------------------------------ undo
    def _snapshot(self) -> dict:
        return {
            "clips": [c.to_dict() for c in self.clips],
            "sounds": [x.to_dict() for x in self.sounds],
            "texts": [x.to_dict() for x in self.texts],
            "overlays": [x.to_dict() for x in self.overlays],
            "overlay_tracks": self.settings.overlay_tracks,
            "multicam": copy.deepcopy(self.settings.multicam),   # its layout follows the angles' clips
            "bins": list(self.settings.bins),
            "footage": {fid: {"log_profile": f.log_profile, "tonemap": f.tonemap, "bin": f.bin,
                              "transcript": copy.deepcopy(f.transcript)}
                        for fid, f in self.footage.items()},
            "voices": {fid: copy.deepcopy(f.transcript) for fid, f in self.voices.items()},
        }

    def _restore(self, snap: dict) -> None:
        self.clips = [TimelineClip.from_dict(d) for d in snap["clips"]]
        self.sounds = [SoundClip.from_dict(d) for d in snap.get("sounds", [])]
        self.texts = [TextClip.from_dict(d) for d in snap.get("texts", [])]
        self.overlays = [OverlayClip.from_dict(d) for d in snap.get("overlays", []) if d.get("footage_id") in self.footage]
        self.settings.overlay_tracks = snap.get("overlay_tracks", self.settings.overlay_tracks)
        if "multicam" in snap:
            self.settings.multicam = copy.deepcopy(snap["multicam"])
        if "bins" in snap:
            self.settings.bins = list(snap["bins"])
        for fid, d in snap["footage"].items():
            f = self.footage.get(fid)
            if f:
                f.log_profile = d["log_profile"]
                f.tonemap = d["tonemap"]
                f.bin = d.get("bin", f.bin)
                f.transcript = d["transcript"]
        for fid, words in snap.get("voices", {}).items():
            if fid in self.voices:
                self.voices[fid].transcript = words

    def set_detected_profile(self, footage_id: str, profile: str, note: str) -> None:
        """Apply an automatically detected log profile.

        Detection finishes in the background after the user may already have edited, so the result is
        written into the undo/redo snapshots too -- otherwise undoing an unrelated edit would revert it.
        """
        f = self.footage.get(footage_id)
        if f is None:
            return
        old = f.log_profile
        f.log_profile, f.log_note, f.log_pending = profile, note, False
        for snap in self._undo + self._redo:
            d = snap["footage"].get(footage_id)
            if d is not None and d["log_profile"] == old:
                d["log_profile"] = profile
        self.dirty = True

    def checkpoint(self) -> None:
        """Call before a modification to make it undoable."""
        self._undo.append(self._snapshot())
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._redo.clear()
        self.dirty = True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self.dirty = True
        return True

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "app": "KangEdit",
            "version": 1,
            "settings": self.settings.to_dict(),
            "footage": [f.to_dict() for f in self.footage.values()],
            "clips": [c.to_dict() for c in self.clips],
            "sounds": [x.to_dict() for x in self.sounds],
            "texts": [x.to_dict() for x in self.texts],
            "overlays": [x.to_dict() for x in self.overlays],
            "voices": [f.to_dict() for f in self.voices.values()],
        }

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        os.replace(tmp, path)
        self.path = path
        self.dirty = False

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        p = Project()
        p.settings = ProjectSettings.from_dict(data.get("settings"))
        for fd in data.get("footage", []):
            f = Footage.from_dict(fd)
            p.footage[f.id] = f
        p.clips = [TimelineClip.from_dict(c) for c in data.get("clips", []) if c.get("footage_id") in p.footage]
        p.sounds = [SoundClip.from_dict(x) for x in data.get("sounds", [])]
        p.texts = [TextClip.from_dict(x) for x in data.get("texts", [])]
        p.overlays = [OverlayClip.from_dict(x) for x in data.get("overlays", []) if x.get("footage_id") in p.footage]
        for fd in data.get("voices", []):
            f = Footage.from_dict(fd)
            p.voices[f.id] = f
        p.path = path
        return p
