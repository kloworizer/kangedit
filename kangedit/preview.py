"""Background frame rendering for the viewer.

Decoding (a long-GOP seek can take a second), grading and reframing all run off the UI thread so
the UI never blocks. Still requests (scrubbing, slipping, edits) are latest-wins. Playback is a
pipeline: one thread only decodes, in timeline order, and hands each decoded frame to a small pool
that converts, grades and reframes it. The UI drains the resulting buffer in sync with the playback
clock. 4K H.265 decodes at roughly the frame rate on its own, so keeping conversion and grading off
the decode thread is what lets playback keep up.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from . import keying, reframe, spherical
from .motion import Motion
from .color import Grade, LutCache, apply_grade, apply_spatial, has_spatial, needs_color, profile_short
from .decoder import DecoderPool
from .edits import blend
from .models import Footage, OverlayClip
from .speed import timemap

READ_AHEAD = 45          # frames buffered (ready or being rendered) during playback
START_FRAMES = 6         # playback's clock starts once this many frames are ready (or the buffer stalls)
STILL_PATIENCE = 0.12    # s a still keeps decoding toward its exact frame once a newer one is waiting
# Conversion and grading release the GIL, so these threads render in parallel with the decoder. Two keep
# up with 30 fps (grading itself is split over more threads, see color.PreviewLut); more only compete with
# the decoder's own frame threads.
RENDER_THREADS = 2
# Decoded frames waiting for the pool hold full-size source pictures (24 MB for 4K 10-bit): cap them.
MAX_INFLIGHT = RENDER_THREADS * 2
# Pre-roll competes with playback for the same cores: it starts only with this many frames buffered and
# gives way (resuming later where it stopped) once the buffer falls below PREROLL_YIELD.
PREROLL_MIN = READ_AHEAD // 3
PREROLL_YIELD = READ_AHEAD // 5
# Clip i decodes on decoder slot i % SLOTS, so while it plays the next clips' decoders can be
# positioned at their first frames (pre-roll): a cut into long-GOP footage costs up to a second
# of decoding from the previous keyframe, far more than the read-ahead buffer covers.
SLOTS = 3
PREROLL = SLOTS - 1      # clips positioned ahead of the one playing


@dataclass(frozen=True)
class Shot:
    """Everything needed to render one timeline clip (a private snapshot, safe to read off-thread)."""
    index: int                   # position on the timeline
    start: float                 # timeline start
    dur: float
    src_in: float
    footage_id: str
    grade: Grade
    log_profile: str
    tonemap: str
    zoom: float
    ox: float
    oy: float
    footage: Footage = field(compare=False, hash=False)
    trans_in: str = ""           # ffmpeg xfade name of the transition into this clip
    half_in: float = 0.0         # its half-length (seconds), centered on the clip's start
    half_out: float = 0.0        # half-length of the transition out of it (the next shot's trans_in)
    rot: float = 0.0             # rotation, degrees clockwise (see reframe.split_rotation)
    box: tuple = ()              # (centre x, centre y, width, height) in parts of the frame: shown in that box
    speed: float = 1.0           # clip speed and ramp template (see speed.py)
    ramp: str = ""
    src_len: float = 0.0         # length of its source range
    view: Optional[spherical.View] = None   # 360° footage: the view of the sphere it shows
    motion: Optional[Motion] = None          # a dynamic zoom / stabilization (see motion.py)

    def src_at(self, off: float) -> float:
        """Source time shown `off` seconds after the shot's start (also before / after it: handles)."""
        if self.ramp or abs(self.speed - 1.0) > 1e-9:
            return self.src_in + timemap(self.src_len, self.speed, self.ramp).src(off)
        return self.src_in + off


@dataclass(frozen=True)
class OverlayShot:
    """An overlay layer's clip (a private copy) and its footage's color settings."""
    clip: OverlayClip
    log_profile: str
    tonemap: str
    footage: Footage = field(compare=False, hash=False)
    motion: Optional[Motion] = None          # a dynamic zoom / stabilization inside its picture (motion.py)

    @property
    def grade(self) -> Grade:
        return self.clip.grade


@dataclass(frozen=True)
class Look:
    """Project-wide render settings."""
    use_proxies: bool
    fit_mode: str
    width: int
    height: int
    decode_h: int
    color: bool = True           # False: show the picture as recorded (no log conversion, LUTs or grade)
    fade_in: float = 0.0         # picture fade from black at the timeline start (seconds)
    fade_out: float = 0.0        # and to black at its end
    end: float = 0.0             # timeline length
    overlays: tuple = ()         # OverlayShots, bottom layer first

    def fade_gain(self, t: float) -> float:
        g = 1.0
        if self.fade_in > 0:
            g = min(g, max(0.0, t / self.fade_in))
        if self.fade_out > 0:
            g = min(g, max(0.0, (self.end - t) / self.fade_out))
        return g


@dataclass
class Rendered:
    image: Optional[np.ndarray]  # BGRA (QImage.Format_RGB32); None: decoding failed, `info` holds the error
    t: float                     # timeline time (playback) or the requested time (stills)
    info: str
    serial: int = 0


@dataclass
class _Pending:
    """A playback frame in the buffer: its image is ready once `future` is done."""
    t: float
    info: str
    future: Future


class PreviewEngine:
    def __init__(self, on_still: Callable[[Rendered], None]):
        self.decoders = DecoderPool()
        self.luts = LutCache()
        self._on_still = on_still
        self._cv = threading.Condition()
        self._still: Optional[tuple] = None      # (serial, shot, src_t, t, look)
        self._still_serial = 0
        self._play: Optional[dict] = None
        self._buf: deque[_Pending] = deque()
        self._inflight = 0                        # submitted playback frames the pool hasn't finished
        self._pool = ThreadPoolExecutor(RENDER_THREADS, thread_name_prefix="preview-render")
        self._lut_wanted: list[tuple] = []        # (grade, log_profile, tonemap, full range), most urgent first
        self._quit = False
        threading.Thread(target=self._run, name="preview", daemon=True).start()
        threading.Thread(target=self._run_luts, name="preview-lut", daemon=True).start()
        threading.Thread(target=self._run_preroll, name="preview-preroll", daemon=True).start()

    # ------------------------------------------------------------------ requests (UI thread)
    def request_still(self, shot: Shot, src_t: float, t: float, look: Look,
                      shots: Optional[list[Shot]] = None) -> int:
        """Render one frame: the timeline at t when `shots` is given (transitions blended), else `shot`'s
        source frame at src_t (slipping / trimming)."""
        with self._cv:
            self._still_serial += 1
            self._still = (self._still_serial, shot, src_t, t, look, shots)
            self._cv.notify_all()
            return self._still_serial

    def start_playback(self, shots: list[Shot], look: Look, t0: float, clock0: Optional[float], fps: float) -> None:
        """Read ahead from t0. With clock0 None the clock waits for set_clock (see ready_to_start)."""
        fps = max(1.0, fps)
        with self._cv:
            self._still = None
            self._clear_buffer()
            self._play = {"shots": shots, "starts": [s.start for s in shots], "look": look,
                          "t0": t0, "clock0": clock0, "fps": fps, "k": int(np.ceil(t0 * fps - 1e-6)),
                          "end": shots[-1].start + shots[-1].dur if shots else 0.0, "prerolled": set()}
            # Build the grading LUTs of the clips ahead in play order, starting at the playhead.
            first = max(0, bisect.bisect_right(self._play["starts"], t0) - 1)
            seen, wanted = set(), []
            for s in (shots[first:] + shots[:first]) if look.color else []:
                full = s.footage.color_range == "pc"
                key = (s.grade.key(), s.log_profile, s.tonemap, full)
                if (key not in seen and needs_color(s.grade, s.log_profile)
                        and self.luts.peek(s.grade, s.log_profile, s.tonemap, full) is None):
                    seen.add(key)
                    wanted.append((s.grade, s.log_profile, s.tonemap, full))
            self._lut_wanted = wanted
            self._cv.notify_all()

    def set_clock(self, clock0: float) -> None:
        with self._cv:
            if self._play is not None:
                self._play["clock0"] = clock0

    def ready_to_start(self) -> bool:
        """Enough frames are ready to start the clock without an immediate stall."""
        with self._cv:
            play = self._play
            if play is None:
                return True
            ready = 0
            for p in self._buf:
                if not p.future.done():
                    break
                ready += 1
            return ready >= START_FRAMES or (ready > 0 and ready == len(self._buf) and (
                len(self._buf) >= READ_AHEAD or play["k"] / play["fps"] >= play["end"]))

    def stop_playback(self) -> None:
        with self._cv:
            self._play = None
            self._clear_buffer()
            self._lut_wanted.clear()
            self._cv.notify_all()

    def take_frame(self, now_t: float) -> Optional[Rendered]:
        """Newest ready playback frame due at timeline time now_t (older ones are discarded)."""
        out = None
        with self._cv:
            last = -1
            for i, p in enumerate(self._buf):
                if p.t > now_t:
                    break
                if p.future.done():
                    last = i
            for _ in range(last + 1):
                out = self._buf.popleft()
                out.future.cancel()   # no-op once done; skips late frames the pool hasn't started
            if out is not None:
                self._cv.notify_all()
        if out is None:
            return None
        try:
            img = out.future.result()
        except Exception:  # noqa: BLE001 - a frame that failed to render is skipped
            return None
        return None if img is None else Rendered(img, out.t, out.info)

    def drop_decoders(self) -> None:
        self.decoders.drop()

    def close(self) -> None:
        with self._cv:
            self._quit = True
            self._play = None
            self._clear_buffer()
            self._cv.notify_all()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.decoders.close()

    def _clear_buffer(self) -> None:
        """(Caller holds _cv.)"""
        for p in self._buf:
            p.future.cancel()
        self._buf.clear()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._quit and self._still is None and not (
                        self._play is not None and len(self._buf) < READ_AHEAD and self._inflight < MAX_INFLIGHT):
                    self._cv.wait()
                if self._quit:
                    return
                still, self._still = self._still, None
                play = self._play if still is None else None
            try:
                if still is not None:
                    self._render_still(*still)
                elif play is not None:
                    self._render_next(play)
            except Exception:  # noqa: BLE001 - a bad file must not kill the preview thread
                if play is not None:
                    play["k"] += 1
                time.sleep(0.01)

    def _render_still(self, serial: int, shot: Shot, src_t: float, t: float, look: Look,
                      shots: Optional[list[Shot]]) -> None:
        started = time.perf_counter()

        def newer_waiting() -> bool:
            return self._still is not None and time.perf_counter() - started > STILL_PATIENCE

        try:
            ovl = []
            if shots:
                layers, mix = self._gather(shots, shot.index, t, look, newer_waiting)
                ovl = self._overlays(t, look, newer_waiting)
            else:
                layers, mix = self._layer(shot, src_t, look, newer_waiting), None
            if not layers:
                return
            out = self._render(layers, mix, look, False, t if shots else None, ovl)
        except Exception as e:  # noqa: BLE001
            self._on_still(Rendered(None, t, f"Cannot decode {shot.footage.name}: {e}", serial))
            return
        self._on_still(Rendered(out, t, self._info(shot, look), serial))

    def _render_next(self, play: dict) -> None:
        """Decode the next playback frame and queue it for rendering on the pool."""
        fps = play["fps"]
        k = play["k"]
        t = k / fps
        if t >= play["end"]:
            with self._cv:
                if self._play is play:
                    self._play = None   # nothing left to read ahead
            return
        # Fell behind the clock (slow decode): jump ahead instead of rendering frames already late. The
        # frames in between still have to be decoded, so aim a little further than the clock.
        clock0 = play["clock0"]
        now_t = play["t0"] + time.perf_counter() - clock0 if clock0 is not None else play["t0"]
        if t < now_t - 1.0 / fps:
            k = int(np.ceil(now_t * fps)) + 3
            t = k / fps
            if t >= play["end"]:
                return
        i = max(0, bisect.bisect_right(play["starts"], t) - 1)
        shot = play["shots"][i]
        look = play["look"]
        layers, mix = self._gather(play["shots"], i, t, look, None)
        ovl = self._overlays(t, look, None)
        fut = self._pool.submit(self._render, layers, mix, look, True, t, ovl) if layers else None
        with self._cv:
            if self._play is not play:
                if fut is not None:
                    fut.cancel()
                return          # stopped or restarted meanwhile
            play["k"] = k + 1
            play["shot"] = i
            if fut is not None:
                self._inflight += 1
                self._buf.append(_Pending(t, self._info(shot, look), fut))
            self._cv.notify_all()
        if fut is not None:
            fut.add_done_callback(self._rendered)

    def _rendered(self, _fut: Future) -> None:
        with self._cv:
            self._inflight -= 1
            self._cv.notify_all()

    def _run_preroll(self) -> None:
        while True:
            with self._cv:
                job = None
                while not self._quit and job is None:
                    play = self._play
                    if play is not None and len(self._buf) >= PREROLL_MIN:
                        cur = play.get("shot", max(0, bisect.bisect_right(play["starts"], play["t0"]) - 1))
                        for j in range(cur + 1, min(cur + 1 + PREROLL, len(play["shots"]))):
                            if j not in play["prerolled"]:
                                play["prerolled"].add(j)
                                job = (play, j)
                                break
                    if job is None:
                        self._cv.wait()
                if self._quit:
                    return
            play, j = job
            shot, look, fps = play["shots"][j], play["look"], play["fps"]

            def starving() -> bool:
                return self._play is not play or len(self._buf) < PREROLL_YIELD

            try:   # position the clip's decoder at its first frame; playback then continues from there
                first_t = np.ceil((shot.start - shot.half_in) * fps - 1e-6) / fps
                dec = self.decoders.get(shot.footage, look.use_proxies, look.decode_h, shot.index % SLOTS)
                _, reached = dec.raw_at(shot.src_at(max(-shot.half_in, first_t - shot.start)), starving)
            except Exception:  # noqa: BLE001
                reached = True
            if not reached:
                with self._cv:   # gave way to playback: finish once the buffer has refilled
                    play["prerolled"].discard(j)

    def _layer(self, shot: Shot, src_t: float, look: Look, interrupt, off: Optional[float] = None) -> list:
        """[(decoder, frame, shot, (seconds into the shot, source time))] of source time src_t."""
        dec = self.decoders.get(shot.footage, look.use_proxies, look.decode_h, shot.index % SLOTS)
        frame, _ = dec.raw_at(src_t, interrupt)
        when = (src_t - shot.src_in if off is None else off, src_t)
        return [] if frame is None else [(dec, frame, shot, when)]

    def _gather(self, shots: list[Shot], i: int, t: float, look: Look, interrupt) -> tuple[list, Optional[tuple]]:
        """Decode the timeline at t: ([(decoder, frame, shot)], None) for shot i, or both shots' layers and
        (transition, progress) inside a transition. No layers when decoding failed."""
        shot = shots[i]
        pair = None
        if shot.half_in > 0 and i > 0 and t < shot.start + shot.half_in:
            pair = (shots[i - 1], shot, shot.trans_in, shot.start, shot.half_in)
        elif shot.half_out > 0 and i + 1 < len(shots) and t >= shot.start + shot.dur - shot.half_out:
            nxt = shots[i + 1]
            pair = (shot, nxt, nxt.trans_in, nxt.start, shot.half_out)
        if pair is not None:
            a, b, kind, cut, half = pair
            # Both sides run into their handles: a past its out point, b before its in point.
            layers = (self._layer(a, a.src_at(t - a.start), look, interrupt, t - a.start)
                      + self._layer(b, b.src_at(t - b.start), look, interrupt, t - b.start))
            if len(layers) == 2:
                return layers, (kind, (t - (cut - half)) / (2 * half))
        off = min(max(0.0, t - shot.start), shot.dur)
        return self._layer(shot, shot.src_at(off), look, interrupt, off), None

    def _overlays(self, t: float, look: Look, interrupt) -> list:
        """Decode the overlay layers showing at timeline time t: [(decoder, frame, OverlayShot)]."""
        out = []
        seen: dict[int, int] = {}
        for ov in look.overlays:
            c = ov.clip
            # Clips on a layer alternate between two decoders, so two overlapping in a transition (or a
            # multicam cross-fade) each keep reading forward instead of seeking back and forth.
            n = seen[c.track] = seen.get(c.track, -1) + 1
            if not (c.start <= t < c.end) or c.alpha_at(t) <= 0:
                continue
            try:
                dec = self.decoders.get(ov.footage, look.use_proxies, look.decode_h, 10 + 2 * c.track + n % 2)
                frame, _ = dec.raw_at(c.src_at(t - c.start), interrupt)
            except Exception:  # noqa: BLE001 - a missing overlay file doesn't stop the picture below
                continue
            if frame is not None:
                out.append((dec, frame, ov))
        return out

    def _render(self, layers: list, mix: Optional[tuple], look: Look, playing: bool,
                t: Optional[float] = None, ovl: list = ()) -> np.ndarray:
        """Convert, grade and reframe decoded layers (on the render pool during playback), then lay the overlay
        layers on top. t: the timeline time, for the fade in / out (None when showing a source frame)."""
        imgs = [self._finish(dec.rgb_of(frame), shot, look, playing, when) for dec, frame, shot, when in layers]
        out = blend(mix[0], imgs[0], imgs[1], mix[1]) if mix is not None else imgs[0]
        if ovl and t is not None:
            out = np.array(out, copy=True) if not out.flags.writeable or out.base is not None else out
            for dec, frame, ov in ovl:
                self._composite(out, dec.rgba_of(frame), ov, look, playing, t)
        gain = look.fade_gain(t) if t is not None else 1.0
        return cv2.convertScaleAbs(out, alpha=gain) if gain < 1.0 else out

    def _composite(self, out: np.ndarray, rgba: np.ndarray, ov: OverlayShot, look: Look, playing: bool,
                   t: float) -> None:
        c = ov.clip
        if spherical.is_360(ov.footage):     # its view of the sphere, about as big as it shows
            vw, vh = spherical.overlay_dims(c)
            k = min(1.0, look.height / vh * max(1.0, c.scale))
            rgba = spherical.reproject(rgba, ov.footage.projection, spherical.view_of(c), max(2, round(vw * k)),
                                       max(2, round(vh * k)), ov.footage.lens_fov)
        fn = self._grader(ov, playing) if look.color else None
        if fn is not None:
            rgba = np.dstack([fn(np.ascontiguousarray(rgba[..., :3])), rgba[..., 3]])
        if c.key_color:
            rgba = keying.apply(rgba, c.key_color, c.key_similarity, c.key_blend, c.key_despill)
        window = None
        if ov.motion is not None:
            p = reframe.overlay_place(rgba.shape[1], rgba.shape[0], look.width, look.height, c.scale, c.box_w,
                                      c.box_h, c.pan_x, c.pan_y)
            window = ov.motion.at_placement(t - c.start, c.src_at(t - c.start), p)
        layer = reframe.overlay_layer(rgba, look.width, look.height, c.scale, c.rotation, c.box_w, c.box_h,
                                      c.pan_x, c.pan_y, window)
        layer = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
        reframe.composite(out, layer, c.x * look.width, c.y * look.height, c.alpha_at(t))

    def _finish(self, img: np.ndarray, shot: Shot, look: Look, playing: bool,
                when: Optional[tuple] = None) -> np.ndarray:
        """when: (seconds into the shot, source time) of the frame, for its movement (see motion.py)."""
        fn = self._grader(shot, playing) if look.color else None
        quarters, tilt = reframe.split_rotation(shot.rot)
        if shot.view is not None:   # 360°: the clip's view of the sphere, as big as the frame (or its box)
            vw, vh = reframe.box_rect(look.width, look.height, shot.box)[2:] if shot.box else (look.width, look.height)
            img = spherical.reproject(img, shot.footage.projection, shot.view, vw, vh, shot.footage.lens_fov)
        img = reframe.rotate_quarters(img, quarters)
        if shot.box:   # a box of the frame, cropped to fill it, black around it
            bx, by, bw, bh = reframe.box_rect(look.width, look.height, shot.box)
            mode, tw, th = "fill", bw, bh
        else:
            mode, tw, th = look.fit_mode, look.width, look.height
        window = None
        if shot.motion is not None and when is not None:
            p = reframe.place(img.shape[1], img.shape[0], tw, th, mode, shot.zoom, shot.ox, shot.oy)
            window = shot.motion.at_placement(when[0], when[1], p)
        out = reframe.render(img, tw, th, mode, shot.zoom, shot.ox, shot.oy, fn, window)
        if look.color and has_spatial(shot.grade):
            out = apply_spatial(out, shot.grade)
        out = reframe.tilt(out, tilt)
        if shot.box:
            canvas = np.zeros((look.height, look.width, 3), np.uint8)
            canvas[by:by + bh, bx:bx + bw] = out
            out = canvas
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGRA)   # Qt's native 32-bit layout: cheapest to paint

    def _grader(self, shot, playing: bool):
        g, lp, tm = shot.grade, shot.log_profile, shot.tonemap
        full = shot.footage.color_range == "pc"
        if not needs_color(g, lp):
            return None
        lut = self.luts.peek(g, lp, tm, full)
        if lut is not None:
            return lut.apply
        if playing:
            return self.luts.get(g, lp, tm, full).apply
        # Adjusting sliders: grade this frame exactly and build its LUT in the background
        # (only the latest grade is worth building while a slider moves).
        with self._cv:
            self._lut_wanted = [(g, lp, tm, full)]
            self._cv.notify_all()
        return lambda a: (apply_grade(a.astype(np.float32) / 255.0, g, lp, tm, full) * 255.0
                          + 0.5).astype(np.uint8)

    def _run_luts(self) -> None:
        while True:
            with self._cv:
                while not self._quit and not self._lut_wanted:
                    self._cv.wait()
                if self._quit:
                    return
                g, lp, tm, full = self._lut_wanted.pop(0)
            try:
                if self.luts.peek(g, lp, tm, full) is None:
                    self.luts.get(g, lp, tm, full)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _info(shot: Shot, look: Look) -> str:
        f = shot.footage
        using_proxy = look.use_proxies and f.has_proxy()
        color = profile_short(f.log_profile) if look.color else "COLOR OFF"
        return f"{f.name}  {'PROXY' if using_proxy else 'ORIGINAL'}" + (f"  {color}" if color else "")
