"""Animated titles and text overlays.

`text_state` is the single description of every animation: the viewer draws each frame from it, and
the export writes the animated parts as one ASS event per output frame (libass renders them with the
same positions, opacity, scale and typed characters), so preview and export move identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .models import TextClip

ANIM_IN = {
    "none": "None (appear)",
    "fade": "Fade in",
    "pop": "Pop (bounce in)",
    "zoom": "Zoom in",
    "slide_up": "Slide up",
    "slide_down": "Slide down",
    "slide_left": "Slide in from the right",
    "slide_right": "Slide in from the left",
    "typewriter": "Typewriter",
}
ANIM_OUT = {
    "none": "None (disappear)",
    "fade": "Fade out",
    "pop": "Pop (shrink away)",
    "zoom": "Zoom out",
    "slide_up": "Slide up",
    "slide_down": "Slide down",
    "slide_left": "Slide out to the left",
    "slide_right": "Slide out to the right",
    "typewriter": "Erase (typewriter)",
}
# Unit direction each slide moves in (x right, y down); in-animations arrive moving this way.
_SLIDE = {"slide_up": (0.0, -1.0), "slide_down": (0.0, 1.0), "slide_left": (-1.0, 0.0), "slide_right": (1.0, 0.0)}
SLIDE_DIST = 0.12        # how far a slide travels, fraction of the frame height
ALIGNS = {"left": "Left", "center": "Centre", "right": "Right"}
FILLS = {"color": "Colour", "video": "Video inside the text"}

# Templates: TextClip field overrides ("font" may list fallbacks, the first installed one is used).
TEXT_PRESETS: dict[str, dict] = {
    "title": dict(label="Big title", text="Your Title", size_pct=11.0, y=0.45, anim_in="pop", anim_out="fade"),
    "lower_third": dict(label="Lower third", text="Name Surname\nPlace · role", size_pct=4.6, x=0.06, y=0.8,
                        align="left", outline=False, box=True, box_color="#101010", box_opacity=70.0,
                        anim_in="slide_right", anim_out="slide_left", duration=4.0),
    "callout": dict(label="Callout", text="WOW!", size_pct=9.0, color="#FFD400", y=0.22, anim_in="pop",
                    anim_out="pop", duration=2.0),
    "typewriter": dict(label="Typewriter", text="Once upon a time…", size_pct=6.0,
                       font=("Courier New", "Menlo", "DejaVu Sans Mono", "Liberation Mono"),
                       anim_in="typewriter", anim_out="fade", anim_dur=1.2, duration=4.0),
    "chapter": dict(label="Chapter", text="Chapter 1", size_pct=5.0, x=0.06, y=0.12, align="left",
                    anim_in="slide_up", anim_out="slide_up"),
    "end_card": dict(label="End card", text="Thanks for watching!", size_pct=8.0, anim_in="zoom",
                     anim_out="fade", anim_dur=0.8, duration=4.0),
    "video_text": dict(label="Video in text", text="WOW", size_pct=30.0, y=0.5, fill="video", bold=True,
                       outline=False, font=("Impact", "Arial Black", "DejaVu Sans"), anim_in="zoom",
                       anim_out="fade", duration=3.0),
    "text_box": dict(label="Text box", text="Add your text", size_pct=5.5, y=0.75, color="#111111",
                     outline=False, box=True, box_color="#FFFFFF", box_opacity=92.0, anim_in="slide_up",
                     anim_out="fade"),
}


def make_text(preset: str, start: float, frame_duration: Optional[float] = None) -> TextClip:
    """A new text clip from a template, starting at `start` (kept inside the timeline when possible)."""
    from .fonts import default_family, first_installed

    cfg = dict(TEXT_PRESETS.get(preset, TEXT_PRESETS["title"]))
    cfg.pop("label", None)
    font = cfg.pop("font", None)
    tc = TextClip(text=cfg.pop("text"), start=max(0.0, start), preset=preset,
                  font=first_installed(*font) if isinstance(font, tuple) else (font or default_family()))
    for k, v in cfg.items():
        setattr(tc, k, v)
    if frame_duration and frame_duration > 0.5:
        tc.duration = min(tc.duration, frame_duration)
        tc.start = min(tc.start, max(0.0, frame_duration - tc.duration))
    return tc


@dataclass
class TextState:
    alpha: float = 1.0       # 0..1
    dx: float = 0.0          # offset, fraction of the frame height (both axes, so slides look the same)
    dy: float = 0.0
    scale: float = 1.0
    chars: Optional[int] = None   # typewriter: characters shown (None = all)


def _ease_out(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


def _back_out(p: float) -> float:
    c = 1.7
    return 1.0 + (c + 1.0) * (p - 1.0) ** 3 + c * (p - 1.0) ** 2


def _apply(st: TextState, kind: str, p: float, entering: bool, n_chars: int) -> None:
    """p: 0 = fully hidden, 1 = settled."""
    p = min(1.0, max(0.0, p))
    if kind == "fade":
        st.alpha *= p
    elif kind == "pop":
        st.scale *= max(0.0, _back_out(p))
        st.alpha *= min(1.0, p * 3.0)
    elif kind == "zoom":
        st.scale *= (0.6 + 0.4 * _ease_out(p)) if entering else (1.0 + 0.35 * (1.0 - _ease_out(p)))
        st.alpha *= _ease_out(p)
    elif kind in _SLIDE:
        ux, uy = _SLIDE[kind]
        k = (1.0 - _ease_out(p)) * SLIDE_DIST * (-1.0 if entering else 1.0)
        st.dx += ux * k
        st.dy += uy * k
        st.alpha *= min(1.0, p * 2.0)
    elif kind == "typewriter":
        shown = int(math.floor(n_chars * p + 1e-9))
        st.chars = shown if st.chars is None else min(st.chars, shown)


def anim_lengths(tc: TextClip) -> tuple[float, float]:
    """Seconds of the in / out animation (each at most half the clip)."""
    d = max(0.0, min(tc.anim_dur, tc.duration / 2))
    return (d if tc.anim_in != "none" else 0.0), (d if tc.anim_out != "none" else 0.0)


def text_state(tc: TextClip, t: float) -> Optional[TextState]:
    """How the text looks at timeline time t, or None when it isn't on screen."""
    lt = t - tc.start
    if lt < 0 or lt >= tc.duration or not tc.text.strip():
        return None
    st = TextState(alpha=max(0.0, min(1.0, tc.opacity / 100.0)))
    a_in, a_out = anim_lengths(tc)
    n = len(tc.text.replace("\n", ""))
    if a_in > 0 and lt < a_in:
        _apply(st, tc.anim_in, lt / a_in, True, n)
    rem = tc.duration - lt
    if a_out > 0 and rem < a_out:
        _apply(st, tc.anim_out, rem / a_out, False, n)
    return st


def visible_lines(tc: TextClip, chars: Optional[int]) -> list[tuple[str, str]]:
    """Each line split into (typed, not yet typed) text."""
    out, left = [], (10 ** 9 if chars is None else chars)
    for line in tc.text.split("\n"):
        k = max(0, min(len(line), left))
        out.append((line[:k], line[k:]))
        left -= len(line)
    return out


# ----------------------------------------------------------------------------- export (ASS)

def _color(hex_color: str, alpha: int = 0) -> str:
    h = (hex_color or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _esc(t: str) -> str:
    return t.replace("\\", "/").replace("{", "(").replace("}", ")")


def _cs_time(cs: int) -> str:
    cs = max(0, cs)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass_metrics(tc: TextClip, height: int) -> tuple[int, int, int]:
    """(font size, outline / box padding, border style) in output pixels; shared with the viewer."""
    size = max(8, int(round(height * tc.size_pct / 100.0)))
    if tc.box:
        return size, max(4, size // 4), 3
    return size, (max(1, size // 14) if tc.outline else 0), 1


def _alignment(tc: TextClip) -> int:
    return {"left": 4, "right": 6}.get(tc.align, 5)    # numpad layout, vertically centred


def is_video_text(tc: TextClip) -> bool:
    return getattr(tc, "fill", "color") == "video"


def video_text_cover(tc: TextClip, st: TextState) -> float:
    """How much (0..1) the background around a video-filled text covers the picture in state st: it comes and
    goes with the text (its opacity and fades)."""
    return max(0.0, min(1.0, st.alpha)) * max(0.0, min(1.0, tc.bg_opacity / 100.0))


def write_titles_ass(path: str, texts: list[TextClip], width: int, height: int, fps: float,
                     mode: str = "titles") -> int:
    """Write the text clips as an ASS file for ffmpeg's subtitles filter. Returns the number of events.

    mode "titles": the texts as they show. For texts filled with video, drawn on black and laid over the
    picture: "mask" (white where the background covers the picture: a full-frame box at the cover's opacity
    with the letters cut out of it in black) and "cover" (the background colour over the whole frame)."""
    fps = max(1.0, fps)
    styles: list[str] = []
    events: list[tuple[int, str]] = []     # (start in centiseconds, line)
    rect = f"m 0 0 l {width} 0 {width} {height} 0 {height}"
    for k, tc in enumerate(texts):
        if not tc.text.strip() or tc.duration <= 0:
            continue
        size, pad, border = ass_metrics(tc, height)
        box_alpha = int(round(255 * (1.0 - max(0.0, min(100.0, tc.box_opacity)) / 100.0)))
        out_col = _color(tc.box_color, box_alpha) if tc.box else "&H00000000"
        name = f"T{k}"
        if mode == "titles":
            styles.append(f"Style: {name},{tc.font},{size},{_color(tc.color)},&H000000FF,{out_col},&H00000000,"
                          f"{-1 if tc.bold else 0},{-1 if tc.italic else 0},0,0,100,100,0,0,{border},{pad},0,"
                          f"{_alignment(tc)},0,0,0,1")
        else:   # the letters alone: no outline, no box
            styles.append(f"Style: {name},{tc.font},{size},&H00000000,&H000000FF,&H00000000,&H00000000,"
                          f"{-1 if tc.bold else 0},{-1 if tc.italic else 0},0,0,100,100,0,0,1,0,0,"
                          f"{_alignment(tc)},0,0,0,1")

        def tags(st: TextState) -> str:
            x = tc.x * width + st.dx * height
            y = tc.y * height + st.dy * height
            a = int(round(255 * (1.0 - max(0.0, min(1.0, st.alpha)))))
            s = max(0.0, st.scale) * 100.0
            return f"{{\\pos({x:.1f},{y:.1f})\\alpha&H{a:02X}&\\fscx{s:.1f}\\fscy{s:.1f}}}"

        def body(st: TextState) -> str:
            if st.chars is None:
                return "\\N".join(_esc(line) for line in tc.text.split("\n"))
            # Untyped characters stay in place but invisible, so the typed ones never shift.
            parts = []
            for typed, rest in visible_lines(tc, st.chars):
                parts.append(_esc(typed) + (f"{{\\alpha&HFF&}}{_esc(rest)}" if rest else ""))
            return "\\N".join(parts)

        def box(color: str, cover: float) -> str:
            a = int(round(255 * (1.0 - max(0.0, min(1.0, cover)))))
            return (f"{{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H{_color(color)[4:]}&\\alpha&H{a:02X}&\\p1}}"
                    f"{rect}{{\\p0}}")

        def event(t0: float, t1: float, st: TextState) -> None:
            c0, c1 = int(round(t0 * 100)), int(round(t1 * 100))
            if c1 <= c0:
                return
            span = f"{_cs_time(c0)},{_cs_time(c1)},{name},,0,0,0,,"
            if mode == "titles":
                if st.alpha > 0.002 and st.scale > 0.002:
                    events.append((c0, f"Dialogue: 0,{span}{tags(st)}{body(st)}"))
                return
            cover = video_text_cover(tc, st)
            if cover <= 0.002:
                return
            if mode == "cover":
                events.append((c0, f"Dialogue: 0,{span}{box(tc.bg_color, 1.0)}"))
                return
            events.append((c0, f"Dialogue: 0,{span}{box('#FFFFFF', cover)}"))
            if st.scale > 0.002:     # the letters, cut out: black (the picture shows through)
                events.append((c0, f"Dialogue: 1,{span}{tags(st)}{body(st)}"))

        a_in, a_out = anim_lengths(tc)
        hold0, hold1 = tc.start + a_in, tc.end - a_out
        # Animated parts: one event per output frame, sampled like the viewer samples them.
        k0, k1 = int(math.ceil(tc.start * fps - 1e-6)), int(math.ceil(tc.end * fps - 1e-6))
        frames = [i / fps for i in range(k0, k1)]
        for i, t in enumerate(frames):
            if hold0 <= t < hold1:
                continue
            st = text_state(tc, t)
            if st is not None:
                event(max(t, tc.start), min(frames[i + 1] if i + 1 < len(frames) else tc.end, tc.end), st)
        if hold1 > hold0:
            k = math.ceil(hold0 * fps - 1e-6) / fps
            e = math.ceil(hold1 * fps - 1e-6) / fps
            st = text_state(tc, (hold0 + hold1) / 2) or TextState()
            event(k, min(e, tc.end), st)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n")
        fh.write(f"PlayResX: {width}\nPlayResY: {height}\n\n")
        fh.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                 "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
                 "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        fh.write("\n".join(styles) + "\n\n")
        fh.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        fh.write("\n".join(line for _, line in sorted(events, key=lambda e: e[0])) + "\n")
    return len(events)
