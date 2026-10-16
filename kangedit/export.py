"""Rendering the timeline with ffmpeg (final export and preview audio)."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

from . import reframe
from .speed import sound_mode, stretch
from .captions import build_captions, caption_layout, write_ass, write_srt
from .color import ffmpeg_halation, ffmpeg_sharpen, needs_color, write_cube
from .config import cache_dir, frame_size
from .edits import ClipPlan, plan
from .fonts import copy_family
from .titles import write_titles_ass
from .ffmpeg import FFmpegError, ProgressCb, ffmpeg_matrix, hw_encoder, hw_encoder_args, hw_encoder_label, run_ffmpeg
from .models import LOOP_CROSSFADE, Footage, OverlayClip, Project

CODECS = {
    "h264": ("H.264 (MP4) - most compatible", ".mp4"),
    "h265": ("H.265 / HEVC (MP4) - smaller files", ".mp4"),
    "prores": ("Apple ProRes 422 HQ (MOV) - for further editing", ".mov"),
}
GPU_STALL = 180.0   # seconds without progress before a GPU export is redone on the CPU
GRADE_FMT = "gbrp16le"   # pixel format clips are graded in (16-bit planar RGB)
QUALITIES = {"High": 18, "Medium": 21, "Small file": 26}
CAPTION_MODES = {
    "none": "No captions",
    "burn": "Burn into video",
    "srt": "Separate .srt file",
    "both": "Burn in + .srt file",
}


@dataclass
class ExportSettings:
    output: str
    aspect: str
    resolution: str
    fps: float
    fit_mode: str = "fill"
    codec: str = "h264"
    quality: str = "High"
    captions: str = "none"
    use_proxies: bool = False   # draft export from proxies
    # Encode on the graphics card when it has a working H.264/H.265 encoder. (GPU *decoding* isn't
    # used here: copying full-size frames back for the CPU filters measured slower than CPU decoding.)
    use_gpu: bool = True
    blend_slowmo: bool = True   # slow motion: blend neighbouring frames instead of repeating them (smoother)


_cubes: dict[str, dict[tuple, str]] = {}      # export folder -> {grade key: .cube already written there}


def _grade_cube(workdir: str, name: str, grade, f: Footage) -> str:
    """Write the grade's .cube LUT for footage f as `name` in workdir, or return the one an earlier clip with
    the same grade and colour settings already wrote (each is a few MB, read by FFmpeg at start)."""
    full = f.color_range == "pc"
    key = (grade.key(), f.log_profile, f.tonemap, full)
    done = _cubes.setdefault(workdir, {})
    if key not in done:
        write_cube(os.path.join(workdir, name), grade, f.log_profile, f.tonemap, full_range=full)
        done[key] = name
    return done[key]


def _clip_inputs(project: Project, use_proxies: bool, i: int, cp: ClipPlan, fps: float) -> tuple[list[str], int, bool, float]:
    """Input args for clip i covering its video and audio ranges (handles and split edits included).
    Returns (args, n_inputs, audio_is_silence, input start time in the source)."""
    clip = project.clips[i]
    f = project.footage[clip.footage_id]
    src = f.proxy_path if (use_proxies and f.has_proxy()) else f.path
    ss = max(0.0, min(cp.v_in, cp.a_in))
    end = max(cp.v_end, cp.a_end)
    if f.is_image:
        args, ss = _image_input(f, ss, end - ss + 0.25, fps)
    else:
        args = ["-ss", f"{ss:.4f}", "-t", f"{end - ss + 0.25:.4f}", "-i", src]
    if f.has_audio and not f.is_image:
        return args, 1, False, ss
    args += ["-f", "lavfi", "-t", f"{cp.a_dur + 0.25:.4f}", "-i", "anullsrc=r=48000:cl=stereo"]
    return args, 2, True, ss


def _image_input(f: Footage, ss: float, dur: float, fps: float) -> tuple[list[str], float]:
    """Input args for an image read from source time ss for dur seconds, and the source time its first frame
    shows: a still repeats its picture, an animated GIF loops from its start (trim the difference)."""
    if f.codec == "gif":
        return ["-ignore_loop", "0", "-t", f"{ss + dur:.4f}", "-i", f.path], 0.0
    return ["-loop", "1", "-framerate", f"{fps:.4f}", "-t", f"{dur:.4f}", "-i", f.path], ss


# Pictures with transparency show black where they are transparent on the main track.
_IMAGE_PREP = "format=rgba,premultiply=inplace=1,"


def _audio_chain(inp: str, out: str, dur: float, volume: float, skip: float = 0.0,
                 segs: Optional[list] = None, ss: float = 0.0, keep_pitch: bool = True) -> str:
    """The sound of a clip: `dur` seconds from `skip` seconds into the input, or at another speed `segs`
    [(source start, source end, speed)] (source times; the input starts at source time `ss`), time-stretched
    with its pitch kept, or resampled (pitch moving with the speed) when not keep_pitch."""
    fade = min(0.02, dur / 4)
    fmt = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
    head, src = "", f"[{inp}]"
    if segs and len(segs) == 1:
        s0, _s1, v = segs[0]
        skip = s0 - ss
        tempo = stretch(v, keep_pitch)
        fmt += tempo + "," if tempo else ""
    elif segs and keep_pitch:
        # A speed ramp, pitch kept: one continuous time-stretch whose tempo changes on the way. Restarting
        # atempo for every piece would lose a little sound at each restart and drift away from the picture.
        # Each atempo stage (0.5..2x) gets its own tempo changes (asendcmd), timed on its own input's clock.
        lo, hi = min(v for *_, v in segs), max(v for *_, v in segs)
        n = max(1, math.ceil(math.log2(max(hi, 1.0 / lo)) - 1e-9))
        chain = []
        for k in range(n):
            name = f"atempo@{out}t{k}"
            t, cmds = 0.0, []
            for i, (s0, s1, v) in enumerate(segs):
                f = v ** (1.0 / n)
                if i:
                    cmds.append(f"{t:.4f} {name} tempo {f:.6f}")
                t += (s1 - s0) / f ** k          # this stage's input runs at the speed of the stages before it
            if cmds:
                chain.append(f"asendcmd=c='{';'.join(cmds)}'")
            chain.append(f"{name}=tempo={segs[0][2] ** (1.0 / n):.6f}")
        skip = segs[0][0] - ss
        fmt += ",".join(chain) + ","
    elif segs:
        # A speed ramp, pitch following the speed: pieces of nearly constant speed, each resampled on its own
        # (exact in length), joined back together.
        n = len(segs)
        head = f"[{inp}]{fmt.rstrip(',')},asplit={n}" + "".join(f"[{out}s{k}]" for k in range(n)) + ";"
        for k, (s0, s1, v) in enumerate(segs):
            tempo = stretch(v, keep_pitch)
            head += (f"[{out}s{k}]atrim=start={max(0.0, s0 - ss):.4f}:end={max(0.0, s1 - ss):.4f},"
                     f"asetpts=PTS-STARTPTS{',' + tempo if tempo else ''}[{out}p{k}];")
        head += "".join(f"[{out}p{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[{out}r];"
        src, skip, fmt = f"[{out}r]", 0.0, ""
    trim = f"atrim=start={skip:.4f},asetpts=PTS-STARTPTS," if skip > 0.0005 else ""
    return (f"{head}{src}{trim}{fmt}"
            f"volume={volume:.3f},afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.4f}:d={fade:.3f},"
            f"apad,atrim=duration={dur:.4f},asetpts=PTS-STARTPTS[{out}]")


def _speed_segments(clip, t0: float, t1: float) -> Optional[list]:
    """A clip's sound over timeline offsets t0..t1 as pieces in source time for its speed (None when it plays
    at normal speed: at 1x, or not synced to the picture)."""
    tm = clip.timemap()
    if tm.normal or sound_mode(clip.audio_mode) == "normal":
        return None
    return [(clip.src_in + a, clip.src_in + b, v) for a, b, v in tm.segments(t0, t1)]


def _clip_audio(ai: str, i: int, clip, cp: ClipPlan, silent: bool, ss: float) -> str:
    mode = sound_mode(clip.audio_mode)
    return _audio_chain(ai, f"a{i}", cp.a_dur, 0.0 if mode == "mute" else clip.volume,
                        0.0 if silent else max(0.0, cp.a_in - ss),
                        None if silent else _speed_segments(clip, cp.a_t0, cp.a_t0 + cp.a_dur), ss, mode != "pitch")


def _curve(fade: float) -> str:
    return ":curve=qsin" if fade > 0.05 else ""


DUCK_THRESHOLD = 0.01    # clip sound above this level (about -40 dBFS) turns the ducked music down


def _mix_sounds(project: Project, inputs: list[str], graph: list[str], k: int, extra: tuple = ()) -> str:
    """Mix the overlay layers' sound (labels in `extra`) and the SFX layer over [acat] (inputs from index k
    on). Sounds marked `duck` (music) are turned down while the edit's own sound plays (sidechain
    compression, at most settings.duck_db). Returns the label of the final audio."""
    dur = project.duration
    base = "acat"
    if extra:
        graph.append(f"[acat]{''.join(f'[{x}]' for x in extra)}amix=inputs={len(extra) + 1}:normalize=0:"
                     f"duration=first:dropout_transition=0[aprog]")
        base = "aprog"
    sounds = [x for x in project.sounds if x.duration > 0.01 and x.start < dur and os.path.exists(x.path)]
    if not sounds:
        return base
    plain, ducked = [], []
    for j, x in enumerate(sounds):
        fi, fo = x.fades()
        fi, fo = max(fi, 0.005), max(fo, min(0.02, x.duration / 4))   # at least a click-free edge
        delay = int(round(max(0.0, x.start) * 1000))
        src = f"[{k + j}:a]"
        if x.looping:
            # One read of the loop, repeated: each repeat crossfades into the next (quarter-sine, even loudness).
            n = x.repeats()
            inputs += ["-ss", f"{x.src_in:.4f}", "-t", f"{x.loop:.4f}", "-i", x.path]
            graph.append(f"{src}aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                         f"asplit={n}" + "".join(f"[lp{j}_{r}]" for r in range(n)))
            cur = f"lp{j}_0"
            for r in range(1, n):
                graph.append(f"[{cur}][lp{j}_{r}]acrossfade=d={LOOP_CROSSFADE:.4f}:c1=qsin:c2=qsin[lpx{j}_{r}]")
                cur = f"lpx{j}_{r}"
            graph.append(f"[{cur}]atrim=duration={x.duration:.4f}[lpj{j}]")
            src = f"[lpj{j}]"
        else:
            inputs += ["-ss", f"{x.src_in:.4f}", "-t", f"{x.duration:.4f}", "-i", x.path]
        # quarter-sine fades: crossfading copies of a sound keep an even loudness
        graph.append(f"{src}aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                     f"volume={x.volume:.3f},afade=t=in:d={fi:.4f}{_curve(fi)},"
                     f"afade=t=out:st={max(0.0, x.duration - fo):.4f}:d={fo:.4f}{_curve(fo)},"
                     f"adelay=delays={delay}:all=1[sfx{j}]")
        (ducked if x.duck else plain).append(f"[sfx{j}]")
    depth = max(0.0, project.settings.duck_db)
    if ducked and depth > 0.05:
        graph.append(f"[{base}]asplit=2[aprogm][aduck_sc]")
        if len(ducked) > 1:
            graph.append(f"{''.join(ducked)}amix=inputs={len(ducked)}:normalize=0:dropout_transition=0[amusic]")
        else:
            graph.append(f"{ducked[0]}anull[amusic]")
        # mix < 1 keeps (1 - mix) of the music however loud the sidechain is: the floor of the ducking.
        mix = 1.0 - 10 ** (-depth / 20.0)
        graph.append(f"[amusic][aduck_sc]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio=20:attack=30:"
                     f"release=400:makeup=1:mix={mix:.4f}[aducked]")
        base = "aprogm"
        plain.append("[aducked]")
    elif ducked:
        plain += ducked
    # normalize=0: sounds add to the edit's audio at their own level instead of ducking it.
    graph.append(f"[{base}]{''.join(plain)}amix=inputs={len(plain) + 1}:normalize=0:duration=first:"
                 f"dropout_transition=0[amix]")
    return "amix"


def _overlay_input(project: Project, c: OverlayClip, use_proxies: bool, fps: float) -> tuple[list[str], float]:
    """Input args for an overlay clip, and how much to trim from the start of its video."""
    f = project.footage[c.footage_id]
    if f.is_image:
        args, ss = _image_input(f, c.src_in, c.duration, fps)
        return args, max(0.0, c.src_in - ss)
    src = f.proxy_path if (use_proxies and f.has_proxy()) else f.path
    # (sound at normal speed under slow motion runs on past the out point)
    return ["-ss", f"{c.src_in:.4f}", "-t", f"{max(c.src_dur, c.duration) + 0.25:.4f}", "-i", src], 0.0


def _overlays(project: Project, use_proxies: bool, fps: float, inputs: list[str], graph: list[str], k: int,
              video: Optional[tuple] = None) -> tuple[int, str, list[str]]:
    """Add the overlay layers (inputs from index k on). video: (frame width, height, workdir, [base label])
    lays their pictures over the base, bottom layer first; None only takes their sound.
    Returns (next input index, video label, overlay audio labels)."""
    dur = project.duration
    vout = video[3] if video else ""
    audio = []
    clips = sorted((c for c in project.overlays if c.footage_id in project.footage and c.duration > 0.01
                    and c.start < dur and os.path.exists(project.footage[c.footage_id].path)),
                   key=lambda c: c.track)
    for j, c in enumerate(clips):
        f = project.footage[c.footage_id]
        sound = f.has_audio and not f.is_image and c.volume > 0.001 and sound_mode(c.audio_mode) != "mute"
        if video is None and not sound:
            continue
        args, skip = _overlay_input(project, c, use_proxies, fps)
        inputs += args
        length = min(c.duration, dur - c.start)
        if sound:
            delay = int(round(max(0.0, c.start) * 1000))
            graph.append(_audio_chain(f"{k}:a:0", f"ova{j}r", length, c.volume, 0.0, _speed_segments(c, 0.0, length),
                                      c.src_in, sound_mode(c.audio_mode) != "pitch"))
            graph.append(f"[ova{j}r]adelay=delays={delay}:all=1[ova{j}]")
            audio.append(f"ova{j}")
        if video is not None:
            tw, th, workdir, _ = video
            trim = f"trim=start={skip:.4f},setpts=PTS-STARTPTS," if skip > 0.0005 else ""
            tm = c.timemap()
            spd = "" if (tm.normal or f.is_image) else f"setpts='{tm.setpts(0.0)}',"
            chain = f"[{k}:v:0]setpts=PTS-STARTPTS,{trim}{spd}fps={fps:.4f},trim=duration={length:.4f},"
            matrix = "bt709" if (use_proxies and f.has_proxy()) else ffmpeg_matrix(f.color_space, min(f.width, f.height))
            if needs_color(c.grade, f.log_profile):
                lut = _grade_cube(workdir, f"lut_ov{j}.cube", c.grade, f)
                chain += f"scale=in_color_matrix={matrix}:in_range=auto,format=rgba64le,lut3d=file={lut}:interp=tetrahedral,"
            elif not f.is_image:
                chain += f"scale=in_color_matrix={matrix}:in_range=auto,format=rgba64le,"
            pl = reframe.overlay_place(f.width, f.height, tw, th, c.scale, c.box_w, c.box_h, c.pan_x, c.pan_y)
            chain += f"scale={pl.scaled_w}:{pl.scaled_h}:flags=lanczos,"
            if (pl.crop_w, pl.crop_h) != (pl.scaled_w, pl.scaled_h):
                chain += f"crop={pl.crop_w}:{pl.crop_h}:{pl.crop_x}:{pl.crop_y},"
            chain += "format=rgba,"
            if abs(c.rotation) >= 0.01:
                a = math.radians(c.rotation)
                chain += f"rotate={a:.6f}:ow=rotw({a:.6f}):oh=roth({a:.6f}):c=none,"
            if c.opacity < 0.999:
                chain += f"colorchannelmixer=aa={max(0.0, c.opacity):.4f},"
            fi, fo = c.fades()
            if fi > 0:
                chain += f"fade=t=in:st=0:d={fi:.4f}:alpha=1,"
            if fo > 0:
                chain += f"fade=t=out:st={max(0.0, c.duration - fo):.4f}:d={fo:.4f}:alpha=1,"
            # To YUV here, as BT.709 (the overlay filter would otherwise convert with its BT.601 default).
            chain += "scale=out_color_matrix=bt709:out_range=tv,format=yuva444p,"
            graph.append(chain + f"setpts=PTS+{c.start:.4f}/TB[ov{j}]")
            graph.append(f"[{vout}][ov{j}]overlay=x='{c.x * tw:.2f}-w/2':y='{c.y * th:.2f}-h/2':eof_action=pass:"
                         f"format=auto[vov{j}]")
            vout = f"vov{j}"
        k += 1
    return k, vout, audio


def _fade_audio(project: Project, graph: list[str], aout: str) -> str:
    """Fade the final mix in / out at the timeline's ends (project settings). Returns the output label."""
    dur = project.duration
    fi, fo = project.settings.fades(dur, sound=True)
    if fi <= 0 and fo <= 0:
        return aout
    fades = []
    if fi > 0:
        fades.append(f"afade=t=in:st=0:d={fi:.4f}")
    if fo > 0:
        fades.append(f"afade=t=out:st={max(0.0, dur - fo):.4f}:d={fo:.4f}")
    graph.append(f"[{aout}]{','.join(fades)}[afaded]")
    return "afaded"


def _join(graph: list[str], plans: list[ClipPlan], fps: float, video: bool) -> None:
    """Join the per-clip streams [v{i}]/[a{i}] into [vcat]/[acat]: transitions overlap neighbours
    (xfade / acrossfade), cuts concatenate. Video and audio are joined separately because split
    edits put their cut points at different times."""
    groups: list[tuple[str, str]] = []
    cur_v = cur_a = ""
    cur_len = 0
    for i, cp in enumerate(plans):
        if cp.xfade and cur_a:
            d_f = cp.pre_f + plans[i - 1].post_f
            d = d_f / fps
            if video:
                graph.append(f"[{cur_v}][v{i}]xfade=transition={cp.xfade}:duration={d:.4f}:"
                             f"offset={(cur_len - d_f) / fps:.4f}[xv{i}]")
                cur_v = f"xv{i}"
            graph.append(f"[{cur_a}][a{i}]acrossfade=d={d:.4f}:c1=tri:c2=tri[xa{i}]")
            cur_a = f"xa{i}"
            cur_len += cp.v_frames - d_f
        else:
            if cur_a:
                groups.append((cur_v, cur_a))
            cur_v, cur_a, cur_len = f"v{i}", f"a{i}", cp.v_frames
    groups.append((cur_v, cur_a))
    if video:
        graph.append("".join(f"[{v}]" for v, _ in groups) + f"concat=n={len(groups)}:v=1:a=0[vcat]")
    graph.append("".join(f"[{a}]" for _, a in groups) + f"concat=n={len(groups)}:v=0:a=1[acat]")


def gpu_encoder(s: ExportSettings) -> Optional[str]:
    """Hardware encoder to use for this export, or None for the CPU encoders."""
    return hw_encoder(s.codec) if s.use_gpu and s.codec in ("h264", "h265") else None


def build_export(project: Project, s: ExportSettings, workdir: str, encoder: Optional[str] = None) -> list[str]:
    tw, th = frame_size(s.aspect, s.resolution)
    inputs: list[str] = []
    graph: list[str] = []
    k = 0
    final_fmt = "yuv422p10le" if s.codec == "prores" else "yuv420p"
    plans = plan(project, s.fps)
    for i, (clip, cp) in enumerate(zip(project.clips, plans)):
        f = project.footage[clip.footage_id]
        args, n_in, silent, ss = _clip_inputs(project, s.use_proxies, i, cp, s.fps)
        inputs += args
        vi, ai = f"{k}:v:0", (f"{k + 1}:a:0" if silent else f"{k}:a:0")
        k += n_in

        skip = max(0.0, cp.v_in - ss)
        trim = f"trim=start={skip:.4f},setpts=PTS-STARTPTS," if skip > 0.0005 else ""
        quarters, tilt = reframe.split_rotation(clip.rotation)
        turn = "," + reframe.ffmpeg_quarters(quarters) if quarters else ""
        prep = _IMAGE_PREP if f.is_image else ""
        tm = clip.timemap()
        spd = "" if (tm.normal or f.is_image) else f"setpts='{tm.setpts(cp.v_in - clip.src_in)}',"
        # framerate (blended frames) keeps the input's timebase; the fps pass after it sets the one xfade needs.
        resample = (f"framerate=fps={s.fps:.4f},fps={s.fps:.4f}" if spd and s.blend_slowmo and min(tm.v_in, tm.v_out, tm.speed) < 0.99
                    else f"fps={s.fps:.4f}")
        chain = [f"[{vi}]setpts=PTS-STARTPTS,{prep}{trim}{spd}{resample}{turn}[v{i}a]"]
        sw, sh = reframe.quarter_dims(f.width, f.height, quarters)
        box = reframe.box_rect(tw, th, (clip.box_x, clip.box_y, clip.box_w, clip.box_h)) \
            if clip.box_w > 0 and clip.box_h > 0 else None
        color = needs_color(clip.grade, f.log_profile)
        ph = box[3] if box else th          # picture height: halation and sharpening scale with it
        halation = ffmpeg_halation(clip.grade, ph, f"v{i}g", f"v{i}h")
        # Graded clips are converted to 16-bit planar RGB (as precise as packed RGB48, faster through the
        # LUT) in the same swscale pass that resizes them, instead of a second full-frame pass.
        convert = ""
        # The source's colour matrix is always named (FFmpeg would take an untagged HD stream for SD BT.601
        # and shift its colours on the way to BT.709).
        matrix = "bt709" if (s.use_proxies and f.has_proxy()) else ffmpeg_matrix(f.color_space, min(f.width, f.height))
        if color or halation:
            convert = f":in_color_matrix={matrix}:in_range=auto|{GRADE_FMT}"
        mode = "fill" if box else s.fit_mode
        chain.append(reframe.ffmpeg_filter(sw, sh, box[2] if box else tw, box[3] if box else th, mode, clip.zoom,
                                           clip.offset_x, clip.offset_y, f"v{i}a", f"v{i}b", f"c{i}", convert))
        post = f"[v{i}b]"
        if convert and f"format={GRADE_FMT}" not in chain[-1].split("overlay")[-1]:
            post += f"scale={convert.split('|')[0][1:]},format={GRADE_FMT},"     # (blurred background)
        if color:
            lut = _grade_cube(workdir, f"lut_{i}.cube", clip.grade, f)
            post += f"lut3d=file={lut}:interp=tetrahedral,"
        if halation:
            chain.append(post + f"null[v{i}g]")
            chain += halation
            post = f"[v{i}h]"
        sharpen = ffmpeg_sharpen(clip.grade, ph)
        rot = reframe.ffmpeg_tilt(tilt)
        pad = f"pad={tw}:{th}:{box[0]}:{box[1]}:black," if box else ""
        to_out = "scale=" + ("" if convert or f.is_image else f"in_color_matrix={matrix}:in_range=auto:")
        post += (f"{to_out}out_color_matrix=bt709:out_range=tv,format={final_fmt},{sharpen + ',' if sharpen else ''}"
                 f"{rot + ',' if rot else ''}{pad}tpad=stop_mode=clone:stop_duration=1,trim=end_frame={cp.v_frames},setpts=PTS-STARTPTS[v{i}]")
        chain.append(post)
        graph += chain
        graph.append(_clip_audio(ai, i, clip, cp, silent, ss))

    _join(graph, plans, s.fps, video=True)
    k, vout, ov_audio = _overlays(project, s.use_proxies, s.fps, inputs, graph, k, (tw, th, workdir, "vcat"))
    aout = _fade_audio(project, graph, _mix_sounds(project, inputs, graph, k, tuple(ov_audio)))
    fonts_dir = os.path.join(workdir, "fonts")
    texts = [x for x in project.texts if x.text.strip() and x.duration > 0 and x.start < project.duration]
    if texts:
        write_titles_ass(os.path.join(workdir, "titles.ass"), texts, tw, th, s.fps)
        found = [copy_family(f, fonts_dir) for f in {x.font for x in texts}]
        graph.append(f"[{vout}]subtitles=titles.ass{':fontsdir=fonts' if any(found) else ''}[vtitles]")
        vout = "vtitles"
    caps = build_captions(project, aspect=s.aspect) if s.captions != "none" else []
    cap_layout = caption_layout(project.settings.captions, tw, th, project.settings.video_format)
    if s.captions in ("burn", "both") and caps:
        write_ass(os.path.join(workdir, "captions.ass"), caps, project.settings.captions, tw, th, cap_layout)
        # Hand libass the font's own files: it may not find fonts installed for this user only by name.
        fontsdir = ":fontsdir=fonts" if copy_family(project.settings.captions.font, fonts_dir) else ""
        graph.append(f"[{vout}]subtitles=captions.ass{fontsdir}[vsub]")
        vout = "vsub"
    fi, fo = project.settings.fades(project.duration)
    if fi > 0 or fo > 0:   # after the titles and captions, so they fade with the picture
        fades = []
        if fi > 0:
            fades.append(f"fade=t=in:st=0:d={fi:.4f}")
        if fo > 0:
            fades.append(f"fade=t=out:st={max(0.0, project.duration - fo):.4f}:d={fo:.4f}")
        graph.append(f"[{vout}]{','.join(fades)}[vfaded]")
        vout = "vfaded"
    if s.captions in ("srt", "both") and caps:
        write_srt(os.path.splitext(s.output)[0] + ".srt", caps, project.settings.captions, cap_layout)

    graph_path = os.path.join(workdir, "graph.txt")
    with open(graph_path, "w", encoding="utf-8") as fh:
        fh.write(";\n".join(graph))

    # Slice-threaded filters (scale, lut3d) keep the filter graph from being the bottleneck.
    out = ["-filter_complex_threads", str(max(2, min(8, os.cpu_count() or 2))),
           "-/filter_complex", "graph.txt", "-map", f"[{vout}]", "-map", f"[{aout}]"]
    if encoder:
        q = QUALITIES[s.quality] + (2 if s.codec == "h265" else 0)
        out += ["-c:v", encoder] + hw_encoder_args(encoder, q) + ["-pix_fmt", "nv12"]
        if s.codec == "h265":
            out += ["-tag:v", "hvc1"]
    elif s.codec == "h265":
        out += ["-c:v", "libx265", "-preset", "medium", "-crf", str(QUALITIES[s.quality] + 2),
                "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]
    elif s.codec == "prores":
        out += ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-pix_fmt", "yuv422p10le"]
    else:
        out += ["-c:v", "libx264", "-preset", "medium", "-crf", str(QUALITIES[s.quality]),
                "-profile:v", "high", "-pix_fmt", "yuv420p"]
    out += ["-r", f"{s.fps:.4f}", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv"]
    if s.codec == "prores":
        out += ["-c:a", "pcm_s16le"]
    else:
        out += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    out += [os.path.abspath(s.output)]
    return inputs + out


def export(project: Project, s: ExportSettings, progress: Optional[ProgressCb] = None,
           cancel: Optional[threading.Event] = None) -> str:
    if not project.clips:
        raise RuntimeError("The timeline is empty")
    encoder = gpu_encoder(s)
    name = os.path.basename(s.output)
    workdir = tempfile.mkdtemp(prefix="export_", dir=str(cache_dir("tmp")))
    try:
        if encoder:
            try:
                # A GPU encoder that hangs (some driver versions never start) is given up after GPU_STALL s.
                run_ffmpeg(build_export(project, s, workdir, encoder), project.duration, progress, cancel,
                           cwd=workdir, label=f"Exporting {name} (GPU: {hw_encoder_label(encoder)})",
                           stall=GPU_STALL)
                return s.output
            except FFmpegError:
                pass   # driver hiccup, out of video memory, a hang, unsupported source...: redo it on the CPU
        run_ffmpeg(build_export(project, s, workdir), project.duration, progress, cancel, cwd=workdir,
                   label=f"Exporting {name}" + (" (CPU)" if s.use_gpu else ""))
    finally:
        _cubes.pop(workdir, None)
        shutil.rmtree(workdir, ignore_errors=True)
    return s.output


def render_audio(project: Project, use_proxies: bool, out_wav: str,
                 cancel: Optional[threading.Event] = None) -> Optional[str]:
    """Mix the timeline audio into a WAV for synced preview playback."""
    if not project.clips:
        return None
    inputs, graph, k = [], [], 0
    fps = project.settings.fps
    plans = plan(project, fps)
    for i, (clip, cp) in enumerate(zip(project.clips, plans)):
        args, n_in, silent, ss = _clip_inputs(project, use_proxies, i, cp, fps)
        inputs += args
        ai = f"{k + 1}:a:0" if silent else f"{k}:a:0"
        k += n_in
        graph.append(_clip_audio(ai, i, clip, cp, silent, ss))
    _join(graph, plans, fps, video=False)
    k, _, ov_audio = _overlays(project, use_proxies, fps, inputs, graph, k)
    aout = _fade_audio(project, graph, _mix_sounds(project, inputs, graph, k, tuple(ov_audio)))
    workdir = tempfile.mkdtemp(prefix="audio_", dir=str(cache_dir("tmp")))
    try:
        with open(os.path.join(workdir, "graph.txt"), "w", encoding="utf-8") as fh:
            fh.write(";\n".join(graph))
        tmp = out_wav + ".part.wav"
        run_ffmpeg(inputs + ["-/filter_complex", "graph.txt", "-map", f"[{aout}]", "-c:a", "pcm_s16le", tmp],
                   project.duration, None, cancel, cwd=workdir)
        os.replace(tmp, out_wav)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return out_wav
