"""A project's health check before export: what would make the export fail or come out other than meant,
each with a fix where there is one (see ui/health_dialog.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import Project

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass
class Issue:
    level: str          # ERROR stops the export; WARNING and INFO don't
    text: str
    fix: str = ""       # what fixes it (see HealthDialog.apply), "" = nothing to do here
    fix_label: str = ""


def _speaks(f) -> bool:
    return f is not None and f.has_audio and not f.is_image


def check(project: Project, captions: str = "none", use_proxies: bool = False) -> list[Issue]:
    """The project's issues for an export (captions: the export's caption choice; use_proxies: a draft from
    the proxies), errors first."""
    from . import motion, stabilize
    out: list[Issue] = []
    if not project.clips:
        return [Issue(ERROR, "The timeline is empty.")]
    used = list(project.clips) + list(project.overlays)
    for fid in dict.fromkeys(c.footage_id for c in used):
        f = project.footage.get(fid)
        if f is not None and not os.path.exists(f.path) and not (use_proxies and f.has_proxy()):
            out.append(Issue(ERROR, f"Missing file: {f.name} ({f.path}) is used in the edit but isn't there "
                                    "(moved, renamed or on a drive that isn't plugged in).", f"locate:{fid}",
                             "Find the file…"))
    gone = [x for x in project.sounds if not os.path.exists(x.path)]
    if gone:
        out.append(Issue(WARNING, f"{len(gone)} sound(s) or music track(s) are missing and won't be heard: "
                                  + ", ".join(x.name or os.path.basename(x.path) for x in gone[:4]),
                         "drop_sounds", "Take them off"))
    if captions != "none":
        todo = {c.footage_id for c in project.clips if _speaks(project.footage.get(c.footage_id))
                and project.footage[c.footage_id].transcript is None}
        if todo:
            out.append(Issue(WARNING, f"Captions are on, but {len(todo)} file(s) in the edit have no transcript "
                                      "yet: their speech gets no captions.", "transcribe", "Transcribe them"))
    if stabilize.missing(project):
        out.append(Issue(INFO, "Stabilized clips whose camera shake isn't measured yet: the export measures it "
                               "first (a little longer).", "measure", "Measure now"))
    faces = [c for c in used if c.motion in motion.FACE_KINDS and c.motion_face is None]
    if faces:
        out.append(Issue(INFO, f"{len(faces)} clip(s) zoom on a face not found yet: they zoom to the middle.",
                         "faces", "Find the faces"))
    heard = any(_speaks(project.footage.get(c.footage_id)) and c.volume > 0.001 and c.audio_mode != "mute"
                for c in used) or any(os.path.exists(x.path) and x.volume > 0.001 for x in project.sounds)
    if not heard:
        out.append(Issue(WARNING, "The video has no sound: no clip has sound (or all are muted) and there is no "
                                  "music or sound effect."))
    muted = [c for c in project.clips if _speaks(project.footage.get(c.footage_id))
             and (c.volume <= 0.001 or c.audio_mode == "mute")]
    if muted and heard:
        out.append(Issue(INFO, f"{len(muted)} clip(s) with sound are muted (volume 0)."))
    late = [x for x in project.texts if x.start >= project.duration - 0.05]
    if late:
        out.append(Issue(WARNING, f"{len(late)} title(s) start after the end of the video and won't show.",
                         "drop_texts", "Take them off"))
    if use_proxies:
        out.append(Issue(WARNING, "This is a draft from the proxies: smaller and softer than the footage.",
                         "originals", "Export from the originals"))
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(out, key=lambda i: order[i.level])
