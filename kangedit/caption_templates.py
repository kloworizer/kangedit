"""Caption style templates: one-click caption looks, built in and saved by the user.

A template is a set of CaptionStyle fields (font, size, colours, animation...). Applying one changes only
the look: whether captions are shown and the speakers setting stay as they are, and the speech model and
language are project settings, not part of the style at all. The user's own templates are kept in the app
settings (as JSON), so they are there in every project.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields

from .models import CaptionStyle

SETTINGS_KEY = "captions/templates"

# The style fields a template sets (not `enabled`, nor the speakers detected for this project).
TEMPLATE_FIELDS = tuple(f.name for f in fields(CaptionStyle) if f.name not in ("enabled", "speakers",
                                                                                   "speaker_colors"))

# Built-in looks. Every combination is one the preview and the export both draw (captions.ANIMATIONS).
BUILTIN: dict[str, dict] = {
    "Clean subtitles": dict(font="Arial", color="#FFFFFF", outline=True, box=False, animation="none",
                            uppercase=False, position="bottom", auto_layout=True),
    "Bold pop": dict(font="Arial", color="#FFFFFF", highlight="#FFD400", outline=True, box=False,
                     animation="pop", uppercase=True, auto_layout=False, position="middle", size_pct=7.5,
                     max_chars=18, max_lines=2, max_duration=2.0),
    "Karaoke": dict(font="Arial", color="#FFFFFF", highlight="#FFD400", outline=True, box=False,
                    animation="highlight", uppercase=False, auto_layout=True),
    "Highlight box": dict(font="Arial", color="#FFFFFF", highlight="#7C4DFF", outline=True, box=False,
                          animation="box", uppercase=True, auto_layout=True),
    "Boxed": dict(font="Arial", color="#FFFFFF", outline=False, box=True, animation="none", uppercase=False,
                  position="bottom", auto_layout=True),
    "Minimal lower": dict(font="Arial", color="#F2F2F2", outline=True, box=False, animation="none",
                          uppercase=False, auto_layout=False, position="bottom", size_pct=4.0, max_chars=42,
                          max_lines=2, max_duration=5.0),
    "One word at a time": dict(font="Arial", color="#FFFFFF", outline=True, box=False, animation="word",
                               uppercase=True, auto_layout=True),
}


def style_fields(style: CaptionStyle) -> dict:
    """The template fields of a style (what "Save current style as…" keeps)."""
    d = asdict(style)
    return {k: d[k] for k in TEMPLATE_FIELDS}


def apply_template(style: CaptionStyle, values: dict) -> None:
    """Set a template's fields on a style (unknown keys, and the ones a template never sets, ignored)."""
    for k, v in values.items():
        if k in TEMPLATE_FIELDS:
            setattr(style, k, v)


def user_templates() -> dict[str, dict]:
    """The templates the user saved, by name (empty when none or the setting can't be read)."""
    from .config import settings
    try:
        raw = settings().value(SETTINGS_KEY, "")
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _store(data: dict[str, dict]) -> None:
    from .config import settings
    s = settings()
    s.setValue(SETTINGS_KEY, json.dumps(data, ensure_ascii=False))
    s.sync()


def save_user_template(name: str, style: CaptionStyle) -> str:
    """Save the style's look under `name` (replacing a saved one of that name). Returns the name used."""
    name = name.strip()
    if not name:
        raise ValueError("A template needs a name")
    if name in BUILTIN:
        raise ValueError(f"“{name}” is a built-in template: choose another name")
    data = user_templates()
    data[name] = style_fields(style)
    _store(data)
    return name


def delete_user_template(name: str) -> bool:
    """Remove a saved template (the built-in ones can't be). Returns whether there was one."""
    data = user_templates()
    if name not in data:
        return False
    del data[name]
    _store(data)
    return True


def all_templates() -> dict[str, dict]:
    """Every template by name: the built-in ones first, then the user's."""
    out = dict(BUILTIN)
    for name, values in user_templates().items():
        out.setdefault(name, values)
    return out


def is_builtin(name: str) -> bool:
    return name in BUILTIN
