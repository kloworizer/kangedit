"""Test setup: settings and the LUT/sound libraries go to a throwaway folder, Qt runs offscreen."""

import os
import tempfile

import pytest

# Before anything imports kangedit: never touch the user's real settings or libraries.
_TMP = tempfile.mkdtemp(prefix="kangedit-tests-")
os.environ["KANGEDIT_SETTINGS_DIR"] = os.path.join(_TMP, "settings")
os.environ["KANGEDIT_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    from kangedit.ui.theme import STYLE
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")   # as the app does (kangedit/app.py): sizes match on every platform
    app.setStyleSheet(STYLE)
    return app


@pytest.fixture
def lut_library():
    """The (test) LUT library folder, emptied before and after the test."""
    from kangedit import lut as lutlib
    from kangedit.config import lut_dir

    def clear():
        for p in lut_dir().iterdir():
            p.unlink()
        lutlib.set_official_choices({})

    clear()
    yield lut_dir()
    clear()


def write_cube(path, size=2, fn=lambda r, g, b: (r, g, b)):
    """A size^3 .cube file mapping (r, g, b) -> fn(r, g, b)."""
    lines = [f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append("%.6f %.6f %.6f" % fn(r / (size - 1), g / (size - 1), b / (size - 1)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def no_blocking_message_boxes(monkeypatch):
    """Tests run unattended: a warning / error / information box (e.g. "Analysis failed" from a background
    task finishing late, with no FFmpeg on the CI runners) must not wait for a click. They are recorded instead.
    A yes / no question answers No unless the test sets its own answer: an unexpected one (such as the offer to
    copy an older cache folder, 0.8 s after a window opens) would otherwise wait for a click forever."""
    shown = []
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:          # Qt libraries missing: the non-UI tests still run
        yield shown
        return
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, _n=name, **k: shown.append((_n, a[1:3])) or QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: shown.append(("question", a[1:3])) or QMessageBox.No))
    try:                        # the offer to transcribe after an import: declined
        from kangedit.ui.main_window import MainWindow
        monkeypatch.setattr(MainWindow, "_ask_transcribe",
                            lambda self, n: shown.append(("transcribe", n)) or (False, False))
    except ImportError:
        pass
    yield shown
