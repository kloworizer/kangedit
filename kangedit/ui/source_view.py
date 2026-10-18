"""The viewer's Source mode: a footage file from the Media panel played on its own, with a mark in and out set at
the playhead (I / O) or on the words of its transcript (select words: the part from the first to the last), and
the marked part put on the timeline."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from .theme import fmt_time

MARKED = QColor(70, 120, 200, 110)       # words inside the marks
SPOKEN = QColor(240, 180, 41, 150)       # the word at the playhead


class SourceBar(QWidget):
    """The whole file as a bar: the marked part shaded, the playhead; click or drag to go there."""
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 1.0
        self.t = 0.0
        self.mark_in: Optional[float] = None
        self.mark_out: Optional[float] = None
        self.setFixedHeight(22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("The whole file: click or drag to go there. The shaded part is between the marks.")

    def x_of(self, t: float) -> float:
        return 4 + (self.width() - 8) * max(0.0, min(1.0, t / max(1e-6, self.duration)))

    def t_of(self, x: float) -> float:
        return max(0.0, min(self.duration, (x - 4) / max(1.0, self.width() - 8) * self.duration))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#3a3d44"))
        p.drawRoundedRect(QRectF(4, 7, self.width() - 8, 8), 3, 3)
        if self.mark_in is not None or self.mark_out is not None:
            a = self.mark_in if self.mark_in is not None else 0.0
            b = self.mark_out if self.mark_out is not None else self.duration
            p.setBrush(QColor(70, 120, 200))
            p.drawRect(QRectF(self.x_of(a), 7, max(2.0, self.x_of(b) - self.x_of(a)), 8))
            p.setPen(QPen(QColor("#9ec1ff"), 2))
            for t in (self.mark_in, self.mark_out):
                if t is not None:
                    p.drawLine(int(self.x_of(t)), 3, int(self.x_of(t)), 19)
        p.setPen(QPen(QColor("#ff5252"), 2))
        x = int(self.x_of(self.t))
        p.drawLine(x, 1, x, self.height() - 1)

    def mousePressEvent(self, e):
        self.seek.emit(self.t_of(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.seek.emit(self.t_of(e.position().x()))


class WordTranscript(QTextEdit):
    """A file's transcript, word by word: select words to mark that part (from the first word's start to the
    last's end); click a word to go there. The marked words and the word at the playhead are highlighted."""
    marked = Signal(float, float)        # source in, out
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(96)
        self.words: list[dict] = []
        self.spans: list[tuple[int, int]] = []          # each word's characters in the text
        self._marks: tuple[Optional[float], Optional[float]] = (None, None)
        self._t = 0.0
        self.selectionChanged.connect(self._selected)
        self.setToolTip("What is said in the file. Select words (drag over them) to mark that part; click a word "
                        "to go there.")

    def set_words(self, words: Optional[list[dict]]) -> None:
        self.words = list(words or [])
        self.spans = []
        text = []
        pos = 0
        for w in self.words:
            s = str(w.get("w", "")).strip()
            self.spans.append((pos, pos + len(s)))
            text.append(s)
            pos += len(s) + 1
        self.blockSignals(True)
        self.setPlainText(" ".join(text))
        self.blockSignals(False)
        self._paint()

    def word_at(self, t: float) -> int:
        """The word being said at source time t (the last started), -1 before the first."""
        k = -1
        for i, w in enumerate(self.words):
            if float(w["s"]) <= t + 1e-6:
                k = i
            else:
                break
        return k

    def show_state(self, t: float, mark_in: Optional[float], mark_out: Optional[float]) -> None:
        if (t, mark_in, mark_out) != (self._t, *self._marks):
            self._t, self._marks = t, (mark_in, mark_out)
            self._paint()

    def _paint(self) -> None:
        sels = []
        a, b = self._marks
        if a is not None or b is not None:
            a = a if a is not None else 0.0
            b = b if b is not None else float("inf")
            for w, (c0, c1) in zip(self.words, self.spans):
                if float(w["s"]) >= a - 0.01 and float(w["e"]) <= b + 0.01:
                    sels.append(self._sel(c0, c1, MARKED, False))
        k = self.word_at(self._t)
        if 0 <= k < len(self.spans) and float(self.words[k]["e"]) >= self._t - 0.3:
            sels.append(self._sel(*self.spans[k], SPOKEN, True))
        self.setExtraSelections(sels)

    def _sel(self, c0: int, c1: int, color: QColor, bold: bool):
        s = QTextEdit.ExtraSelection()
        cur = QTextCursor(self.document())
        cur.setPosition(c0)
        cur.setPosition(c1, QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        fmt.setBackground(QBrush(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        s.cursor, s.format = cur, fmt
        return s

    def _words_in(self, c0: int, c1: int) -> list[int]:
        return [i for i, (a, b) in enumerate(self.spans) if a < c1 and b > c0]

    def _selected(self) -> None:
        cur = self.textCursor()
        if not cur.hasSelection():
            return
        got = self._words_in(cur.selectionStart(), cur.selectionEnd())
        if got:
            self.marked.emit(float(self.words[got[0]]["s"]), float(self.words[got[-1]]["e"]))

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        cur = self.textCursor()
        if not cur.hasSelection():
            got = self._words_in(cur.position() - 1, cur.position() + 1)
            if got:
                self.seek.emit(float(self.words[got[0]]["s"]))


class SourcePanel(QWidget):
    """Under the viewer in Source mode: the file's bar, the marks and what to do with them, and its words."""
    mark = Signal(str)                 # "in" | "out" | "clear"
    use = Signal(str)                  # "insert" | "append": put the marked part on the timeline
    transcribe = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bar = SourceBar()
        self.words = WordTranscript()
        b_in = QPushButton("⟦ In")
        b_in.setToolTip("Mark the start of the part to use at the playhead (I)")
        b_in.clicked.connect(lambda: self.mark.emit("in"))
        b_out = QPushButton("Out ⟧")
        b_out.setToolTip("Mark the end of the part to use at the playhead (O)")
        b_out.clicked.connect(lambda: self.mark.emit("out"))
        b_clear = QPushButton("Clear")
        b_clear.setToolTip("Clear the marks: the whole file")
        b_clear.clicked.connect(lambda: self.mark.emit("clear"))
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("font-weight: 600;")
        self.b_insert = QPushButton("⤓ Insert at playhead")
        self.b_insert.setToolTip("Put the marked part on the timeline where its playhead is (between the clips "
                                 "nearest to it)")
        self.b_insert.clicked.connect(lambda: self.use.emit("insert"))
        self.b_append = QPushButton("Append")
        self.b_append.setToolTip("Put the marked part at the end of the timeline")
        self.b_append.clicked.connect(lambda: self.use.emit("append"))
        self.b_transcribe = QPushButton("Transcribe this file")
        self.b_transcribe.clicked.connect(self.transcribe.emit)
        self.lbl_words = QLabel("")
        self.lbl_words.setStyleSheet("color: #9aa0ab;")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for w in (b_in, b_out, b_clear):
            w.setFocusPolicy(Qt.NoFocus)
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.b_insert, self.b_append):
            w.setFocusPolicy(Qt.NoFocus)
            row.addWidget(w)
        words_row = QHBoxLayout()
        words_row.setContentsMargins(0, 0, 0, 0)
        words_row.addWidget(self.lbl)
        words_row.addWidget(self.lbl_words, 1)
        words_row.addWidget(self.b_transcribe)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(self.bar)
        lay.addLayout(row)
        lay.addLayout(words_row)
        lay.addWidget(self.words)

    def show_file(self, footage) -> None:
        self.bar.duration = max(0.01, footage.duration)
        words = footage.transcript or []
        self.words.set_words(words)
        self.words.setVisible(bool(words))
        self.b_transcribe.setVisible(footage.transcript is None and footage.has_audio and not footage.is_image)
        if words:
            self.lbl_words.setText("· select words to mark that part, click one to go there")
        elif footage.is_image or not footage.has_audio:
            self.lbl_words.setText("No sound in this file.")
        elif footage.transcript is None:
            self.lbl_words.setText("No transcript yet: transcribe it to mark parts by their words.")
        else:
            self.lbl_words.setText("Nobody speaks in this file.")

    def show_state(self, t: float, mark_in: Optional[float], mark_out: Optional[float], fps: float) -> None:
        self.bar.t, self.bar.mark_in, self.bar.mark_out = t, mark_in, mark_out
        self.bar.update()
        self.words.show_state(t, mark_in, mark_out)
        a = mark_in if mark_in is not None else 0.0
        b = mark_out if mark_out is not None else self.bar.duration
        self.lbl.setText(f"In {fmt_time(a, fps)} · Out {fmt_time(b, fps)} · {max(0.0, b - a):.2f} s"
                         + ("" if mark_in is not None or mark_out is not None else " (the whole file)"))
