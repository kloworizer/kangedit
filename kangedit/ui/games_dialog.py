"""Tools → Label the gaming plays: the words shown at a double, triple or bigger multi-kill (see games.py)."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

from ..games import LABEL_WORDS


class PlayLabelsDialog(QDialog):
    """The label words (one per play size; {n}: how many bursts) and whether single plays get one too."""

    def __init__(self, words: dict, singles: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label the gaming plays")
        lay = QVBoxLayout(self)
        intro = QLabel("A title goes where several plays (kills, hits, big sounds and flashes) come one after another "
                       "in the edit. Labelling again replaces the labels made before; Undo takes them off.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.edits: dict[int, QLineEdit] = {}
        for n, label in ((2, "Two in a row:"), (3, "Three in a row:"), (4, "Four or more ({n} = how many):"),
                         (1, "A single play:")):
            e = QLineEdit(str(words.get(n, LABEL_WORDS[n])))
            e.setPlaceholderText(LABEL_WORDS[n])
            self.edits[n] = e
            form.addRow(label, e)
        lay.addLayout(form)
        self.chk_singles = QCheckBox("Also label single plays")
        self.chk_singles.setChecked(singles)
        self.chk_singles.toggled.connect(self.edits[1].setEnabled)
        self.edits[1].setEnabled(singles)
        lay.addWidget(self.chk_singles)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Label the plays")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def words(self) -> dict[int, str]:
        return {n: (e.text().strip() or LABEL_WORDS[n]) for n, e in self.edits.items()}

    def singles(self) -> bool:
        return self.chk_singles.isChecked()
