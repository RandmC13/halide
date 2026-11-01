"""The line under the picture: what is true of the roll, as chips that stay, and what is happening now,
as dim text that fades. A chip opens a short list over the bottom of the picture (inside the window:
a separate pop-up can't be placed reliably on Wayland), and a row there goes to its frame. Errors
are red chips that stay until dismissed or replaced, so the next message can't push them out."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from halide.gui import theme
from halide.gui.widgets import ElidedLabel

ACTIVITY_MS = 6000  # how long "what just happened" stays before it fades
_PANEL_ROWS = 8  # rows a notice's list shows before "…and N more"


@dataclass
class Notice:
    chip: str  # the chip's text: "⚠ 17 scanned differently"
    title: str  # the list's heading: "17 frames were scanned differently from most of the roll"
    rows: list[tuple[str, str, int | None]] = field(default_factory=list)  # (label, text, frame index or None)
    error: bool = False
    footer: str = ""  # a last, dim line: what to do about it


def _chip_style(error: bool, open_: bool) -> str:
    colour = theme.AGREE_RED if error else theme.TEXT_WARNING
    border = colour if open_ else theme.BORDER
    background = theme.BORDER if open_ else theme.BACKGROUND_ALT
    return (
        f"QPushButton {{ color: {colour}; background: {background}; border: 1px solid {border};"
        f" border-radius: 10px; padding: 1px 9px; font-weight: 600; }}"
        f"QPushButton:hover {{ border-color: {colour}; }}"
    )


class _Panel(QFrame):
    """The open notice's list, laid over the bottom of the picture."""

    frameRequested = Signal(int)
    closed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("noticePanel")
        self.setStyleSheet(
            f"QFrame#noticePanel {{ background: {theme.BACKGROUND_ALT}; border: 1px solid {theme.BORDER};"
            f" border-radius: 4px; }}"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(3)
        self.hide()

    def show_notice(self, notice: Notice) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
                item.widget().deleteLater()
            elif item.layout() is not None:
                while item.layout().count():
                    child = item.layout().takeAt(0).widget()
                    if child is not None:
                        child.setParent(None)
                        child.deleteLater()
        head = QHBoxLayout()
        title = QLabel(notice.title)
        title.setWordWrap(True)
        title.setStyleSheet(f"color: {theme.AGREE_RED if notice.error else theme.TEXT}; font-weight: 600;")
        head.addWidget(title, 1)
        close = QPushButton("✕")
        close.setProperty("role", "remove")
        close.setToolTip("Close this list")
        close.setFixedWidth(22)
        close.clicked.connect(self.closed)
        head.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        self._layout.addLayout(head)
        for label, text, frame in notice.rows[:_PANEL_ROWS]:
            self._layout.addWidget(self._row(label, text, frame))
        extra = len(notice.rows) - _PANEL_ROWS
        footer = " · ".join(t for t in (f"…and {extra} more" if extra > 0 else "", notice.footer) if t)
        if footer:
            note = QLabel(footer)
            note.setWordWrap(True)
            note.setProperty("role", "dim")
            self._layout.addWidget(note)
        self.show()
        self.raise_()

    def _row(self, label: str, text: str, frame: int | None) -> QWidget:
        row = QPushButton() if frame is not None else QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(10)
        name = QLabel(label)
        name.setFont(theme.edge_font(12))
        name.setStyleSheet(f"color: {theme.EDGE_PRINT};")
        name.setFixedWidth(name.fontMetrics().horizontalAdvance("FRAME 00") + 10)
        layout.addWidget(name)
        body = ElidedLabel(text)  # one line a row, the whole reason in its tooltip
        layout.addWidget(body, 1)
        if frame is not None:
            row.setFlat(True)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            row.setToolTip("Go to this frame")
            row.setStyleSheet("QPushButton { text-align: left; border: none; padding: 0; min-height: 0; }"
                              f"QPushButton:hover {{ background: {theme.BORDER}; }}")
            row.setFixedHeight(name.sizeHint().height() + 4)
            row.clicked.connect(lambda: self.frameRequested.emit(frame))
        return row


class NoticeBar(QWidget):
    """Chips on the left (`set_notice`/`clear_notice`), activity on the right (`show_activity`)."""

    frameRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._notices: dict[str, Notice] = {}
        self._chips: dict[str, QPushButton] = {}
        self._open: str | None = None
        self._host: QWidget | None = None
        self._panel: _Panel | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._chip_row = QHBoxLayout()
        self._chip_row.setSpacing(6)
        row.addLayout(self._chip_row)
        self.activity = ElidedLabel("")
        self.activity.setProperty("role", "dim")
        self.activity.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.activity, 1)
        self.setFixedHeight(24)
        self._fade = QTimer(self)
        self._fade.setSingleShot(True)
        self._fade.timeout.connect(lambda: self.activity.set_full_text(""))

    def set_host(self, host: QWidget) -> None:
        """The widget the open list is laid over (the picture), at its bottom edge."""
        self._host = host
        self._panel = _Panel(host)
        self._panel.frameRequested.connect(self._on_frame)
        self._panel.closed.connect(self.close_list)

    # --- activity ---------------------------------------------------------------------------------

    def show_activity(self, text: str, lasting: bool = False) -> None:
        """What is happening now. It fades after a few seconds unless `lasting` (a loading line that
        the next one replaces)."""
        self.activity.set_full_text(text)
        if lasting or not text:
            self._fade.stop()
        else:
            self._fade.start(ACTIVITY_MS)

    # --- notices ----------------------------------------------------------------------------------

    def notices(self) -> dict[str, Notice]:
        return dict(self._notices)

    def set_notice(self, key: str, notice: Notice) -> None:
        self._notices[key] = notice
        chip = self._chips.get(key)
        if chip is None:
            chip = QPushButton()
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda: self.toggle(key))
            self._chips[key] = chip
            self._chip_row.addWidget(chip)
        chip.setText(notice.chip)
        chip.setToolTip(notice.title)
        chip.setStyleSheet(_chip_style(notice.error, key == self._open))
        if key == self._open and self._panel is not None:
            self._panel.show_notice(notice)
            self._place_panel()

    def clear_notice(self, key: str) -> None:
        self._notices.pop(key, None)
        chip = self._chips.pop(key, None)
        if chip is not None:
            chip.setParent(None)
            chip.deleteLater()
        if self._open == key:
            self.close_list()

    def clear_errors(self) -> None:
        for key in [k for k, n in self._notices.items() if n.error]:
            self.clear_notice(key)

    def clear_all(self) -> None:
        for key in list(self._notices):
            self.clear_notice(key)

    def toggle(self, key: str) -> None:
        if self._open == key:
            self.close_list()
            return
        self.close_list()
        if key not in self._notices or self._panel is None:
            return
        self._open = key
        self._chips[key].setStyleSheet(_chip_style(self._notices[key].error, True))
        self._panel.show_notice(self._notices[key])
        self._place_panel()

    def close_list(self) -> None:
        key, self._open = self._open, None
        if key in self._chips:
            self._chips[key].setStyleSheet(_chip_style(self._notices[key].error, False))
        if self._panel is not None:
            self._panel.hide()
        if key is not None and key.startswith("error") and key in self._notices:
            self.clear_notice(key)  # an error, once read, is done with

    def list_open(self) -> str | None:
        return self._open

    def _place_panel(self) -> None:
        if self._panel is None or self._host is None:
            return
        width = min(self._host.width() - 24, 620)
        self._panel.setFixedWidth(width)
        layout = self._panel.layout()
        layout.activate()
        # Wrapped text: ask for the height at this width, not the unwrapped size hint.
        wanted = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else layout.totalSizeHint().height()
        height = min(wanted, self._host.height() - 24)
        self._panel.setGeometry(12, self._host.height() - height - 12, width, height)

    def _on_frame(self, index: int) -> None:
        self.close_list()
        self.frameRequested.emit(index)
