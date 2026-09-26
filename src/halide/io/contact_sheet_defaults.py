"""Contact sheet defaults, kept apart from io/contact_sheet.py (which needs numpy and Pillow) so
the CLI can show them in its flags' help without importing either."""

DEFAULT_FRAME_WIDTH = 900  # px per frame cell — a 6-across sheet is ~6000 px wide
DEFAULT_COLUMNS = 6
