"""Shared numeric guards for the core pipeline.

Real scans occasionally contain zero or slightly negative pixels (sensor noise floor,
in-camera black-level clipping). log/power/reciprocal operations are undefined there, so every
stage that touches raw transmittance clamps to this floor first. It is a numerical safety net,
not a content decision — it never engages for well-exposed scans.
"""

MIN_TRANSMITTANCE = 1e-7
