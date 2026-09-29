# Extra findings (orchestrator, while checking 2.7)

### 8x-1: Contact-sheet and GUI typography depend on system fonts that macOS doesn't have
- Severity: S4
- Where: src/halide/io/contact_sheet.py:110-122 (DejaVu/Liberation looked up by file name), src/halide/gui/theme.py:48 (`font-family: "DejaVu Sans"`)
- Evidence: `_font` tries DejaVuSans(-Bold)/LiberationSans by bare file name, then `ImageFont.load_default(size=)` (Pillow's built-in font: no bold). Neither DejaVu nor Liberation ships with macOS (or Windows), so there the orange edge print loses its bold film-lettering look and the GUI falls back to whatever sans-serif Qt picks. The contact sheet's appearance (R-094, R-016) then depends on the machine it runs on.
- Requirement: R-016, R-094
- Suggestion: vendor one font family (DejaVu is under a permissive licence; or a condensed typeface chosen for the edge print) in src/halide/assets/fonts, load it by path in both places, add its licence next to the others (S).
- Confidence: likely (code read; not run on macOS)

### 8x-2: No dependency has a version range, and the ICC/sRGB code rebuilds colour-science's internals
- Severity: S2 (future silent colour change / breakage on a fresh install)
- Where: pyproject.toml [project].dependencies; src/halide/io/icc.py:~205-230 (`matrix_chromatic_adaptation_VonKries`, `xy_to_xyY`, `RGB_COLOURSPACES`, `CCS_ILLUMINANTS`), io/raster.py srgb_matrix
- Evidence: every dependency is bare ("numpy", "colour-science", …). working_space_matrices() reproduces how colour.XYZ_to_RGB builds its matrices internally (D1), so a colour-science release that changes those internals or defaults (CAT, illuminant table keys) changes halide's colour or breaks import, with nothing to stop a new user getting it. test_icc cross-validates only against the installed version.
- Requirement: R-001, R-010
- Suggestion: lower bounds for everything plus an upper bound (next major) on colour-science and numpy; record the tested versions in the README (S).
- Confidence: likely
