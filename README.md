# halide

**Turn camera or scanner captures of colour negative film into faithful positive images.**

halide inverts colour negatives the way a darkroom print does: it balances the film's orange mask
and its three dye layers, inverts, and prints the result through the measured response curve of a
real photographic paper. The goal is colour that matches what the film actually recorded, not a
"look". Once a film stock is calibrated, a whole roll develops consistently, with no per-frame
tweaking.

- **Calibrate once, reuse everywhere.** Click neutral objects (a white wall, a grey road, a black
  tyre) on any frames of a roll in a small GUI. halide fits them together, shows how well each point
  agrees with the rest in CC filter values, and saves the result as a named profile for that film
  stock, process and scanner.
- **Real paper, fitted per frame.** The default output is printed through a Kodak Endura paper
  curve. Exposure and contrast grade are fitted to each negative, like a printer choosing exposure
  and paper grade on the enlarger. You can also pin them yourself.
- **Colour managed end to end.** Input is a linear TIFF with an embedded ICC profile. halide works
  in ACEScg and writes tagged ACEScg TIFFs, then exports sRGB PNG/JPEG for sharing.
- **Built for whole rolls.** Parallel batch processing sized to your machine's memory, scan
  consistency checks (shutter speed, white balance, edits), and contact sheets styled after a real
  contact print.
- **A flat output for editing.** `--output flat` gives an unclipped linear positive for editing in
  darktable, and `halide print` prints it onto the paper afterwards.

## Install

The easiest way is [uv](https://docs.astral.sh/uv/). It installs halide as a command you can run
from anywhere, and fetches a suitable Python for it if you don't have one. You also need
[git](https://git-scm.com/downloads).

**1. Install uv** (skip this if you already have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                        # macOS and Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

**2. Install halide:**

```bash
uv tool install git+https://github.com/RandmC13/halide
```

**3. Check it worked:**

```bash
halide --help
```

If your terminal says `halide` isn't found, run `uv tool update-shell` and open a new terminal.

**Tab completion** sets itself up in zsh, bash and fish. The first time you run halide in a
terminal, it prints a one-line note. From the next terminal you open, Tab completes commands,
flags, saved profile names, and scans (TIFFs only where a scan is expected, folders where a roll
is). It stays up to date by itself when halide gains new flags. For zsh and bash, halide adds a
short, marked block to `~/.zshrc` or `~/.bashrc` (`~/.bash_profile` on macOS). fish needs no
edit: halide adds `~/.config/fish/completions/halide.fish`. To opt out, delete that block or file
(it won't come back), or set `HALIDE_NO_COMPLETION=1`. Other shells, such as PowerShell on
Windows, don't get completion yet.

Later, `uv tool upgrade halide` updates to the latest version and `uv tool uninstall halide`
removes it. Use the full GitHub address above: a different, unrelated package called `halide` is on
PyPI, so `uv tool install halide` on its own installs the wrong thing.

halide is developed and tested on Linux. It is plain Python (numpy, Qt via PySide6), so macOS and
Windows should work, but they haven't been tested yet.

<details>
<summary>Without uv (pip)</summary>

With Python 3.11 or newer:

```bash
git clone https://github.com/RandmC13/halide.git
cd halide
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install .
```

The `halide` command then works whenever that virtual environment is active.
</details>

**Optional:** install [exiftool](https://exiftool.org/) (`sudo apt install libimage-exiftool-perl`,
`brew install exiftool`) so developed images keep their camera metadata. Without it halide still
works; it just doesn't copy EXIF.

<details>
<summary>Linux: the calibration window won't open?</summary>

Minimal Linux installs can be missing libraries Qt needs. On Debian/Ubuntu:

```bash
sudo apt install libegl1 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 \
                 libxcb-shape0 libxcb-xkb1
```
</details>

## Preparing your scans

halide reads **linear TIFFs with an embedded ICC profile**, the kind a raw developer exports. It
doesn't read raw files directly: your raw developer already handles demosaicing, cropping, lens
correction and dust removal well.

In darktable or RawTherapee, for every frame of the roll:

1. Crop to the image, leaving out the film holder and rebate. An opaque edge left in the crop
   throws off the print fit.
2. Leave tone and colour modules **off** (filmic, sigmoid, curves, local contrast, and so on). The
   export has to stay linear.
3. Export a 16-bit or 32-bit float TIFF in a **linear** wide-gamut profile, such as darktable's
   *linear Rec2020 RGB*, with the profile embedded.
4. Keep camera settings and white balance the same across the roll if you can. `halide check`
   shows you whether you did.

If a file isn't suitable (gamma-encoded, no profile, the wrong kind of profile), halide rejects it
and explains why.

## Quick start

A typical roll:

```bash
# 1. Check the scans were made consistently (reads file headers only, so it's fast)
halide check roll16/

# 2. Pick neutral points across the roll and save them as a profile
halide calibrate roll16/

# 3. Develop the whole roll with that profile
halide batch roll16/ roll16-developed/ --profile "Portra 400"

# 4. Make sRGB copies for sharing, plus a contact sheet
halide export roll16-developed/ roll16-jpeg/ --format jpg
halide contact roll16-developed/ roll16-sheet.jpg
```

Run `invert` or `batch` without a calibration flag and halide offers your saved profiles, the
picker, or an automatic estimate. For a single frame:

```bash
halide invert IMG_0158.tif IMG_0158-positive.tif --pick    # click neutrals on this frame
```

Want to try settings before committing disk space? `halide batch roll16/ --contact-sheet
preview.jpg` develops every frame at full quality but keeps only the contact sheet.

## Commands

| Command | What it does |
|---|---|
| `halide calibrate [roll]` | GUI: pick neutral points on any frames, see a live positive, save a profile |
| `halide invert IN OUT` | Develop one negative (`--profile`, `--pick`, `--auto-density`, `--output flat`, …) |
| `halide batch IN_DIR [OUT_DIR]` | Develop a folder in parallel; `--contact-sheet` for a preview or proof |
| `halide print IN OUT` | Print a flat positive, for example after editing it in darktable |
| `halide export IN OUT` | ACEScg TIFF (or folder of them) to sRGB PNG/JPEG |
| `halide contact DIR SHEET` | Contact sheet of developed frames |
| `halide check DIR` | Report inconsistent scan settings across a roll |
| `halide profile list\|show\|edit\|rename\|delete` | Manage saved profiles (stored in `~/.config/halide/profiles/`) |

Every command has `--help`. Every output TIFF records how it was printed (profile, grade,
exposure) in its metadata, and `invert` prints it too.

## How it works

The method is Aaron Buchler's, from
[*Scanning Color Negative Film*](https://abpy.github.io/2023/08/20/color-neg.html):

1. **White balance.** A per-channel multiply that neutralises the film base and light source.
2. **Density balance.** A per-channel power function that equalises the contrast of the three
   dye layers, so neutrals stay neutral from shadows to highlights.
3. **Invert.** Take the reciprocal, as the enlarger and paper would.
4. **Print.** Map the result through a measured paper characteristic curve, with exposure and
   grade fitted to the frame's density range.

Steps 1 and 2 are the calibration: two numbers per channel, solved from the neutral points you
pick. All of this runs in linear ACEScg, because the result depends on the working space.

The reasoning behind each design choice, with measurements from real rolls, is recorded in
[`CLAUDE.md`](CLAUDE.md) (under "Decisions and why"). Longer plans and investigations are in
[`docs/`](docs/).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Code layout: `src/halide/core/` holds the pure image maths, `io/` handles TIFF/ICC/export,
`calibration/` handles profiles and fitting, `batch/` does parallel processing, `cli/` holds the
commands and `gui/` the Qt picker. `legacy/` keeps the original single-file script for comparison.

## Acknowledgements

halide exists because of other people's generous, openly shared work.

- **[Aaron Buchler](https://github.com/abpy)**. halide implements the method from his post
  [*Scanning Color Negative Film*](https://abpy.github.io/2023/08/20/color-neg.html), a careful,
  colorimetrically grounded account of how colour negative film records light and how to reverse
  it. The paper curve halide prints through and the reference LUTs its tests check against come
  from his [color-neg-resources](https://github.com/abpy/color-neg-resources) (MIT). Its density
  balance is verified against his reference script.
- **[Alchemy Color](https://www.alchemycolor.com)**, whose
  [color-negative-inversion](https://github.com/alchemy-color/color-negative-inversion) workflow
  builds on Aaron's method and was a second reference for this implementation.
- **[Elle Stone](https://github.com/ellelstone/elles_icc_profiles)**, for the well-behaved linear
  ACEScg ICC profile (`ACEScg-elle-V4-g10`) that halide tags its output with (CC BY-SA 3.0).
- The open-source libraries it's built on, especially
  [colour-science](https://www.colour-science.org/), [numpy](https://numpy.org/),
  [tifffile](https://github.com/cgohlke/tifffile), [PySide6/Qt](https://doc.qt.io/qtforpython/)
  and [Pillow](https://python-pillow.org/). Thanks also to [darktable](https://www.darktable.org/)
  and [RawTherapee](https://rawtherapee.com/), which produce the scans halide starts from.

## License

halide is free software, released under the [GNU General Public License v3.0 or later](LICENSE).
You can use, study, share and modify it. If you distribute a modified version, it has to stay open
under the same licence.

The bundled third-party files keep their own licences, which sit next to them:
`src/halide/assets/tone_curves/` and `tests/golden/luts/` (MIT, Aaron Buchler), and
`src/halide/assets/icc_profiles/` (CC BY-SA 3.0, Elle Stone).
