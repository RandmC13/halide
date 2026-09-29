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

Tested with numpy 2.5, colour-science 0.4.7, tifffile 2026.9, PySide6 6.11, Pillow 12.3, psutil 7.2 and
shtab 1.12. Older versions within halide's declared ranges should work but aren't tested.

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

Every frame needs the same treatment, in whichever raw developer you use.

### Exporting your scans

The menu names below are from memory of recent darktable and RawTherapee releases. I couldn't check
them against the current manuals, so where a name differs in your version, follow the intent: a
**TIFF**, **32-bit float** (or 16-bit), in a **linear** profile (gamma 1.0), with the profile
embedded and no tone or colour edits applied.

**Before exporting, for both programs:** crop to the image, leaving out the film holder and rebate
(an opaque edge left in the crop throws off the print fit), and turn tone and colour modules
**off**: filmic, sigmoid, curves, levels, local contrast and so on. Keep the same
white balance and camera settings across the roll if you can. `halide check` tells you afterwards
whether you did.

**darktable** (check your version for exact names)

1. In the darkroom, look at the history stack: filmic rgb, sigmoid, base curve, tone curve,
   rgb curve, local contrast, and colour balance should not be active. Basic exposure, crop, lens
   correction, denoise and spot removal are fine.
2. In the darkroom's *output color profile* module, set the profile to *linear Rec2020 RGB*.
3. In the lighttable's *export* module, set *target storage* to *file on disk* and *format* to
   *TIFF*. Set *bit depth* to *32 bit (float)* (or 16 bit), *compression* to *uncompressed* or
   *deflate*, and *profile* to *linear Rec2020 RGB*.
4. Export. Some darktable versions have written the wrong profile into TIFFs; halide notices
   (see Troubleshooting).

**RawTherapee** (check your version for exact names)

1. In the *Color Management* tab of the editor, set *Output profile* to *RTv4_Rec2020*, and
   *Output profile > TRC* to linear (gamma 1.0, slope 0). If your version has no linear TRC option
   for it, choose a linear Rec2020 profile in *Preferences > Color Management > Output profile*
   instead.
2. In the *Exposure* and *Tone Mapping* panels leave curves, tone mapping and local contrast off.
3. In *Save* (Ctrl+S), choose *TIFF* with *32-bit floating-point* (or 16-bit), and save.

If a file isn't suitable (gamma-encoded, no profile, the wrong kind of profile), halide rejects it
and explains why. It will say, for example, "this profile is gamma-encoded (e.g. for display use).
Re-export using your raw processor's linear gamma/tone-curve option".

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
| `halide profile list\|show\|rename\|delete\|edit` | Manage saved profiles (see Troubleshooting for where they live) |

Every command has `--help`, with examples. `halide --version` prints the version. `batch`, `invert`,
`print` and `export` never write over a scan. If an output already exists they ask (in a terminal)
or refuse (in a script); add `--overwrite` to replace it, or `--skip-existing` to develop only
the frames that are missing (resuming an interrupted roll). Outputs are written whole or not at
all, so a cancelled run leaves no half-written file.

## What you get

- **Print** (the default): a TIFF printed through the paper curve, with exposure and grade fitted
  to the frame. Use it for finished pictures.
- **Flat** (`--output flat`): the film's own recorded contrast, unclipped and linear, for
  editing in darktable. Edit only exposure, crop, spot removal, lens and denoise. Then
  `halide print` prints it onto the paper.
- **Provenance:** every output TIFF records in its metadata how it was made: profile, grade,
  exposure, scan settings, and whether the CPU or GPU did it. `invert` prints the grade and
  exposure too.
- **Contact sheet:** `halide contact` (or `batch --contact-sheet`) makes a sheet styled after a
  real contact print, each frame captioned with its printing decision, for comparing settings
  across a roll.

## Troubleshooting

- **"this profile is gamma-encoded", "no embedded profile", or another colour-profile error.** The
  export wasn't linear, or the profile wasn't embedded. Re-export following "Exporting your scans"
  above. To see which profile a TIFF really carries, run `exiftool -icc_profile:all file.tif`
  if you have exiftool.
- **"is already a halide positive".** You gave `invert` a picture halide already made. To print a
  flat positive again, use `halide print`.
- **Developed images have no camera metadata (EXIF).** Install exiftool (see Install). Without it
  halide works but doesn't copy EXIF. If exiftool is installed but fails on one file, that frame
  is still developed and you get a warning.
- **The calibration window won't open (Qt errors).** See the Linux libraries note under Install.
- **The GPU wasn't used.** Run `halide gpu` to see what halide finds. If a frame doesn't fit on
  the card, or the GPU fails part-way, halide develops that frame on the CPU and prints a warning;
  the result is the same picture. `--device cpu` (or `HALIDE_DEVICE=cpu`) skips the GPU entirely.
  A GPU batch needs Python 3.13 or newer and room in `/dev/shm`; otherwise it falls back to
  per-worker GPU use, and the run sheet says why.
- **Where saved profiles live.** Linux: `~/.config/halide/profiles/` (or `$XDG_CONFIG_HOME/halide/
  profiles/`). macOS: `~/Library/Application Support/halide/profiles/` (unless `XDG_CONFIG_HOME` is
  set). Each profile is a small JSON file you can back up or copy to another machine.
- **Plain output for logs.** `NO_COLOR=1`, or piping halide's output to a file, gives plain text
  lines with no colour or moving progress display.



## GPU acceleration (optional)

If you have an NVIDIA card, halide can develop frames on it instead of the CPU. Measured on one
machine (RTX 3070, 8-core CPU, 7.6 GiB of free RAM, a 37-frame roll of full-resolution scans):
one `invert` took 2.7 s instead of 3.4 s, and a whole `batch` about 17 s instead of 33 s on the
CPU. In a batch, one halide process runs the GPU for all the workers, which read and write the
files; when each worker loaded NVIDIA's libraries itself, they filled the RAM and the same batch
took 25-27 s. Most of what's left is reading and writing the TIFFs, which a GPU
can't speed up. It needs an NVIDIA GPU; there's nothing to gain on other hardware, so halide
doesn't mention this unless it finds one.

```bash
halide gpu             # what halide sees: your card, and whether GPU support is installed
halide gpu --install   # add it (downloads CuPy and NVIDIA's CUDA libraries, about 1 GB)
```

GPU support isn't installed by default because that download is large. Once it's there, halide uses
it automatically (`--device auto`, the default) whenever it's usable, and falls back to the CPU on
its own if it isn't. Force the CPU with `--device cpu` or `HALIDE_DEVICE=cpu`;
`HALIDE_GPU_SERVICE=0` makes each batch worker use the GPU itself again (for troubleshooting). GPU
and CPU output aren't bit-for-bit identical — the two round the last digit of some calculations
differently — but halide's GPU tests require them to match to within 1 part in 100,000, and 8-bit
exports to within one code value. Each output TIFF records which one made it. `halide gpu
--install` also prints the command to remove it again later.

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
