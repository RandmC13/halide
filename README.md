# halide

**Turn camera or scanner captures of colour negative film into faithful positive images.**

halide inverts colour negatives the way a darkroom print does: it balances the film's orange mask
and its three dye layers, inverts, and prints the result through the measured response curve of a
real photographic paper. The goal is colour that matches what the film actually recorded, not a
"look". Once a film stock is calibrated, a whole roll develops consistently, with no per-frame
tweaking.

- **A whole roll in one window.** `halide open` loads a roll as a filmstrip. Click neutral objects
  (a white wall, a grey road, a black tyre) on any frames; halide fits them together and shows how
  well each point agrees with the rest in CC filter values. Then develop the whole roll from the
  same window, with a contact sheet and JPEG or PNG copies for sharing. No command line needed.
- **Calibrate once, reuse everywhere.** Save the calibration as a named profile for that film stock,
  process and scanner, and later rolls of the same film develop with no picking at all, in the
  window or from the command line.
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

To update to the latest version, run the install command again with `--force`:

```bash
uv tool install --force git+https://github.com/RandmC13/halide
```

`uv tool uninstall halide` removes it. Always use the full GitHub address: a different, unrelated
package called `halide` is on PyPI, so `uv tool install halide` on its own installs the wrong thing,
and `uv tool upgrade halide` can replace halide with it (it then says "No executables are provided
by package `halide`" and removes the tool). Your saved profiles aren't affected either way.

Tested with numpy 2.5, colour-science 0.4.7, tifffile 2026.10, PySide6 6.12, Pillow 13.0, psutil 7.2 and
shtab 1.13. Older versions within halide's declared ranges should work but aren't tested.

halide is developed and tested on Linux. It is plain Python (numpy, Qt via PySide6), so macOS and
Windows should work, but they haven't been tested yet.

<details>
<summary>Without uv (pip)</summary>

With Python 3.12 or newer:

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
`brew install exiftool`) so developed images and their JPEG/PNG copies keep their camera metadata
(date taken, camera, lens). Without it halide still works; it just doesn't copy EXIF. Shared copies
leave out halide's processing record, the scan's file path and darktable's edit history.

<details>
<summary>Linux: the halide window won’t open?</summary>

Minimal Linux installs can be missing libraries Qt needs. On Debian/Ubuntu:

```bash
sudo apt install libgl1 libegl1 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 \
                 libxcb-shape0 libxcb-xkb1
```
</details>

## Preparing your scans

halide reads **linear TIFFs with an embedded ICC profile**, the kind a raw developer exports. It
doesn't read raw files directly: your raw developer already handles demosaicing, cropping, lens
correction and dust removal well.

Every frame needs the same treatment, in whichever raw developer you use.

### Exporting your scans

Export a **TIFF**, **32-bit float** (or 16-bit), in a **linear** profile (gamma 1.0), with the profile
embedded and no tone or colour edits applied. Before exporting, crop to the image, leaving out the
film holder and rebate: an opaque edge left in the crop throws off the print fit. Every frame of a
roll needs the same treatment, and the same white balance. `halide check` tells you afterwards
whether that's what you got.

**darktable**

halide comes with a darktable style that sets all of this up:
[`darktable/halide-linear-scan.dtstyle`](darktable/halide-linear-scan.dtstyle). It switches off every
tone and colour module (exposure, filmic, sigmoid, color calibration, curves, levels, color balance,
local contrast, negadoctor, monochrome, sharpen, framing and so on) and sets the output profile to
linear Rec2020. Crop, rotation, perspective, lens correction and denoise are left to you.

White balance stays darktable's own: *camera reference*, which darktable sets on every raw it
imports with its default *modern* chromatic adaptation (preferences, *processing* tab). It works
the frame's own camera white balance back to the same D65 reference, so every frame matches even if
the camera was on auto white balance while scanning. Check the white balance module says *camera
reference*; if it doesn't, click that button on each frame. Don't copy white balance from one frame
and paste it onto others: the pasted values belong to the first frame, and the others come out a
few CC off.

1. Download the style file. In the lighttable's *styles* module, click *import* and choose it.
2. Select every frame of the roll and apply the style *halide linear scan* to them from the same
   module. Apply it in the lighttable (or the darkroom) rather than through the export module's own
   style option, so the change is recorded in each image's history, where `halide check` reads it.
3. Crop each frame (and straighten it if needed).
4. In the lighttable's *export* module, set *format* to *TIFF*, *bit depth* to *32 bit (float)*,
   *compression* to *deflate*, and *profile* to *linear Rec2020 RGB*. In the export module's
   metadata preferences, keep *develop history* ticked, so `halide check` can see how each frame
   was made.
5. Export.

White balance belongs to the profile: calibrate a new profile for scans made this way, rather than
reusing one made from scans with a different white balance.

**RawTherapee** (check your version for exact names)

1. Choose a **linear** Rec.2020 (or other linear, wide-gamut) output profile and embed it in the
   TIFF. The exact menu path is unverified and differs between versions: look for the output
   profile in the *Color Management* tab, or in *Preferences*. The name *RTv4_Rec2020* may not be
   linear, so don't assume it is.
2. Leave curves, tone mapping and local contrast off.
3. Save (Ctrl+S) as *TIFF*, 32-bit floating point if offered (or 16-bit).
4. To check the result: halide rejects a non-linear profile with a specific message, and
   `exiftool -icc_profile:all file.tif` shows what's embedded.

If a file isn't suitable (gamma-encoded, no profile, the wrong kind of profile), halide rejects it
and explains why. It will say, for example, "this profile is gamma-encoded (e.g. for display use).
Re-export using your raw processor's linear gamma/tone-curve option".

## Quick start

**In the window**, a roll from start to finish:

```bash
halide open portra-roll-03/
```

1. Click neutral points on a few frames (see "The halide window" below for what to pick).
2. *Save profile*, and name it after the film ("Portra 400").
3. *Develop roll…*: choose where the prints go and whether to also save JPEG or PNG copies, then
   *Develop*. halide checks the scans, develops every frame and shows how the roll came out;
   *Contact sheet* then prints one beside the prints.

On the next roll of the same film, open it, choose "Portra 400" in *Profiles*, *Use for this roll*,
and develop straight away.

**From the command line**, the same roll, useful for scripts and for rolls you already have a
profile for:

```bash
# 1. Check the scans were made consistently (reads file headers only, so it's fast)
halide check portra-roll-03/

# 2. Develop the whole roll with a saved profile
halide batch portra-roll-03/ portra-roll-03-developed/ --profile "Portra 400"

# 3. Make sRGB copies for sharing, plus a contact sheet
halide export portra-roll-03-developed/ portra-roll-03-jpeg/ --format jpg
halide contact portra-roll-03-developed/ portra-roll-03-sheet.jpg
```

The window runs these same commands underneath, so its prints are identical to the terminal's.

Run `invert` or `batch` without a calibration flag and halide offers your saved profiles, the
picking in the halide window, or an automatic estimate. The automatic estimate (`--auto-density`
for one frame, `--auto-density-roll` for a roll) guesses the neutrals statistically: fine for a
first look, but approximate. For a roll, every frame makes its own guess and the roll takes the
middle one, so one odd frame can't pull the rest off; it's best on a full roll and rough under six
frames. Frames whose colour runs well away from the rest (a scene under warm light, say) are named
on the run sheet: they print that way, as the film recorded them. For finished pictures, pick
neutral points. For a single frame:

```bash
halide invert IMG_0158.tif IMG_0158-positive.tif --pick    # click neutrals on this frame
```

Want to see what a run would do first? Add `--dry-run` to `batch` (or to `print`/`export` on a folder):
halide shows the frames, outputs, calibration and worker count it would use, instantly, and develops,
writes and asks nothing.

Want to try settings before committing disk space? `halide batch portra-roll-03/ --contact-sheet
preview.jpg` develops every frame at full quality but keeps only the contact sheet.

## The halide window

`halide open portra-roll-03/` opens the roll in the halide window: a filmstrip of its frames along
the top (a ⚠ marks a frame scanned differently from the rest), the frame you're on below it, and
the calibration panel on the right. Left and right arrow keys step through the frames.

**Pick neutral points.** Click things that were white, grey or black in the scene: a white wall, a
grey road, a black tyre, a cloud. You're looking at the negative, so brightness is reversed: real
whites look dark. Pick on any frames of the roll, and spread the points across the tones, from deep
shadow to bright highlight; three to six good points is usually plenty. *Show likely neutrals*,
above the picture, marks candidates, as a hint only. The step wedge at the top of the panel
(*Coverage of the roll's densities*) shows where your points sit in density; points spread along it
give a fit that holds for the whole roll.

**Read the agreement.** From three points on, each point gets a reading: how far it is from what the
other points agree neutral is, as a colour-printing filter pack on a dichroic head (CC 10 = 0.10
density; *what's CC?* in the panel explains it). Up to about CC 5 is good; CC 5 to 10 (amber) is worth a second look; over CC 10 (red)
usually means the object wasn't neutral after all (a cream wall, a warm-lit cloud, skin). The note
under the point list names the point that disagrees most; deal with that one first, since one bad point
pulls the others a few CC off too. Ctrl+Z takes a point back.

**Check the positive.** Once two points are far enough apart in density (0.1 D), the
**Negative | Positive** switch above the picture shows the frame printed with your calibration.
Before that, the positive is the roll's rough automatic estimate and says so; it settles as the
frames load. Once they have, a chip under the picture (*2 run warm*) names frames whose colour runs
away from the rest of the roll, and the strip marks them WARM, COOL, GREEN or MAGENTA. Neutrals
picked on those frames may be tinted by their light. *Raw negative* shows the scan
as it is, orange mask and all. If the fit only holds over part of the density
range, the window says where ("Fit reliable over D 0.4-1.2 only") and suggests where to add a point.

**The drawers** under the point list hold the rest: *Extra information* (film stock, process and
scanner, saved with the profile), *Print* (exposure and grade sliders, previewed on this frame; *Use
for the whole roll* pins them for every frame) and *Details* (the numbers behind the fit).

**Save.** *Save profile* names the calibration ("Portra 400"), so you can reuse it on every later
roll of that film, process and scanner. You can also use a calibration once without saving it.

**Develop the roll.** *Develop roll…* turns the panel over to the darkroom side:
- *Into*: the folder the prints go to;
- *Output*: *Print* (through the paper) or *Flat* (the film's own contrast, for editing elsewhere);
- *Names*: how the prints are named, from the scan's name, the roll, the frame number or the stock;
- *Existing*: what to do with prints already in the folder (ask, replace, or skip them);
- *Even out*: corrects scans digitized at different shutter speeds, when the roll has them;
- *For sharing*: also save a JPEG or PNG of every print;
- *The command*: the `halide batch` command the window will run, to copy into a terminal if you like.

*Develop* starts it. Each frame on the filmstrip is printed as it's done, and you can cancel at any
point (the frames in hand finish first). Afterwards the panel shows how the roll came out, with buttons for
*Contact sheet* (printed beside the prints), *Open folder*, *Export JPEG or PNG…* and, after a cancel,
*Carry on* to develop the frames still to do.

**Later rolls.** Open the next roll, choose your profile in the *Profiles* window (top right) and
*Use for this roll*. If its colour looks slightly off on the new roll, click a few more points to refine it, then
update the profile or save it as a new one. From the command line, `--profile "Portra 400"` does the
same for `invert` and `batch`.

## Scanning a roll consistently

One calibration is exactly right for a whole roll only if every frame was digitized the same way:
the same manual camera exposure (shutter, aperture, ISO), the same white balance, and no tone or
colour edits in the raw developer. `halide check portra-roll-03/` reads the file headers and tells
you, in seconds; `batch` runs the same check before it starts.

If the frames were digitized at different shutter speeds (an auto-exposure mode on the copy camera,
say), add `--match-scan-exposure` to `invert` or `batch`. It evens out the difference exactly, from
each scan's EXIF, against the exposure the profile was made at (or `--scan-reference FRAME`). A
different white balance between frames can't be corrected after export: re-export with one fixed
white balance (the darktable style does this for you).

## Commands

| Command | What it does |
|---|---|
| `halide open [roll]` | The halide window: load a roll, pick neutral points or use a saved profile, see a live positive, save a profile, and develop the whole roll |
| `halide invert IN OUT` | Develop one negative (`--profile`, `--pick`, `--auto-density`, `--output flat`, …) |
| `halide batch IN_DIR [OUT_DIR]` | Develop a folder in parallel; `--contact-sheet` for a preview or proof |
| `halide print IN OUT` | Print a flat positive, for example after editing it in darktable |
| `halide export IN OUT` | ACEScg TIFF (or folder of them) to sRGB PNG/JPEG |
| `halide contact DIR SHEET` | Contact sheet of developed frames |
| `halide check DIR` | Report inconsistent scan settings across a roll |
| `halide profile list\|show\|rename\|delete\|edit` | Manage saved profiles (see Troubleshooting for where they live) |
| `halide gpu [--install]` | Show whether an NVIDIA GPU can be used, and add GPU support (see below) |

Every command has `--help`, with examples. `halide --version` prints the version. `batch`, `invert`,
`print` and `export` never write over a scan. If an output already exists they ask (in a terminal)
or refuse (in a script); add `--overwrite` to replace it, or `--skip-existing` to develop only
the frames that are missing (resuming an interrupted roll). Outputs are written whole or not at
all, so a cancelled run leaves no half-written file.

### A folder of rolls

`batch --recursive` develops every folder of scans under the input folder as its own roll, one
after another, each with its own run sheet, scan checks and log. `--into` says where each roll's
prints go, relative to its scan folder:

```bash
# Holiday/Roll1/pre-processed/*.tif  →  Holiday/Roll1/inversion/
halide batch Holiday/ --recursive --into ../inversion --profile "Portra 400"
```

With an output folder instead (`halide batch Holiday/ Prints/ --recursive`), each roll gets a
subfolder named after it. Folders of halide's own prints are skipped, so running the same command
again with `--skip-existing` develops only new frames. You're asked about existing outputs, or a
missing calibration, once for the whole run.

### Naming outputs

`--name` names each output from a template instead of the scan's own name; halide adds the
extension. It works on `batch` and on folder `print` and `export`.

| Field | Is |
|---|---|
| `{stem}` | the scan's name, without its extension |
| `{roll}` | the roll's name (the input folder, or the roll's folder with `--recursive`) |
| `{n}` | the frame's number in the roll, from 1; `{n:02}` pads it to two digits |
| `{stock}` | the profile's film stock |

`--name "{roll}_{n:02}"` gives `Roll1_01.tif`, `Roll1_02.tif`, and so on. Frames are numbered in
natural order (`frame_2` before `frame_10`), over every scan of the roll, so a resumed run keeps
each frame's number. A template that would give two frames the same name is refused before
anything is written. `--dry-run` lists each frame's new name.

### After a run

When `batch`, `print` or `export` finishes a folder, a few lines under the "Developed" line tell
you how the roll came out: the range of print grades and exposures, any frame whose grade is well
below the roll's typical one (often a crop with something bright left in it), and any warnings,
grouped. halide also keeps a notebook: every run appends to `halide-log.txt` in the output
folder, with the command you ran, the settings, and a line per frame. It is written as the run
goes, so a run that is killed still leaves a record, and the next run notes that it didn't
finish. Use `--log FILE` to put the notebook somewhere else, or `--no-log` to skip it.

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
- **The halide window won’t open (Qt errors).** See the Linux libraries note under Install.
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

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Code layout: `src/halide/core/` holds the pure image maths, `io/` handles TIFF/ICC/export,
`calibration/` handles profiles and fitting, `batch/` does parallel processing, `cli/` holds the
commands and `gui/` the Qt window. `benchmarks/gpu_bench.py` times CPU against GPU on your own
machine.

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
