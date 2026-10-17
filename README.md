# KangEdit

KangEdit is a desktop video editor for Windows, macOS and Linux. It analyzes your footage, picks the best moments and builds a short-form or long-form edit for you. You then fine-tune the edit, color, captions and sound effects, and export.

Everything runs on your own computer. Analysis, speech recognition and rendering work offline. The only network use is the one-time FFmpeg and Whisper model downloads, the optional sound-effect search and the optional background-music download.

---

## Contents

- [Features](#features)
- [System requirements](#system-requirements)
- [Install and run](#install-and-run)
- [Quick start](#quick-start)
- [Keyboard and mouse](#keyboard-and-mouse)
- [Video layers and multicam](#video-layers-and-multicam)
- [Sound effects](#sound-effects)
- [Official camera LUTs](#official-camera-luts)
- [Where files are stored](#where-files-are-stored)
- [Troubleshooting](#troubleshooting)
- [Development and releases](#development-and-releases)
- [Project layout](#project-layout)
- [License](#license)
- [Third-party software](#third-party-software)

---

## Features

### Automatic editing

| | |
|---|---|
| **Best moments** | Each clip is analyzed at 4 fps for sharpness, motion and shake, exposure, colorfulness, faces, audio loudness and scene cuts. Candidate windows are scored and picked greedily. Analysis runs on several CPU lanes plus the GPU video decoder (NVIDIA, AMD or Intel; two GPU lanes on CPUs with 8 threads or more). |
| **Quick first look** | New footage is first looked at from its keyframes only, all files at once, many times faster than the full analysis: thumbnails, scrubbing, bins and even an auto edit work at once. The full analysis then runs in the background and replaces each quick look as it finishes (an auto edit made meanwhile says how many files it took from a quick look; regenerate it afterwards for the best edit). |
| **Smarter analysis** | *Tools → Smarter analysis (AI)* (on by default): two small neural networks run offline on the CPU. **MobileNetV2** describes what each second of a shot shows, so the same view filmed again, a little to the side, zoomed or darker, is recognised even across files (a phone and a camera filming the same thing), where the plain method can't tell it from a different view. **YAMNet** hears speech, laughter, cheering and applause, crowds, music, wind, nature and vehicles: moments people enjoyed score higher (more so for the hook), wind on the microphone lower, and scenic shots are found by "nobody speaking" rather than by loudness (loud waves are a view). Turn it off on a slow computer for the lighter analysis; turning it on adds it to existing analyses without decoding the video again. |
| **Short / Long format** | *Short-form*: 9:16, about 30 s, 1.2–3 s cuts, strongest moment first as the hook. *Long-form*: 16:9, 4–10 s shots in story order. Its default length follows the footage, at about a quarter of it, between 1 and 20 minutes. *Custom*: set everything yourself. |
| **Story order** | Clips are ordered by recording time, then by file name (C0002 before C0013). The recording time is the oldest date the file carries (its creation time tag, its file creation date, its modification date or a date in its file name such as `VID_20260501_091005`), as copying renews the creation date and an unset camera clock gives a wrong tag. The same date decides where the opening and closing are looked for and how *Bins by date* sorts. |
| **Scenes** | Footage recorded close together is one scene: a gap of more than 30 minutes between recordings (another day) or, when the files carry a location (phones, action cameras, drones), more than 1 km between them (another place) starts a new one. In a long-form edit each scene after the first opens with a scenic / travel shot from that scene (a view with nobody's face in it and no one talking, sharp, colourful, steady or gently panning), taken first from the **B-roll bin** (right-click media → *Add to B-roll bin*), and its natural sound fades in over the end of the scene before. *Transitions between scenes* play between these scenes. |
| **Finishing** | Options in ✨ Auto Edit, and *Tools → Finish the edit* for any timeline: **whole sentences** (long form: every clip with speech starts and ends on complete sentences of the transcript); **cut between words** (never mid-word: the transcript's word timings, else a quieter moment); **cut on the beat** of the music on the timeline; **pacing** (steady, build-up, trailer or calm finish: clip lengths change along the edit); **remove fillers** ("um / eh / anu"), long pauses and false starts using the transcripts; **vary the shot sizes** (no three close-ups or wide shots in a row); **even out the loudness** of the clips. Moments with the lens covered, a blown-out picture or the focus hunting are avoided. |
| **Highlight reel** | *Tools → Make a highlight reel (9:16)* turns the edit into a 30, 45 or 60 s vertical reel of its best moments: a hook first, trailer pacing, on the music's beat, each clip framed on its face. Undo brings the full edit back. |
| **Opening & closing** | Looks for the greeting that opens the video and the sign-off that closes it, in English and Indonesian: "hello…", "welcome…", "assalamualaikum…", "thanks for watching", "sampai jumpa"… It searches intro/outro-named files and the first and last recordings, and uses the whole spoken passage. |
| **Transitions & J/L cuts** | Optional transitions between scenes or at every cut: dissolve, dip to black or white, wipe, slide, zoom, or a mix. For long-form, J-cuts let the next scene's sound lead in, and L-cuts let a voice finish over the next picture. |
| **Reuse rules** | Footage can be used several times, but its segments never overlap, always keep a minimum gap, and near-identical shots are rejected. Clips that break the rule after manual edits get a red outline. |
| **Edit from a script** | ✨ Auto Edit → **📜 Script** (or *Tools → Auto Edit from script…*): paste the script and each line is found in the footage's transcripts by text similarity, then the clips are laid out in the script's order. The **similarity threshold** (30–100 %) decides how close a line must be; speech-recognition slips and small ad-libs still match. A line said in several takes uses the best match or the last good take. Clips are cut on word boundaries with adjustable padding, lines said one after the other in the same take become one clip, and a live preview lists every line with its score, file and time (lines not found in red). Files without a transcript are transcribed first. |
| **Regenerate shorter / longer** | **− Shorter** / **+ Longer** on the toolbar rebuild the edit about 25% shorter or longer with the same settings. *Tools → Regenerate auto edit at length…* asks for an exact length. |

### Editing

| | |
|---|---|
| **Smooth preview** | Playback is a pipeline: one thread only decodes, while a pool of threads converts, grades and reframes, so 4K H.265 keeps up with the frame rate. The next clips are prepared while one plays, and that preparation steps aside whenever playback needs the CPU. Scrubbing never freezes the window. |
| **Color on / off in the viewer** | The viewer menu below the picture (or *View → Color in viewer*, **Ctrl+Shift+B**) turns the log conversion, LUTs and grading off, always or only while playing, for lighter playback. Export always includes them. |
| **Timeline** | Drag to reorder, drag edges to trim, **Alt/Shift+drag** to slip. **Ctrl+click** / **Shift+click** / **Ctrl+A** select several clips (Ctrl+A and Ctrl+click include clips on the video layers). Delete, duplicate, auto color, transitions and *Other moment* work on the whole selection. **↶ Undo** / **↷ Redo** buttons sit at the left of the timeline toolbar. |
| **Transitions & split edits** | Per clip, in the Inspector or the right-click menu. Transitions are centered on the cut and use the footage around each clip, so the timeline length and captions don't move. |
| **Bins and filters** | The row above the Media list picks a **bin** and filters by **type** (videos, photos / images) and **picture shape** (landscape, portrait, square, 16:9, 9:16, 4:3, 21:9). Right-click → *Bins*: new bin, move the selected media into a bin, rename or delete a bin, **Clear all bins**, or **Bins by date** to sort the media by recording day (the oldest date the file carries, a date in the file name included). Ctrl+A in the list selects everything it shows. Right-click a timeline clip → **Show in Media panel** to find its footage. |
| **Transcripts** | After an import KangEdit offers to transcribe every file with sound that has no transcript yet (*Tools → Offer to transcribe after importing* turns it off). *Tools → Full transcript…* (or right-click media → *Show the full transcript…*) merges the transcripts in recording order, a sentence per line, with or without file names and times, to copy or save as .txt. |
| **Media list views** | The **☰** / **▦** buttons above the Media list switch between the detail view (duration, format, camera, log profile, status) and a thumbnail grid. In the grid, move the mouse across a thumbnail to scrub through the clip; Ctrl+wheel resizes the thumbnails. |
| **Recent projects** | *File → Open recent* lists the last 10 projects you opened or saved. |
| **Sound-effect layer** | A layer of sound effects under the video track, placed freely in time. Drag to move, drag edges to trim, set the volume by right-clicking. Search and add sounds from MyInstants in the **Sounds** tab, or drop any audio file on the timeline. |
| **Multicam** | Two or more cameras (or a phone and a camera) on the same scene: turn on *Multicam* in ✨ Auto Edit and give each file its angle. The angles are synced by their sound (onset matching over the whole recording, then sample-accurate matching in windows across it, with clock drift corrected), the main angle is edited as usual and the others follow on the video layers, cut at the same moments. Layouts: picture in picture, main video + talking cam, stacked, side by side, 2 × 2 grid; boxes are cropped around each angle's face. The angles keep in sync as you trim, slip, split or swap moments; *Tools → Multicam layout* changes the layout later. |
| **Images** | Import JPG, PNG, GIF, BMP, WebP and TIFF pictures along with your footage. On the main track an image is a 5-second clip (trim it to any length); an animated GIF loops; PNG transparency is kept on video layers. |
| **Video layers** | Layers **V2, V3…** above the main track for picture in picture, logos, B-roll and images on top. Drag footage or an image from Media onto a layer (or right-click it → *Add on a video layer*). Drag the clip in the viewer to place it and use the mouse wheel over it to resize it; the **🧩 Layer** tab sets position, size, rotation, opacity, fades and the clip's own volume, with one-click corners for picture in picture. Higher layers sit in front. Add layers from the layer's right-click menu. A layer clip is refined like a main-track clip: with it selected, the **🎨 Color** tab sets its log conversion, exposure and color, ✨ Auto, look LUT, source in / out, speed and volume. |
| **Slow & fast motion, speed ramps** | Right-click a clip → *Speed* (0.25× to 4×, or any speed from 0.1× to 8×), or the Inspector's **Speed** group. *Speed ramp* templates change the speed along the clip: *Hero moment*, *Bullet time*, *Slow → fast*, *Fast → slow*, *Slow finish*, *Speed burst*, *Pulse*, *Montage*. *Sound at this speed* keeps the sound in sync (pitch kept, or pitch following the speed like a tape), plays it at normal speed, or mutes it. Captions and multicam angles follow, and the export can blend frames for smoother slow motion. |
| **Rotation** | Any clip (Reframe → *Rotation*, or right-click → *Rotate 90°*) and any layer clip can be rotated. Quarter turns rotate the footage before it fills the frame, so a phone clip recorded sideways fills it upright. |
| **Lane heights and zoom** | Drag the bottom edge of the video track, the video layers or the sound lane to make them taller (double-click the edge to reset), or use **↕ Lanes** (sliders and *Fit to panel*). **⇔ Fit** (Shift+Z) zooms so the whole edit, every lane included, fits the width; **− / +** (Ctrl+− / Ctrl+=) zoom out / in. |
| **Used in the edit** | The Media list's **In edit** column and a green **✓** on the thumbnails show which footage the timeline uses, and how many times. |
| **Auto ducking** | Music turns down by itself while the clips have speech or sound and comes back up in between (sidechain compression in the preview mix and the export). Music added from the NCS list is ducked; switch it on for any sound in the **Sounds** tab or the sound's right-click menu, and set how far it drops (*Duck by*, 12 dB by default). |
| **Updates** | *Help → Check for updates…* looks for a newer release on GitHub; KangEdit also checks once a day at startup (*Help → Check for updates at startup*). A packaged build downloads its new version, checks it against the release's checksums and restarts with it; a source copy replaces its files and reinstalls its packages; a git checkout runs `git pull`. |

### Color

| | |
|---|---|
| **Log → Rec.709** | Sony S-Log3/S-Log2, Panasonic V-Log, Canon C-Log2/3, ARRI LogC3/4, Fujifilm F-Log/F-Log2, Nikon N-Log, DJI D-Log, Apple Log, Blackmagic Film Gen5 and HLG, using the published curves and gamut matrices (full-range files read correctly). The filmic output keeps hues (bright reds stay red, skies don't turn cyan) and fades the brightest colours to white, and colours outside Rec.709 are compressed smoothly into it instead of clipping. DJI D-Log M, D-Log2 and the D-Log / D-Log2 *Vivid* looks are in the same menu and use DJI's official LUTs (see [Official camera LUTs](#official-camera-luts)). |
| **Camera auto-detect** | The camera is identified from the file's metadata, and its log profile is chosen from the metadata or the picture. **Detect all** re-runs it for every file in the background. |
| **Official camera LUTs** | Use the makers' own conversion LUTs (DJI D-Log M, Sony, Panasonic, Fujifilm, Nikon, ARRI…) instead of the built-in curves. See [Official camera LUTs](#official-camera-luts). |
| **LUTs (.cube)** | Import 1D/3D `.cube` LUTs (or the `.zip` they come in) as a log conversion or as a creative look with a strength slider. |
| **Exposure & color** | Exposure, contrast, highlights, shadows, temperature, tint, saturation, vibrance. **Auto** sets exposure, white balance and contrast. *Auto color all clips* runs in the background with progress and Cancel. Preview and export share the same color math. |
| **Processing order** | The industry order: ① log curve and gamut to scene-linear Rec.709 (or the conversion LUT, with exposure and white balance applied before it on the log signal) → ② exposure and white balance in linear light → ③ contrast, shadows and highlights in stops around middle grey → ④ saturation and vibrance in linear light → ⑤ output tone map (filmic or clip) → ⑥ creative look LUT (expects Rec.709) at its strength → ⑦ halation and sharpening. |

### Captions

| | |
|---|---|
| **Automatic captions** | Offline speech recognition with faster-whisper. The default model, **Auto**, is `large-v3` (the most accurate) on an NVIDIA GPU and `large-v3-turbo` (almost as accurate, several times faster) on the CPU, which is where it runs on AMD and Intel graphics. On the CPU it uses one thread per physical core and decodes several stretches of speech at once. Each word's start and end are then set on the sound (Whisper's own times are often 0.1-0.3 s off), so cuts between words, on whole sentences and around fillers land in the pauses. Footage with no speech (views, B-roll) is skipped in seconds. Transcripts are kept in the cache by file, so the same footage in another project is never transcribed again. Captions follow every trim and reorder, and manual fixes are kept. |
| **Word animations** | Karaoke highlight, highlight box, pop, reveal, or one word at a time, timed to the speech. They look the same in the viewer and in the burned-in export. |
| **Automatic placement** | Size and position follow the frame ratio and format. 9:16: lower middle, clear of the app buttons. 1:1 / 4:5: lower third. 16:9: subtitles at the bottom. |
| **Output** | Burn in, save as `.srt`, or both. |

### Export

| | |
|---|---|
| **Ratios & resolutions** | 16:9, 9:16, 1:1, 4:5, 4:3, 2.39:1 at 720p, 1080p, 1440p or 4K. Fill (crop with per-clip pan/zoom), fit, or blurred-background fit. |
| **Codecs** | H.264, H.265 or ProRes 422 HQ. |
| **GPU encoding** | H.264/H.265 use the graphics card's encoder when it has one (NVIDIA NVENC, AMD AMF, Intel Quick Sync, Apple VideoToolbox), and fall back to the CPU automatically. H.265 exports were about 3.7× faster on an AMD RX 480. |
| **Proxies** | Optional 360–1080p short-GOP H.264 copies for lighter editing. They are decoded and encoded on the graphics card when it can (same detection as analysis and export), and redone on the CPU for footage the GPU can't decode. Exports use the originals unless you choose *Draft export*. |

---

## System requirements

### Ready-made builds

| | Minimum | Notes |
|---|---|---|
| **Windows** | Windows 10 version 1809 or Windows 11, 64-bit (x64) | Windows on ARM isn't tested. |
| **macOS** | macOS 14 Sonoma, Apple Silicon (M1 or newer) | No Intel build. The app says so on older macOS versions. |
| **Linux** | x86-64 with glibc 2.35 or newer: Ubuntu 22.04, Debian 12, Fedora 36, Linux Mint 21, RHEL / Rocky / Alma 10, or newer | X11 or Wayland desktop. No ARM64 build. RHEL 9 and its clones (glibc 2.34) are too old. Needs the Qt libraries listed under [Linux](#linux-x86-64-or-arm64). |

These are the oldest systems the bundled libraries support: Qt 6.12, onnxruntime, NumPy and OpenCV in the macOS build need macOS 14, and the Linux build is made on Ubuntu 22.04 (glibc 2.35).

### From source

Python **3.10 or newer** on 64-bit Windows 10 / 11, macOS or Linux (x86-64 or ARM64). Each release is tested on Windows (latest), macOS 14 and Ubuntu 22.04 with Python 3.10 and 3.12. On older systems (macOS 12 or 13, Intel Macs, Linux with glibc 2.28 to 2.33) pip installs older versions of Qt and the other packages that still support them. That should work, but it isn't tested.

### Hardware

| | Minimum | Recommended |
|---|---|---|
| **CPU** | 64-bit, 4 cores | 8 cores or more for 4K |
| **Memory** | 8 GB | 16 GB or more for 4K or long edits |
| **Disk** | about 1.5 GB for the app, FFmpeg and the speech model, plus space for analysis, proxies and exports | an SSD for the footage and the cache |
| **Graphics** | any (preview and export work on the CPU) | a GPU with a video encoder (NVIDIA, AMD, Intel, Apple) for faster proxies and exports; an NVIDIA card with CUDA for faster captions |

An internet connection is needed once, to download FFmpeg and the speech model, and later for updates, NCS music, sound effects and official LUTs.

---

## Install and run

### Ready-made builds

Each [GitHub release](../../releases) has a packaged KangEdit for **Windows (x64)**, **macOS (Apple Silicon, macOS 14 or newer)** and **Linux (x64)**: no Python needed (see [System requirements](#system-requirements)). Unpack it and start `KangEdit.exe`, `KangEdit.app` or `KangEdit/KangEdit`. FFmpeg is downloaded on the first start. The builds aren't code-signed: on Windows choose *More info → Run anyway*; on macOS right-click the app → *Open* the first time. `SHA256SUMS.txt` lists the archives' checksums.

### From source

You need **Python 3.10 or newer**. The launcher sets up everything else on the first start:

1. a private Python environment in `.venv`;
2. the Python packages from `requirements.txt` (PySide6, NumPy, OpenCV, PyAV, faster-whisper);
3. **FFmpeg**, downloaded into the `ffmpeg/` folder if it isn't already on your `PATH`.

The first start takes a few minutes and needs about 1.5 GB of disk space. Later starts are instant. When `requirements.txt` changes, the packages are updated automatically.

### Windows 10 / 11

1. Install Python from [python.org](https://www.python.org/downloads/). Tick **"Add python.exe to PATH"** in the installer.
2. Double-click **`run.bat`**.

### macOS (Apple Silicon or Intel)

1. Install Python: `brew install python@3.12`, or the installer from [python.org](https://www.python.org/downloads/).
2. In Terminal, in the KangEdit folder:
   ```sh
   chmod +x run.sh run.command   # once
   ./run.sh
   ```
   After that you can double-click **`run.command`** in Finder.

### Linux (x86-64 or ARM64)

1. Install Python with venv support and the libraries Qt needs:
   ```sh
   # Debian / Ubuntu
   sudo apt install python3 python3-venv libxcb-cursor0 libxkbcommon-x11-0 libegl1 libgl1 libpulse0
   # Fedora
   sudo dnf install python3 xcb-util-cursor libxkbcommon-x11 mesa-libEGL pulseaudio-libs
   # Arch
   sudo pacman -S python xcb-util-cursor libxkbcommon-x11 libglvnd libpulse
   ```
2. Start it:
   ```sh
   chmod +x run.sh   # once
   ./run.sh
   ```

### Manual setup

```sh
python install.py             # create/update .venv, install packages, download FFmpeg if needed
python install.py --no-ffmpeg # same, but use the FFmpeg on your PATH
python install.py --run       # set up if needed, then start KangEdit
```

To start KangEdit directly afterwards, run `.venv/bin/python main.py` (macOS/Linux) or `.venv\Scripts\python main.py` (Windows). You can pass a project file or video files: `main.py project.kgproj` or `main.py clip1.mp4 clip2.mov`.

**FFmpeg.** KangEdit uses the FFmpeg in its own `ffmpeg/` folder first, then the one on your `PATH`. If neither exists when the app starts, it offers to download it. To install FFmpeg yourself instead: `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (macOS), or your package manager (Linux).

**GPU speech recognition** is used automatically on NVIDIA cards with CUDA. Everything else runs on the CPU.

---

## Quick start

1. **Import** footage: use the button, or drag files and folders onto the Media panel. Analysis runs in the background.
2. Check the log profile. It's usually detected automatically (see the *Camera* and *Log* columns); **Detect all** in the Inspector re-runs it. For DJI D-Log M and other makers' own conversions, set up [Official camera LUTs](#official-camera-luts).
3. Optional: **Generate proxies** for lighter editing of 4K or 10-bit H.265.
4. **✨ Auto Edit**: choose *Short-form* or *Long-form*, check the transitions, J/L cuts and opening/closing options, then click *Create edit*. Use **− Shorter** / **+ Longer** to try other lengths.
5. Refine the timeline: reorder, trim, slip, transitions (right-click a clip), and sound effects (**Sounds** tab).
6. **Auto color all clips**, then fine-tune the selected clip in the Inspector.
7. **Generate captions**. The first run downloads the Whisper model. Choose the model, language, animation and placement under *Captions → Style…*.
8. **⬆ Export**: choose the ratio, resolution, codec, GPU encoding and caption mode.

---

## Keyboard and mouse

| Key | Action |
|---|---|
| Space | Play / pause |
| ← / → | Previous / next frame (Shift: 10 frames) |
| ↑ / ↓ | Previous / next clip |
| Home / End | Start / end |
| S | Split at the playhead |
| Del | Delete the selected clips (or the selected sound) |
| Ctrl+D | Duplicate clip |
| Ctrl+A | Select all clips, video layers included (in the Media panel: all the media it shows) |
| R | Other moment: swap the clip for the best moment close by (its length follows the new moment) |
| Ctrl+Shift+− / Ctrl+Shift+= | Regenerate the auto edit shorter / longer |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |
| Shift+Z | Fit the timeline |
| Ctrl+Shift+R | Reset the layout (*View → Reset layout*) |
| Ctrl+− / Ctrl+= | Zoom the timeline out / in |
| Ctrl+Shift+A | Auto Edit… |
| Ctrl+Shift+B | Color adjustments in the viewer on / off |
| F1 | Shortcut list |

| Mouse on the timeline | Action |
|---|---|
| Drag a clip | Reorder |
| Drag a clip's edge | Trim |
| Alt/Shift+drag (or *Slip* mode) | Slide the moment inside the clip |
| Ctrl+click / Shift+click | Add to the selection / select a range |
| Right-click a clip | Transition, J/L cut, color, duplicate, delete… |
| Drag / drag edges of a sound | Move / trim the sound |
| Right-click a sound | Volume, fade in / out, fit to timeline, move to the playhead, delete |
| Ctrl+wheel | Zoom |
| Shift+wheel | Scroll the lanes up / down (when they are taller than the panel) |
| Drag a lane's bottom edge | Make the video track, the video layers or the sound lane taller |
| Drag footage from Media | Onto a video layer: an overlay there · onto the main track: a clip at that cut |
| Drag a layer clip | Move it in time, or up / down to another layer; drag its edges to trim |

| Mouse on the viewer | Action |
|---|---|
| Drag a title or a layer clip | Place it |
| Wheel over the selected layer clip | Resize it |

On macOS, use ⌘ where this table says Ctrl.

---

## Video layers and multicam

The lanes **V2, V3…** above the main track lay pictures on top of the edit: picture in picture, a logo, B-roll,
an image, a second camera angle. Higher layers are in front of lower ones.

**Add something to a layer**

- Drag footage or an image from the **Media** panel onto a layer. (Dropped on the main track, it becomes a clip
  there instead, at the nearest cut.)
- Or right-click footage in Media → *Add on a video layer at the playhead*, or right-click an empty layer →
  *Add the selected media here*.
- New videos start as picture in picture in the bottom-right corner, images in the middle at half size. A layer
  that is busy at that time sends the new clip to the next free layer.
- More layers: right-click an empty part of a layer → *Add a video layer* (up to 8), or *View → Add a video
  layer*.

**Place it**

- Drag the clip in the **viewer** to move it (it snaps to the centre lines); the mouse wheel over the selected
  clip resizes it.
- The **🧩 Layer** tab (double-click the clip, or select it) has one-click positions (full frame, centre and the
  four picture-in-picture corners), X / Y, size, rotation (⟲ / ⟳ 90°), opacity, fade in / out, start, length,
  speed and speed ramp, layer and the clip's own volume.
- Clips cropped to fill a box (the multicam layouts) also get **Crop X / Y**, to choose which part of the picture
  fills it.

**Refine it**

- With a layer clip selected, the **🎨 Color** (Clip) tab works on it as on a main-track clip: log conversion and
  camera detection, exposure and color, ✨ Auto, look LUT, source in / out, speed and volume. The 🧩 Layer tab's
  *Color, log, LUT, trim & speed* button gets you there; right-click the clip → *Color* for auto color, reset and
  *Copy this grade to all clips*.
- *✨ Auto all clips* and *Copy to all clips* cover the main track and the video on every layer (pictures on the
  layers, such as logos, keep their own look).

**Time it**

- On the timeline, drag a layer clip to move it in time, or up / down to another layer; drag its edges to trim.
  It snaps to the playhead, the cuts and other layer clips.
- Right-click it for *Move to playhead*, *Split at playhead*, *Duplicate*, positions, rotation, *Speed*,
  *Speed ramp*, fades, layer up / down and *Delete*.
- A layer clip's sound is mixed in at its own volume (0 % mutes it). PNG transparency is kept.

**Multicam: several cameras of the same moments**

1. Import every camera's files (each needs its own sound).
2. In **✨ Auto Edit**, tick *Multicam*, and give each file its angle: *Main angle*, *Angle 2*, *Angle 3*,
   *Angle 4* (KangEdit guesses one angle per camera model).
3. Pick a **layout**, whose **sound** to use, whether to keep to moments every angle recorded, and whether to
   use the **whole recording** instead of its best moments, then *Create edit*.

The angles are synced by their sound, each to the longest recording (and through it to the main angle): first by matching where the loudness rises over the whole recordings
(which works with different microphones and levels), then sample by sample in pieces spread across them, which
also corrects the slow clock drift between cameras. The main angle is edited as usual and every other angle is
cut at exactly the same moments on the layers above. The status bar shows each angle's offset; an angle whose
sound doesn't match is left out with a warning.

| Layout | |
|---|---|
| Picture in picture | The main angle fills the frame; the others are small, whole pictures in the corners. |
| Main video + talking cam | Landscape: a square box in the corner, cropped around the speaker's face. Vertical / square: the main angle on top, the speaker below. |
| Stacked, top / bottom | A full-width band per angle, each cropped around its subject (for 9:16). |
| Side by side | A full-height column per angle (for 16:9). |
| Grid (2 × 2) | Up to four angles. |
| Switching | Like a live multicam edit: one angle fills the frame at a time and the edit cuts between them at the best moments: to whoever is speaking (each camera's microphone is loudest for the person nearest it), the closer face, the steadier and sharper picture. At a laugh, applause or an exclamation it cuts to the others' reaction. Shots last between the edit's shortest and longest clip length: an angle held longer gives way to another for a shot (a reaction shot), as often as needed, unless no other camera is recording. With *Use the whole recording* the main track is cut at every angle change, so the edit's transitions play between the angles. The sound stays on the main angle. |

Two angles take the main track and one layer, three angles two layers, and so on: in the boxed layouts the main
angle's clips are shown in their box on the main track itself. Switching uses one layer for every other angle,
one clip per shot.

Transitions between main-track clips include the angles: a dissolve (or wipe, slide, zoom) cross-fades every
angle along with the main picture, and a dip to black or white fades them out and in.

The angles are **linked** to the main clip they were cut with (🔗 on the layer clip). Trims, slips, splits,
*Other moment*, a new hook and reordering on the main track re-time them, and the angle clips are handles on the
same group: drag one to slip every angle together, drag its edges to trim them all, press **R** with it
selected for another moment. Positions you changed by hand are kept. **🔗 Link** in the timeline toolbar
(Ctrl+L, also in the right-click menus and the Layer tab) unlinks the selected clip's angles, or a selected
angle, so they can be moved and trimmed on their own; linking them again puts them back in sync. *Tools → Multicam
layout* (or a layer clip's right-click menu) switches to another layout later; the **− / +** auto edit buttons
regenerate the edit with its angles.

---

## Sound effects

The **Sounds** tab searches [MyInstants](https://www.myinstants.com), a large collection of short meme and effect sounds.

- **▶ Preview** plays a sound.
- **＋ Add at playhead** downloads it into your sound library and puts it on the sound-effect layer.
- The **Library** list holds every sound you added or imported. Drag one onto the timeline, or add it at the playhead.
- **Import…** adds your own audio files (MP3, WAV, OGG, M4A, FLAC…). You can also drop audio files straight onto the timeline.

Sound effects are mixed into the preview and the export at their own volume (right-click → *Volume*), each with its own fade in and fade out (right-click → *Fade in* / *Fade out*, or the sliders in the **Sounds** tab). *Fit to timeline* lays a sound over the whole edit as one clip: a shorter sound loops with 2-second crossfades between repeats (dashed lines on the clip), a longer one is cut to the edit's length with a fade out.

> MyInstants sounds are uploaded by its users and may be copyrighted. Make sure you're allowed to use a sound before you publish a video with it.

### Background music (NCS)

At the top of the **Sounds** tab, pick the kind of video (Vlog, Cinematic, Aftermovie / party, Travel, Chill, Gaming / action) for suggested [NoCopyrightSounds](https://www.youtube.com/@NoCopyrightSounds) tracks, or paste any YouTube link. **♪ Add to timeline** downloads the track with [yt-dlp](https://github.com/yt-dlp/yt-dlp), extracts its sound to an .mp3 in your sound library, and lays it under the whole edit at 50% volume, auto-ducked under the clips' sound. **▶ Preview** (or double-click a track) streams it so you can listen first without downloading it, and **■ Stop** stops it; previews also stop when the timeline starts playing. **YouTube** opens the track in your browser.

NCS music is free to use in videos as long as the description credits it: the credit is copied to the clipboard when you add a track. Music from other YouTube channels needs its owner's permission.

> YouTube only serves media to apps that solve its JavaScript challenge. KangEdit uses [Deno](https://deno.com), Node.js or Bun when one is installed; without one, some downloads may fail.

---

## Official camera LUTs

Camera makers publish their own log → Rec.709 conversion LUTs. **Tools → Official camera LUTs…** lists them with a **Download page…** button for each maker:

| Maker | Log | Where |
|---|---|---|
| DJI | D-Log M, D-Log, D-Log2 (+ Vivid) | [dji.com/lut](https://www.dji.com/lut): **Download** in the dialog fetches any camera's LUT directly |
| Sony | S-Log3 / S-Gamut3(.Cine), S-Log2 | [Sony LUT library](https://pro.sony/bp_BR/technology/professional-video-lut-look-up-table) |
| Panasonic | V-Log, V-Log L | [LUMIX downloads](https://av.jpn.support.panasonic.com/support/global/cs/dsc/download/index3.html) |
| Fujifilm | F-Log, F-Log2 | [fujifilm-x.com LUTs](https://www.fujifilm-x.com/en-gb/support/download/lut/) |
| Nikon | N-Log | [Nikon Download Center](https://downloadcenter.nikonimglib.com/en/download/sw/258.html) |
| ARRI | LogC3, LogC4 | [arri.com/log-c](https://www.arri.com/log-c) |
| Canon | C-Log2, C-Log3 | Your camera's page on Canon's support site → Downloads → "Canon lookup table" |
| Blackmagic | Film Gen5 | Installed with DaVinci Resolve (**Import from DaVinci Resolve**) |

1. **DJI:** click **Download…** in a DJI row and pick your camera. Every log → Rec.709 LUT on [dji.com/lut](https://www.dji.com/lut) is listed (32 cameras and variants), downloaded from DJI's own servers into the LUT library, and selected. Two entries without a direct file (Mavic 3 D-Log, Zenmuse X9) open their download page instead.
   **Other makers:** download the LUT from the maker's page (KangEdit can't download these itself, because the pages require accepting the maker's terms).
2. Click **Import LUT files (.cube / .zip)…** and choose the file as downloaded. Recognized official files are marked ★ and selected automatically.
3. Keep **Use these LUTs automatically when a camera's log profile is detected** ticked, and optionally switch existing footage now.

DJI cameras record D-Log M (Mini 4/5 Pro, Air 3/3S, Avata 2/360, Flip, Osmo Pocket 3, Osmo Action 4–6, Osmo 360, Osmo Nano), D-Log (Mavic 3/4 Pro, Osmo Pocket 4/4P, Zenmuse, Phantom 3/4, Inspire 1) or D-Log2 (Osmo Pocket 4P). DJI has published no curve for D-Log M or D-Log2, so those menu entries convert with DJI's LUT. Until it's imported they are approximated with the D-Log curve and marked *LUT not imported, approximated*; choosing one offers to get the LUT.

---

## Where files are stored

| What | Where |
|---|---|
| Cache: proxies, analysis, temporary files | `cache/` in the KangEdit folder. If that folder isn't writable, it goes in the user data folder below. |
| Downloaded FFmpeg | `ffmpeg/` in the KangEdit folder |
| LUT library and sound library | User data folder: `%LOCALAPPDATA%\KangEdit` (Windows), `~/Library/Application Support/KangEdit` (macOS), `~/.local/share/KangEdit` (Linux) |
| Whisper models | The Hugging Face cache (`~/.cache/huggingface`) |
| Settings | The system settings store (registry on Windows, `~/Library/Preferences` on macOS, `~/.config` on Linux) |

Caches are keyed by file, so they're reused across projects. **Tools → Cache folder & clearing…** shows how much space each kind uses, clears them, and changes the cache folder. When you change the folder, KangEdit offers to **copy** the existing proxies and analysis into the new one; the old folder is left untouched. Clearing the cache never affects the LUT or sound libraries.

Earlier versions kept the cache in the user data folder. On the first start after updating, KangEdit offers once to copy it into the new location.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| "Python 3.10 or newer is required" | Install a current Python. On Windows, tick *Add python.exe to PATH*. |
| Linux: Qt "could not load the xcb platform plugin" | Install the system libraries listed under [Linux](#linux-x86-64-or-arm64). |
| Linux: `ensurepip is not available` | `sudo apt install python3-venv` (Debian/Ubuntu). |
| macOS: "run.sh: Permission denied" | Run `chmod +x run.sh run.command` once, or start it with `sh run.sh`. |
| FFmpeg download failed (offline, proxy, firewall) | Install FFmpeg yourself and put it on `PATH`. KangEdit uses it automatically. |
| Playback starts after a short pause | The first frame of long-GOP footage has to be decoded from the previous keyframe. Proxies make the pause much shorter. |
| Choppy playback on a slow computer | Set the viewer to *Color: off while playing* (Ctrl+Shift+B toggles color), and turn on proxies (toolbar → *Use proxies*), then **Generate proxies**. |
| Proxies aren't faster with the GPU | Older cards (e.g. AMD RX 400/500) decode 4K H.265 no faster than a many-core CPU, and can't decode high-bitrate files at all (KangEdit then uses the CPU). Untick *Use the graphics card* in the Proxies dialog. |
| Export uses the CPU | Your GPU has no H.264/H.265 encoder, its driver is too old, or ProRes is selected. Untick *Encode on the graphics card* to force the CPU. |
| Captions are slow | Choose *Auto* or `large-v3-turbo` (or `small`) under *Captions → Style…*: without an NVIDIA GPU, `large-v3` runs on the CPU and is several times slower. |
| Analysis is slow on an older computer | Turn off *Tools → Smarter analysis (AI)*. |
| Sound search fails | MyInstants is unreachable from your network. Import audio files instead. |
| Reporting a problem | *Help → Copy system info* copies your app version, system, library versions, graphics card support and folders. Paste it into your report, along with the log file it names. |

---

## Development and releases

```sh
pip install -r requirements-dev.txt   # app packages + pytest, ruff, PyInstaller
ruff check .                          # lint
pytest                                # tests (settings and libraries go to a temporary folder)
python main.py --self-test            # every module loads and the main window opens (offscreen)
python scripts/build.py               # packaged build for this platform → dist/KangEdit-<version>-<platform>.zip
```

**Branches.** `main` is development: features and fixes go there through pull requests. `stable` holds the latest released version. Use `stable` for a dependable copy, and `main` for the newest changes.

**Versioning.** `kangedit/__init__.py` holds the version (semantic versioning: `MAJOR.MINOR.PATCH`). `python scripts/bump_version.py patch|minor|major|X.Y.Z` changes it. Packaged builds also show the commit and build date in *Help → About* and `--version`.

**Releasing.** A release is a pull request from `main` into `stable`:

1. On `main`, bump the version (`python scripts/bump_version.py minor`), and in `CHANGELOG.md` rename `## [Unreleased]` to the new version (`## [0.2.0] - YYYY-MM-DD`) with a fresh `## [Unreleased]` above it. Commit and merge into `main`.
2. Open a pull request from `main` into `stable` and merge it once CI passes. Its *release-check* job (`python scripts/release_check.py`) makes sure the version is newer than `stable`'s and has a CHANGELOG section.
3. The merge starts the *Release* workflow: it tests, builds every platform and publishes the GitHub Release `v0.2.0` (creating the tag) with the archives, `SHA256SUMS.txt` and the version's CHANGELOG section as the notes.

To build locally instead: `python scripts/build.py` on each platform.

**GitHub Actions** (`.github/workflows/`):

| Workflow | When | What |
|---|---|---|
| **CI** | Every pull request, and pushes to `main` and `stable` | Lint, tests and the self-test on Windows, macOS and Linux with Python 3.10 and 3.12. Pull requests into `stable` also run *release-check*. |
| **Build** | Called by Release, or by hand (*Actions → Build → Run workflow*) | PyInstaller builds for Windows x64, macOS arm64 and Linux x64, kept as workflow artifacts for 30 days. Each build runs its own self-test before it is archived. |
| **Release** | Pushes to `stable` (or by hand) | If `stable`'s version has no GitHub Release yet: tests, builds every platform and publishes `vX.Y.Z` with the archives, `SHA256SUMS.txt`, the CHANGELOG section and generated notes. Otherwise does nothing. |

---

## Project layout

```
main.py                 entry point:  python main.py [project.kgproj | video files…] [--version | --self-test]
install.py              cross-platform setup (venv, packages, FFmpeg)
run.bat / run.sh / run.command   launchers for Windows / macOS + Linux / macOS Finder
requirements.txt        app packages;  requirements-dev.txt: + test, lint and packaging tools
pyproject.toml          pytest and ruff settings
scripts/                build.py (packaged build), bump_version.py (versioning), release_check.py + changelog.py (release PR check)
CHANGELOG.md            release history
packaging/              PyInstaller spec
tests/                  pytest suite
.github/workflows/      CI, Build, Release
kangedit/
  app.py                startup (offers the FFmpeg download when it's missing)
  config.py             paths (app folder, cache, user data), constants, open_path()
  ffmpeg.py             FFmpeg discovery, probing, progress runner, GPU encoder/decoder detection
  ffmpeg_setup.py       FFmpeg download and unpacking (standard library only)
  analysis.py           per-footage signal extraction (OpenCV + FFmpeg, parallel CPU/GPU decode), quick first look
  smart.py              smarter analysis: MobileNetV2 (what shots show) and YAMNet (what is heard), ONNX on the CPU
  autoedit.py           scoring, best-moment selection, presets, transition / J-L placement
  scenes.py             recording dates, scenes (time groups) and their scenic openers
  polish.py             finishing an edit: cuts between words / on the beat, pacing, fillers, shot sizes, loudness
  scriptedit.py         edit from a script: line matching against transcripts (text similarity), clip layout
  bookends.py           story order and opening / closing detection
  edits.py              transitions and split edits shared by preview, export and preview audio
  captions.py           Whisper transcription, caption building, layouts, animations, SRT/ASS
  color.py              log curves, gamut matrices, gamut compression, tone map, grading, preview LUTs, auto color
  camera.py             camera identification and log-profile choice
  lut.py                .cube LUTs, the LUT library and the official camera LUT catalog
  sfx.py                sound library, MyInstants search and download
  images.py             still images and animated GIFs as footage
  speed.py              slow / fast motion and speed ramps (timeline ↔ source time map, FFmpeg setpts / atempo)
  sync.py               syncing recordings by their sound (onset envelope + GCC-PHAT, drift fit)
  multicam.py           multicam layouts and keeping the angles in step with the edit
  faces.py              face detection (YuNet model in assets/) for Zoom to face and best-moment scoring
  updater.py            update check and install from the GitHub releases
  export.py             FFmpeg filtergraph rendering (+ preview audio mix)
  preview.py            background preview engine: decode thread + render pool, read-ahead, pre-roll, transitions
  decoder.py            PyAV frame-accurate decoding
  reframe.py            aspect-ratio crop/fit geometry (shared by preview and export)
  proxy.py              proxy generation (GPU decode/encode with CPU fallback)
  models.py             project model, undo, save/load (.kgproj)
  bins.py               Media panel bins (by hand or by recording date) and filters
  tasks.py              background task queue
  ui/                   PySide6 interface: viewer, timeline, media bin, inspector, video layers,
                        captions, sounds, dialogs
```

---

## License

KangEdit is free to use, change and share under the [MIT License with the Commons Clause](LICENSE).

| | |
|---|---|
| **Allowed** | Using it for anything: personal videos, professional and client work, monetized YouTube / TikTok / Instagram channels, schools, companies. Changing it, and sharing it or your changed version for free (with the license). The videos you make with it are yours. |
| **Not allowed** | Selling KangEdit: selling copies or a changed version of it, charging for it as part of a product, or offering a paid service (hosting, support, consulting) whose value comes mainly from KangEdit. |

To sell KangEdit or build a paid product or service on it, ask for a separate license: esha@angkasa.dev.

The Commons Clause makes this "source available" rather than open source in the OSI sense. Third-party software
used by KangEdit keeps its own license: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Third-party software

KangEdit is written in Python and uses [PySide6 / Qt](https://www.qt.io/qt-for-python), [NumPy](https://numpy.org), [OpenCV](https://opencv.org), [PyAV](https://pyav.org) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper models via CTranslate2).

Face detection uses the [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) model by Shiqi Yu et al. (MIT licence), included as `kangedit/assets/face_detection_yunet_2023mar.onnx`.

FFmpeg isn't included in this repository. When needed, setup downloads static GPL builds from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (Windows, Linux) or [Martin Riedl's build server](https://ffmpeg.martin-riedl.de/) (macOS). Those builds are covered by FFmpeg's own license.
