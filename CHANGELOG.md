# Changelog

All notable changes to KangEdit are listed here. Versions follow [semantic versioning](https://semver.org).
Note changes under [Unreleased]; a release (a pull request from `main` into `stable`) renames it to the new version,
and the Release workflow uses that section as the GitHub release notes.

## [Unreleased]

## [0.6.5] - 2026-10-17

### Fixed
- A long timeline (hundreds of clips, such as an edit with fillers taken out) played without sound and
  couldn't be exported: the whole timeline went to one FFmpeg run with an input per clip, more than a command
  line may hold on Windows (or files a program may open on macOS). The preview sound and the export of a long
  timeline are now made in parts, cut only at straight cuts, and joined; music, titles, captions and fades go
  on in a final pass. When the preview sound can't be made, the status bar now says why.

### Changed
- The hook of a long-form auto edit takes each of its moments from a different scene (another time, day or
  place) while there are scenes left, so it shows more of the trip.

## [0.6.4] - 2026-10-17

### Added
- **Whole sentences** (✨ Auto Edit → Finishing, on by default; *Tools → Finish the edit → Cut on whole
  sentences*): in a long-form edit every clip with speech starts and ends on complete sentences of the
  transcript. A sentence a clip cuts into is taken in whole, or left out when the clip would get too long.
- **Full transcript** (*Tools → Full transcript…*, or right-click media → *Show the full transcript…*): every
  file's transcript merged in recording order, a sentence per line, with or without file names and times;
  *Copy all* or save as .txt, and *Transcribe the rest* for files without one.
- After an import, KangEdit offers to transcribe every file with sound that has no transcript yet (*Don't ask
  again*, or *Tools → Offer to transcribe after importing*).
- Right-click a timeline clip (main track or video layer) → **Show in Media panel**: its footage is selected
  there, clearing a bin or filter that hid it.
- Dates in file names (`VID_20260501_091005`, `PXL_…`, `DJI_…`, `2026-05-01 09.10.05`, `IMG-20260501-WA0001`…)
  count as a recording date: *Bins by date*, story order and scenes use them.

### Fixed
- *Use every footage at least once* now uses every usable file (0.6.3 stopped at 90% once past the target
  length). With more files than fit even at the shortest clip length, the edit runs longer than the target.
- "Clips overlap / too close to another clip": the pieces *Remove fillers* leaves of one take (cut straight
  together, one after the other) were counted, so an edit could show dozens of warnings with nothing repeated.
  They aren't any more, and an auto edit now leaves out any clip that would still show a moment another clip
  shows (closer than the gap between reuses).
- Cuts on the beat no longer shorten a clip into its last words.

## [0.6.3] - 2026-10-16

### Changed
- Auto edit: the target length shows minutes (and hours) once it passes two minutes, e.g. "12 min 30 s"; you can
  type "12:30", "12 min", "1h 5m" or plain seconds.
- Auto edit: on a screen wide enough (1360 px or more) the window opens in two columns, so it needs no scrolling.
- *Use every footage at least once*: about 90% of the usable files now get in, not a minority of them. Clips
  are shortened to share the target length between files, and the edit may run a little past the target to
  fit more of them in.

### Fixed
- Media panel in thumbnail view: widening the panel made thumbnails smaller, with the same number per row.
  They now grow with the panel, keeping the same margins.
- The refine panel could not be made narrower than its widest tab, too wide on a Full HD screen. It can now be
  narrowed to 300 px (wide tabs scroll sideways), and its width is remembered.

## [0.6.2] - 2026-10-16

### Fixed
- Media panel → *Transcribe for captions* transcribed every selected file again, even the ones already
  transcribed. It now does only those without a transcript (saying how many it kept); when all of them have one,
  it asks before transcribing them again.

## [0.6.1] - 2026-10-16

### Fixed
- Captions: the speech model is loaded once (downloaded on first use), then every file is transcribed in one
  task with file-by-file progress; it no longer showed "Loading speech model" for each file. The auto edit's
  quick listening model no longer pushes out the captions model (they reloaded each other), a file that fails
  doesn't stop the others, and cancelling keeps the files already done.
- Auto edit: footage shorter than about 2 s made every auto edit fail; *Remove fillers* failed when a clip ended
  on a filler word.
- Finishing steps (pacing, shot variety, cuts between words / on the beat) could move clips onto moments the
  edit must not show (the rest of the timeline, parts not every camera recorded, what a highlight reel's edit
  left out) and ignored the gap between reuses; shot variety could swap a clip to before the one it follows.
- B-roll footage could split a scene in two or shorten the edit for a scene opener that never came: scenes now
  come from the edited footage, and B-roll joins the scene it was recorded in.
- Windows updates: an accented user name, or %, & or parentheses in a path, could make the update helper copy
  nothing (it now takes every path from its environment).
- Script edit: a speaker label on a line of its own ("ANNA:") was treated as a line to find.
- The quick listening used to find the opening and closing kept transcribing after giving up; a model download
  could hold up other transcriptions.
- The preview's grading crashed on empty, RGBA or very wide pictures.

## [0.6.0] - 2026-10-16

### Added
- Long-form auto edit: **scenes** are time groups of the footage (a gap of over 30 minutes between recordings,
  such as another day or place, starts a new one). Each scene after the first opens with a scenic / travel shot
  from that scene (a view without faces or talking, sharp, colourful, steady or gently panning) when it has one,
  and *Transitions between scenes* play between these scenes rather than at every change of file.
- Media panel: right-click → *Bins* → **Clear all bins** (the media stays in the project; undoable).
- **Finishing** options in ✨ Auto Edit (and *Tools → Finish the edit* for any timeline):
  - **Cut between words**: a cut that falls inside a word moves to the pause next to it (the transcript's word
    timings, else a clearly quieter moment), up to 0.6 s.
  - **Cut on the beat**: with music on the timeline, each cut moves to its nearest beat (detected from the music,
    looping music included), never into a word.
  - **Pacing**: steady, build-up (long shots first, faster and faster), trailer (quick, longer, quick) or calm
    finish; the clip lengths change along the edit within the clip lengths set.
  - **Remove fillers, long pauses and false starts** ("um", "eh", "anu", silences over 0.9 s, words said twice in
    a row) from the clips using the transcripts; missing transcripts are made first. They become jump cuts.
  - **Vary the shot sizes**: where three close-ups, medium or wide shots would follow each other, the middle one
    is swapped for a nearby moment of another size that scores nearly as well.
  - **Even out the loudness**: each clip's sound is brought to the edit's typical level (at most ±9 dB).
- Moments with the lens covered (a finger, the camera in a pocket), a blown-out picture or the focus hunting are
  avoided when choosing the best moments.
- **Highlight reel**: *Tools → Make a highlight reel (9:16)* turns the edit into a 30, 45 or 60 s vertical reel of
  its best moments (hook first, trailer pacing, on the beat, framed on the faces); Undo brings the edit back.
- Scenes also split by **place**: files that carry a location (phones, action cameras, drones) more than 1 km
  apart are different scenes.
- **B-roll bin**: right-click media → *Add to B-roll bin*. Scenic scene openers come from it first, even when it
  wasn't among the files chosen for the edit.
- A scene's scenic opener brings its natural sound in 1.2 s early, under the end of the scene before (also
  through a transition: the sound crossfade starts before the picture one).
- Multicam switching cuts to the other angles at a laugh, applause or an exclamation (a short burst in the
  sound), most of all to an angle whose picture moves.

### Changed
- Auto ducking listens to the voice band of the edit's sound, so speech turns the music down and wind or rumble
  doesn't.
- The recording date of a file is the oldest one it carries: its creation time tag, its file creation date or its
  modification date (copying a file renews its creation date; a camera with its clock unset writes a wrong tag,
  and dates before 1995 are ignored). Bins by date, story order and the choice of the files searched for the
  opening and closing all use it.

## [0.5.1] - 2026-10-16

### Fixed
- Multicam, *Use the whole recording*: the target length now follows the main angle's recording (and can't be
  changed, as the whole recording is used); turning the option off brings back the length set before.
- Multicam switching: an angle could stay on screen far longer than the clip lengths in the Auto Edit settings
  (one angle for 30 s at the start, layer clips much longer than the longest clip). Every shot now lasts
  between the shortest and longest clip length: an angle held longer gives way to another as often as needed
  (only where no other camera is recording can one stay longer).
- Multicam switching with the whole recording: the transitions from the Auto Edit settings weren't applied.
  The main track is now cut at every angle change, so the transitions (and the angle shots) play there; the
  recording keeps its length and stays in sync.
- Windows updates: the restart helper could leave some files of the old version in place (robocopy skips files
  of the same size and time stamp, which a new build can have). It now copies every file, and writes what it
  did to `update.log` in the logs folder.

## [0.5.0] - 2026-10-16

### Added
- **Edit from a script**: ✨ Auto Edit → **📜 Script** (or *Tools → Auto Edit from script…*). Paste the script;
  each line is found in the footage's transcripts by text similarity and the clips are laid out in the script's
  order. An adjustable **similarity threshold** decides how close a line must be (recognition slips such as
  "their" / "there" or a dropped word still match), a line said in several takes uses the best match or the last
  good take, clips are cut on word boundaries with adjustable padding (never into the next words), and lines said
  one after the other in the same take become one clip. A live preview lists each line with its score, file and
  time, and lines that weren't found are listed in red and reported afterwards. Footage without a transcript is
  transcribed first. Replaces the timeline or adds after it; undoable.
- Media panel **bins**: a *Bin* list above the media (all media, not in a bin, each bin with its count), a *Bin*
  column, and right-click → *Bins*: new bin, move the selected media to a bin (or out of it), rename or delete a
  bin (its media stays), and **Bins by date**, which sorts the media into a bin per recording day (from the
  file's creation time, else its date). Bins are saved with the project and undoable.
- Media panel **filters**: show videos or photos / images only, and only one picture shape: landscape, portrait,
  square, 16:9, 9:16, 4:3, 3:4 or 21:9 (as shown: a phone clip filmed upright is portrait).
- Multicam: *Use the whole recording (no best-moment cuts)* in ✨ Auto Edit → Multicam puts the main angle's
  whole recording on the timeline, so Switching cuts between the cameras all the way through, like editing a
  full multicam event.
- Multicam sync uses the longest recording as the reference: every file is synced to it and each angle's offset
  to the main angle is worked out through it (files that don't match it are synced directly, as before).
- Ctrl+A selects the clips on the video layers too (multicam angles, picture in picture...), and Ctrl+click adds
  a layer clip to the selection: color, LUT, log, speed and volume changes and Delete apply to them as well.
- DaVinci Resolve LUTs with a 1D shaper before the 3D LUT (LUT_1D_SIZE + LUT_3D_SIZE in one .cube) import and
  apply correctly.

### Changed
- Log → Rec.709 conversion is more accurate and looks better:
  - The filmic tone map keeps hues: it maps each colour's brightness and keeps its channel ratios, so bright
    saturated colours no longer drift (red to orange, blue sky to cyan; hue shifts in bright colours went from
    about 16° on average, up to 41°, to about 6°), and the brightest colours fade to white like film instead of
    clipping as flat patches. Middle grey and the grey scale are unchanged.
  - Colours the camera recorded outside Rec.709 (neon, LED lights, deep blues, flowers) are compressed smoothly
    into it, as the ACES reference gamut compression does, instead of clipping, so they keep their gradations.
  - Files recorded in full range (0–1023 code values) are read with the camera's curve correctly; they came out
    flat before (raised blacks, dull highlights).
  - The export LUT is 65 points per side instead of 33, and follows the exact grade 7 times more closely (mean
    error under half a 10-bit level); clips with the same grade share one LUT file.
  - The preview grades with every blue level exact and smooth (interpolated) red and green, which removes the
    banding the old table showed in gradients (error about one 8-bit level on a grey ramp, before up to 3 with
    steps of 6), and is a little faster.
- Export: grading runs at 16 bits per channel in planar RGB, and the colour conversion is done in the same pass
  as the resize, which exports faster (about 6 % in a test project) and more precisely.
- Proxies are encoded to decode about 1.8× faster (smoother scrubbing on slow computers), and big downscales
  (4K → 540p) average the pixels, so fine detail doesn't shimmer.

### Fixed
- The preview could show a different frame from the export (up to half a second off) for recordings whose video
  starts after the audio: the preview now counts time from the file's start, as the export does.
- Export: HD clips without colour tags and picture-in-picture / multicam layer clips were converted with the
  wrong colour matrix (slightly off colours, mostly in greens and skin tones).
- The update check at startup only ran once every 20 hours (and a manual check reset that), so a new release
  often showed no popup. It now runs at every start (not again within 15 minutes).
- Windows: after *Update now*, a console window with `find "<number>"` appeared and the update could hang:
  the restart helper ran without a console, so each of its commands opened a window of its own and its pause
  failed. It now runs hidden, waits with a command that works there, and gives up waiting after two minutes.
- Ctrl+A in the Media panel selects all the media shown there (it selected the timeline's clips instead); in a
  text field it selects the text.
- A proxy that hangs on one file no longer turns GPU proxies off for every file: a GPU way that hung is skipped
  for that file (and for all only after it hung on two). GPU attempts give up after 30 s without progress, and
  a CPU attempt after 5 minutes with an error instead of staying at 0 %.
- Camera RAW files FFmpeg can't decode (Sony X-OCN in MXF, Blackmagic RAW, REDCODE, ARRIRAW...) now say so and
  what to do (convert to ProRes, DNxHR or H.264 / H.265 first) instead of "Could not decode frames".

## [0.4.0] - 2026-10-13

### Added
- Slow and fast motion: any clip (main track or video layer) plays from 0.1× to 8×. Right-click → *Speed* for
  0.25×, 0.5×, 0.75×, 1.5×, 2×, 3×, 4× or a custom speed, or use the Inspector's new **Speed** group (one-click
  presets and a slider; the Layer tab has the same for layer clips). The clip keeps its in and out points and gets
  longer or shorter on the timeline; its sound is time-stretched with its pitch kept.
- Speed ramps: templates that change the speed along the clip, smoothly: *Hero moment* (fast in, slow motion in
  the middle, fast out), *Bullet time*, *Slow → fast*, *Fast → slow*, *Slow finish*, *Speed burst*, *Pulse* and
  *Montage*. A ramp combines with the clip's speed (a 2× *Pulse* is twice as fast throughout). Ramps follow the
  footage frame for frame in the preview and the export, and the sound follows the ramp.
- Sound at another speed, per clip (right-click → *Speed* → *Sound at this speed*, the Inspector's Speed group
  or the Layer tab): *Synced, pitch kept* (the default), *Synced, pitch follows speed* (slow motion sounds deeper,
  fast motion higher, like a tape), *Not synced (normal speed)* (the sound plays at normal speed from the in point,
  whatever the picture does) or *Muted*. Captions follow the sound.
- Smooth slow motion in the export: slowed-down parts blend neighbouring frames instead of repeating them (an
  option in the Export dialog, on by default).
- Speed works with everything else: transitions and J / L cuts use the footage around the clip at its speed,
  captions follow the new timing, the timeline's filmstrip and waveform show the clip as it plays (and its speed in
  the clip header), trims and slips move at the clip's speed, and multicam angles play at their main clip's speed.
- Refine clips on every video layer, not only the main track: select a clip on V2, V3… and the **🎨 Color** /
  Clip tab refines it like a main-track clip: log conversion and camera detection, exposure and color, ✨ Auto,
  look LUT, source in / out, speed and volume (its position, size and opacity stay in the 🧩 Layer tab, which
  has a button to the color tools). Right-click a layer clip → *Color* for color, auto color, reset and copying
  its grade. *Auto all clips* and *Copy to all clips* now cover the video on every layer too (pictures on the
  layers, such as logos, are left as they are).
- Multicam **Switching** layout (✨ Auto Edit → Multicam → Layout, or *Tools → Multicam layout*): one camera fills
  the frame at a time and the edit cuts between the angles at the best moments, like a live multicam edit,
  instead of showing every angle together. Each angle is scored along the recording (who is speaking, judged
  by which camera's microphone is loudest, the closer face, the subject's activity, sharpness, shake) and the
  cuts are chosen with a minimum shot length of 2 s, no cutting on noise, and a short cutaway when one angle
  is held too long. The shots follow trims, slips and splits like the other layouts; the sound stays on the
  main angle.
- README: *System requirements* (minimum Windows, macOS and Linux versions for the ready-made builds and from
  source, and hardware).

### Fixed
- Generating proxies could stay at 0 % forever: on some machines (seen with Intel Quick Sync on Windows) the
  GPU decode + encode never starts, and the stuck FFmpeg also blocked every task queued after it. A GPU attempt
  that makes no progress for 60 s is now stopped and the proxy is made the next way (CPU decoding, then fully on
  the CPU); a way that hung is skipped for the rest of the session. A GPU export that hangs is redone on the
  CPU the same way (after 3 minutes), and *Cancel* now stops an FFmpeg that has gone silent.
- Transitions in multicam edits: the camera angles on the layers cut hard while only the main angle
  dissolved, so a transition hardly showed (in the boxed layouts most of the frame jumped). The angles now take
  part: a dissolve-like transition cross-fades them too, a dip fades them out and in.
- Undo after changing the multicam layout left the new layout's settings behind, so the next edit rebuilt the
  angles in the undone layout. The multicam settings are now part of undo.
- The preview gives overlapping clips on one video layer their own decoders, so a cross-fade between two
  clips of the same camera plays without seeking back and forth.
- The macOS build now declares macOS 14 as its minimum (it said 12, but the Qt, onnxruntime and NumPy it bundles
  need macOS 14), so older Macs get a clear message instead of a crash.

## [0.3.0] - 2026-10-12

### License
- KangEdit now has a license: the MIT License with the Commons Clause (`LICENSE`). It may be used for anything,
  including paid and monetized video work, changed and shared for free, but KangEdit itself may not be sold
  (copies, changed versions, paid products or services built mainly on it). Third-party licenses are listed in
  `THIRD_PARTY_NOTICES.md`; both files ship with the packaged builds, and *Help → About* summarizes the terms.

### Added
- Multicam in ✨ Auto Edit: give each recording an angle (main, angle 2, 3, 4; guessed from the camera), and the
  angles are synced by their sound to within a millisecond, clock drift between cameras included. The main
  angle is edited as usual and every other angle follows on the video layers, cut at the same moments. Layout
  templates: picture in picture, main video + talking cam (a square box cropped around the speaker's face;
  main on top and speaker below in vertical video), stacked top / bottom, side by side and a 2 × 2 grid.
  Choose whose sound is used, and whether to keep to moments every angle recorded. The angles stay in sync
  after trims, slips, splits, *Other moment* and new hooks; *Tools → Multicam layout* (or a layer clip's
  right-click menu) switches the layout afterwards. Layer clips can now be cropped to fill a box (Crop X / Y).
- Multicam angles are linked to their main clip: dragging an angle slips the whole group, its edges trim it and
  R (Other moment) works on it, all angles staying in sync, also while dragging. **🔗 Link** (Ctrl+L, in the
  timeline toolbar, the right-click menus and the Layer tab) unlinks angles to edit them on their own and links
  them back in sync. N angles take the main track and N - 1 layers: in the boxed layouts the main angle is shown
  in its box on the main track instead of a copy on a layer.
- Images: import JPG, PNG, GIF, BMP, WebP and TIFF. An image on the main track is a 5-second clip that can be
  trimmed to any length; animated GIFs loop; PNG transparency is kept on video layers.
- Video layers (V2, V3…) above the main track for picture in picture, logos, B-roll and images on top: drag
  footage or images from Media onto a layer, place them by dragging in the viewer (mouse wheel resizes), and
  set position, size, rotation, opacity, fades and volume in the new **Layer** tab. Layer video sound is mixed in.
- Rotation for clips (Reframe → Rotation, ⟲ / ⟳ 90° and the right-click menu) and for layer clips. Quarter
  turns turn the footage before it fills the frame.
- Adjustable lane heights: drag the bottom edge of the video track, the video layers or the sound lane, or use
  **↕ Lanes** (sliders, *Fit to panel*). The lanes scroll up / down when taller than the panel.
- Zoom − / + buttons (Ctrl+− / Ctrl+=) next to the zoom slider; **Fit** now fits every lane, not only the main
  track.
- Auto ducking: music turns down under the clips' speech and sound (on for NCS music; per sound in the Sounds
  tab or its right-click menu, with a project-wide *Duck by* amount).
- The Media list shows which footage the edit uses (**In edit** column, green ✓ on the thumbnails); footage can
  be dragged onto the timeline.
- *Fit to timeline* makes one sound clip that loops (with crossfades between repeats) instead of several
  overlapping copies on two rows; a 🔁 ×N label and dashed lines show the repeats, and it can be made longer
  by dragging its end.
- Help for video layers: the **Layer** tab explains how to add, place and time layer clips (and shows quick
  tips while one is selected), the guided tour has a *Video layers* step, and the README a *Video layers and
  multicam* section.
- *View → Reset layout* (Ctrl+Shift+R): panels, the viewer / timeline split, lane heights and tabs back to
  their usual sizes, and the timeline fitted.
- *Help → About* shows the GitHub repository and a contact address.
- Updates from GitHub: *Help → Check for updates…*, and a daily check at startup. Packaged builds download,
  verify and install the new release and restart; source copies update their files; git checkouts `git pull`.

### Changed
- Color is processed in the order colorists use: log curve and gamut → exposure and white balance (in
  linear light) → contrast, shadows and highlights (in stops around middle grey) → saturation and vibrance
  (linear light) → output tone map → creative look LUT → halation and sharpening. With a conversion LUT,
  exposure and white balance now act on the log signal before the LUT, so highlights the LUT would clip can be
  recovered. Graded clips may look slightly different than in 0.2.

### Fixed
- Without FFmpeg, analysis and syncing report "FFmpeg was not found" instead of failing with an internal error.
- With several clips selected (e.g. Ctrl+A), *Import LUT (.cube)…* in the clip's Camera profile or Look list
  applied the imported LUT to one clip only; it now applies to every selected clip. A camera detection that
  finishes after you picked a profile by hand no longer overwrites your choice.
- After *Fit*, the timeline no longer has a horizontal scroll bar (it only shows when there is more to see),
  and long edits (over ~10 minutes) now fit too.
- Zoom to face finds faces far more often: it now uses OpenCV's YuNet face detector (faces turned away, tilted,
  small, blurred, dark or in flat log footage), looks at 9 frames at 720p instead of 5 at 480p, follows the
  clip's rotation, and picks the face seen in the most frames. With OpenCV 5 (what a new install gets) the old
  detector was missing entirely, so no face was ever found; the best-moment face score had the same problem.

## [0.2.1] - 2026-10-12

### Fixed
- On smaller displays, the toolbar and the timeline's buttons wrap onto a second row instead of hiding in
  a » menu (which could not be clicked): Export, Hook, Other moment, Zoom and Fit are always visible.

## [0.2.0] - 2026-10-12

### Added
- *Help → Copy system info*: app version, OS, CPU, memory, library versions, FFmpeg, GPU decode / encode
  support, screens and folders, ready to paste into a bug report.

### Changed
- The window opens maximized on the display the mouse pointer is on.
- The Media and Refine panels are fixed in place (no more undocking); drag their borders to resize them.
- The timeline toolbar moves buttons that don't fit into a » menu instead of widening the window.

### Fixed
- On smaller displays (e.g. 1440×900) the window no longer grows past the screen: every Refine tab
  scrolls to its bottom and the panels can be resized.
- Python 3.10 (PyAV 17): a crash when two preview threads converted the same frame.

## [0.1.0] - 2026-10-12

First public release.

### Automatic editing
- Best-moment analysis (sharpness, motion and shake, exposure, colorfulness, faces, loudness, scene cuts) on several CPU lanes plus the GPU video decoder.
- Short-form (9:16), long-form (16:9) and custom presets, story order from recording time, and opening / closing detection in English and Indonesian.
- Transitions, J/L cuts, reuse rules, hooks, and *Regenerate shorter / longer*.

### Editing
- Pipelined preview that keeps up with 4K H.265, timeline with trim, slip, multi-select, undo / redo, autosave with rolling backups and recovery.
- Animated titles and text overlays with templates.
- Platform overlays (Instagram Reels, TikTok) and full-screen viewer.

### Color
- Log → Rec.709 for Sony, Panasonic, Canon, ARRI, Fujifilm, Nikon, DJI, Apple and Blackmagic, with camera auto-detection.
- Official camera LUTs, `.cube` LUT import, exposure and color controls, auto color; color on / off in the viewer.

### Captions
- Offline captions with faster-whisper, word animations, automatic placement, burn-in and `.srt` output.

### Sound
- Sound-effect layer with MyInstants search, volume, fades and crossfades.
- Background music: NCS suggestions by mood, YouTube audio download, preview and fit to timeline.

### Export
- 16:9, 9:16, 1:1, 4:5, 4:3, 2.39:1 up to 4K in H.264, H.265 or ProRes 422 HQ, with GPU encoding and CPU fallback.
- Optional proxies for lighter editing.

### App
- App and tray icons, crash and error logging, *Help → About* with version and build.
