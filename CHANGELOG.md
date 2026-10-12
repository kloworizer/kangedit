# Changelog

All notable changes to KangEdit are listed here. Versions follow [semantic versioning](https://semver.org).
Note changes under [Unreleased]; a release (a pull request from `main` into `stable`) renames it to the new version,
and the Release workflow uses that section as the GitHub release notes.

## [Unreleased]

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
