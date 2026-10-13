# PyInstaller spec for KangEdit (one-folder build; a .app bundle on macOS).
# Built by scripts/build.py, which also writes kangedit/_build_info.py and archives the result.

import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
with open(os.path.join(ROOT, "kangedit", "__init__.py"), encoding="utf-8") as fh:
    VERSION = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)

ASSETS = os.path.join(ROOT, "kangedit", "assets")
datas = [(os.path.join(ASSETS, f), "kangedit/assets") for f in os.listdir(ASSETS) if f.endswith((".png", ".onnx"))]   # icons, face detector
datas += collect_data_files("faster_whisper")           # Silero VAD model used by transcription
datas += collect_data_files("yt_dlp_ejs")               # YouTube challenge solver scripts (music download)
binaries = collect_dynamic_libs("ctranslate2")          # Whisper inference runtime
hiddenimports = collect_submodules("kangedit")          # UI modules are imported lazily
hiddenimports += collect_submodules("yt_dlp") + collect_submodules("yt_dlp_ejs")   # extractors load lazily

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
              "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtDesigner"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KangEdit",
    console=False,               # GUI app; --version / --self-test report via $KANGEDIT_SELFTEST_LOG
    disable_windowed_traceback=False,
    argv_emulation=False,
    upx=False,
    icon=os.path.join(ASSETS, "kangedit.ico"),
)

coll = COLLECT(exe, a.binaries, a.datas, name="KangEdit", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="KangEdit.app",
        icon=os.path.join(ASSETS, "kangedit.icns"),
        bundle_identifier="app.kangedit.KangEdit",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "14.0",   # the bundled Qt 6, onnxruntime and NumPy need macOS 14
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "KangEdit project",
                "CFBundleTypeExtensions": ["kgproj"],
                "CFBundleTypeRole": "Editor",
            }],
        },
    )
