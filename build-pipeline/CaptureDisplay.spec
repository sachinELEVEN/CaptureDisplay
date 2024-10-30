# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['../main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['keyboard-listener','save_copied_text_to_file','one-time-cursor-info','menu_bar_app','fast-screen-recording','utils'],  # Add dependencies here
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='CaptureDisplay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['../assets/capturedisplay.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CaptureDisplay',
)
app = BUNDLE(
    coll,
    name='CaptureDisplay.app',
    icon='../assets/capturedisplay.icns',
    bundle_identifier='com.capturedisplay.macapp',
)
