# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('./assets/', 'assets')],
    hiddenimports=['keyboard-listener','save_copied_text_to_file','one-time-cursor-info','menu_bar_app','fast-screen-recording','utils','settings_file_manager'],  # Add dependencies here
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CaptureDisplayX77',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CaptureDisplayX77',
)
app = BUNDLE(
    coll,
    name='CaptureDisplayX77.app',
    icon='./assets/CaptureDisplayX.icns',
    info_plist={
        'com.apple.security.input-method': True,
    },
    bundle_identifier='com.CaptureDisplayX77.macapp',
)
