# -*- mode: python ; coding: utf-8 -*-
import os

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
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CaptureDisplay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    runtime_node_dir='CaptureDisplay-v1.1.5',
    
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['../assets/CaptureDisplay.icns'],
)
app = BUNDLE(
    exe,
    name='CaptureDisplay.app',
    icon='./assets/CaptureDisplayX.icns',
    info_plist={
        'NSAppleEventsUsageDescription': 'CaptureDisplay needs accessibility permission so that you can control the app via keyboard shortcuts',
        'com.apple.security.input-method': True,
        'NSDesktopFolderUsageDescription': 'CaptureDisplay will store logs and settings file here.',
        'CFBundleIdentifier': 'com.CaptureDisplayv3.macapp',
    },
    bundle_identifier='com.CaptureDisplayv3.macapp',
)
