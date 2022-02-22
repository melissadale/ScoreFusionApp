# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Building Exe

* `python -m PyInstaller --name Fusion ..\ScoreFusion\main.py --noconfirm`
*  update spec file
* `python -m PyInstaller .\Fusion.spec --noconfirm`

### Spec File Update ###

---

```
from kivy_deps import sdl2, glew
# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['..\\ScoreFusion\\main.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=['win32timezone'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='Fusion',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe, Tree('..\\ScoreFusion\\'),
               a.binaries,
               a.zipfiles,
               a.datas, 
               *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Fusion')
```


### Debugging ###
`cd C:\Users\melis\Documents\pythonProject\FusionApp\dist\Fusion`
`.\Fusion.exe`