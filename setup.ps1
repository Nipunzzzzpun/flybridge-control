# One-time Windows setup: creates a Python 3.12 venv, installs everything, and makes a desktop shortcut.
# Run from the repo folder:   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$dir = $PSScriptRoot
py -3.12 -m venv "$dir\venv"
if ($LASTEXITCODE -ne 0) { Write-Host "Python 3.12 not found. Install it from python.org first."; exit 1 }
$py = "$dir\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r "$dir\requirements.txt"
Set-Content "$dir\Run_FlyLab.bat" "@echo off`r`ncd /d `"$dir`"`r`n`"$py`" flylab.py`r`npause"
$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$env:USERPROFILE\Desktop\FlyLab.lnk")
$s.TargetPath = "$dir\Run_FlyLab.bat"; $s.WorkingDirectory = $dir; $s.Save()
Write-Host "Done. Double-click the FlyLab shortcut on your desktop."
