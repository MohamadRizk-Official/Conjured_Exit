# setup_cv_env.ps1 -- create or refresh the `cf64` venv for the Pathcaster tracker (OpenCV).
#
# Why a second venv: PyPI has NO win_arm64 wheels for opencv-python / opencv-contrib-python
# (or their headless variants), so on the Windows-on-ARM64 dev laptop (Snapdragon X) OpenCV can
# only run inside an x64 CPython under Windows' x64 emulation (Prism).  `cf64` is that x64
# Python 3.12 venv: opencv-contrib-python + numpy + scipy + bleak, so the tracker and the BLE
# link can live in one process.  On ordinary x64 machines the normal 3.12 interpreter is
# already x64 and this script simply uses it.  Measured on the ARM64 laptop (tools\cv_perf.py):
# a full 2-camera tracking cycle costs ~4 ms under emulation, well inside the 15 ms / 30 Hz budget.
#
# Run from PowerShell in the project root:   .\setup_cv_env.ps1
#   (if scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
#   -Recreate   delete and rebuild cf64 (e.g. it was created from the wrong interpreter)
#   -NoInstall  only discover an interpreter; never install one
#
# Idempotent.  Does NOT touch PATH, the default `python`, the `py` launcher default, the `cf`
# venv (bleak/cflib, native ARM64) or any Bluetooth setting.  The x64 Python it may install is a
# per-user python.org build under %LOCALAPPDATA%\Programs\Python\Python312 with PrependPath=0.
#
# Afterwards:   .\cf64\Scripts\Activate.ps1
#               python tools\cv_perf.py          # OpenCV speed check (needs < 15 ms per cycle)
#               python tools\camera_probe.py     # which camera index / backend to use
param(
    [switch]$Recreate,
    [switch]$NoInstall
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$venv = Join-Path $root "cf64"
$venvPy = Join-Path $venv "Scripts\python.exe"
$packages = @("opencv-contrib-python", "numpy", "bleak", "scipy")

# $true when $exe is a CPython 3.12 built for x64 (AMD64).  We check the compiler tag in
# sys.version because platform.machine() reports the HOST (ARM64) even inside an emulated
# x64 process, and the py launcher's "-64" flag also matches ARM64 builds.
function Test-X64Py312([string]$exe) {
    if (-not $exe) { return $false }
    if (-not (Test-Path $exe)) { return $false }
    try {
        $out = & $exe -c "import sys; print(sys.version_info[:2] == (3, 12) and 'AMD64' in sys.version)"
        return ("$out".Trim() -eq "True")
    } catch {
        return $false
    }
}

# Look everywhere an x64 CPython 3.12 normally lives.  First match wins.
function Find-X64Py312 {
    $cands = New-Object System.Collections.Generic.List[string]
    # python.org / winget per-user and all-users default locations
    $cands.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"))
    $cands.Add("C:\Program Files\Python312\python.exe")
    # Registry: PythonCore\3.12 is the x64 build (3.12-arm64 and 3.12-32 are the other tags)
    foreach ($key in @("HKCU:\Software\Python\PythonCore\3.12\InstallPath",
                       "HKLM:\Software\Python\PythonCore\3.12\InstallPath")) {
        try {
            $props = Get-ItemProperty -Path $key -ErrorAction Stop
            if ($props.ExecutablePath) { $cands.Add($props.ExecutablePath) }
            if ($props."(default)") { $cands.Add((Join-Path $props."(default)" "python.exe")) }
        } catch { }
    }
    # Whatever the py launcher / PATH call 3.12 (on x64 machines this is just the normal 3.12)
    try { $p = & py -3.12 -c "import sys; print(sys.executable)"; if ($p) { $cands.Add("$p".Trim()) } } catch { }
    foreach ($name in @("python3.12", "python3", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $cands.Add($cmd.Source) }
    }
    foreach ($c in $cands) {
        if (Test-X64Py312 $c) { return $c }
    }
    return $null
}

# Per-user x64 CPython 3.12 via winget.  --architecture x64 matters: on ARM64, winget otherwise
# treats the ARM64 3.12 as "already installed".  Include_launcher=0 keeps the user's py launcher.
function Install-X64Py312 {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw ("No x64 CPython 3.12 found and winget is unavailable. Install it manually from " +
               "https://www.python.org/downloads/windows/ -> 3.12.x 'Windows installer (64-bit)', " +
               "per-user, untick 'Add python.exe to PATH'. Then rerun this script.")
    }
    Write-Host "No x64 CPython 3.12 found -> installing per-user via winget (no PATH / launcher change)"
    & winget install --id Python.Python.3.12 --architecture x64 `
        --accept-package-agreements --accept-source-agreements --disable-interactivity `
        --override "/quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_launcher=0"
}

# ---------------------------------------------------------------------------- venv
if ($Recreate -and (Test-Path $venv)) {
    Write-Host "Removing $venv"
    Remove-Item -Recurse -Force $venv
}
if (Test-Path $venvPy) {
    if (Test-X64Py312 $venvPy) {
        Write-Host "Reusing existing venv cf64"
    } else {
        Write-Host "cf64 exists but is not an x64 Python 3.12 -> rebuilding it"
        Remove-Item -Recurse -Force $venv
    }
}
if (-not (Test-Path $venvPy)) {
    $py = Find-X64Py312
    if (-not $py) {
        if ($NoInstall) { throw "No x64 CPython 3.12 found (rerun without -NoInstall to let winget install it)." }
        Install-X64Py312
        $py = Find-X64Py312
        if (-not $py) { throw "x64 CPython 3.12 still not found after the install; check the winget output above." }
    }
    Write-Host "Creating venv cf64 from $py"
    & $py -m venv $venv
    if (-not (Test-Path $venvPy)) { throw "venv creation failed" }
}

& $venvPy -c "import os, platform, struct, sys; print('cf64 Python', sys.version.split()[0], '| build', 'x64' if 'AMD64' in sys.version else platform.machine(), '| host', os.environ.get('PROCESSOR_ARCHITEW6432') or platform.machine(), '|', str(struct.calcsize('P')*8) + '-bit')"

# ---------------------------------------------------------------------------- packages
& $venvPy -m pip install --upgrade pip --quiet --disable-pip-version-check
& $venvPy -m pip install --disable-pip-version-check $packages
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }

# ---------------------------------------------------------------------------- verify
# Everything the tracker needs, importable together with bleak in ONE process.
& $venvPy -c "import cv2, numpy, scipy, bleak; d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); assert hasattr(cv2, 'triangulatePoints') and hasattr(cv2, 'undistortPoints') and hasattr(cv2, 'solvePnP'); print('OpenCV', cv2.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| aruco OK | bleak OK')"
if ($LASTEXITCODE -ne 0) { throw "cf64 verification failed" }

Write-Host ""
Write-Host "cf64 OK.  Activate with:  .\cf64\Scripts\Activate.ps1"
Write-Host "Then:  python tools\cv_perf.py      python tools\camera_probe.py"
