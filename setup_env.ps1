# setup_env.ps1 -- create or refresh the `cf` venv for Pathcaster.
# Run from PowerShell in the project root:  .\setup_env.ps1
# (if scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Works on x64 and on Windows-on-ARM64 (where libusb-package has no wheel and is shimmed).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".\cf\Scripts\python.exe")) {
    # prefer the native ARM64 interpreter on ARM64 laptops (py -3.12 may resolve to the x64 build)
    $py = (py -3.12-arm64 -c "import sys; print(sys.executable)" 2>$null)
    if (-not $py) { $py = (py -3.12 -c "import sys; print(sys.executable)") }
    Write-Host "Creating venv cf from $py"
    & $py -m venv cf
}
$python = ".\cf\Scripts\python.exe"
& $python -c "import sys, platform, struct; print('Python', sys.version.split()[0], platform.machine(), str(struct.calcsize('P')*8) + '-bit')"

& $python -m pip install --upgrade pip --quiet
& $python -m pip install bleak numpy scipy pyusb pyyaml packaging
& $python -m pip install --no-deps cflib

# The real libusb-package only has x64/x86 wheels. Try it; fall back to our shim on ARM64.
& $python -m pip install libusb-package
if ($LASTEXITCODE -ne 0) {
    Write-Host "libusb-package unavailable on this CPU -> using shims\libusb_package"
    $site = & $python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    Set-Content -Path (Join-Path $site "pathcaster_shims.pth") -Value (Join-Path $root "shims") -Encoding ascii
}

& $python -c "import cflib.crtp, bleak, numpy, scipy; print('cf venv OK')"
Write-Host "Activate with:  .\cf\Scripts\Activate.ps1"
