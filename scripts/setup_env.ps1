param(
    [ValidateSet("cpu", "cu121")]
    [string]$Torch = "cpu",
    [string]$PythonCmd = "py"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
Set-Location $root

if (-not (Test-Path ".venv")) {
    & $PythonCmd -m venv .venv
}

$py = Resolve-Path ".venv/Scripts/python.exe"
& $py -m pip install --upgrade pip setuptools wheel

if ($Torch -eq "cpu") {
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    & $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

& $py -m pip install -r requirements.txt
& $py -m pip install -e third_party/Real-ESRGAN-0.3.0 --no-deps

Write-Host "Environment ready."
Write-Host "Run app: .\\scripts\\run_dev.ps1"
Write-Host "Build EXE: .\\scripts\\build_exe.ps1"
