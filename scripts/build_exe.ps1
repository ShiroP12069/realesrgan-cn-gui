param(
    [switch]$IncludeLocalWeights
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
Set-Location $root

if (-not (Test-Path ".venv/Scripts/python.exe")) {
    throw "Missing .venv. Run .\\scripts\\setup_env.ps1 first."
}

$py = Resolve-Path ".venv/Scripts/python.exe"

$args = @(
    "--noconfirm",
    "--clean",
    "--name", "RealESRGAN_CN_GUI",
    "--windowed",
    "--paths", "src",
    "--paths", "third_party/Real-ESRGAN-0.3.0",
    "--add-data", "assets;assets",
    "--collect-all", "realesrgan",
    "--collect-all", "basicsr",
    "--collect-all", "gfpgan",
    "--collect-all", "facexlib",
    "main.py"
)

# Default build keeps EXE/package and model weights separated.
if ($IncludeLocalWeights) {
    $args += @("--add-data", "third_party/Real-ESRGAN-0.3.0/weights;weights")
}

& $py -m PyInstaller @args

# Prevent accidental launch of temporary build exe.
$tempExe = Join-Path $root "build/RealESRGAN_CN_GUI/RealESRGAN_CN_GUI.exe"
if (Test-Path $tempExe) {
    Remove-Item -Force $tempExe
}

Write-Host "Build done: dist/RealESRGAN_CN_GUI/RealESRGAN_CN_GUI.exe"
if ($IncludeLocalWeights) {
    Write-Host "Mode: bundled local weights."
} else {
    Write-Host "Mode: no bundled weights (models downloaded at runtime or by download_models.ps1)."
}
