param(
    [string[]]$Models = @("RealESRGAN_x4plus"),
    [switch]$All
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
Set-Location $root

if (-not (Test-Path ".venv/Scripts/python.exe")) {
    throw "Missing .venv. Run .\\scripts\\setup_env.ps1 first."
}

$py = Resolve-Path ".venv/Scripts/python.exe"

$allModels = @(
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x2plus",
    "realesr-animevideov3",
    "realesr-general-x4v3"
)

if ($All) {
    $Models = $allModels
}

$modelJson = $Models | ConvertTo-Json -Compress
$pyCode = @"
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from realesrgan_gui.engine import RealESRGANEngine

models = json.loads(r'''$modelJson''')
engine = RealESRGANEngine()

for m in models:
    print(f"Prepare model: {m}")
    engine.download_model(m, "", print)

print("Done.")
"@

& $py -c $pyCode

