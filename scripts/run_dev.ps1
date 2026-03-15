$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
Set-Location $root

$py = Resolve-Path ".venv/Scripts/python.exe"
& $py main.py

