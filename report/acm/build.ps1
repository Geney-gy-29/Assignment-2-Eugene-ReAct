# [A3-IMPROVEMENT] Build the ACM report with Tectonic.
# Tectonic (tools/tectonic.exe) is gitignored -- download it fresh:
#   https://github.com/tectonic-typesetting/tectonic/releases
# and place tectonic.exe in tools/ before running this script.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tectonic = Join-Path $root "tools\tectonic.exe"
if (-not (Test-Path $tectonic)) {
    throw "tools/tectonic.exe not found. Download from https://github.com/tectonic-typesetting/tectonic/releases and place it there."
}
Push-Location $PSScriptRoot
try {
    & $tectonic main.tex
    Write-Host "Built report/acm/main.pdf"
} finally {
    Pop-Location
}
