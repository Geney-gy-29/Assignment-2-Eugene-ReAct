# [A3-IMPROVEMENT] Build the presentation deck with Marp CLI.
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    npx --yes @marp-team/marp-cli slides.md --pdf --allow-local-files -o build/slides.pdf
    npx --yes @marp-team/marp-cli slides.md --pptx --allow-local-files -o build/slides.pptx
    Write-Host "Built slides/build/slides.pdf and slides/build/slides.pptx"
} finally {
    Pop-Location
}
