$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:RHEA_PYTHON) { $env:RHEA_PYTHON } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
    & $Python scripts\train_phase2.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 training failed with exit code $LASTEXITCODE" }
    & $Python scripts\export_phase2.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 export failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_phase2.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 validation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
