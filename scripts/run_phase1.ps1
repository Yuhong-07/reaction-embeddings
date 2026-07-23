$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RawDir = Join-Path $ProjectRoot "data\raw\rhea"
$Python = if ($env:RHEA_PYTHON) { $env:RHEA_PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

$Rxn = Join-Path $RawDir "rhea-rxn.tar.gz"
$Tsv = Join-Path $RawDir "rhea-tsv.tar.gz"

if (-not (Test-Path -LiteralPath $Rxn)) {
    Invoke-WebRequest -Uri "https://ftp.expasy.org/databases/rhea/ctfiles/rhea-rxn.tar.gz" -OutFile $Rxn
}
if (-not (Test-Path -LiteralPath $Tsv)) {
    Invoke-WebRequest -Uri "https://ftp.expasy.org/databases/rhea/tsv/rhea-tsv.tar.gz" -OutFile $Tsv
}

& $Python -c "import rdkit, pyarrow, yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

Push-Location $ProjectRoot
try {
    & $Python scripts\prepare_rhea.py --config configs\data\rhea_phase1.yaml
    if ($LASTEXITCODE -ne 0) { throw "Rhea preparation failed with exit code $LASTEXITCODE" }
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_phase1.py --config configs\data\rhea_phase1.yaml
    if ($LASTEXITCODE -ne 0) { throw "Validation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
