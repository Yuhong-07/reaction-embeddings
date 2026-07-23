$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:RHEA_PYTHON) { $env:RHEA_PYTHON } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
    & $Python scripts\train_phase2.py `
        --model-config configs\model\phase2_structure_only.yaml `
        --train-config configs\train\phase2_v2_pilot_cpu.yaml
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 v2 training failed with exit code $LASTEXITCODE" }
    & $Python scripts\export_phase2.py --export-config configs\export\phase2_v2_pilot.yaml
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 v2 export failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_phase2_v2.py
    if ($LASTEXITCODE -ne 0) { throw "Phase 2 v2 validation failed with exit code $LASTEXITCODE" }
    & $Python scripts\diagnose_duplicate_embeddings.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_pilot.npy `
        --ids artifacts\embeddings\reaction_ids_v2_pilot.tsv `
        --output-dir artifacts\reports\duplicate_embedding_diagnosis_v2_pilot
    if ($LASTEXITCODE -ne 0) { throw "Duplicate diagnosis failed with exit code $LASTEXITCODE" }
    & $Python scripts\analyze_embedding_quality.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_pilot.npy `
        --ids artifacts\embeddings\reaction_ids_v2_pilot.tsv `
        --output-dir artifacts\reports\semantic_quality_v2_pilot `
        --duplicate-diagnosis artifacts\reports\duplicate_embedding_diagnosis_v2_pilot\duplicate_embedding_diagnosis.json `
        --checkpoint artifacts\checkpoints\phase2_v2_pilot.pt
    if ($LASTEXITCODE -ne 0) { throw "Semantic quality analysis failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_embedding_quality_report.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_pilot.npy `
        --report-dir artifacts\reports\semantic_quality_v2_pilot
    if ($LASTEXITCODE -ne 0) { throw "Semantic quality validation failed with exit code $LASTEXITCODE" }
    & $Python scripts\audit_phase2_v2_inputs.py
    if ($LASTEXITCODE -ne 0) { throw "Input audit failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
