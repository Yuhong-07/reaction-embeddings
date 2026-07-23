$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:RHEA_PYTHON) { $env:RHEA_PYTHON } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
    & $Python scripts\train_phase2.py `
        --model-config configs\model\phase2_structure_only.yaml `
        --train-config configs\train\phase2_v2_full_ecaux0_cpu.yaml
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus training failed with exit code $LASTEXITCODE" }
    & $Python scripts\export_phase2.py --export-config configs\export\phase2_v2_full_ecaux0.yaml
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus export failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_phase2_v2.py --export-config configs\export\phase2_v2_full_ecaux0.yaml
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus export validation failed with exit code $LASTEXITCODE" }
    & $Python scripts\diagnose_duplicate_embeddings.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_full_ecaux0.npy `
        --ids artifacts\embeddings\reaction_ids_v2_full_ecaux0.tsv `
        --output-dir artifacts\reports\duplicate_embedding_diagnosis_v2_full_ecaux0
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus duplicate diagnosis failed with exit code $LASTEXITCODE" }
    & $Python scripts\analyze_embedding_quality.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_full_ecaux0.npy `
        --ids artifacts\embeddings\reaction_ids_v2_full_ecaux0.tsv `
        --output-dir artifacts\reports\semantic_quality_v2_full_ecaux0 `
        --duplicate-diagnosis artifacts\reports\duplicate_embedding_diagnosis_v2_full_ecaux0\duplicate_embedding_diagnosis.json `
        --checkpoint artifacts\checkpoints\phase2_v2_full_ecaux0.pt
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus semantic analysis failed with exit code $LASTEXITCODE" }
    & $Python scripts\validate_embedding_quality_report.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_full_ecaux0.npy `
        --report-dir artifacts\reports\semantic_quality_v2_full_ecaux0
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus semantic validation failed with exit code $LASTEXITCODE" }
    & $Python scripts\audit_phase2_v2_inputs.py `
        --checkpoint artifacts\checkpoints\phase2_v2_full_ecaux0.pt `
        --output artifacts\reports\phase2_v2_full_input_audit.json
    if ($LASTEXITCODE -ne 0) { throw "Full-corpus input audit failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
