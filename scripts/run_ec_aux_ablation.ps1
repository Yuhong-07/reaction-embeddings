$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = if ($env:RHEA_PYTHON) { $env:RHEA_PYTHON } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python scripts\train_phase2.py `
        --model-config configs\model\phase2_structure_only.yaml `
        --train-config configs\train\phase2_v2_pilot_ecaux0_cpu.yaml
    if ($LASTEXITCODE -ne 0) { throw "EC auxiliary weight=0 training failed with exit code $LASTEXITCODE" }
    & $Python scripts\export_phase2.py --export-config configs\export\phase2_v2_pilot_ecaux0.yaml
    if ($LASTEXITCODE -ne 0) { throw "EC auxiliary weight=0 export failed with exit code $LASTEXITCODE" }
    & $Python scripts\diagnose_duplicate_embeddings.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_pilot_ecaux0.npy `
        --ids artifacts\embeddings\reaction_ids_v2_pilot_ecaux0.tsv `
        --output-dir artifacts\reports\duplicate_embedding_diagnosis_v2_pilot_ecaux0
    if ($LASTEXITCODE -ne 0) { throw "EC auxiliary weight=0 duplicate diagnosis failed with exit code $LASTEXITCODE" }
    & $Python scripts\analyze_embedding_quality.py `
        --embedding artifacts\embeddings\reaction_embeddings_v2_pilot_ecaux0.npy `
        --ids artifacts\embeddings\reaction_ids_v2_pilot_ecaux0.tsv `
        --output-dir artifacts\reports\semantic_quality_v2_pilot_ecaux0 `
        --duplicate-diagnosis artifacts\reports\duplicate_embedding_diagnosis_v2_pilot_ecaux0\duplicate_embedding_diagnosis.json `
        --checkpoint artifacts\checkpoints\phase2_v2_pilot_ecaux0.pt
    if ($LASTEXITCODE -ne 0) { throw "EC auxiliary weight=0 semantic analysis failed with exit code $LASTEXITCODE" }
    & $Python scripts\compare_ec_aux_ablation.py
    if ($LASTEXITCODE -ne 0) { throw "EC auxiliary ablation comparison failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
