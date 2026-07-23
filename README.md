# Rhea Reaction Embeddings

A reproducible Python project for preparing Rhea reactions and generating fixed-length, structure-only metabolic reaction embeddings.

The current corpus contains **18,072 strictly deduplicated reactions from Rhea release 141**. The Phase 2 v2 encoder produces 256-dimensional embeddings from molecular structure, signed stoichiometry, reaction direction, and optional cofactor-role information. EC numbers are never used as encoder inputs.

## Project status

| Stage | Status | Result |
|---|---|---|
| Phase 1: data preparation | Complete | 18,072 canonical reactions |
| Phase 2: baseline encoder | Archived baseline | 673 duplicate vectors; collapse check failed |
| Phase 2 v2: structure-only pilot | Complete | Selected `ec_auxiliary=0`; independent EC evidence retained |
| EC auxiliary ablation | Complete | `weight=0` recommended over `weight=0.2` |
| Full-corpus structure-only training | Complete | 18,072/18,072 reactions trained; effective rank 149.57; collapse check passed |

## Key design rules

- The authoritative input is a structured reaction record, not reaction SMILES alone.
- Reactants and products retain signed stoichiometric coefficients.
- Reaction direction is an explicit model input.
- EC numbers are evaluation labels or optional auxiliary targets only; they are not concatenated into the embedding input.
- Missing optional metadata uses explicit masks and never causes a reaction to be discarded.
- Embedding quality is checked with nearest neighbors, EC consistency, effective rank, duplicate-vector diagnosis, and collapse tests.

## Repository layout

```text
.
├── .github/workflows/       # GitHub Actions continuous integration
├── artifacts/               # Generated checkpoints, embeddings, and reports
├── configs/                 # Data, model, training, and export configuration
├── data/
│   ├── raw/                 # Immutable downloaded archives
│   ├── interim/             # Caches and transformation logs
│   ├── processed/           # Canonical reaction corpus
│   └── manifests/           # Versions, checksums, and provenance
├── docs/                    # Design decisions and experiment notes
├── schemas/                 # Reaction JSON Schemas
├── scripts/                 # Reproducible command-line entry points
├── src/rhea_embedding/      # Installable Python package
│   ├── chemistry/           # Molecular graph construction
│   ├── data/                # Rhea preparation and dataset loading
│   ├── models/              # D-MPNN and reaction encoder
│   ├── quality_control/     # Embedding invariance checks
│   └── training/            # Training, losses, export, and manifests
├── tests/                   # Automated unit tests
├── agent.md                 # Code and experiment change ledger
├── AGENTS.embedding.md      # Project-specific modeling requirements
├── pyproject.toml           # Package metadata and build configuration
└── requirements.txt         # Runtime dependencies
```

Generated data and model artifacts remain in the local layout above but are excluded from normal Git commits by `.gitignore`. Publish large deliverables with a release asset, object storage, or Git LFS rather than ordinary Git blobs.

## Download data and generated outputs

The [v0.3.0 GitHub Release](https://github.com/Yuhong-07/reaction-embeddings/releases/tag/v0.3.0) provides two versioned archives:

- `rhea-release-141-raw-data.zip`: the immutable Rhea release 141 source archives under `data/raw/`.
- `reaction-embeddings-v0.3.0-output-data.zip`: processed reaction tables plus checkpoints, embedding exports, and validation reports under `data/processed/` and `artifacts/`.

Download `SHA256SUMS.txt` from the same release and verify the archives before use. The rebuildable `data/interim/` molecular-graph cache is intentionally excluded.

## Model detail 
See the [Embedding results summary](docs/embedding_results_summary.md) for the detail of the model and the validation results.
## Requirements

- Python 3.11 or newer
- NumPy, pandas, PyArrow, PyYAML, RDKit, and PyTorch
- PowerShell for the bundled end-to-end runners

Create an environment and install the package from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For a non-editable dependency-only installation:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce the data

Run Phase 1 from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1
```

The runner downloads the two Rhea archives when they are absent, prepares the corpus, runs the unit tests, and validates the export. To use an existing archive set:

```powershell
python scripts\prepare_rhea.py --config configs\data\rhea_phase1.yaml
python scripts\validate_phase1.py --config configs\data\rhea_phase1.yaml
```

Primary Phase 1 outputs:

- `data/processed/rhea_reactions.parquet`
- `data/processed/rhea_reactions.jsonl`
- `data/processed/reaction_id_map.tsv`
- `data/processed/rhea_source_records.parquet`
- `data/processed/rhea_direction_variants.parquet`
- `data/manifests/rhea_release_141_manifest.json`
- `artifacts/reports/data_quality_report.json`

The canonical master rows use `direction=undefined` because a Rhea master identifier does not itself assert a condition-specific net direction. Explicit left-to-right, right-to-left, and reversible alternatives are stored in `rhea_direction_variants.*` with source evidence.

## Train and export embeddings

Run the deterministic structure-only Phase 2 v2 pilot:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase2_v2_pilot.ps1
```

Run the controlled `ec_auxiliary=0` versus `0.2` ablation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ec_aux_ablation.ps1
```

Run the complete 18,072-reaction fixed-epoch training and validation pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase2_v2_full.ps1
```

The current recommended full-corpus output is:

- checkpoint: `artifacts/checkpoints/phase2_v2_full_ecaux0.pt`
- embedding matrix: `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.npy`
- tabular export: `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.parquet`
- reaction IDs: `artifacts/embeddings/reaction_ids_v2_full_ecaux0.tsv`
- quality report: `artifacts/reports/embedding_quality_report_v2_full_ecaux0.json`
- semantic report: `artifacts/reports/semantic_quality_v2_full_ecaux0/embedding_semantic_quality_report.md`
- ablation report: `artifacts/reports/ec_aux_ablation/ec_aux_ablation_report.md`

The original `phase2_mvp.*` files are retained as a failed-collapse baseline. Pilot v2 files are retained as the reaction-disjoint EC evaluation evidence used to select the full-training objective.

## Validate

Run the complete unit-test suite:

```powershell
python -m unittest discover -s tests -v
```

Validate the recommended v2 checkpoint and export:

```powershell
python scripts\validate_phase2_v2.py `
  --export-config configs\export\phase2_v2_full_ecaux0.yaml
```

The current test suite contains 17 passing tests. The recommended full-corpus export has no NaN/Inf values, no unexplained duplicate vectors, effective rank 149.57, and passes the embedding-collapse checks.

## Reproducibility and audit trail

- `agent.md` records every material code, configuration, data, and experiment change.
- `data/manifests/` stores source and export checksums.
- `artifacts/reports/` stores training histories and validation results.
- Random seeds and frozen split indices are saved with checkpoints.
- The pilot EC evaluation excludes training and model-selection reactions from both queries and candidates. Full-corpus EC metrics are explicitly labelled transductive because every reaction structure was used for self-supervised training.


## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the schema, preprocessing, model inputs, loss functions, evaluation split, or exported artifacts. Every accepted change must also be appended to `agent.md`.

## License

No repository license has been selected yet. Add an explicit license before publishing or accepting external contributions.
