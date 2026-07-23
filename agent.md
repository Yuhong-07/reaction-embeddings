# Agent code and experiment ledger

Last updated: 2026-07-23 (Europe/Berlin)

This file is the single project ledger for code, configuration, experiment, validation, and output changes under `D:\Codex\2026-07-21`. The source files remain executable authority; every future code or experiment change must append an entry here with the date, reason, changed files, validation, outputs, and rollback information.

## 1. Current approved state

- Corpus: 18,072 strictly deduplicated Rhea release 141 reactions.
- Embedding dimension: 256.
- Current model: Phase 2 v2 structure-only reaction encoder.
- Main embedding inputs: reactant structure, product structure, signed stoichiometry, reaction direction, and optional cofactor role with a missing mask.
- Forbidden main input: EC number. The model raises an error if `use_ec` or `ec_as_input` is enabled.
- Recommended objective: structure-only self-supervision with `ec_auxiliary=0`.
- Recommended checkpoint: `artifacts/checkpoints/phase2_v2_full_ecaux0.pt`.
- Recommended matrix: `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.npy`.
- Recommended tabular export: `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.parquet`.
- Full-corpus status: all 18,072 reactions trained for five fixed epochs; final export and all required checks passed.
- EC evaluation policy: retain the pilot reaction-disjoint EC report as independent evidence; label full-corpus EC metrics as transductive/descriptive because every reaction structure was used for self-supervised training.

## 2. Chronological change record

### 2026-07-23 17:30 — GitHub Release data publication

Purpose:

- Publish the immutable Rhea source archives and generated project outputs as downloadable ZIP assets without adding large binary files to Git history.

Changed files:

- `README.md`: added the v0.3.0 release download guide and archive contents.
- `.gitignore`: excluded the local `release_assets/` staging directory from normal Git commits.
- `agent.md`: recorded release contents, exclusions, sizes, and checksums.

Release:

- Tag: `v0.3.0`.
- Repository: `https://github.com/Yuhong-07/reaction-embeddings`.
- Published release: `https://github.com/Yuhong-07/reaction-embeddings/releases/tag/v0.3.0`.
- Published from commit: `ce53e02e1bfddde10c744fe47840fd679d9f315f`.
- `rhea-release-141-raw-data.zip`: 24.91 MB; contains `data/raw/`.
- `reaction-embeddings-v0.3.0-output-data.zip`: 211.62 MB; contains `data/processed/`, `artifacts/checkpoints/`, `artifacts/embeddings/`, and `artifacts/reports/`.
- `SHA256SUMS.txt`: checksum sidecar for both archives.

Checksums:

- Raw data ZIP: `7784781BB27CCAB07192EA51952A78866E8A18F2B2EBD1AAD2EA443B9FD4C066`.
- Output data ZIP: `9C75B0301F92694C336DCA36767D106393337945D4911E923361957A03D4ADD8`.

Data/checkpoint compatibility:

- Archive creation did not modify any source dataset, processed table, model checkpoint, embedding, or report.
- `data/interim/` is excluded because it is a rebuildable molecular-graph cache rather than source or final output data.

Validation:

- Listed both ZIP archives successfully after creation and confirmed their expected top-level paths.
- Generated SHA-256 hashes for both assets.
- Published the GitHub Release and verified that all three requested assets have independent download links.
- GitHub reports the uploaded ZIP digests as the same SHA-256 values recorded above.

Decision and rollback:

- GitHub Release assets are the canonical downloadable binary distribution for v0.3.0; the Git repository remains source-only.
- To roll back publication, delete the v0.3.0 Release assets; local files and Git history remain usable.

### 2026-07-23 17:00 — GitHub repository publication

Purpose:

- Publish the reproducible project source, configuration, tests, documentation, and manifests as a public GitHub repository.
- Keep generated datasets, model checkpoints, reports, and embedding matrices outside normal Git history because they are reproducible artifacts and may exceed GitHub file-size limits.

Changed files:

- `pyproject.toml`: added the canonical repository and documentation URLs.
- `agent.md`: recorded the publication policy and destination.

Repository:

- `https://github.com/Yuhong-07/reaction-embeddings`
- Default branch: `main`.
- Visibility: public.

Data/checkpoint compatibility:

- No model, data, checkpoint, or embedding content changed.
- Existing `.gitignore` rules exclude `data/raw`, `data/interim`, `data/processed`, `artifacts/checkpoints`, `artifacts/embeddings`, and `artifacts/reports`; their lightweight README/manifests remain versioned where applicable.

Validation:

- Scanned versioned project paths for common secret, token, password, and private-key patterns before staging.
- Audited staged paths and file sizes before the initial commit and remote push.

Decision and rollback:

- GitHub is the canonical source-code remote; large generated artifacts remain local and are documented by manifests/checksums.
- To roll back publication, remove or privatize the GitHub repository; local computation outputs are unaffected.

### 2026-07-23 16:15 — Embedding output and code-usage documentation

Purpose:

- Explain why the full-corpus pipeline produces multiple output files and identify the minimum files required for each common use case.
- Add copyable examples for reading embeddings, preserving row-to-ID alignment, retrieving a reaction, joining embeddings to a metabolic-network reaction table, viewing precomputed neighbors, validating artifacts, and reproducing training.

Changed files:

- `docs/embedding_results_summary.md`: added output-role explanations and a complete code usage guide. The guide recommends Parquet for inspection and joins, NPY plus TSV for numerical training, `many_to_one` validation for network-to-Rhea joins, explicit missing masks for unmapped reactions, and retention of organism/compartment/bounds as separate context.
- `agent.md`: recorded the documentation change.

Validation:

- Examples use the current recommended `phase2_v2_full_ecaux0` paths and actual Parquet/TSV column names.
- The usage guide distinguishes routine consumption from destructive retraining and states that the full runner replaces full-corpus outputs.

Rollback:

- Remove the output explanation and code usage section; no model, data, checkpoint, or embedding file is affected.

### 2026-07-23 15:45 — Full-corpus structure-only training and export

Purpose:

- Train the selected structure-only `ec_auxiliary=0` objective on every one of the 18,072 deduplicated Rhea reactions.
- Export a new recommended full-corpus checkpoint and embedding matrix while preserving the pilot files as reaction-disjoint EC evidence.

Changed code and configuration:

- `src/rhea_embedding/training/phase2.py`: added a validated zero-validation mode so a fixed final fit can use all selected reactions; when `validation_fraction=0`, every selected reaction is trained and the final epoch is retained.
- `src/rhea_embedding/data/reaction_dataset.py`: incompatible or unreadable graph caches now rebuild automatically. This was required because the pre-GitHub-reorganization cache referenced the former `chemistry.*` module path and could not be unpickled under `rhea_embedding.*`.
- `scripts/analyze_embedding_quality.py`: added explicit reaction-disjoint versus full-corpus transductive EC evaluation modes, dynamic limitations, and corrected the Markdown effective-rank pass/fail label.
- `tests/test_phase2.py`: added zero-validation split, deterministic positive split, and incompatible-cache rebuild tests. Test count increased from 14 to 17.
- `configs/train/phase2_v2_full_ecaux0_cpu.yaml`: full 18,072-reaction, five-epoch, weight-zero, final-epoch configuration.
- `configs/export/phase2_v2_full_ecaux0.yaml`: full-corpus output paths.
- `scripts/run_phase2_v2_full.ps1`: complete tests, training, export, validation, duplicate diagnosis, semantic analysis, semantic validation, and input-audit runner.
- `pyproject.toml` and `src/rhea_embedding/__init__.py`: package version advanced from 0.2.0 to 0.3.0 for the full-corpus release.

Training record:

- Random seed: 42.
- Device: CPU, eight Torch threads.
- Training reactions: 18,072.
- Validation/model-selection reactions: 0.
- Epochs: 5 fixed; final epoch retained.
- Epoch-1 total loss: 0.6443330.
- Epoch-5 total loss: 0.2960448.
- Training time: 696.532 seconds.
- EC input: disabled.
- EC auxiliary weight: 0.

Direction audit:

- Canonical source field remains undefined, but model direction is derived from `supported_directions` evidence.
- Model categories: left-to-right 6,554; reversible 7,627; right-to-left 287; undefined 3,604.
- Masking direction changed 18,072/18,072 embeddings; median L2 difference 5.91.
- EC removal and excluded-metadata perturbation both produced maximum absolute embedding difference 0.

Results:

- Matrix: `[18072, 256]`, float32, 0 NaN/Inf.
- Exact duplicate-vector excess: 51 across 40 groups.
- Equivalent-input duplicate excess: 51.
- Unexplained encoder collisions: 0.
- Effective rank: 149.57.
- Components for 90% variance: 125.
- First-component variance: 2.000%.
- Top-10 cumulative variance: 15.718%.
- Random-pair cosine mean/std: 0.0481 / 0.0920.
- Strict collapse acceptance: pass.
- All invariance, output-hash, nearest-neighbor-table, and input-policy checks: pass.
- Full-corpus descriptive exact EC precision@1: 16.788%; this is transductive and is not the independent EC estimate.

Recommended outputs and checksums:

- `artifacts/checkpoints/phase2_v2_full_ecaux0.pt`: `6216d9d86a0fb75d8a3335482a50a6ea5409446687d2e34d9428148a09c093b2`.
- `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.npy`: `d53db0ca8ca97c43c5a0aad4da7d0a93b0baf8d7d814bdb46b076d21ddf09e35`.
- `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.parquet`: `80ae07da11c63a6f80932982898f222c8a22a6c0740387b55bc6e5b3fead8562`.
- `artifacts/embeddings/reaction_ids_v2_full_ecaux0.tsv`: `1f0b4bd4e9fceb39d387d5d22a73e379ec7aa91ee14d8d52b0c5cdc66850a343`.
- `data/manifests/phase2_v2_full_ecaux0_embedding_manifest.json`.
- `artifacts/reports/phase2_v2_full_training_report.md`.
- `artifacts/reports/embedding_quality_report_v2_full_ecaux0.json`.
- `artifacts/reports/semantic_quality_v2_full_ecaux0/`.
- `artifacts/reports/duplicate_embedding_diagnosis_v2_full_ecaux0/`.
- `artifacts/reports/phase2_v2_full_input_audit.json`.

Documentation updates:

- `README.md`: full-corpus status, runner, recommended outputs, validation, and EC evaluation interpretation.
- `docs/embedding_results_summary.md`: replaced the pilot-only report with the full-corpus English report and corrected direction handling.
- `docs/phase2_pilot.md`: marked as historical.

Rollback:

- Restore the recommended pointers to `phase2_v2_pilot_ecaux0.*`; no pilot checkpoint, independent EC report, raw data, or processed corpus was removed.

### 2026-07-21 17:30 — English colleague-facing embedding report

Purpose:

- Provide a self-contained English document covering methods, data, model inputs, outputs, validation results, reporting limitations, and recommended next steps.
- Define which claims are appropriate for internal reporting and prevent the pilot checkpoint from being described as full-corpus production training.

Changed files:

- `docs/embedding_results_summary.md`: added the English technical report, recommended colleague-facing statement, exact output paths and checksums, leakage-controlled EC results, and explicit direction/cofactor/pilot limitations.
- `README.md`: linked the English report from the reproducibility and audit section.
- `agent.md`: recorded the documentation change.

Validation:

- Metrics were cross-checked against `embedding_quality_report_v2_pilot_ecaux0.json`, the semantic-quality report, the Phase 2 v2 adjustment report, the Rhea release manifest, and the EC auxiliary ablation report.
- The report identifies the recommended model as `ec_auxiliary=0`, training as a single-seed 1,900/100 pilot split, and the export as 18,072 reactions by 256 dimensions.

Rollback:

- Remove `docs/embedding_results_summary.md` and its README link. No code, data, checkpoint, or embedding artifact is affected.

### 2026-07-21 17:00 — GitHub-standard repository reorganization

Purpose:

- Convert the project to a conventional installable Python `src` layout without moving generated datasets, checkpoints, embeddings, or reports.
- Make repository metadata, dependency installation, continuous integration, and contribution rules suitable for GitHub.
- Replace the Phase 1-only README with a complete project landing page covering the current Phase 2 v2 state.

Changed files and paths:

- Moved `src/chemistry`, `src/data`, `src/models`, `src/training`, and `src/quality_control` under the namespace package `src/rhea_embedding/`.
- Added `src/rhea_embedding/__init__.py` with project version `0.2.0`.
- Updated all package, script, and test imports to use `rhea_embedding.*`.
- Renamed `requirements-data.txt` to the conventional `requirements.txt`; updated the Phase 1 runner and error guidance.
- Added `pyproject.toml` for editable installation, package discovery, dependencies, and the `rhea-prepare` console command.
- Added `.gitignore`, `.gitattributes`, `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `data/README.md`, and `artifacts/README.md`.
- Rewrote `README.md` as the GitHub project landing page with status, architecture rules, repository layout, installation, reproduction, validation, audit, contribution, and license sections.
- Updated `scripts/validate_phase2_v2.py` so the EC auxiliary weight defaults to the selected checkpoint value; explicit `--expected-ec-weight` still overrides it.
- Removed generated `__pycache__` directories from the source tree; future caches are ignored by Git.

Compatibility:

- Data, configuration, checkpoint, embedding, manifest, and report paths are unchanged.
- Existing checkpoints remain compatible because model class behavior and state-dict key names did not change.
- Direct imports such as `from models...` are intentionally replaced by `from rhea_embedding.models...`.
- Editable installation is now `python -m pip install -e .`.

Validation:

- `python -m unittest discover -s tests -v`: 14/14 passed after the package move.
- `scripts/validate_phase1.py`: PASS; 18,072 reactions and nine checksums verified.
- `scripts/validate_phase2_v2.py --export-config configs/export/phase2_v2_pilot_ecaux0.yaml`: PASS; `[18072, 256]`, EC auxiliary weight 0, 51 expected exact duplicates, invariance and output hashes verified.

Rollback:

- Move the five package subdirectories back under `src/`, restore the former non-namespaced imports, rename `requirements.txt` back to `requirements-data.txt`, and remove the new repository metadata files. No data or model artifact rollback is required.

### 2026-07-21 — Phase 1 Rhea data preparation

Purpose:

- Freeze Rhea release 141 into a reproducible structured corpus.
- Preserve reaction participants, signed stoichiometry, compound identity, direction evidence, EC annotations, provenance, and source mappings.
- Retain 18,072 strictly deduplicated chemical reactions for embedding.

Implemented code and configuration:

- `src/rhea_embedding/data/prepare_rhea.py`: parses RXN and TSV archives, standardizes records, constructs strict and loose duplicate keys, exports processed data and audit records.
- `scripts/prepare_rhea.py`: command-line entry point.
- `scripts/run_phase1.ps1`: end-to-end Phase 1 runner.
- `scripts/validate_phase1.py`: validates counts, schema, checksums, directions, and exported tables.
- `configs/data/rhea_phase1.yaml`: release and output configuration.
- `schemas/reaction.schema.json`: canonical reaction schema.
- `schemas/direction_variant.schema.json`: direction-variant schema.
- `tests/test_prepare_rhea.py`: strict-key order invariance, stoichiometric scale invariance, and loose-key behavior.

Key decisions:

- Strict duplicate key uses compound identity, reduced signed stoichiometry, compartment, and direction class.
- Loose keys that ignore proton, water, direction, or compartment are report-only and never auto-merge.
- Rhea master IDs are undefined-direction members; canonical records do not default to reversible.
- Direction-aware LR, RL, and reversible variants are exported separately.
- Source-specific EC and cross-reference annotations remain in the source-record mapping.

Results:

- Source master records: 18,558.
- Successfully parsed master records: 18,550.
- Missing RXN records: 8, retained as explicit failures outside the embedding corpus.
- Strict duplicate groups: 184.
- Merged source records: 478.
- Final reactions: 18,072.
- Failed participant SMILES: 0.
- Unsanitized but retained molecular structure: 1.
- Imbalanced reactions: 6.
- EC coverage: 41.0359%.
- Direction-evidence coverage: 80.0575%.

Primary records:

- `docs/data_decisions.md`
- `data/interim/transformation_log.jsonl`
- `data/manifests/rhea_release_141_manifest.json`
- `artifacts/reports/data_quality_report.json`

### 2026-07-21 — Original Phase 2 MVP pilot

Purpose:

- Validate the minimum D-MPNN reaction-embedding and export pipeline on a deterministic 2,000-reaction CPU subset.

Implemented code and configuration:

- `src/rhea_embedding/chemistry/graph.py`: RDKit molecular graphs, atom/bond features, graph batching, segment operations.
- `src/rhea_embedding/data/reaction_dataset.py`: reaction examples, graph cache, vocabularies, batching, side swapping, participant permutation.
- `src/rhea_embedding/models/dmpnn.py`: shared directed message-passing molecular encoder.
- `src/rhea_embedding/models/reaction_encoder.py`: participant tokens, side pooling, reaction fusion, direction handling, projections, and auxiliary heads.
- `src/rhea_embedding/training/phase2.py`: reproducibility, training, checkpointing, export, manifests, and embedding reports.
- `src/rhea_embedding/quality_control/embedding_checks.py`: order, batch, repeat, randomized-SMILES, and reversible-swap checks.
- `scripts/train_phase2.py`, `scripts/export_phase2.py`, `scripts/run_phase2.ps1`, `scripts/validate_phase2.py`.
- `configs/model/phase2_mvp.yaml`, `configs/train/phase2_mvp_cpu.yaml`, `configs/export/phase2_mvp.yaml`, `configs/data/phase2_rhea.yaml`.
- `tests/test_phase2.py`.

Original architecture and objective:

- Four-step D-MPNN with normalized attention molecular readout.
- Participant-side attention pooling plus a signed-sum delta branch.
- EC and other optional metadata could enter reaction fusion directly.
- Losses: contrastive, masked participant reconstruction, stoichiometry regression, and direction prediction.

Training record:

- Random seed: 42.
- Selected reactions: 2,000.
- Training: 1,900.
- Validation: 100.
- Epochs: 3.
- Best validation loss: 0.4506716 at epoch 3.
- Checkpoint: `artifacts/checkpoints/phase2_mvp.pt`.
- Metadata: `artifacts/checkpoints/phase2_mvp.metadata.json`.

### 2026-07-21 — Baseline semantic-quality diagnostics

Purpose:

- Add exact cosine nearest neighbors, EC consistency, random baselines, variance-spectrum analysis, and collapse checks.

Added code:

- `scripts/analyze_embedding_quality.py`: exact top-10 cosine neighbors, EC precision/hit rates, analytical random baselines, SVD diagnostics, effective rank, duplicate-vector counts, similarity distributions, JSON/CSV/Markdown/PNG reports.
- `scripts/validate_embedding_quality_report.py`: row counts, query coverage, rank ordering, self-neighbor exclusion, hashes, and report consistency.

Baseline findings:

- Raw exact duplicate vectors: 673.
- Duplicate groups: 134.
- Equivalent-input duplicate excess: 14.
- Unexplained encoder-collision excess: 659.
- Effective rank: 10.55.
- Components required for 90% variance: 9.
- Top-10 cumulative variance: 91.86%.
- Random-pair cosine mean: about 0.458.
- Strict collapse acceptance: failed.
- EC entered the original model input, so original EC agreement was not independent validation.

Primary outputs:

- `artifacts/reports/semantic_quality/`
- `artifacts/reports/duplicate_embedding_diagnosis/`

### 2026-07-21 — Duplicate-vector root-cause investigation

Purpose:

- Determine whether duplicate vectors were caused by parse failures, empty/default graphs, identical preprocessed inputs, missing direction, pooling collisions, or unknown placeholders.

Added diagnostic code:

- `scripts/diagnose_duplicate_embeddings.py`: groups bit-identical embeddings, compares structure and model-input signatures, classifies expected equivalence versus unexplained collision, and records wildcard/unparsed/shared-identity involvement.
- `scripts/trace_encoder_collisions.py`: traces collisions through molecular embeddings.
- `scripts/trace_fusion_collisions.py`: captures fusion input and layer outputs to locate exact collision stages.

Findings:

- No duplicate group was caused by an empty graph or default vector.
- The one unsanitized molecule is retained as an 18-atom graph and does not appear in a duplicate group.
- Exact `*` participants are the generic R-group `CHEBI:13193`, not a universal missing-compound replacement.
- Many old collisions occurred before the final fusion layer.
- The original attention readout was mean-like and insensitive to molecular size/repeated local environments in some homologous and polymer-like structures.
- Reaction direction was already connected to fusion; it was not missing.
- Dummy-atom isotope/attachment labels such as `[1*]` and `[2*]` were absent from the original atom feature vector.

### 2026-07-21 — Phase 2 v2 structure-only redesign

Purpose:

- Remove EC leakage, improve molecular/set distinguishability, and add explicit anti-collapse regularization.

Changed model code:

- `src/rhea_embedding/models/reaction_encoder.py`
  - Removed EC, reaction type, participant role, and compartment from main fusion.
  - Added a hard configuration error for `use_ec=true` or `ec_as_input=true`.
  - Retained structures, side, signed stoichiometry, direction, and optional cofactor with a missing mask.
  - Replaced single participant attention with attention + normalized sum + participant count + absolute stoichiometry sum.
  - Retained the signed delta branch as a complementary feature.
  - Added EC prediction only as an output-side auxiliary head.
- `src/rhea_embedding/models/dmpnn.py`
  - Added `attention_sum` readout: attention + normalized atom sum + log atom count.
- `src/rhea_embedding/chemistry/graph.py`
  - Added isotope/attachment-label features so `[1*]`, `[2*]`, and `*` remain distinguishable.
  - Incremented graph-feature version.
- `src/rhea_embedding/data/reaction_dataset.py`
  - Added graph-feature-version validation to the cache and automatic cache rebuilding on feature changes.
- `src/rhea_embedding/training/phase2.py`
  - Creates two independently dropped/permuted legal views.
  - Adds InfoNCE contrastive loss.
  - Adds VICReg-style variance penalty.
  - Adds off-diagonal covariance penalty.
  - Adds optional multi-label EC BCE prediction from the embedding.
  - Saves exact train and validation indices in checkpoints.
  - Reports input policy and exact duplicate counts during export.

Loss convention:

```text
total_loss =
    1.0  * contrastive_loss
  + 1.0  * variance_penalty
  + 0.04 * covariance_penalty
  + w_ec * ec_auxiliary_bce
```

All terms are non-negative penalties minimized with addition. A negative sign would only be correct if variance/covariance were defined as rewards rather than penalties.

Added configuration and runners:

- `configs/model/phase2_structure_only.yaml`
- `configs/train/phase2_v2_pilot_cpu.yaml`
- `configs/export/phase2_v2_pilot.yaml`
- `scripts/run_phase2_v2_pilot.ps1`
- `scripts/audit_phase2_v2_inputs.py`
- `scripts/validate_phase2_v2.py`

Added/updated tests:

- EC changes do not change the main embedding.
- EC input configuration is rejected.
- Direction changes irreversible embeddings.
- Reversible swap invariance holds.
- Participant order invariance holds.
- Missing optional metadata remains finite.
- Attention-sum readout distinguishes molecule sizes.
- Dummy-atom isotope labels remain distinguishable.
- Variance and covariance penalties are finite.

Phase 2 v2 pilot training (`w_ec=0.2`):

- Seed: 42.
- Training/validation: 1,900/100.
- Epochs: 5.
- Best validation total loss: 0.4781011 at epoch 5.
- Checkpoint: `artifacts/checkpoints/phase2_v2_pilot.pt`.
- Full export: `artifacts/embeddings/reaction_embeddings_v2_pilot.npy`.

Phase 2 v2 quality results:

- Raw exact duplicate vectors: 51.
- Equivalent-input duplicate vectors: 51.
- Unexplained encoder collisions: 0.
- Effective rank: 52.91.
- Components required for 90% variance: 45.
- Top principal-component variance: about 5.21%.
- Top-10 cumulative variance: about 42.07%.
- Strict collapse acceptance: passed.

Input audit:

- Removing every EC value changes the embedding by exactly 0.
- Altering excluded metadata changes the embedding by exactly 0.
- Masking direction changes 18,072/18,072 embeddings.
- Direction-mask L2 difference median: 4.51.
- Direction counts: LR 6,554; RL 287; reversible 7,627; undefined 3,604.
- Current cofactor-role annotations: 0 participants; the model receives the missing mask.

Primary reports:

- `artifacts/reports/phase2_v2_adjustment_report.md`
- `artifacts/reports/phase2_v2_input_audit.json`
- `artifacts/reports/semantic_quality_v2_pilot/`
- `artifacts/reports/duplicate_embedding_diagnosis_v2_pilot/`

### 2026-07-21 — Independent EC evaluation policy

Purpose:

- Prevent EC auxiliary labels or checkpoint selection from leaking into EC nearest-neighbor evaluation.

Changed code:

- `scripts/analyze_embedding_quality.py`
  - Loads `train_indices` and `validation_indices` from a checkpoint.
  - Excludes their union from both nearest-neighbor query and candidate pools.
  - Writes `ec_consistency_training_excluded.csv`.
  - Uses the independent pool as the primary reported EC metric.

Frozen pilot split:

- Training reactions: 1,900.
- Validation/model-selection reactions: 100.
- Independent EC pool: 16,072.
- EC-annotated independent queries: 6,596.

Rule for future training:

- Never evaluate EC retrieval on reactions whose EC labels contributed to training or checkpoint selection.
- Keep the same frozen EC split when comparing objectives or model changes.

### 2026-07-21 — EC auxiliary weight ablation

Purpose:

- Compare `ec_auxiliary=0` with `0.2` while holding all other conditions fixed.

Added code and configuration:

- `configs/train/phase2_v2_pilot_ecaux0_cpu.yaml`
- `configs/export/phase2_v2_pilot_ecaux0.yaml`
- `scripts/run_ec_aux_ablation.ps1`
- `scripts/compare_ec_aux_ablation.py`
- `scripts/validate_phase2_v2.py` now accepts the expected EC weight.

Controlled conditions:

- Identical train indices, validation indices, model configuration, random seed, epochs, batch size, learning rate, and non-EC loss weights.
- Both models use the same independent EC query/candidate pool.
- EC remains absent from main embedding input in both models.

Weight-zero training record:

- Seed: 42.
- Training/validation: 1,900/100.
- Epochs: 5.
- Best validation total loss: 0.4811307 at epoch 5.
- Checkpoint: `artifacts/checkpoints/phase2_v2_pilot_ecaux0.pt`.
- Full export: `artifacts/embeddings/reaction_embeddings_v2_pilot_ecaux0.npy`.

Ablation results:

| Metric | `w_ec=0` | `w_ec=0.2` | Difference (`0.2-0`) |
|---|---:|---:|---:|
| Exact EC precision@1 | 15.631% | 15.221% | -0.409% |
| EC level-1 precision@1 | 75.076% | 74.970% | -0.106% |
| Effective rank | 52.87 | 52.91 | +0.04 |
| Components for 90% variance | 45 | 45 | 0 |
| Unexplained duplicate vectors | 0 | 0 | 0 |
| Collapse acceptance | pass | pass | — |

Paired exact-EC analysis:

- Annotated queries: 6,596.
- Weight 0.2 improves 41 queries, worsens 68, and leaves 6,487 unchanged.
- Paired precision@1 difference 95% interval: `[-0.719%, -0.099%]`.
- Current recommendation: use `ec_auxiliary=0` for the next stage and retain the independent EC pool.
- Limitation: this recommendation is based on one random seed and a 2,000-reaction pilot.

Primary outputs:

- `artifacts/reports/ec_aux_ablation/ec_aux_ablation_report.md`
- `artifacts/reports/ec_aux_ablation/ec_aux_ablation_report.json`
- `artifacts/reports/ec_aux_ablation/ec_aux_ablation_metrics.csv`

## 3. Current code inventory

### Data and chemistry

| File | Responsibility |
|---|---|
| `src/rhea_embedding/data/prepare_rhea.py` | Rhea parsing, normalization, deduplication, provenance, manifests, and Phase 1 exports |
| `src/rhea_embedding/data/reaction_dataset.py` | Corpus loading, vocabularies, graph cache, batching, participant permutation, and orientation swapping |
| `src/rhea_embedding/chemistry/graph.py` | RDKit graph construction, atom/bond/isotope features, graph batching, segment sum/softmax |

### Models and training

| File | Responsibility |
|---|---|
| `src/rhea_embedding/models/dmpnn.py` | Directed message passing and molecular readout |
| `src/rhea_embedding/models/reaction_encoder.py` | Participant tokens, hybrid side pooling, direction-aware reaction fusion, projection, EC auxiliary head |
| `src/rhea_embedding/training/phase2.py` | Losses, training loop, deterministic split, checkpoints, export, manifests, and quality metadata |
| `src/rhea_embedding/quality_control/embedding_checks.py` | Embedding invariance and reproducibility checks |

### Entry points, diagnostics, and validation

| File | Responsibility |
|---|---|
| `scripts/prepare_rhea.py` | Phase 1 CLI |
| `scripts/train_phase2.py` | Phase 2 training CLI |
| `scripts/export_phase2.py` | Full embedding export CLI |
| `scripts/analyze_embedding_quality.py` | Nearest neighbors, EC metrics, random baselines, SVD/collapse diagnostics |
| `scripts/diagnose_duplicate_embeddings.py` | Expected-equivalence versus encoder-collision classification |
| `scripts/trace_encoder_collisions.py` | Molecular-encoder collision tracing |
| `scripts/trace_fusion_collisions.py` | Reaction fusion-layer collision tracing |
| `scripts/audit_phase2_v2_inputs.py` | EC exclusion and direction sensitivity audit |
| `scripts/compare_ec_aux_ablation.py` | Controlled EC-weight comparison and paired confidence interval |
| `scripts/validate_phase1.py` | Phase 1 artifact validation |
| `scripts/validate_phase2.py` | Original MVP validation |
| `scripts/validate_phase2_v2.py` | v2 checkpoint/export/input-policy validation |
| `scripts/validate_embedding_quality_report.py` | Semantic report, neighbor table, and hash validation |
| `scripts/run_phase1.ps1` | Phase 1 end-to-end runner |
| `scripts/run_phase2.ps1` | Original MVP runner |
| `scripts/run_phase2_v2_pilot.ps1` | v2 training, export, diagnosis, semantic analysis, validation, and input audit |
| `scripts/run_ec_aux_ablation.ps1` | EC weight-zero training/export/analysis and comparison runner |
| `scripts/run_phase2_v2_full.ps1` | Full-corpus tests, training, export, diagnostics, validation, and input audit |

### Tests

| File | Coverage |
|---|---|
| `tests/test_prepare_rhea.py` | Deduplication key invariance and loose/strict key behavior |
| `tests/test_phase2.py` | Graph shapes, pooling, EC exclusion, direction, isotope labels, invariance, missing data, collapse penalties |

### Configurations and schemas

| File | Purpose |
|---|---|
| `configs/data/rhea_phase1.yaml` | Phase 1 source and output configuration |
| `configs/data/phase2_rhea.yaml` | Phase 2 corpus and graph-cache paths |
| `configs/model/phase2_mvp.yaml` | Superseded original model baseline |
| `configs/model/phase2_structure_only.yaml` | Current structure-only v2 model |
| `configs/train/phase2_mvp_cpu.yaml` | Superseded original pilot training |
| `configs/train/phase2_v2_pilot_cpu.yaml` | v2 pilot with `ec_auxiliary=0.2` |
| `configs/train/phase2_v2_pilot_ecaux0_cpu.yaml` | Recommended v2 pilot with `ec_auxiliary=0` |
| `configs/train/phase2_v2_full_ecaux0_cpu.yaml` | Recommended full-corpus fixed five-epoch training with `ec_auxiliary=0` |
| `configs/export/phase2_mvp.yaml` | Original MVP export |
| `configs/export/phase2_v2_pilot.yaml` | v2 weight-0.2 export |
| `configs/export/phase2_v2_pilot_ecaux0.yaml` | v2 weight-zero export |
| `configs/export/phase2_v2_full_ecaux0.yaml` | Recommended full-corpus weight-zero export |
| `schemas/reaction.schema.json` | Canonical reaction schema |
| `schemas/direction_variant.schema.json` | Direction-variant schema |

### Repository metadata and documentation

| File | Purpose |
|---|---|
| `README.md` | GitHub landing page, status, setup, layout, execution, validation, and audit guidance |
| `pyproject.toml` | Installable package metadata, dependencies, package discovery, and tool configuration |
| `requirements.txt` | Conventional dependency list for non-editable installation |
| `.gitignore` | Excludes environments, caches, generated data, checkpoints, embeddings, and reports |
| `.gitattributes` | Normalizes text line endings and marks binary artifact formats |
| `.github/workflows/ci.yml` | Python 3.11 editable-install and unit-test workflow |
| `CONTRIBUTING.md` | Development, validation, data-safety, and change-ledger rules |
| `data/README.md` | Data lifecycle and publication policy |
| `artifacts/README.md` | Generated-artifact layout and publication policy |

### Package markers

The following files are intentionally empty Python package markers and therefore contain no executable logic: `src/rhea_embedding/data/__init__.py`, `src/rhea_embedding/chemistry/__init__.py`, `src/rhea_embedding/models/__init__.py`, `src/rhea_embedding/training/__init__.py`, and `src/rhea_embedding/quality_control/__init__.py`. The root `src/rhea_embedding/__init__.py` defines package metadata.

Generated `__pycache__` files are runtime caches and are not source-code records.

## 4. Reproduction commands

Set the project root to `D:\Codex\2026-07-21` and configure Python/RDKit/PyTorch dependencies before running.

Phase 1:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1
```

Phase 2 v2 pilot (`ec_auxiliary=0.2` baseline):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase2_v2_pilot.ps1
```

EC auxiliary ablation and recommended weight-zero model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_ec_aux_ablation.ps1
```

Recommended full-corpus training and export:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase2_v2_full.ps1
```

Validate the recommended weight-zero export:

```powershell
python scripts\validate_phase2_v2.py `
  --export-config configs\export\phase2_v2_full_ecaux0.yaml `
  --expected-ec-weight 0

python scripts\validate_embedding_quality_report.py `
  --embedding artifacts\embeddings\reaction_embeddings_v2_full_ecaux0.npy `
  --report-dir artifacts\reports\semantic_quality_v2_full_ecaux0
```

## 5. Validation status

- Unit tests: 17 passed.
- Recommended full-corpus embedding shape: `[18072, 256]`.
- Dtype: float32.
- NaN/Inf: 0.
- Unique reaction IDs: 18,072.
- Unexpected duplicate embeddings: 0.
- Expected equivalent-input duplicate excess: 51.
- Participant-order invariance: pass.
- Reversible-swap invariance: pass.
- Batch/single equivalence: pass within tolerance.
- Randomized-SMILES equivalence: pass.
- EC removal exact invariance: pass, maximum absolute difference 0.
- Output and checkpoint hashes: verified.
- Top-10 nearest-neighbor table: 180,720 rows, 10 per query, no self-neighbors.
- Full-corpus and both EC ablation models pass the strict collapse check.
- Full-corpus effective rank: 149.57; 125 components explain 90% variance.
- Full-corpus direction audit: LR 6,554; reversible 7,627; RL 287; undefined 3,604.

## 6. Known limitations and open decisions

- Full-corpus training is complete but uses one random seed and a fixed five-epoch schedule inherited from the pilot.
- The full-corpus fit has no validation/model-selection subset; the final epoch is intentionally retained so all 18,072 reactions participate in training.
- Full-corpus EC metrics are transductive descriptive diagnostics. The pilot report remains the reaction-disjoint EC evidence.
- The current corpus has no populated cofactor-role annotations; only the missing mask is active.
- Atom-mapped/CGR and reaction-center branches are not implemented.
- No downstream biological task has been used as a final acceptance criterion.
- The project directory is not currently a Git repository; this ledger, manifests, configs, reports, and checksums are the available audit trail.
- Multiple-seed replication remains recommended before external comparative claims.

## 7. Required format for every future entry

Append new entries above the inventory using this template:

```markdown
### YYYY-MM-DD HH:MM — Short change title

Purpose:
- Why the change was needed.

Changed files:
- `path/to/file`: exact behavioral change.

Configuration:
- Old value -> new value.

Data/checkpoint compatibility:
- Whether caches/checkpoints must be rebuilt.

Validation:
- Commands run and pass/fail results.

Outputs:
- New or replaced artifacts and checksums/manifests.

Decision and rollback:
- Accepted/rejected conclusion and how to restore the previous state.
```

Maintenance rule: no model, data-processing, loss, split, evaluation, or export change is considered complete until this `agent.md` entry is updated.
