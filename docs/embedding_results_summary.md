# Full-Corpus Reaction Embeddings: Methods, Data, Inputs, Outputs, and Validation

**Project:** Rhea Reaction Embeddings  
**Report date:** 23 July 2026  
**Recommended model:** Phase 2 v2 full-corpus structure-only model with `ec_auxiliary=0`  
**Reporting status:** Full-corpus self-supervised training and export complete; downstream biological validation remains pending

## Executive summary

We prepared 18,072 strictly deduplicated chemical reactions from Rhea release 141, trained the structure-only encoder on all 18,072 reactions for five fixed epochs, and exported one 256-dimensional embedding per reaction. No validation subset was removed from the final fit because the architecture, loss weights, and epoch count had already been selected in the preceding pilot and EC auxiliary ablation.

The main embedding uses molecular structures, participant side, signed stoichiometry, evidence-derived reaction direction, and an optional cofactor-role branch with an explicit missing-value mask. EC numbers are neither encoder inputs nor training targets in the recommended model (`ec_auxiliary=0`).

The final matrix has shape `[18,072, 256]`, contains no NaN or infinite values, and passes all numerical, invariance, duplicate-vector, and anti-collapse checks. Its effective rank is 149.57, compared with 52.87 for the preceding weight-zero pilot. There are 51 exact duplicate vectors; all 51 are explained by reactions with identical permitted model inputs, leaving zero unexplained encoder collisions.

This export is the recommended chemical-reaction representation for subsequent network integration. It is not yet a condition-specific, organism-specific, compartment-aware, or flux-aware representation.

## 1. Data

### Source and preparation

- Source database: Rhea
- Release: 141, dated 10 June 2026
- Source license: CC BY 4.0
- Input archives: Rhea RXN and TSV distributions
- Python: 3.11.7
- RDKit: 2025.09.6

RXN records supplied participants and molecular structures. TSV records supplied identifiers, EC annotations, cross-references, reaction SMILES, and direction evidence. Strict deduplication required matching compound identities, reduced signed stoichiometry, compartment, and direction class. Loose matches were reported but never merged automatically.

| Data statistic | Value |
|---|---:|
| Rhea master source records | 18,558 |
| Successfully parsed master records | 18,550 |
| Records merged by strict deduplication | 478 |
| Final reactions | 18,072 |
| Unique molecular structures | 14,051 |
| Missing RXN records | 8 |
| Failed participant SMILES | 0 |
| Retained unsanitized structures | 1 |
| Reactions flagged as imbalanced | 6 |
| EC-annotated reactions | 7,416 (41.0%) |

### Direction handling

The stored canonical Rhea master field is `direction=undefined`, because a master identifier does not itself assert condition-specific net flux. The model does not simply use that field for every reaction. `ReactionCorpus` derives a direction policy from the available `supported_directions` evidence and then applies the configured orientation policy.

| Model direction category | Reaction count |
|---|---:|
| Left-to-right | 6,554 |
| Reversible | 7,627 |
| Right-to-left | 287 |
| Undefined | 3,604 |

Masking direction changed all 18,072 embeddings by more than `1e-6`; the median L2 change was 5.91. These categories represent database direction evidence, not organism- or condition-specific flux. The export must not be presented as a flux-direction model.

## 2. Model inputs

This section describes the values actually consumed by the current full-corpus code path: `ReactionCorpus`, `ReactionEncoder`, and `configs/model/phase2_structure_only.yaml`.

### Inputs consumed by the current encoder

- **Participant molecular graphs.** Each participant's `canonical_smiles` is parsed with RDKit and converted to a graph; the SMILES text itself is not tokenized by the model.
- **Atom features.** Atomic number, formal charge, degree, aromaticity, hybridization, chirality, total hydrogen count, and isotope information.
- **Directed-bond features.** Bond type, conjugation, ring membership, and bond stereochemistry.
- **Participant side.** A learned two-class embedding distinguishes reactants from products.
- **Stoichiometry.** `log1p(abs(coefficient))` and a stoichiometry-missing indicator enter each participant token. The signed raw coefficient is also used in the molecular delta branch, while each side pool receives `log1p` of its total absolute stoichiometry.
- **Per-side aggregate features.** Reactants and products are pooled separately using an attention-weighted token pool, a participant-count-normalized token sum, `log1p(participant_count)`, and `log1p(total_absolute_stoichiometry)`.
- **Evidence-derived direction.** `supported_directions` is reduced to `left_to_right`, `right_to_left`, `reversible`, or `undefined`. The encoder consumes a learned direction embedding and an undefined-direction indicator. Right-to-left reactions are encoded after swapping sides and coefficient signs; reversible reactions average the original and swapped encodings.
- **Cofactor-role branch.** The active configuration has `use_cofactor: true`, so the encoder consumes a cofactor-role embedding and a cofactor-missing indicator for every participant. The current corpus has no populated cofactor-role annotations; consequently, this branch receives the unknown index and `cofactor_missing=1` for all participants and contributes no observed role categories.

The fusion layer receives the reactant-side representation, product-side representation, `products - reactants`, the signed stoichiometric molecular delta, the direction embedding, and the direction-missing indicator. It outputs the 256-dimensional reaction embedding.

### Values not consumed by the main embedding encoder

- EC numbers; `ReactionEncoder` raises an error if `use_ec` or `ec_as_input` is enabled
- Reaction ID or canonical reaction ID
- Raw reaction SMILES or atom-mapped reaction SMILES
- Reaction type
- Participant `role` metadata
- Compartment
- Atom mapping and bond-change features
- Organism, tissue, condition, pathway, network topology, flux, bounds, or other biological context

EC values may be loaded by the dataset for auxiliary prediction or evaluation, but the selected full-corpus training configuration sets `ec_auxiliary: 0.0`; therefore, EC is neither an encoder input nor an active training target. The input audit measured exactly zero embedding change after removing EC or altering other excluded metadata.

## 3. Model and training methods

Each molecule is encoded by a shared four-step D-MPNN with hidden dimension 256 and dropout 0.1. Molecular readout combines attention, a normalized atom sum, and log atom count. Reactants and products are pooled separately using attention, a normalized participant sum, participant count, and absolute stoichiometry sum. A signed stoichiometric delta branch supplies complementary information. The fused representation is projected to 256 dimensions.

For reversible reactions, the representation is averaged across forward and side-swapped encodings. Right-to-left reactions swap sides and coefficient signs. Undefined reactions retain the stored orientation and use the undefined-direction embedding.

### Full-corpus training configuration

| Setting | Value |
|---|---:|
| Training reactions | 18,072 |
| Validation reactions | 0 |
| Checkpoint selection | Final epoch; no validation model selection |
| Random seed | 42 |
| Epochs | 5 |
| Batch size | 64 |
| Learning rate | 0.0003 |
| Weight decay | 0.00001 |
| Temperature | 0.1 |
| Device | CPU, 8 threads |
| Training time | 696.5 seconds |

The fixed objective was selected from the preceding controlled pilot ablation:

```text
total_loss =
    1.00 * contrastive_loss
  + 1.00 * variance_penalty
  + 0.04 * covariance_penalty
  + 0.00 * ec_auxiliary_loss
```

All terms are non-negative penalties and are added. Training loss decreased from 0.6443 in epoch 1 to 0.2960 in epoch 5.

### Why EC auxiliary weight is zero

The pilot compared `ec_auxiliary=0` with `0.2` using identical train/validation indices, seed, model, and reaction-disjoint EC evaluation pool. Exact EC precision@1 was 15.631% for weight zero and 15.221% for weight 0.2. The paired difference (`0.2 - 0`) was -0.409 percentage points with a reaction-level 95% interval of `[-0.719, -0.099]`. Both models passed collapse checks, so the simpler structure-only objective was selected for the full fit.

This ablation is based on one random seed; the reaction-level interval does not replace uncertainty across repeated training seeds.

## 4. Recommended outputs

The project produces several files because numerical data, row identifiers, model weights, reproducibility metadata, and quality evidence have different formats and use cases. Most users do **not** need every file.

### Which files do I actually need?

| Use case | Minimum required files |
|---|---|
| Inspect or analyze embeddings with pandas | `reaction_embeddings_v2_full_ecaux0.parquet` only |
| Train a NumPy/PyTorch/scikit-learn model | `reaction_embeddings_v2_full_ecaux0.npy` plus `reaction_ids_v2_full_ecaux0.tsv` |
| Attach embeddings to a metabolic network | Parquet only, or NPY plus the ID table; a network reaction-to-Rhea mapping is also required |
| Reproduce embeddings | Checkpoint, configs, processed reaction corpus, and export manifest |
| Verify quality or report results | Quality, semantic, duplicate, and input-audit reports |

The NPY and Parquet files contain the same embedding values in different representations. The NPY file is compact and efficient for numerical training; the TSV file identifies each NPY row. The Parquet file stores `reaction_id` and `embedding` together and is usually the simplest file for inspection and database joins.

Checkpoint and report files are not additional embedding datasets:

- The checkpoint contains trained model parameters.
- The manifest records inputs, configurations, and checksums.
- The quality report verifies shape, finite values, invariance, and output hashes.
- The semantic report contains nearest-neighbor, EC, effective-rank, and collapse diagnostics.
- The duplicate report explains repeated vectors.
- The input audit verifies that EC and excluded metadata do not enter the embedding.

| Output | Location |
|---|---|
| NumPy embedding matrix | `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.npy` |
| Parquet embedding table | `artifacts/embeddings/reaction_embeddings_v2_full_ecaux0.parquet` |
| Reaction-ID mapping | `artifacts/embeddings/reaction_ids_v2_full_ecaux0.tsv` |
| Full checkpoint | `artifacts/checkpoints/phase2_v2_full_ecaux0.pt` |
| Checkpoint metadata | `artifacts/checkpoints/phase2_v2_full_ecaux0.metadata.json` |
| Export quality report | `artifacts/reports/embedding_quality_report_v2_full_ecaux0.json` |
| Semantic/collapse report | `artifacts/reports/semantic_quality_v2_full_ecaux0/embedding_semantic_quality_report.json` |
| Duplicate diagnosis | `artifacts/reports/duplicate_embedding_diagnosis_v2_full_ecaux0/duplicate_embedding_diagnosis.json` |
| Input audit | `artifacts/reports/phase2_v2_full_input_audit.json` |
| Export manifest | `data/manifests/phase2_v2_full_ecaux0_embedding_manifest.json` |

### SHA-256 checksums

| File | SHA-256 |
|---|---|
| Full checkpoint | `6216d9d86a0fb75d8a3335482a50a6ea5409446687d2e34d9428148a09c093b2` |
| NPY embedding | `d53db0ca8ca97c43c5a0aad4da7d0a93b0baf8d7d814bdb46b076d21ddf09e35` |
| Parquet embedding | `80ae07da11c63a6f80932982898f222c8a22a6c0740387b55bc6e5b3fead8562` |
| Reaction IDs | `1f0b4bd4e9fceb39d387d5d22a73e379ec7aa91ee14d8d52b0c5cdc66850a343` |

## 5. Validation results

### Numerical integrity and invariance

| Check | Full-corpus result |
|---|---:|
| Matrix shape | `[18072, 256]` |
| Data type | Float32 |
| NaN or infinite values | 0 |
| Unique reaction IDs | 18,072 |
| Repeat-export maximum difference | 0.0 |
| Participant-order maximum difference | `1.01e-6` |
| Batch-versus-single maximum difference | `9.54e-7` |
| Reversible-swap maximum difference | 0.0 |
| Randomized-SMILES cosine similarity | 1.0 |
| Automated tests | 17/17 passed |

### Collapse and duplicate diagnostics

| Metric | Full model | Weight-zero pilot |
|---|---:|---:|
| Effective rank | 149.57 | 52.87 |
| Components explaining 90% variance | 125 | 45 |
| First-component explained variance | 2.000% | 5.412% |
| Top-10 cumulative explained variance | 15.718% | 42.066% |
| Random-pair cosine mean | 0.0481 | 0.1343 |
| Random-pair cosine standard deviation | 0.0920 | 0.1533 |
| Exact duplicate-vector excess | 51 | 51 |
| Input-equivalent duplicate excess | 51 | 51 |
| Unexplained duplicate excess | 0 | 0 |
| Strict collapse acceptance | Pass | Pass |

The 51 repeated vectors arise from 40 groups whose permitted model inputs are identical. They should not be forced apart using reaction IDs or EC labels.

### EC neighborhood diagnostics

The full model used every reaction structure but no EC labels. Therefore, its EC neighborhood metrics are **transductive descriptive diagnostics**, not reaction-disjoint generalization estimates.

| Full-corpus descriptive metric | Result | Random baseline |
|---|---:|---:|
| Exact EC precision@1 | 16.788% | 0.004638% |
| Exact EC hit@10 | 26.092% | 0.046342% |
| EC level-1 precision@1 | 75.526% | 10.397% |

The earlier pilot remains the appropriate reaction-disjoint EC evidence: after excluding 1,900 training and 100 validation/model-selection reactions from both queries and candidates, exact EC precision@1 was 15.631%, exact EC hit@10 was 22.968%, and EC level-1 precision@1 was 75.076%.

EC agreement describes embedding geometry; it does not establish performance on an organism-specific, pathway, flux, or other downstream biological task.

## 6. Reporting limitations

1. Full-corpus training used one random seed and a fixed five-epoch schedule.
2. No validation split was used in the final fit; model and epoch choices were inherited from the pilot.
3. Full-corpus EC metrics are transductive because all reaction structures were used during self-supervised training, although EC labels were not used.
4. The representation contains no organism, compartment, condition, or flux context.
5. Direction categories come from Rhea-linked direction evidence and are not condition-specific flux directions.
6. Cofactor-role, atom-map, context, reaction-type, and compartment fields remain unpopulated.
7. No downstream biological task has been used as a final acceptance criterion.
8. The project directory is not currently a Git repository, so `git_revision` is null in manifests.
