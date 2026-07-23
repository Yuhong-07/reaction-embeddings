# Phase 2 MVP pilot

> Historical note: this document describes the pre-adjustment pilot. Full-corpus structure-only training on all 18,072 reactions was completed on 2026-07-23. See `docs/embedding_results_summary.md` and `artifacts/reports/phase2_v2_full_training_report.md` for the recommended output.

> Superseded baseline: this document describes the pre-adjustment checkpoint. See
> `artifacts/reports/phase2_v2_adjustment_report.md` and
> `configs/model/phase2_structure_only.yaml` for the current structure-only v2 pilot.

This checkpoint validates the minimum deep reaction-embedding pipeline on CPU.
It uses a shared four-step D-MPNN molecular encoder, stoichiometry-aware participant
tokens, order-invariant attention pooling for reaction sides, signed molecular
difference aggregation, direction handling, optional metadata missing masks, and
a 256-dimensional reaction projection.

Training uses a deterministic 2,000-reaction subset: 1,900 training reactions and
100 validation reactions. The loss combines contrastive views, masked participant
reconstruction in embedding space, masked stoichiometry regression, and direction
classification. The trained checkpoint is then applied to all 18,072 canonical
reactions.

This is a pilot representation. It is suitable for verifying data/model interfaces,
invariance, export reproducibility, and downstream plumbing. It is not a substitute
for full-corpus self-supervised pretraining or downstream biological validation.

Direction policy:

- LR evidence only: retain stored LR orientation.
- RL evidence only: swap sides and negate coefficients.
- BI evidence or both LR and RL: average forward and swapped encodings.
- No directional evidence: retain stored orientation and use the undefined token.
