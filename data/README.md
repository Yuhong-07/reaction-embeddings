# Data directory

This directory follows a reproducible data-lifecycle layout:

- `raw/`: immutable source archives downloaded from Rhea.
- `interim/`: graph caches, candidate matches, and transformation logs.
- `processed/`: validated and deduplicated reaction records used for embedding.
- `manifests/`: source versions, provenance, coverage, and checksums.

Large source and generated data files are intentionally excluded from ordinary Git commits. Recreate them with `scripts/run_phase1.ps1` or obtain the frozen release artifacts and verify them against the manifests.
