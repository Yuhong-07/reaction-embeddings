# Data decisions

## 2026-07-21 — strict chemical deduplication

The release contains 18,550 successfully parsed Rhea master records. The strict
canonical key groups records only when compound identity, reduced signed
stoichiometry, compartment, and direction class are identical. This produces
18,072 canonical chemical reactions by merging 478 source records across 184
groups.

The project owner confirmed that the 18,072 deduplicated chemical reactions are
the embedding corpus. Because 102 duplicate groups have different EC annotation
sets, source-specific EC numbers and external cross-references are retained in
`data/processed/rhea_source_records.*`. The canonical corpus may contain an
aggregate EC list for optional missing-aware model input, but the source mapping
table remains authoritative for source-level annotation.

Loose equivalence keys that ignore proton, water, direction, or compartment are
never auto-merged. They are exported only as review candidates.

Eight master records listed in the direction table have no corresponding
left-to-right RXN file in release 141. They are retained in the source mapping
with `embedding_included=false` and an explicit error, but are not inserted into
the canonical embedding corpus.

## 2026-07-21 — direction representation

Rhea master IDs are UN (undefined-direction) members of a four-ID quartet. The
canonical chemical record therefore keeps `direction=undefined`; setting all
masters to reversible would be incorrect. Direction-aware input is exported as
three linked variants per canonical reaction (LR, RL, BI). Cross-reference counts
from Rhea external mappings and UniProtKB/Swiss-Prot distinguish a formally
available quartet variant from a direction that is actually used by a linked
record. LR/RL side orientation is handled by `swap_sides`; BI requests reversible
symmetrization in the model.
