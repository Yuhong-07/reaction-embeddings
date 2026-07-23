from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Rhea Phase 1 export")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: root / value for key, value in config["paths"].items()}
    report = json.loads(paths["report"].read_text(encoding="utf-8"))

    canonical = pq.read_table(paths["processed_parquet"]).to_pylist()
    source = pq.read_table(paths["source_records_parquet"]).to_pylist()
    direction_variants = pq.read_table(paths["direction_variants_parquet"]).to_pylist()
    canonical_ids = {row["reaction_id"] for row in canonical}
    assert len(canonical) == report["reaction_count"] == 18072
    assert len(canonical_ids) == len(canonical), "canonical reaction IDs are not unique"
    assert len(source) == report["source_record_count"] == 18558
    assert len(direction_variants) == len(canonical) * 3 == 54216
    assert len({row["direction_variant_id"] for row in direction_variants}) == len(direction_variants)
    by_canonical = {}
    for row in direction_variants:
        by_canonical.setdefault(row["canonical_reaction_id"], set()).add(row["direction"])
    assert set(by_canonical) == canonical_ids
    assert all(values == {"left_to_right", "right_to_left", "reversible"} for values in by_canonical.values())

    for row in canonical:
        participants = row["participants"]
        assert participants and any(p["side"] == "reactant" for p in participants)
        assert any(p["side"] == "product" for p in participants)
        for participant in participants:
            assert participant["canonical_smiles"]
            if participant["side"] == "reactant":
                assert participant["coefficient"] < 0
            else:
                assert participant["coefficient"] > 0

    for row in source:
        if row["embedding_included"]:
            assert row["canonical_reaction_id"] in canonical_ids
        else:
            assert row["canonical_reaction_id"] is None
            assert row["parse_error"]

    with paths["id_map"].open("r", encoding="utf-8", newline="") as handle:
        id_map = list(csv.DictReader(handle, delimiter="\t"))
    assert len(id_map) == len(source)
    assert len({row["source_reaction_id"] for row in id_map}) == len(source)

    for relative, expected in report["output_checksums"].items():
        actual = sha256_file(root / relative)
        assert actual == expected, f"checksum mismatch: {relative}"

    print(json.dumps({
        "status": "ok",
        "canonical_reaction_count": len(canonical),
        "source_record_count": len(source),
        "direction_variant_count": len(direction_variants),
        "failed_smiles_count": report["failed_smiles_count"],
        "unsanitized_smiles_count": report["unsanitized_smiles_count"],
        "imbalanced_reaction_count": report["imbalanced_reaction_count"],
        "unknown_balance_count": report["unknown_balance_count"],
        "verified_checksum_count": len(report["output_checksums"]),
    }, indent=2))


if __name__ == "__main__":
    main()
