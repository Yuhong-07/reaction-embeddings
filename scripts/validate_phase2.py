from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 2 embedding outputs")
    parser.add_argument("--export-config", type=Path, default=Path("configs/export/phase2_mvp.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.export_config if args.export_config.is_absolute() else root / args.export_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: root / value for key, value in config.items() if key != "batch_size"}
    report = json.loads(paths["quality_report"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))

    matrix = np.load(paths["embedding_npy"], allow_pickle=False)
    parquet = pq.read_table(paths["embedding_parquet"]).to_pylist()
    with paths["reaction_ids_tsv"].open("r", encoding="utf-8", newline="") as handle:
        id_rows = list(csv.DictReader(handle, delimiter="\t"))
    ids = [row["reaction_id"] for row in id_rows]

    assert matrix.shape == (18072, 256)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert len(ids) == len(set(ids)) == matrix.shape[0]
    assert [row["reaction_id"] for row in parquet] == ids
    parquet_matrix = np.asarray([row["embedding"] for row in parquet], dtype=np.float32)
    assert np.array_equal(parquet_matrix, matrix)
    assert report["matrix_shape"] == [18072, 256]
    assert report["nan_or_inf_count"] == 0
    assert report["invariance_checks"]["passed"] is True
    assert report["training_scope"] == "pilot_subset"
    assert report["selected_training_and_validation_count"] == 2000
    assert report["train_reaction_count"] == 1900
    assert report["validation_reaction_count"] == 100

    for relative, expected in report["output_checksums"].items():
        assert sha256_file(root / relative) == expected, f"Checksum mismatch: {relative}"
    assert sha256_file(paths["checkpoint_path"]) == report["checkpoint_sha256"]
    assert manifest["checkpoint"]["sha256"] == report["checkpoint_sha256"]

    print(json.dumps({
        "status": "ok",
        "matrix_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "unique_reaction_ids": len(set(ids)),
        "nan_or_inf_count": int((~np.isfinite(matrix)).sum()),
        "invariance_passed": report["invariance_checks"]["passed"],
        "training_scope": report["training_scope"],
        "train_reaction_count": report["train_reaction_count"],
        "validation_reaction_count": report["validation_reaction_count"],
        "verified_output_checksums": len(report["output_checksums"]),
    }, indent=2))


if __name__ == "__main__":
    main()

