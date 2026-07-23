from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 2 v2 structure-only embedding outputs")
    parser.add_argument("--export-config", default="configs/export/phase2_v2_pilot.yaml")
    parser.add_argument(
        "--expected-ec-weight",
        type=float,
        default=None,
        help="Expected EC auxiliary weight; defaults to the value stored in the checkpoint.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    export_config = yaml.safe_load((root / args.export_config).read_text(encoding="utf-8"))
    paths = {key: root / value for key, value in export_config.items() if key != "batch_size"}
    checkpoint = torch.load(paths["checkpoint_path"], map_location="cpu", weights_only=False)
    checkpoint_ec_weight = float(checkpoint["train_config"]["loss_weights"]["ec_auxiliary"])
    expected_ec_weight = checkpoint_ec_weight if args.expected_ec_weight is None else args.expected_ec_weight
    report = json.loads(paths["quality_report"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
    matrix = np.load(paths["embedding_npy"], allow_pickle=False)
    parquet = pq.read_table(paths["embedding_parquet"]).to_pylist()
    with paths["reaction_ids_tsv"].open("r", encoding="utf-8", newline="") as handle:
        ids = [row["reaction_id"] for row in csv.DictReader(handle, delimiter="\t")]

    model_config = checkpoint["model_config"]
    assert model_config.get("ec_as_input", False) is False
    assert model_config.get("use_ec", False) is False
    assert model_config["molecule_readout"] == "attention_sum"
    assert matrix.shape == (18072, 256)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert len(ids) == len(set(ids)) == matrix.shape[0]
    assert [row["reaction_id"] for row in parquet] == ids
    assert np.array_equal(np.asarray([row["embedding"] for row in parquet], dtype=np.float32), matrix)
    assert report["invariance_checks"]["passed"] is True
    assert checkpoint["train_config"]["loss_weights"] == {
        "contrastive": 1.0, "variance": 1.0, "covariance": 0.04,
        "ec_auxiliary": expected_ec_weight,
    }
    for relative, expected in report["output_checksums"].items():
        assert sha256_file(root / relative) == expected
    assert manifest["checkpoint"]["sha256"] == sha256_file(paths["checkpoint_path"])

    unique_vectors = int(np.unique(matrix, axis=0).shape[0])
    print(json.dumps({
        "status": "PASS",
        "matrix_shape": list(matrix.shape),
        "ec_as_embedding_input": False,
        "ec_auxiliary_weight": expected_ec_weight,
        "molecule_readout": model_config["molecule_readout"],
        "exact_duplicate_vector_count": int(len(matrix) - unique_vectors),
        "invariance_passed": True,
        "output_hashes_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
