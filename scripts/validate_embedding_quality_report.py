from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate semantic embedding quality report artifacts")
    parser.add_argument("--embedding", default="artifacts/embeddings/reaction_embeddings.npy")
    parser.add_argument("--report-dir", default="artifacts/reports/semantic_quality")
    args = parser.parse_args()
    report_dir = root / args.report_dir
    summary = json.loads((report_dir / "embedding_semantic_quality_report.json").read_text(encoding="utf-8"))
    assert summary["embedding_shape"] == [18072, 256]
    assert summary["embedding_sha256"] == sha256_file(root / args.embedding)
    for filename, expected_hash in summary["outputs"].items():
        assert sha256_file(report_dir / filename) == expected_hash, filename

    counts: Counter[str] = Counter()
    previous_rank: dict[str, int] = {}
    previous_score: dict[str, float] = {}
    row_count = 0
    with (report_dir / "nearest_neighbors_top10.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query = row["query_reaction_id"]
            rank = int(row["rank"])
            score = float(row["cosine_similarity"])
            assert query != row["neighbor_reaction_id"]
            assert rank == previous_rank.get(query, 0) + 1
            assert score <= previous_score.get(query, float("inf")) + 1e-7
            counts[query] += 1
            previous_rank[query] = rank
            previous_score[query] = score
            row_count += 1
    assert row_count == 180720
    assert len(counts) == 18072
    assert set(counts.values()) == {10}

    criteria = summary["collapse_check"]["criterion_results"]
    assert summary["collapse_check"]["passed"] == all(criteria.values())
    assert len(summary["ec_consistency"]) == 12
    if summary.get("ec_consistency_training_excluded") is not None:
        assert len(summary["ec_consistency_training_excluded"]) == 12
        assert summary["ec_evaluation_pool_reaction_count"] < summary["reaction_count"]
        assert (report_dir / "ec_consistency_training_excluded.csv").exists()
    print(json.dumps({
        "status": "PASS",
        "nearest_neighbor_rows": row_count,
        "queries": len(counts),
        "neighbors_per_query": 10,
        "self_neighbor_count": 0,
        "report_hashes_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
