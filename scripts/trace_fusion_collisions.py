from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rhea_embedding.data.reaction_dataset import ReactionCorpus
from rhea_embedding.models.reaction_encoder import ReactionEncoder


def unique_rows(tensor: torch.Tensor) -> int:
    return int(np.unique(tensor.detach().cpu().numpy(), axis=0).shape[0])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the layer where exact reaction-embedding collisions occur")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/phase2_mvp.pt")
    parser.add_argument("--embedding", default="artifacts/embeddings/reaction_embeddings.npy")
    parser.add_argument("--ids", default="artifacts/embeddings/reaction_ids.tsv")
    parser.add_argument("--data-config", default="configs/data/phase2_rhea.yaml")
    parser.add_argument("--output-dir", default="artifacts/reports/duplicate_embedding_diagnosis")
    args = parser.parse_args()

    data_config = yaml.safe_load((PROJECT_ROOT / args.data_config).read_text(encoding="utf-8"))
    corpus = ReactionCorpus(PROJECT_ROOT / data_config["reaction_parquet"], PROJECT_ROOT / data_config["graph_cache"])
    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = ReactionEncoder(checkpoint["model_config"], checkpoint["vocab_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    matrix = np.load(PROJECT_ROOT / args.embedding, allow_pickle=False)
    with (PROJECT_ROOT / args.ids).open("r", encoding="utf-8", newline="") as handle:
        reaction_ids = [row["reaction_id"] for row in csv.DictReader(handle, delimiter="\t")]
    example_by_id = {example.reaction_id: example for example in corpus.examples}
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(matrix):
        groups[row.tobytes()].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]

    captures: dict[str, torch.Tensor] = {}
    handles = [
        model.fusion[0].register_forward_pre_hook(lambda module, values: captures.__setitem__("fusion_input", values[0].detach())),
        model.fusion[0].register_forward_hook(lambda module, values, output: captures.__setitem__("linear1", output.detach())),
        model.fusion[1].register_forward_hook(lambda module, values, output: captures.__setitem__("relu", output.detach())),
        model.fusion[3].register_forward_hook(lambda module, values, output: captures.__setitem__("linear2", output.detach())),
        model.fusion[4].register_forward_hook(lambda module, values, output: captures.__setitem__("layernorm", output.detach())),
    ]
    rows = []
    for group_id, indices in enumerate(duplicate_groups, 1):
        examples = [example_by_id[reaction_ids[index]] for index in indices]
        direction_indices = {example.direction_index for example in examples}
        if 2 in direction_indices:
            continue
        batch = corpus.collate(examples)
        molecules = model.encode_molecules(batch)
        output = model.encode_canonical(molecules, batch).embedding
        rows.append({
            "group_id": group_id,
            "group_size": len(indices),
            "direction_indices": ";".join(map(str, sorted(direction_indices))),
            "fusion_input_unique_rows": unique_rows(captures["fusion_input"]),
            "linear1_unique_rows": unique_rows(captures["linear1"]),
            "relu_unique_rows": unique_rows(captures["relu"]),
            "linear2_unique_rows": unique_rows(captures["linear2"]),
            "layernorm_unique_rows": unique_rows(captures["layernorm"]),
            "canonical_output_unique_rows": unique_rows(output),
            "fusion_input_max_range": float((captures["fusion_input"].max(dim=0).values - captures["fusion_input"].min(dim=0).values).max()),
            "linear1_max_range": float((captures["linear1"].max(dim=0).values - captures["linear1"].min(dim=0).values).max()),
            "relu_max_range": float((captures["relu"].max(dim=0).values - captures["relu"].min(dim=0).values).max()),
            "linear2_max_range": float((captures["linear2"].max(dim=0).values - captures["linear2"].min(dim=0).values).max()),
        })
    for handle in handles:
        handle.remove()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fusion_collision_trace.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    first_collapsed = Counter()
    for row in rows:
        stages = [
            ("fusion_input", row["fusion_input_unique_rows"]),
            ("linear1", row["linear1_unique_rows"]),
            ("relu", row["relu_unique_rows"]),
            ("linear2", row["linear2_unique_rows"]),
            ("layernorm", row["layernorm_unique_rows"]),
        ]
        collapsed_stage = next((name for name, count in stages if count == 1), "not_exact_before_canonical")
        first_collapsed[collapsed_stage] += 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duplicate_groups_total": len(duplicate_groups),
        "groups_traced_excluding_reversible": len(rows),
        "first_exact_collision_stage_counts": dict(first_collapsed),
        "interpretation": (
            "Distinct fusion inputs that become identical after a linear layer indicate small structural differences "
            "are below float32 resolution after the trained projection, not empty/default graph replacement."
        ),
    }
    (output_dir / "fusion_collision_trace.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
