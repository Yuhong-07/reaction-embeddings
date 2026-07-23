from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rhea_embedding.data.reaction_dataset import ReactionCorpus
from rhea_embedding.models.reaction_encoder import ReactionEncoder


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit which fields can affect Phase 2 v2 embeddings")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/phase2_v2_pilot.pt")
    parser.add_argument("--data-config", default="configs/data/phase2_rhea.yaml")
    parser.add_argument("--output", default="artifacts/reports/phase2_v2_input_audit.json")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_config = yaml.safe_load((PROJECT_ROOT / args.data_config).read_text(encoding="utf-8"))
    corpus = ReactionCorpus(PROJECT_ROOT / data_config["reaction_parquet"], PROJECT_ROOT / data_config["graph_cache"])
    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = ReactionEncoder(checkpoint["model_config"], checkpoint["vocab_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(corpus, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=corpus.collate_fn())

    direction_differences = []
    ec_max_abs_difference = 0.0
    excluded_metadata_max_abs_difference = 0.0
    for batch in loader:
        molecules = model.encode_molecules(batch)
        base = model.encode_canonical(molecules, batch).embedding
        without_direction = model.encode_canonical(molecules, batch, mask_direction_input=True).embedding
        direction_differences.extend(torch.linalg.vector_norm(base - without_direction, dim=1).cpu().tolist())

        without_ec = replace(
            batch,
            ec_index=torch.empty((0,), dtype=torch.long),
            ec_reaction_index=torch.empty((0,), dtype=torch.long),
        )
        ec_difference = (base - model.encode_canonical(molecules, without_ec).embedding).abs().max()
        ec_max_abs_difference = max(ec_max_abs_difference, float(ec_difference))

        altered_excluded_metadata = replace(
            batch,
            role_index=torch.zeros_like(batch.role_index),
            compartment_index=torch.zeros_like(batch.compartment_index),
            reaction_type_index=torch.zeros_like(batch.reaction_type_index),
        )
        metadata_difference = (base - model.encode_canonical(molecules, altered_excluded_metadata).embedding).abs().max()
        excluded_metadata_max_abs_difference = max(excluded_metadata_max_abs_difference, float(metadata_difference))

    differences = np.asarray(direction_differences)
    direction_counts = Counter(example.direction_index for example in corpus.examples)
    direction_names = {0: "left_to_right", 1: "right_to_left", 2: "reversible", 3: "undefined"}
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reaction_count": len(corpus),
        "main_embedding_inputs": [
            "reactant_structure", "product_structure", "signed_stoichiometry",
            "reaction_direction", "cofactor_role_with_missing_mask",
        ],
        "forbidden_or_excluded_inputs": ["ec_number", "reaction_type", "role", "compartment"],
        "ec_removal_max_abs_embedding_difference": ec_max_abs_difference,
        "excluded_metadata_max_abs_embedding_difference": excluded_metadata_max_abs_difference,
        "direction_mask_l2_difference": {
            "min": float(differences.min()),
            "median": float(np.median(differences)),
            "mean": float(differences.mean()),
            "max": float(differences.max()),
            "changed_reaction_count_at_1e_6": int(np.sum(differences > 1e-6)),
        },
        "direction_distribution": {
            direction_names[index]: direction_counts.get(index, 0) for index in range(4)
        },
        "cofactor_annotated_participant_count": sum(
            index != 0 for example in corpus.examples for index in example.cofactor_indices
        ),
        "passed": bool(
            ec_max_abs_difference == 0.0
            and excluded_metadata_max_abs_difference == 0.0
            and np.sum(differences > 1e-6) == len(corpus)
        ),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
