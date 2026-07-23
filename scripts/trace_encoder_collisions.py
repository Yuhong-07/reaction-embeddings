from __future__ import annotations

import argparse
import csv
import hashlib
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

from rhea_embedding.chemistry.graph import collate_molecule_graphs
from rhea_embedding.data.reaction_dataset import ReactionCorpus
from rhea_embedding.models.reaction_encoder import ReactionEncoder


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded_input_signature(example, molecule_keys: list[str]) -> str:
    participants = sorted(zip(
        example.sides,
        [round(float(value), 8) for value in example.coefficients],
        [molecule_keys[index] for index in example.molecule_indices],
        example.role_indices,
        example.cofactor_indices,
        example.compartment_indices,
    ))
    if example.direction_index == 1:
        participants = sorted((1 - side, -coefficient, molecule, role, cofactor, compartment)
                              for side, coefficient, molecule, role, cofactor, compartment in participants)
    if example.direction_index == 2:
        reversed_participants = sorted((1 - side, -coefficient, molecule, role, cofactor, compartment)
                                       for side, coefficient, molecule, role, cofactor, compartment in participants)
        participants = min(participants, reversed_participants)
    payload = {
        "participants": participants,
        "direction_index": example.direction_index,
        "reaction_type_index": example.reaction_type_index,
        "ec_indices": sorted(example.ec_indices),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Trace duplicate reaction embeddings through the molecular encoder")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/phase2_mvp.pt")
    parser.add_argument("--embedding", default="artifacts/embeddings/reaction_embeddings.npy")
    parser.add_argument("--ids", default="artifacts/embeddings/reaction_ids.tsv")
    parser.add_argument("--data-config", default="configs/data/phase2_rhea.yaml")
    parser.add_argument("--output-dir", default="artifacts/reports/duplicate_embedding_diagnosis")
    parser.add_argument("--molecule-batch-size", type=int, default=256)
    args = parser.parse_args()

    data_config = yaml.safe_load((PROJECT_ROOT / args.data_config).read_text(encoding="utf-8"))
    corpus = ReactionCorpus(
        PROJECT_ROOT / data_config["reaction_parquet"],
        PROJECT_ROOT / data_config["graph_cache"],
    )
    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = ReactionEncoder(checkpoint["model_config"], checkpoint["vocab_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    molecule_rows = []
    for start in range(0, len(corpus.graphs), args.molecule_batch_size):
        graph_batch = collate_molecule_graphs(corpus.graphs[start:start + args.molecule_batch_size])
        molecule_rows.append(model.molecule_projection(model.molecule_encoder(graph_batch)).cpu())
    molecule_matrix = torch.cat(molecule_rows).numpy().astype(np.float32, copy=False)
    molecule_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(molecule_matrix):
        molecule_groups[row.tobytes()].append(index)
    duplicate_molecule_groups = [indices for indices in molecule_groups.values() if len(indices) > 1]
    duplicate_molecule_groups.sort(key=lambda values: (-len(values), corpus.smiles[values[0]]))
    molecule_keys = [digest_bytes(row.tobytes()) for row in molecule_matrix]

    matrix = np.load(PROJECT_ROOT / args.embedding, allow_pickle=False)
    with (PROJECT_ROOT / args.ids).open("r", encoding="utf-8", newline="") as handle:
        reaction_ids = [row["reaction_id"] for row in csv.DictReader(handle, delimiter="\t")]
    example_by_id = {example.reaction_id: example for example in corpus.examples}
    reaction_vector_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(matrix):
        reaction_vector_groups[row.tobytes()].append(index)
    duplicate_reaction_groups = [indices for indices in reaction_vector_groups.values() if len(indices) > 1]

    encoded_explained = 0
    encoded_partially_collided = 0
    for indices in duplicate_reaction_groups:
        examples = [example_by_id[reaction_ids[index]] for index in indices]
        signatures = {encoded_input_signature(example, molecule_keys) for example in examples}
        if len(signatures) == 1:
            encoded_explained += 1
        distinct_molecules = {molecule for example in examples for molecule in example.molecule_indices}
        if len({molecule_keys[index] for index in distinct_molecules}) < len(distinct_molecules):
            encoded_partially_collided += 1

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    group_path = output_dir / "duplicate_molecule_encoder_groups.csv"
    with group_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["group_id", "group_size", "smiles"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for group_id, indices in enumerate(duplicate_molecule_groups, 1):
            for index in indices:
                writer.writerow({"group_id": group_id, "group_size": len(indices), "smiles": corpus.smiles[index]})

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "molecule_count": len(corpus.smiles),
        "exact_unique_molecule_embeddings": len(molecule_groups),
        "duplicate_molecule_embedding_excess_count": len(corpus.smiles) - len(molecule_groups),
        "duplicate_molecule_embedding_group_count": len(duplicate_molecule_groups),
        "molecules_in_duplicate_embedding_groups": sum(map(len, duplicate_molecule_groups)),
        "molecule_duplicate_group_size_histogram": dict(sorted(Counter(map(len, duplicate_molecule_groups)).items())),
        "duplicate_reaction_embedding_group_count": len(duplicate_reaction_groups),
        "reaction_groups_fully_explained_by_identical_encoded_inputs": encoded_explained,
        "reaction_groups_containing_molecular_encoder_collisions": encoded_partially_collided,
        "diagnosis": (
            "Exact molecular-encoder collisions indicate finite-radius message passing plus attention/mean-like readout "
            "is unable to distinguish some structurally different molecules, especially homologous or repetitive structures."
        ),
    }
    (output_dir / "encoder_collision_trace.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
