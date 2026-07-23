from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def model_canonical_smiles(smiles: str) -> str:
    """Canonicalize exactly as RDKit graph parsing does, including implicit removal of ordinary explicit H."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return smiles
        mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def orientation_participants(participants: list[dict[str, Any]], direction: str) -> list[tuple[Any, ...]]:
    swap = direction == "right_to_left"
    rows = []
    for participant in participants:
        side = participant["side"]
        coefficient = float(participant["coefficient"])
        if swap:
            side = "product" if side == "reactant" else "reactant"
            coefficient = -coefficient
        rows.append((
            side,
            round(coefficient, 8),
            model_canonical_smiles(participant["canonical_smiles"]),
            participant.get("cofactor_role"),
        ))
    return sorted(rows)


def direction_policy(supported: list[str]) -> str:
    values = set(supported)
    if "reversible" in values or {"left_to_right", "right_to_left"}.issubset(values):
        return "reversible"
    if values == {"left_to_right"}:
        return "left_to_right"
    if values == {"right_to_left"}:
        return "right_to_left"
    return "undefined"


def signatures(record: dict[str, Any]) -> tuple[str, str, str]:
    direction = direction_policy(record.get("supported_directions") or [])
    oriented = orientation_participants(record["participants"], direction)
    if direction == "reversible":
        reverse = sorted((
            "product" if side == "reactant" else "reactant",
            round(-coefficient, 8),
            smiles,
            cofactor,
        ) for side, coefficient, smiles, cofactor in oriented)
        structure_core = min(oriented, reverse)
    else:
        structure_core = oriented
    structure_signature = stable_hash({"participants": structure_core, "direction": direction})
    main_v2_signature = stable_hash({
        "participants": structure_core,
        "direction": direction,
        "cofactor_missing": [row[3] is None for row in structure_core],
    })
    legacy_signature = stable_hash({
        "participants": sorted((
            side,
            coefficient,
            smiles,
            cofactor,
            participant.get("role"),
            participant.get("compartment"),
        ) for (side, coefficient, smiles, cofactor), participant in zip(
            orientation_participants(record["participants"], direction),
            sorted(record["participants"], key=lambda p: (
                p["side"], round(float(p["coefficient"]), 8), p["canonical_smiles"]
            )),
        )),
        "direction": direction,
        "reaction_type": record.get("reaction_type"),
        "ec_numbers": sorted(record.get("ec_numbers") or []),
    })
    return structure_signature, main_v2_signature, legacy_signature


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace exact duplicate embeddings to reaction inputs")
    parser.add_argument("--embedding", default="artifacts/embeddings/reaction_embeddings.npy")
    parser.add_argument("--ids", default="artifacts/embeddings/reaction_ids.tsv")
    parser.add_argument("--reactions", default="data/processed/rhea_reactions.parquet")
    parser.add_argument("--output-dir", default="artifacts/reports/duplicate_embedding_diagnosis")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    matrix = np.load(root / args.embedding, allow_pickle=False)
    with (root / args.ids).open("r", encoding="utf-8", newline="") as handle:
        id_rows = list(csv.DictReader(handle, delimiter="\t"))
    reaction_ids = [row["reaction_id"] for row in id_rows]
    columns = ["reaction_id", "participants", "ec_numbers", "reaction_type", "supported_directions"]
    records = pq.read_table(root / args.reactions, columns=columns).to_pylist()
    record_by_id = {record["reaction_id"]: record for record in records}
    if matrix.shape[0] != len(reaction_ids):
        raise ValueError("Embedding and reaction ID counts differ")

    vector_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(matrix):
        vector_groups[row.tobytes()].append(index)
    duplicate_groups = [indices for indices in vector_groups.values() if len(indices) > 1]
    duplicate_groups.sort(key=lambda values: (-len(values), reaction_ids[values[0]]))

    all_smiles_to_compounds: dict[str, set[str]] = defaultdict(set)
    unsanitized_smiles: set[str] = set()
    wildcard_smiles: set[str] = set()
    for record in records:
        for participant in record["participants"]:
            smiles = participant["canonical_smiles"]
            all_smiles_to_compounds[smiles].add(participant["compound_id"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                unsanitized_smiles.add(smiles)
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None and any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                wildcard_smiles.add(smiles)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    member_path = output_dir / "duplicate_embedding_members.csv"
    group_path = output_dir / "duplicate_embedding_groups.csv"
    member_fields = [
        "group_id", "group_size", "reaction_id", "direction", "participant_count",
        "structure_signature", "main_v2_signature", "legacy_input_signature",
        "contains_unparsed_smiles", "contains_wildcard_atom", "contains_shared_smiles_identity",
        "reaction_type", "ec_numbers", "participant_summary",
    ]
    group_rows = []
    member_rows = []
    duplicate_reaction_indices: set[int] = set()
    for group_id, indices in enumerate(duplicate_groups, 1):
        duplicate_reaction_indices.update(indices)
        structure_signatures = set()
        main_v2_signatures = set()
        legacy_signatures = set()
        directions = set()
        any_unparsed = False
        any_wildcard = False
        any_shared_identity = False
        for index in indices:
            reaction_id = reaction_ids[index]
            record = record_by_id[reaction_id]
            structure_signature, main_v2_signature, legacy_signature = signatures(record)
            structure_signatures.add(structure_signature)
            main_v2_signatures.add(main_v2_signature)
            legacy_signatures.add(legacy_signature)
            direction = direction_policy(record.get("supported_directions") or [])
            directions.add(direction)
            smiles_values = [participant["canonical_smiles"] for participant in record["participants"]]
            contains_unparsed = any(smiles in unsanitized_smiles for smiles in smiles_values)
            contains_wildcard = any(smiles in wildcard_smiles for smiles in smiles_values)
            contains_shared_identity = any(len(all_smiles_to_compounds[smiles]) > 1 for smiles in smiles_values)
            any_unparsed |= contains_unparsed
            any_wildcard |= contains_wildcard
            any_shared_identity |= contains_shared_identity
            summary = ";".join(
                f"{p['side']}:{float(p['coefficient']):g}:{p['compound_id']}:{p['canonical_smiles']}"
                for p in record["participants"]
            )
            member_rows.append({
                "group_id": group_id,
                "group_size": len(indices),
                "reaction_id": reaction_id,
                "direction": direction,
                "participant_count": len(record["participants"]),
                "structure_signature": structure_signature,
                "main_v2_signature": main_v2_signature,
                "legacy_input_signature": legacy_signature,
                "contains_unparsed_smiles": contains_unparsed,
                "contains_wildcard_atom": contains_wildcard,
                "contains_shared_smiles_identity": contains_shared_identity,
                "reaction_type": record.get("reaction_type") or "",
                "ec_numbers": ";".join(record.get("ec_numbers") or []),
                "participant_summary": summary,
            })
        group_rows.append({
            "group_id": group_id,
            "group_size": len(indices),
            "same_structure_signature": len(structure_signatures) == 1,
            "same_main_v2_signature": len(main_v2_signatures) == 1,
            "same_legacy_input_signature": len(legacy_signatures) == 1,
            "direction_values": ";".join(sorted(directions)),
            "contains_unparsed_smiles": any_unparsed,
            "contains_wildcard_atom": any_wildcard,
            "contains_shared_smiles_identity": any_shared_identity,
        })

    with member_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=member_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(member_rows)
    with group_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(group_rows)

    group_size_histogram = Counter(len(indices) for indices in duplicate_groups)
    equivalent_excess = sum(
        int(row["group_size"]) - 1 for row in group_rows if row["same_main_v2_signature"]
    )
    unexplained_excess = sum(
        int(row["group_size"]) - 1 for row in group_rows if not row["same_main_v2_signature"]
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_rows": len(matrix),
        "exact_unique_vectors": len(vector_groups),
        "duplicate_vector_excess_count": len(matrix) - len(vector_groups),
        "duplicate_vector_group_count": len(duplicate_groups),
        "reactions_in_duplicate_groups": len(duplicate_reaction_indices),
        "duplicate_group_size_histogram": dict(sorted(group_size_histogram.items())),
        "duplicate_groups_same_structure_input": sum(row["same_structure_signature"] for row in group_rows),
        "duplicate_groups_same_main_v2_input": sum(row["same_main_v2_signature"] for row in group_rows),
        "duplicate_vector_excess_same_main_v2_input": equivalent_excess,
        "duplicate_vector_excess_different_main_v2_input": unexplained_excess,
        "duplicate_groups_same_legacy_input": sum(row["same_legacy_input_signature"] for row in group_rows),
        "duplicate_groups_with_unparsed_smiles": sum(row["contains_unparsed_smiles"] for row in group_rows),
        "duplicate_groups_with_wildcard_atoms": sum(row["contains_wildcard_atom"] for row in group_rows),
        "duplicate_groups_with_shared_smiles_identity": sum(row["contains_shared_smiles_identity"] for row in group_rows),
        "corpus_unparsed_smiles_count": len(unsanitized_smiles),
        "corpus_wildcard_smiles_count": len(wildcard_smiles),
        "smiles_shared_by_multiple_compound_ids": sum(len(ids) > 1 for ids in all_smiles_to_compounds.values()),
        "notes": {
            "unparsed_smiles": "RDKit sanitized parsing failed; these are not replaced by an empty graph in the current graph builder.",
            "shared_smiles_identity": "Different compound IDs have the same canonical structure; this is not automatically an error.",
            "legacy_input_signature": "Includes structure, side, stoichiometry, direction, cofactor/role/compartment, reaction type, and EC.",
        },
    }
    summary_path = output_dir / "duplicate_embedding_diagnosis.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
