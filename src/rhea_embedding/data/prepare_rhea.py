from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions


PIPELINE_NAME = "rhea_phase1_prepare"
DIRECTION_ENUM = {"LR": "left_to_right", "RL": "right_to_left", "BI": "reversible", "UN": "undefined"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_tsv_from_tar(archive: Path, basename: str) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == basename), None)
        if member is None:
            raise FileNotFoundError(f"{basename} not found in {archive}")
        stream = tar.extractfile(member)
        if stream is None:
            raise OSError(f"Cannot read {member.name} from {archive}")
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        return list(csv.DictReader(text, delimiter="\t"))


def read_headerless_tsv_from_tar(archive: Path, basename: str, fieldnames: list[str]) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if Path(m.name).name == basename), None)
        if member is None:
            raise FileNotFoundError(f"{basename} not found in {archive}")
        stream = tar.extractfile(member)
        if stream is None:
            raise OSError(f"Cannot read {member.name} from {archive}")
        text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
        return list(csv.DictReader(text, fieldnames=fieldnames, delimiter="\t"))


def parse_direction_rows(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for row in rows:
        master = int(row["RHEA_ID_MASTER"])
        result[master] = {
            "master": master,
            "left_to_right": int(row["RHEA_ID_LR"]),
            "right_to_left": int(row["RHEA_ID_RL"]),
            "reversible": int(row["RHEA_ID_BI"]),
        }
    return result


def index_values(rows: Iterable[dict[str, str]], value_column: str) -> dict[int, list[str]]:
    values: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        values[int(row["MASTER_ID"])].add(row[value_column])
    return {key: sorted(items) for key, items in values.items()}


def index_xrefs(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, list[str]]]:
    result: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        result[int(row["MASTER_ID"])][row["DB"]].add(row["ID"])
    return {master: {db: sorted(ids) for db, ids in sorted(dbs.items())} for master, dbs in result.items()}


def direction_evidence_from_rows(
    rows: Iterable[dict[str, str]], database_column: str | None = None, fixed_database: str | None = None
) -> dict[int, dict[str, dict[str, set[str]]]]:
    evidence: dict[int, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    for row in rows:
        code = row["DIRECTION"]
        if code not in DIRECTION_ENUM:
            continue
        database = fixed_database or (row[database_column] if database_column else "unknown")
        evidence[int(row["MASTER_ID"])][DIRECTION_ENUM[code]][database].add(row["ID"])
    return evidence


def index_reaction_smiles(rows: Iterable[dict[str, str]]) -> dict[int, str]:
    result = {}
    for row in rows:
        first = row.get("RHEA_ID") or next(iter(row.values()))
        second = row.get("REACTION_SMILES") or list(row.values())[1]
        result[int(first)] = second
    return result


def split_source_smiles(reaction_smiles: str | None) -> tuple[list[str], list[str]]:
    if not reaction_smiles or ">>" not in reaction_smiles:
        return [], []
    left, right = reaction_smiles.split(">>", 1)
    return (left.split(".") if left else [], right.split(".") if right else [])


@dataclass
class ParsedMol:
    compound_id: str
    original_smiles: str | None
    canonical_smiles: str | None
    atom_counts: Counter[str] | None
    formal_charge: int | None
    unknown_atoms: int
    error: str | None
    sanitization_status: str


def canonicalize_mol(mol: Chem.Mol, source_smiles: str | None) -> ParsedMol:
    compound_id = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "UNKNOWN"
    candidate = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(candidate)
        canonical = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
        counts: Counter[str] = Counter()
        unknown = 0
        charge = 0
        for atom in candidate.GetAtoms():
            symbol = atom.GetSymbol()
            if atom.GetAtomicNum() == 0:
                unknown += 1
                symbol = "*"
            counts[symbol] += 1
            # Explicit hydrogen atoms are counted as atoms above; excluding
            # neighbours avoids counting the same hydrogen a second time.
            counts["H"] += atom.GetTotalNumHs(includeNeighbors=False)
            charge += atom.GetFormalCharge()
        if counts.get("H") == 0:
            counts.pop("H", None)
        return ParsedMol(compound_id, source_smiles, canonical, counts, charge, unknown, None, "sanitized")
    except Exception as exc:  # RDKit raises several exception types across versions
        # Rhea's source SMILES are independently generated by RDKit. They are a
        # reliable same-release fallback for the rare CT-file valence issue.
        fallback = Chem.MolFromSmiles(source_smiles) if source_smiles else None
        if fallback is not None:
            try:
                canonical = Chem.MolToSmiles(fallback, canonical=True, isomericSmiles=True)
                counts: Counter[str] = Counter()
                charge = 0
                unknown = 0
                for atom in fallback.GetAtoms():
                    symbol = atom.GetSymbol()
                    if atom.GetAtomicNum() == 0:
                        unknown += 1
                        symbol = "*"
                    counts[symbol] += 1
                    counts["H"] += atom.GetTotalNumHs(includeNeighbors=False)
                    charge += atom.GetFormalCharge()
                if counts.get("H") == 0:
                    counts.pop("H", None)
                return ParsedMol(compound_id, source_smiles, canonical, counts, charge, unknown, f"rxn_fallback: {type(exc).__name__}: {exc}", "sanitized")
            except Exception:
                pass
        unsanitized = Chem.MolFromSmiles(source_smiles, sanitize=False) if source_smiles else None
        if unsanitized is not None:
            try:
                canonical = Chem.MolToSmiles(unsanitized, canonical=True, isomericSmiles=True)
                unknown = sum(atom.GetAtomicNum() == 0 for atom in unsanitized.GetAtoms())
                return ParsedMol(
                    compound_id, source_smiles, canonical, None, None, unknown,
                    f"unsanitized_fallback: {type(exc).__name__}: {exc}", "unsanitized"
                )
            except Exception:
                pass
        return ParsedMol(compound_id, source_smiles, None, None, None, 0, f"{type(exc).__name__}: {exc}", "failed")


def aggregate_side(
    mols: list[Chem.Mol],
    source_smiles: list[str],
    side: str,
    reaction_id: str,
    transformations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[ParsedMol]]:
    parsed: list[ParsedMol] = []
    for index, mol in enumerate(mols):
        source = source_smiles[index] if index < len(source_smiles) else None
        item = canonicalize_mol(mol, source)
        parsed.append(item)
        if item.error and item.error.startswith("rxn_fallback:"):
            transformations.append({
                "reaction_id": reaction_id,
                "compound_id": item.compound_id,
                "field": "smiles_source",
                "original_value": "RXN mol block",
                "processed_value": item.original_smiles,
                "transformation_name": "fallback_to_same_release_rhea_smiles",
                "software_version": f"RDKit {rdBase.rdkitVersion}",
                "confidence": "high",
            })
        if item.sanitization_status == "unsanitized":
            transformations.append({
                "reaction_id": reaction_id,
                "compound_id": item.compound_id,
                "field": "smiles",
                "original_value": item.original_smiles,
                "processed_value": item.canonical_smiles,
                "transformation_name": "canonicalize_without_sanitization",
                "software_version": f"RDKit {rdBase.rdkitVersion}",
                "confidence": "low",
            })
        if item.original_smiles != item.canonical_smiles:
            transformations.append({
                "reaction_id": reaction_id,
                "compound_id": item.compound_id,
                "field": "smiles",
                "original_value": item.original_smiles,
                "processed_value": item.canonical_smiles,
                "transformation_name": "rdkit_canonical_isomeric_smiles",
                "software_version": f"RDKit {rdBase.rdkitVersion}",
                "confidence": "high" if item.canonical_smiles else "low",
            })

    grouped: dict[tuple[str, str | None], list[ParsedMol]] = defaultdict(list)
    for item in parsed:
        grouped[(item.compound_id, item.canonical_smiles)].append(item)

    participants = []
    sign = -1.0 if side == "reactant" else 1.0
    for (compound_id, canonical), members in sorted(grouped.items()):
        coefficient = sign * len(members)
        participant = {
            "compound_id": compound_id,
            "original_smiles": members[0].original_smiles,
            "canonical_smiles": canonical,
            "smiles_sanitization_status": (
                "failed" if canonical is None else
                "unsanitized" if any(m.sanitization_status == "unsanitized" for m in members) else
                "sanitized"
            ),
            "smiles_missing": canonical is None,
            "mapped_smiles": None,
            "original_coefficient": coefficient,
            "coefficient": coefficient,
            "side": side,
            "compartment": None,
            "role": None,
            "cofactor_role": None,
            "cofactor_missing": True,
            "compartment_missing": True,
            "role_missing": True,
        }
        participants.append(participant)
        if len(members) > 1:
            transformations.append({
                "reaction_id": reaction_id,
                "compound_id": compound_id,
                "field": "coefficient",
                "original_value": [sign] * len(members),
                "processed_value": coefficient,
                "transformation_name": "aggregate_duplicate_participants",
                "software_version": PIPELINE_NAME,
                "confidence": "exact",
            })
    return participants, parsed


def balance_status(parsed_reactants: list[ParsedMol], parsed_products: list[ParsedMol]) -> tuple[bool | None, str, dict[str, Any]]:
    all_mols = parsed_reactants + parsed_products
    if any(
        item.compound_id.startswith(("POLYMER:", "RHEA-COMP:")) or item.unknown_atoms > 0
        for item in all_mols
    ):
        return None, "unknown", {"reason": "generic_or_polymeric_participant"}
    if any(item.atom_counts is None or item.formal_charge is None for item in all_mols):
        return None, "unknown", {"reason": "smiles_parse_failure"}

    left_atoms: Counter[str] = Counter()
    right_atoms: Counter[str] = Counter()
    left_charge = 0
    right_charge = 0
    for item in parsed_reactants:
        left_atoms.update(item.atom_counts or {})
        left_charge += item.formal_charge or 0
    for item in parsed_products:
        right_atoms.update(item.atom_counts or {})
        right_charge += item.formal_charge or 0

    delta = {element: right_atoms[element] - left_atoms[element] for element in sorted(set(left_atoms) | set(right_atoms))}
    delta = {element: value for element, value in delta.items() if value}
    charge_delta = right_charge - left_charge
    unknown_atoms = sum(item.unknown_atoms for item in all_mols)
    balanced = not delta and charge_delta == 0
    return balanced, "balanced" if balanced else "imbalanced", {
        "element_delta_json": json.dumps(delta, sort_keys=True),
        "charge_delta": charge_delta,
        "unknown_atom_count": unknown_atoms,
    }


def reduced_coefficients(participants: list[dict[str, Any]]) -> list[tuple[str, int, str | None]]:
    integers = [int(abs(p["coefficient"])) for p in participants]
    divisor = reduce(gcd, integers) if integers else 1
    divisor = max(divisor, 1)
    return sorted((p["compound_id"], int(p["coefficient"]) // divisor, p["compartment"]) for p in participants)


def build_keys(record: dict[str, Any], ignored_loose: set[str]) -> tuple[str, str]:
    strict_payload = {"participants": reduced_coefficients(record["participants"]), "direction": record["direction"]}
    loose_participants = [p for p in record["participants"] if p["compound_id"] not in ignored_loose]
    loose_payload = {"participants": [(cid, coef) for cid, coef, _ in reduced_coefficients(loose_participants)]}
    return stable_hash(strict_payload), stable_hash(loose_payload)


def parse_release_from_rxn(block: str) -> int | None:
    match = re.search(r"RHEA:release=(\d+)", block)
    return int(match.group(1)) if match else None


def build_records(config: dict[str, Any], project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    paths = {key: project_root / value for key, value in config["paths"].items()}
    tsv_archive = paths["tsv_archive"]
    rxn_archive = paths["rxn_archive"]

    directions = parse_direction_rows(read_tsv_from_tar(tsv_archive, "rhea-directions.tsv"))
    ec_by_master = index_values(read_tsv_from_tar(tsv_archive, "rhea2ec.tsv"), "ID")
    xrefs_by_master = index_xrefs(read_tsv_from_tar(tsv_archive, "rhea2xrefs.tsv"))
    source_smiles = index_reaction_smiles(
        read_headerless_tsv_from_tar(tsv_archive, "rhea-reaction-smiles.tsv", ["RHEA_ID", "REACTION_SMILES"])
    )

    transformations: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    observed_releases: set[int] = set()

    with tarfile.open(rxn_archive, "r:gz") as tar:
        members = {Path(member.name).name: member for member in tar.getmembers() if member.isfile()}
        for master, variants in sorted(directions.items()):
            lr_id = variants["left_to_right"]
            member = members.get(f"{lr_id}.rxn")
            reaction_id = f"RHEA:{master}"
            if member is None:
                failures.append({"reaction_id": reaction_id, "stage": "rxn_lookup", "error": f"Missing {lr_id}.rxn"})
                continue
            stream = tar.extractfile(member)
            if stream is None:
                failures.append({"reaction_id": reaction_id, "stage": "rxn_read", "error": f"Unreadable {member.name}"})
                continue
            block = stream.read().decode("utf-8", errors="replace")
            release = parse_release_from_rxn(block)
            if release is not None:
                observed_releases.add(release)
            try:
                reaction = rdChemReactions.ReactionFromRxnBlock(block, sanitize=False, removeHs=False)
                if reaction is None:
                    raise ValueError("RDKit returned no reaction")
            except Exception as exc:
                failures.append({"reaction_id": reaction_id, "stage": "rxn_parse", "error": f"{type(exc).__name__}: {exc}"})
                continue

            original_reaction_smiles = source_smiles.get(lr_id)
            left_source, right_source = split_source_smiles(original_reaction_smiles)
            reactants, parsed_reactants = aggregate_side(
                list(reaction.GetReactants()), left_source, "reactant", reaction_id, transformations
            )
            products, parsed_products = aggregate_side(
                list(reaction.GetProducts()), right_source, "product", reaction_id, transformations
            )
            participants = reactants + products
            is_balanced, balance_check, balance_details = balance_status(parsed_reactants, parsed_products)
            left = ".".join(p["canonical_smiles"] or "" for p in reactants)
            right = ".".join(p["canonical_smiles"] or "" for p in products)
            ecs = ec_by_master.get(master)
            variant_strings = {key: f"RHEA:{value}" for key, value in variants.items()}
            record = {
                "reaction_id": reaction_id,
                "canonical_reaction_id": reaction_id,
                "equivalence_group_id": f"RHEA_EQ:{master}",
                "source_ids": {
                    "rhea": sorted(set(variant_strings.values())),
                    "metanetx": [],
                    "modelseed": [],
                    "bigg": [],
                },
                "equivalent_rhea_master_ids": [reaction_id],
                "direction_variant_ids": variant_strings,
                "participants": participants,
                "reaction_smiles": f"{left}>>{right}",
                "source_reaction_smiles": original_reaction_smiles,
                "atom_mapped_reaction_smiles": None,
                "direction": "undefined",
                "supported_directions": [],
                "direction_missing": True,
                "direction_evidence_summary_json": "{}",
                "reaction_type": None,
                "ec_numbers": ecs,
                "is_balanced": is_balanced,
                "balance_check_status": balance_check,
                "balance_element_delta_json": balance_details.get("element_delta_json"),
                "balance_charge_delta": balance_details.get("charge_delta"),
                "balance_unknown_atom_count": balance_details.get("unknown_atom_count", 0),
                "context": None,
                "external_xrefs_json": json.dumps(xrefs_by_master.get(master, {}), ensure_ascii=False, sort_keys=True),
                "missing_masks": {
                    "reaction_type": True,
                    "cofactor": True,
                    "ec_numbers": not bool(ecs),
                    "atom_map": True,
                    "compartment": True,
                    "context": True,
                    "flux": True,
                    "direction": True,
                },
                "provenance": [{
                    "source": "Rhea",
                    "source_release": str(config["source"]["release"]),
                    "source_reaction_id": reaction_id,
                    "rxn_direction_used": "left_to_right",
                    "match_confidence": "exact",
                }],
            }
            strict_key, loose_key = build_keys(record, set(config["processing"]["ignored_in_loose_key"]))
            record["canonical_key_sha256"] = strict_key
            record["loose_key_sha256"] = loose_key
            raw_records.append(record)

    expected_release = int(config["source"]["release"])
    if observed_releases != {expected_release}:
        raise ValueError(f"RXN release mismatch: expected {expected_release}, observed {sorted(observed_releases)}")

    strict_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        strict_groups[record["canonical_key_sha256"]].append(record)

    deduplicated: list[dict[str, Any]] = []
    id_map: list[dict[str, Any]] = []
    conflicting_strict_groups = 0
    for _, group in sorted(strict_groups.items(), key=lambda item: min(int(r["reaction_id"].split(":")[1]) for r in item[1])):
        group.sort(key=lambda r: int(r["reaction_id"].split(":")[1]))
        winner = deepcopy(group[0])
        if len(group) > 1:
            balance_states = {r["balance_check_status"] for r in group}
            if len(balance_states) > 1:
                conflicting_strict_groups += 1
                for record in group:
                    deduplicated.append(record)
                    id_map.append({"source_reaction_id": record["reaction_id"], "canonical_reaction_id": record["reaction_id"], "status": "kept_conflict"})
                continue
            winner["equivalent_rhea_master_ids"] = [r["reaction_id"] for r in group]
            winner["source_ids"]["rhea"] = sorted({item for r in group for item in r["source_ids"]["rhea"]})
            all_ec = sorted({ec for r in group for ec in (r["ec_numbers"] or [])})
            winner["ec_numbers"] = all_ec or None
            winner["missing_masks"]["ec_numbers"] = not bool(all_ec)
            merged_xrefs: dict[str, set[str]] = defaultdict(set)
            for record in group:
                for db, ids in json.loads(record["external_xrefs_json"]).items():
                    merged_xrefs[db].update(ids)
            winner["external_xrefs_json"] = json.dumps({db: sorted(ids) for db, ids in sorted(merged_xrefs.items())}, sort_keys=True)
            winner["provenance"] = [entry for r in group for entry in r["provenance"]]
        deduplicated.append(winner)
        for index, record in enumerate(group):
            id_map.append({
                "source_reaction_id": record["reaction_id"],
                "canonical_reaction_id": winner["reaction_id"],
                "status": "kept" if index == 0 else "merged_strict_exact",
            })

    loose_groups: dict[str, list[str]] = defaultdict(list)
    for record in deduplicated:
        loose_groups[record["loose_key_sha256"]].append(record["reaction_id"])
    loose_candidates = [
        {"loose_key_sha256": key, "reaction_ids": ids, "candidate_count": len(ids), "action": "report_only_not_merged"}
        for key, ids in sorted(loose_groups.items()) if len(ids) > 1
    ]

    stats = {
        "source_master_reaction_count": len(directions),
        "parsed_reaction_count": len(raw_records),
        "retained_reaction_count": len(deduplicated),
        "strict_duplicate_group_count": sum(1 for group in strict_groups.values() if len(group) > 1),
        "strict_group_ec_annotation_conflict_count": sum(
            len({tuple(r["ec_numbers"] or []) for r in group}) > 1
            for group in strict_groups.values() if len(group) > 1
        ),
        "strict_group_xref_annotation_conflict_count": sum(
            len({r["external_xrefs_json"] for r in group}) > 1
            for group in strict_groups.values() if len(group) > 1
        ),
        "deduplication_count": len(raw_records) - len(deduplicated),
        "conflicting_strict_group_count": conflicting_strict_groups,
        "loose_candidate_group_count": len(loose_candidates),
        "reaction_parse_failure_count": len(failures),
        "failures": failures,
    }
    return deduplicated, transformations, stats, id_map + [{"_loose_candidates": loose_candidates}]


def field_coverage(records: list[dict[str, Any]]) -> dict[str, float]:
    total = len(records)
    participant_total = sum(len(r["participants"]) for r in records)
    def fraction(count: int, denominator: int = total) -> float:
        return round(count / denominator, 6) if denominator else 0.0
    return {
        "reaction_smiles": fraction(sum(bool(r["reaction_smiles"]) for r in records)),
        "direction_defined": fraction(sum(r["direction"] != "undefined" for r in records)),
        "direction_evidence_available": fraction(sum(bool(r["supported_directions"]) for r in records)),
        "ec_numbers": fraction(sum(bool(r["ec_numbers"]) for r in records)),
        "reaction_type": fraction(sum(r["reaction_type"] is not None for r in records)),
        "atom_map": fraction(sum(r["atom_mapped_reaction_smiles"] is not None for r in records)),
        "context": fraction(sum(r["context"] is not None for r in records)),
        "participant_compound_id": fraction(sum(bool(p["compound_id"]) for r in records for p in r["participants"]), participant_total),
        "participant_canonical_smiles": fraction(sum(bool(p["canonical_smiles"]) for r in records for p in r["participants"]), participant_total),
        "participant_compartment": fraction(sum(p["compartment"] is not None for r in records for p in r["participants"]), participant_total),
        "participant_role": fraction(sum(p["role"] is not None for r in records for p in r["participants"]), participant_total),
        "participant_cofactor_role": fraction(sum(p["cofactor_role"] is not None for r in records for p in r["participants"]), participant_total),
    }


def ensure_parent(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the preferred Parquet output; install requirements.txt") from exc

    participant_type = pa.struct([
        ("compound_id", pa.string()), ("original_smiles", pa.string()), ("canonical_smiles", pa.string()),
        ("smiles_sanitization_status", pa.string()), ("smiles_missing", pa.bool_()),
        ("mapped_smiles", pa.string()), ("original_coefficient", pa.float64()), ("coefficient", pa.float64()),
        ("side", pa.string()), ("compartment", pa.string()), ("role", pa.string()), ("cofactor_role", pa.string()),
        ("cofactor_missing", pa.bool_()), ("compartment_missing", pa.bool_()), ("role_missing", pa.bool_()),
    ])
    schema = pa.schema([
        ("reaction_id", pa.string()), ("canonical_reaction_id", pa.string()), ("equivalence_group_id", pa.string()),
        ("source_ids", pa.struct([("rhea", pa.list_(pa.string())), ("metanetx", pa.list_(pa.string())), ("modelseed", pa.list_(pa.string())), ("bigg", pa.list_(pa.string()))])),
        ("equivalent_rhea_master_ids", pa.list_(pa.string())),
        ("direction_variant_ids", pa.struct([("master", pa.string()), ("left_to_right", pa.string()), ("right_to_left", pa.string()), ("reversible", pa.string())])),
        ("participants", pa.list_(participant_type)), ("reaction_smiles", pa.string()), ("source_reaction_smiles", pa.string()),
        ("atom_mapped_reaction_smiles", pa.string()), ("direction", pa.string()), ("reaction_type", pa.string()),
        ("supported_directions", pa.list_(pa.string())), ("direction_missing", pa.bool_()),
        ("direction_evidence_summary_json", pa.string()),
        ("ec_numbers", pa.list_(pa.string())), ("is_balanced", pa.bool_()), ("balance_check_status", pa.string()),
        ("balance_element_delta_json", pa.string()), ("balance_charge_delta", pa.int64()), ("balance_unknown_atom_count", pa.int64()),
        ("context", pa.string()), ("external_xrefs_json", pa.string()),
        ("missing_masks", pa.struct([("reaction_type", pa.bool_()), ("cofactor", pa.bool_()), ("ec_numbers", pa.bool_()), ("atom_map", pa.bool_()), ("compartment", pa.bool_()), ("context", pa.bool_()), ("flux", pa.bool_()), ("direction", pa.bool_())])),
        ("provenance", pa.list_(pa.struct([("source", pa.string()), ("source_release", pa.string()), ("source_reaction_id", pa.string()), ("rxn_direction_used", pa.string()), ("match_confidence", pa.string())]))),
        ("canonical_key_sha256", pa.string()), ("loose_key_sha256", pa.string()),
    ])
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def build_source_records(
    config: dict[str, Any], project_root: Path, id_map: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tsv_archive = project_root / config["paths"]["tsv_archive"]
    directions = parse_direction_rows(read_tsv_from_tar(tsv_archive, "rhea-directions.tsv"))
    ec_by_master = index_values(read_tsv_from_tar(tsv_archive, "rhea2ec.tsv"), "ID")
    xrefs_by_master = index_xrefs(read_tsv_from_tar(tsv_archive, "rhea2xrefs.tsv"))
    map_by_source = {row["source_reaction_id"]: row for row in id_map}
    failure_by_source = {row["reaction_id"]: row for row in failures}
    rows = []
    for master, variants in sorted(directions.items()):
        source_id = f"RHEA:{master}"
        mapping = map_by_source.get(source_id)
        failure = failure_by_source.get(source_id)
        rows.append({
            "source_reaction_id": source_id,
            "canonical_reaction_id": mapping["canonical_reaction_id"] if mapping else None,
            "embedding_included": mapping is not None,
            "deduplication_status": mapping["status"] if mapping else "excluded_missing_rxn",
            "parse_status": "parsed" if mapping else (failure["stage"] if failure else "not_parsed"),
            "parse_error": failure["error"] if failure else None,
            "direction_variant_ids": {
                "master": f"RHEA:{variants['master']}",
                "left_to_right": f"RHEA:{variants['left_to_right']}",
                "right_to_left": f"RHEA:{variants['right_to_left']}",
                "reversible": f"RHEA:{variants['reversible']}",
            },
            "ec_numbers": ec_by_master.get(master),
            "external_xrefs_json": json.dumps(xrefs_by_master.get(master, {}), ensure_ascii=False, sort_keys=True),
        })
    return rows


def write_source_records_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("source_reaction_id", pa.string()), ("canonical_reaction_id", pa.string()),
        ("embedding_included", pa.bool_()), ("deduplication_status", pa.string()),
        ("parse_status", pa.string()), ("parse_error", pa.string()),
        ("direction_variant_ids", pa.struct([
            ("master", pa.string()), ("left_to_right", pa.string()),
            ("right_to_left", pa.string()), ("reversible", pa.string()),
        ])),
        ("ec_numbers", pa.list_(pa.string())), ("external_xrefs_json", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd", use_dictionary=True)


def build_direction_variants(
    config: dict[str, Any], project_root: Path, canonical_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tsv_archive = project_root / config["paths"]["tsv_archive"]
    directions = parse_direction_rows(read_tsv_from_tar(tsv_archive, "rhea-directions.tsv"))
    external_evidence = direction_evidence_from_rows(
        read_tsv_from_tar(tsv_archive, "rhea2xrefs.tsv"), database_column="DB"
    )
    uniprot_evidence = direction_evidence_from_rows(
        read_tsv_from_tar(tsv_archive, "rhea2uniprot_sprot.tsv"), fixed_database="UniProtKB_Swiss-Prot"
    )

    variants = []
    supported_reaction_count = 0
    support_counts = Counter()
    for record in canonical_records:
        masters = [int(value.split(":")[1]) for value in record["equivalent_rhea_master_ids"]]
        aggregate: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for master in masters:
            for source in (external_evidence, uniprot_evidence):
                for direction, databases in source.get(master, {}).items():
                    for database, identifiers in databases.items():
                        aggregate[direction][database].update(identifiers)

        summary = {
            direction: {database: len(identifiers) for database, identifiers in sorted(databases.items())}
            for direction, databases in sorted(aggregate.items())
        }
        supported = [
            direction for direction in ("left_to_right", "right_to_left", "reversible")
            if sum(summary.get(direction, {}).values()) > 0
        ]
        record["supported_directions"] = supported
        record["direction_missing"] = not bool(supported)
        record["direction_evidence_summary_json"] = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        record["missing_masks"]["direction"] = not bool(supported)
        if supported:
            supported_reaction_count += 1

        undefined_count = sum(summary.get("undefined", {}).values())
        for direction in ("left_to_right", "right_to_left", "reversible"):
            database_counts = summary.get(direction, {})
            evidence_count = sum(database_counts.values())
            support_counts[direction] += int(evidence_count > 0)
            rhea_ids = [
                f"RHEA:{directions[master][direction]}" for master in masters
            ]
            variants.append({
                "direction_variant_id": f"{record['canonical_reaction_id']}|{direction}",
                "canonical_reaction_id": record["canonical_reaction_id"],
                "direction": direction,
                "rhea_directional_ids": sorted(rhea_ids),
                "swap_sides": direction == "right_to_left",
                "reversible_symmetrization": direction == "reversible",
                "directional_evidence_count": evidence_count,
                "has_directional_evidence": evidence_count > 0,
                "evidence_by_database_json": json.dumps(database_counts, ensure_ascii=False, sort_keys=True),
                "undefined_evidence_count": undefined_count,
                "semantics": (
                    "net_flux_left_to_right" if direction == "left_to_right" else
                    "net_flux_right_to_left" if direction == "right_to_left" else
                    "equilibrium_bidirectional"
                ),
            })
    stats = {
        "direction_variant_count": len(variants),
        "canonical_reactions_with_directional_evidence": supported_reaction_count,
        "direction_variant_evidence_coverage": {
            direction: support_counts[direction] for direction in ("left_to_right", "right_to_left", "reversible")
        },
    }
    return variants, stats


def write_direction_variants(parquet_path: Path, tsv_path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("direction_variant_id", pa.string()), ("canonical_reaction_id", pa.string()),
        ("direction", pa.string()), ("rhea_directional_ids", pa.list_(pa.string())),
        ("swap_sides", pa.bool_()), ("reversible_symmetrization", pa.bool_()),
        ("directional_evidence_count", pa.int64()), ("has_directional_evidence", pa.bool_()),
        ("evidence_by_database_json", pa.string()), ("undefined_evidence_count", pa.int64()),
        ("semantics", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(records, schema=schema), parquet_path, compression="zstd", use_dictionary=True)
    fieldnames = [field.name for field in schema]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["rhea_directional_ids"] = ";".join(row["rhea_directional_ids"])
            writer.writerow(row)


def git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = {key: project_root / value for key, value in config["paths"].items()}
    for required in (paths["rxn_archive"], paths["tsv_archive"]):
        if not required.exists():
            raise FileNotFoundError(f"Missing raw input: {required}")

    records, transformations, stats, mapping_bundle = build_records(config, project_root)
    loose_candidates = mapping_bundle[-1]["_loose_candidates"]
    id_map = mapping_bundle[:-1]
    ensure_parent([
        paths["processed_parquet"], paths["processed_jsonl"], paths["id_map"], paths["transformation_log"],
        paths["source_records_parquet"], paths["source_records_jsonl"],
        paths["direction_variants_parquet"], paths["direction_variants_tsv"],
        paths["loose_candidates"], paths["report"], paths["manifest"],
    ])
    direction_variants, direction_stats = build_direction_variants(config, project_root, records)
    write_jsonl(paths["processed_jsonl"], records)
    write_parquet(paths["processed_parquet"], records)
    write_direction_variants(paths["direction_variants_parquet"], paths["direction_variants_tsv"], direction_variants)
    write_jsonl(paths["transformation_log"], transformations)
    source_records = build_source_records(config, project_root, id_map, stats["failures"])
    write_jsonl(paths["source_records_jsonl"], source_records)
    write_source_records_parquet(paths["source_records_parquet"], source_records)

    mapped_source_ids = {row["source_reaction_id"] for row in id_map}
    for row in source_records:
        if row["source_reaction_id"] not in mapped_source_ids:
            id_map.append({
                "source_reaction_id": row["source_reaction_id"],
                "canonical_reaction_id": "",
                "status": row["deduplication_status"],
            })
    id_map.sort(key=lambda row: int(row["source_reaction_id"].split(":")[1]))

    with paths["id_map"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_reaction_id", "canonical_reaction_id", "status"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(id_map)
    with paths["loose_candidates"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loose_key_sha256", "reaction_ids", "candidate_count", "action"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in loose_candidates:
            writer.writerow({**row, "reaction_ids": ";".join(row["reaction_ids"])})

    failed_participants = sum(p["canonical_smiles"] is None for r in records for p in r["participants"])
    unsanitized_participants = sum(
        p["smiles_sanitization_status"] == "unsanitized" for r in records for p in r["participants"]
    )
    rxn_smiles_fallback_count = sum(t["transformation_name"] == "fallback_to_same_release_rhea_smiles" for t in transformations)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": PIPELINE_NAME,
        "project_version": config["project_version"],
        "source": config["source"],
        "reaction_count": len(records),
        "field_coverage": field_coverage(records),
        "failed_smiles_count": failed_participants,
        "unsanitized_smiles_count": unsanitized_participants,
        "source_record_count": len(source_records),
        "rxn_smiles_fallback_count": rxn_smiles_fallback_count,
        "imbalanced_reaction_count": sum(r["balance_check_status"] == "imbalanced" for r in records),
        "unknown_balance_count": sum(r["balance_check_status"] == "unknown" for r in records),
        **direction_stats,
        **stats,
        "output_checksums": {},
    }

    output_paths = [
        paths["processed_parquet"], paths["processed_jsonl"], paths["id_map"],
        paths["source_records_parquet"], paths["source_records_jsonl"],
        paths["direction_variants_parquet"], paths["direction_variants_tsv"],
        paths["transformation_log"], paths["loose_candidates"],
    ]
    report["output_checksums"] = {str(path.relative_to(project_root)).replace("\\", "/"): sha256_file(path) for path in output_paths}
    with paths["report"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": report["generated_at_utc"],
        "pipeline": {"name": PIPELINE_NAME, "version": config["project_version"], "git_revision": git_revision(project_root)},
        "source": config["source"],
        "inputs": [
            {"path": str(paths["rxn_archive"].relative_to(project_root)).replace("\\", "/"), "url": config["source"]["rxn_url"], "size_bytes": paths["rxn_archive"].stat().st_size, "sha256": sha256_file(paths["rxn_archive"])},
            {"path": str(paths["tsv_archive"].relative_to(project_root)).replace("\\", "/"), "url": config["source"]["tsv_url"], "size_bytes": paths["tsv_archive"].stat().st_size, "sha256": sha256_file(paths["tsv_archive"])},
        ],
        "outputs": [
            {"path": name, "sha256": checksum} for name, checksum in sorted(report["output_checksums"].items())
        ] + [{"path": str(paths["report"].relative_to(project_root)).replace("\\", "/"), "sha256": sha256_file(paths["report"])}],
        "record_count": len(records),
        "deduplication_policy": {
            "strict": "auto-merge exact compound ID + reduced signed stoichiometry + compartment + direction",
            "loose": "report only; never auto-merged",
        },
        "software": {"python": __import__("sys").version.split()[0], "rdkit": rdBase.rdkitVersion},
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "reaction_count": report["reaction_count"],
        "failed_smiles_count": report["failed_smiles_count"],
        "imbalanced_reaction_count": report["imbalanced_reaction_count"],
        "deduplication_count": report["deduplication_count"],
        "loose_candidate_group_count": report["loose_candidate_group_count"],
        "report": str(paths["report"]),
    }, indent=2))
    return report


def main(project_root: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Rhea release data for reaction embedding")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = project_root or Path.cwd()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)


if __name__ == "__main__":
    main()
