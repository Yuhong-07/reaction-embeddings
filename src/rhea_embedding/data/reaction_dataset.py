from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from rhea_embedding.chemistry.graph import (
    GRAPH_FEATURE_VERSION,
    MoleculeGraph,
    MoleculeGraphBatch,
    collate_molecule_graphs,
    smiles_to_graph,
)


DIRECTION_TO_INDEX = {
    "left_to_right": 0,
    "right_to_left": 1,
    "reversible": 2,
    "undefined": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction_policy(supported: list[str]) -> str:
    supported_set = set(supported)
    if "reversible" in supported_set or {"left_to_right", "right_to_left"}.issubset(supported_set):
        return "reversible"
    if supported_set == {"left_to_right"}:
        return "left_to_right"
    if supported_set == {"right_to_left"}:
        return "right_to_left"
    return "undefined"


@dataclass
class ReactionExample:
    reaction_id: str
    molecule_indices: list[int]
    coefficients: list[float]
    sides: list[int]
    role_indices: list[int]
    cofactor_indices: list[int]
    compartment_indices: list[int]
    ec_indices: list[int]
    reaction_type_index: int
    direction_index: int
    supported_directions: list[str]


@dataclass
class ReactionBatch:
    molecule_graphs: MoleculeGraphBatch
    participant_molecule_index: torch.Tensor
    participant_reaction_index: torch.Tensor
    participant_ptr: torch.Tensor
    side: torch.Tensor
    coefficient: torch.Tensor
    role_index: torch.Tensor
    cofactor_index: torch.Tensor
    compartment_index: torch.Tensor
    ec_index: torch.Tensor
    ec_reaction_index: torch.Tensor
    reaction_type_index: torch.Tensor
    direction_index: torch.Tensor
    reaction_ids: list[str]

    @property
    def reaction_count(self) -> int:
        return len(self.reaction_ids)

    def to(self, device: torch.device | str) -> "ReactionBatch":
        return ReactionBatch(
            molecule_graphs=self.molecule_graphs.to(device),
            participant_molecule_index=self.participant_molecule_index.to(device),
            participant_reaction_index=self.participant_reaction_index.to(device),
            participant_ptr=self.participant_ptr.to(device),
            side=self.side.to(device),
            coefficient=self.coefficient.to(device),
            role_index=self.role_index.to(device),
            cofactor_index=self.cofactor_index.to(device),
            compartment_index=self.compartment_index.to(device),
            ec_index=self.ec_index.to(device),
            ec_reaction_index=self.ec_reaction_index.to(device),
            reaction_type_index=self.reaction_type_index.to(device),
            direction_index=self.direction_index.to(device),
            reaction_ids=self.reaction_ids,
        )

    def swapped_orientation(self) -> "ReactionBatch":
        return replace(self, side=1 - self.side, coefficient=-self.coefficient)

    def permuted_participants(self) -> "ReactionBatch":
        order_parts = []
        for reaction_index in range(self.reaction_count):
            start = int(self.participant_ptr[reaction_index])
            end = int(self.participant_ptr[reaction_index + 1])
            order_parts.append(torch.arange(end - 1, start - 1, -1, device=self.side.device))
        order = torch.cat(order_parts)
        return replace(
            self,
            participant_molecule_index=self.participant_molecule_index[order],
            participant_reaction_index=self.participant_reaction_index[order],
            side=self.side[order],
            coefficient=self.coefficient[order],
            role_index=self.role_index[order],
            cofactor_index=self.cofactor_index[order],
            compartment_index=self.compartment_index[order],
        )


class ReactionCorpus(Dataset[ReactionExample]):
    def __init__(
        self,
        parquet_path: Path,
        graph_cache_path: Path | None = None,
        rebuild_graph_cache: bool = False,
    ) -> None:
        self.parquet_path = parquet_path
        columns = ["reaction_id", "participants", "ec_numbers", "reaction_type", "supported_directions"]
        records = pq.read_table(parquet_path, columns=columns).to_pylist()

        smiles_values = sorted({
            participant["canonical_smiles"]
            for record in records
            for participant in record["participants"]
        })
        self.smiles_to_index = {smiles: index for index, smiles in enumerate(smiles_values)}
        self.smiles = smiles_values

        role_values = sorted({
            participant["role"] for record in records for participant in record["participants"]
            if participant["role"] is not None
        })
        cofactor_values = sorted({
            participant["cofactor_role"] for record in records for participant in record["participants"]
            if participant["cofactor_role"] is not None
        })
        compartment_values = sorted({
            participant["compartment"] for record in records for participant in record["participants"]
            if participant["compartment"] is not None
        })
        ec_values = sorted({ec for record in records for ec in (record["ec_numbers"] or [])})
        reaction_type_values = sorted({record["reaction_type"] for record in records if record["reaction_type"] is not None})
        self.role_vocab = {value: index + 1 for index, value in enumerate(role_values)}
        self.cofactor_vocab = {value: index + 1 for index, value in enumerate(cofactor_values)}
        self.compartment_vocab = {value: index + 1 for index, value in enumerate(compartment_values)}
        self.ec_vocab = {value: index + 1 for index, value in enumerate(ec_values)}
        self.reaction_type_vocab = {value: index + 1 for index, value in enumerate(reaction_type_values)}

        self.examples = []
        for record in records:
            participants = record["participants"]
            policy = direction_policy(record["supported_directions"] or [])
            self.examples.append(ReactionExample(
                reaction_id=record["reaction_id"],
                molecule_indices=[self.smiles_to_index[p["canonical_smiles"]] for p in participants],
                coefficients=[float(p["coefficient"]) for p in participants],
                sides=[0 if p["side"] == "reactant" else 1 for p in participants],
                role_indices=[self.role_vocab.get(p["role"], 0) for p in participants],
                cofactor_indices=[self.cofactor_vocab.get(p["cofactor_role"], 0) for p in participants],
                compartment_indices=[self.compartment_vocab.get(p["compartment"], 0) for p in participants],
                ec_indices=[self.ec_vocab[ec] for ec in (record["ec_numbers"] or [])],
                reaction_type_index=self.reaction_type_vocab.get(record["reaction_type"], 0),
                direction_index=DIRECTION_TO_INDEX[policy],
                supported_directions=list(record["supported_directions"] or []),
            ))

        fingerprint = sha256_file(parquet_path)
        self.graph_cache_path = graph_cache_path
        self.graphs = self._load_or_build_graphs(fingerprint, rebuild_graph_cache)

    def _load_or_build_graphs(self, fingerprint: str, rebuild: bool) -> list[MoleculeGraph]:
        if self.graph_cache_path and self.graph_cache_path.exists() and not rebuild:
            try:
                payload = torch.load(self.graph_cache_path, map_location="cpu", weights_only=False)
            except (AttributeError, EOFError, ModuleNotFoundError, OSError, pickle.UnpicklingError, RuntimeError):
                payload = None
            if isinstance(payload, dict):
                if (
                    payload.get("data_sha256") == fingerprint
                    and payload.get("smiles") == self.smiles
                    and payload.get("graph_feature_version") == GRAPH_FEATURE_VERSION
                ):
                    return payload["graphs"]
        graphs = [smiles_to_graph(smiles) for smiles in self.smiles]
        if self.graph_cache_path:
            self.graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "data_sha256": fingerprint,
                "graph_feature_version": GRAPH_FEATURE_VERSION,
                "smiles": self.smiles,
                "graphs": graphs,
            }, self.graph_cache_path)
        return graphs

    def vocab_sizes(self) -> dict[str, int]:
        return {
            "role": len(self.role_vocab) + 1,
            "cofactor": len(self.cofactor_vocab) + 1,
            "compartment": len(self.compartment_vocab) + 1,
            "ec": len(self.ec_vocab) + 1,
            "reaction_type": len(self.reaction_type_vocab) + 1,
        }

    def vocabularies(self) -> dict[str, dict[str, int]]:
        return {
            "role": self.role_vocab,
            "cofactor": self.cofactor_vocab,
            "compartment": self.compartment_vocab,
            "ec": self.ec_vocab,
            "reaction_type": self.reaction_type_vocab,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ReactionExample:
        return self.examples[index]

    def collate(self, examples: Iterable[ReactionExample]) -> ReactionBatch:
        examples = list(examples)
        unique_molecules: dict[int, int] = {}
        for example in examples:
            for global_index in example.molecule_indices:
                if global_index not in unique_molecules:
                    unique_molecules[global_index] = len(unique_molecules)
        local_to_global = sorted(unique_molecules, key=unique_molecules.get)
        molecule_batch = collate_molecule_graphs([self.graphs[index] for index in local_to_global])

        participant_molecule = []
        participant_reaction = []
        side = []
        coefficient = []
        role = []
        cofactor = []
        compartment = []
        ptr = [0]
        ec_index = []
        ec_reaction_index = []
        for reaction_index, example in enumerate(examples):
            participant_molecule.extend(unique_molecules[index] for index in example.molecule_indices)
            participant_reaction.extend([reaction_index] * len(example.molecule_indices))
            side.extend(example.sides)
            coefficient.extend(example.coefficients)
            role.extend(example.role_indices)
            cofactor.extend(example.cofactor_indices)
            compartment.extend(example.compartment_indices)
            ptr.append(len(participant_molecule))
            ec_index.extend(example.ec_indices)
            ec_reaction_index.extend([reaction_index] * len(example.ec_indices))

        return ReactionBatch(
            molecule_graphs=molecule_batch,
            participant_molecule_index=torch.tensor(participant_molecule, dtype=torch.long),
            participant_reaction_index=torch.tensor(participant_reaction, dtype=torch.long),
            participant_ptr=torch.tensor(ptr, dtype=torch.long),
            side=torch.tensor(side, dtype=torch.long),
            coefficient=torch.tensor(coefficient, dtype=torch.float32),
            role_index=torch.tensor(role, dtype=torch.long),
            cofactor_index=torch.tensor(cofactor, dtype=torch.long),
            compartment_index=torch.tensor(compartment, dtype=torch.long),
            ec_index=torch.tensor(ec_index, dtype=torch.long),
            ec_reaction_index=torch.tensor(ec_reaction_index, dtype=torch.long),
            reaction_type_index=torch.tensor([example.reaction_type_index for example in examples], dtype=torch.long),
            direction_index=torch.tensor([example.direction_index for example in examples], dtype=torch.long),
            reaction_ids=[example.reaction_id for example in examples],
        )

    def collate_fn(self) -> Callable[[list[ReactionExample]], ReactionBatch]:
        return self.collate
