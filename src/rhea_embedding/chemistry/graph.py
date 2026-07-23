from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from rdkit import Chem


ATOM_NUM_CHOICES = list(range(119))
FORMAL_CHARGE_CHOICES = list(range(-5, 6))
DEGREE_CHOICES = list(range(7))
HYBRIDIZATION_CHOICES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
HYDROGEN_CHOICES = list(range(5))
BOND_TYPE_CHOICES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_CHOICES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
GRAPH_FEATURE_VERSION = 2


def one_hot_unknown(value: Any, choices: list[Any]) -> list[float]:
    result = [0.0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = len(choices)
    result[index] = 1.0
    return result


def safe_call(function, default):
    try:
        return function()
    except Exception:
        return default


def atom_features(atom: Chem.Atom) -> list[float]:
    atomic_num = safe_call(atom.GetAtomicNum, 0)
    formal_charge = safe_call(atom.GetFormalCharge, 0)
    degree = safe_call(atom.GetDegree, 0)
    hybridization = safe_call(atom.GetHybridization, Chem.rdchem.HybridizationType.UNSPECIFIED)
    chirality = safe_call(atom.GetChiralTag, Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    hydrogen_count = safe_call(lambda: atom.GetTotalNumHs(includeNeighbors=False), 0)
    isotope = safe_call(atom.GetIsotope, 0)
    return (
        one_hot_unknown(atomic_num, ATOM_NUM_CHOICES)
        + one_hot_unknown(formal_charge, FORMAL_CHARGE_CHOICES)
        + one_hot_unknown(degree, DEGREE_CHOICES)
        + [float(safe_call(atom.GetIsAromatic, False))]
        + one_hot_unknown(hybridization, HYBRIDIZATION_CHOICES)
        + one_hot_unknown(chirality, CHIRAL_CHOICES)
        + one_hot_unknown(min(int(hydrogen_count), 4), HYDROGEN_CHOICES)
        + [float(isotope == 0), min(float(isotope), 300.0) / 300.0]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        one_hot_unknown(bond.GetBondType(), BOND_TYPE_CHOICES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + one_hot_unknown(bond.GetStereo(), BOND_STEREO_CHOICES)
    )


ATOM_FEATURE_DIM = (
    len(ATOM_NUM_CHOICES) + 1
    + len(FORMAL_CHARGE_CHOICES) + 1
    + len(DEGREE_CHOICES) + 1
    + 1
    + len(HYBRIDIZATION_CHOICES) + 1
    + len(CHIRAL_CHOICES) + 1
    + len(HYDROGEN_CHOICES) + 1
    + 2
)
BOND_FEATURE_DIM = len(BOND_TYPE_CHOICES) + 1 + 2 + len(BOND_STEREO_CHOICES) + 1


@dataclass
class MoleculeGraph:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    reverse_edge: torch.Tensor
    smiles: str
    sanitization_status: str


@dataclass
class MoleculeGraphBatch:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    reverse_edge: torch.Tensor
    atom_to_molecule: torch.Tensor
    num_molecules: int

    def to(self, device: torch.device | str) -> "MoleculeGraphBatch":
        return MoleculeGraphBatch(
            atom_features=self.atom_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            reverse_edge=self.reverse_edge.to(device),
            atom_to_molecule=self.atom_to_molecule.to(device),
            num_molecules=self.num_molecules,
        )


def molecule_from_smiles(smiles: str) -> tuple[Chem.Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "sanitized"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    mol.UpdatePropertyCache(strict=False)
    return mol, "unsanitized"


def smiles_to_graph(smiles: str) -> MoleculeGraph:
    mol, status = molecule_from_smiles(smiles)
    atoms = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    if atoms.numel() == 0:
        raise ValueError(f"SMILES has no atoms: {smiles}")

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_feature_rows: list[list[float]] = []
    reverse: list[int] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        forward = len(edge_sources)
        backward = forward + 1
        edge_sources.extend([begin, end])
        edge_targets.extend([end, begin])
        edge_feature_rows.extend([features, features])
        reverse.extend([backward, forward])

    if edge_sources:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edges = torch.tensor(edge_feature_rows, dtype=torch.float32)
        reverse_tensor = torch.tensor(reverse, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edges = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
        reverse_tensor = torch.empty((0,), dtype=torch.long)
    return MoleculeGraph(atoms, edge_index, edges, reverse_tensor, smiles, status)


def collate_molecule_graphs(graphs: Iterable[MoleculeGraph]) -> MoleculeGraphBatch:
    graphs = list(graphs)
    if not graphs:
        raise ValueError("Cannot collate an empty molecule graph list")
    atom_rows = []
    edge_indices = []
    edge_rows = []
    reverse_rows = []
    atom_to_molecule = []
    atom_offset = 0
    edge_offset = 0
    for molecule_index, graph in enumerate(graphs):
        atom_rows.append(graph.atom_features)
        atom_to_molecule.append(torch.full((graph.atom_features.shape[0],), molecule_index, dtype=torch.long))
        if graph.edge_index.shape[1]:
            edge_indices.append(graph.edge_index + atom_offset)
            edge_rows.append(graph.edge_features)
            reverse_rows.append(graph.reverse_edge + edge_offset)
        atom_offset += graph.atom_features.shape[0]
        edge_offset += graph.edge_index.shape[1]
    return MoleculeGraphBatch(
        atom_features=torch.cat(atom_rows, dim=0),
        edge_index=torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.cat(edge_rows, dim=0) if edge_rows else torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32),
        reverse_edge=torch.cat(reverse_rows, dim=0) if reverse_rows else torch.empty((0,), dtype=torch.long),
        atom_to_molecule=torch.cat(atom_to_molecule, dim=0),
        num_molecules=len(graphs),
    )


def segment_sum(values: torch.Tensor, index: torch.Tensor, segment_count: int) -> torch.Tensor:
    output_shape = (segment_count,) + tuple(values.shape[1:])
    output = values.new_zeros(output_shape)
    output.index_add_(0, index, values)
    return output


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, segment_count: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("segment_softmax expects one-dimensional scores")
    maxima = scores.new_full((segment_count,), float("-inf"))
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    shifted = scores - maxima[index]
    exponentials = shifted.exp()
    denominators = scores.new_zeros((segment_count,))
    denominators.index_add_(0, index, exponentials)
    return exponentials / denominators[index].clamp_min(1e-12)
