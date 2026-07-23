from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from rdkit import Chem

from rhea_embedding.chemistry.graph import collate_molecule_graphs, smiles_to_graph
from rhea_embedding.data.reaction_dataset import ReactionCorpus
from rhea_embedding.models.reaction_encoder import ReactionEncoder


@torch.no_grad()
def run_embedding_invariance_checks(
    model: ReactionEncoder,
    corpus: ReactionCorpus,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    examples = [corpus[index] for index in range(min(8, len(corpus)))]
    batch = corpus.collate(examples).to(device)
    molecule_embeddings = model.encode_molecules(batch)
    base = model.encode_canonical(molecule_embeddings, batch).embedding

    permuted = batch.permuted_participants()
    permuted_embedding = model.encode_canonical(molecule_embeddings, permuted).embedding
    participant_order_max_abs_diff = float((base - permuted_embedding).abs().max().cpu())

    repeated = model.encode_canonical(molecule_embeddings, batch).embedding
    repeat_max_abs_diff = float((base - repeated).abs().max().cpu())

    single = corpus.collate([examples[0]]).to(device)
    single_molecules = model.encode_molecules(single)
    single_embedding = model.encode_canonical(single_molecules, single).embedding[0]
    batch_single_max_abs_diff = float((base[0] - single_embedding).abs().max().cpu())

    reversible_index = next(index for index, example in enumerate(corpus.examples) if example.direction_index == 2)
    reversible_batch = corpus.collate([corpus[reversible_index]]).to(device)
    reversible_molecules = model.encode_molecules(reversible_batch)
    reversible_forward = model.encode_canonical(reversible_molecules, reversible_batch).embedding
    reversible_swapped = model.encode_canonical(
        reversible_molecules, reversible_batch.swapped_orientation()
    ).embedding
    reversible_swap_max_abs_diff = float((reversible_forward - reversible_swapped).abs().max().cpu())

    optional_missing_embedding = model.encode_canonical(
        molecule_embeddings, batch, mask_optional_metadata=True
    ).embedding
    optional_missing_finite = bool(torch.isfinite(optional_missing_embedding).all().cpu())

    randomized_similarity = None
    for smiles in corpus.smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
        if randomized == smiles:
            continue
        graph_batch = collate_molecule_graphs([smiles_to_graph(smiles), smiles_to_graph(randomized)]).to(device)
        molecule_pair = model.molecule_projection(model.molecule_encoder(graph_batch))
        randomized_similarity = float(F.cosine_similarity(molecule_pair[0:1], molecule_pair[1:2]).cpu())
        break
    if randomized_similarity is None:
        raise RuntimeError("Could not construct a randomized SMILES quality-control pair")

    tolerances = {
        "participant_order": 1e-5,
        "repeat": 0.0,
        "batch_single": 1e-5,
        "reversible_swap": 1e-6,
        "randomized_smiles_cosine_min": 0.999,
    }
    passed = (
        participant_order_max_abs_diff <= tolerances["participant_order"]
        and repeat_max_abs_diff <= tolerances["repeat"]
        and batch_single_max_abs_diff <= tolerances["batch_single"]
        and reversible_swap_max_abs_diff <= tolerances["reversible_swap"]
        and randomized_similarity >= tolerances["randomized_smiles_cosine_min"]
        and optional_missing_finite
    )
    return {
        "passed": passed,
        "tolerances": tolerances,
        "participant_order_max_abs_diff": participant_order_max_abs_diff,
        "repeat_export_max_abs_diff": repeat_max_abs_diff,
        "batch_single_max_abs_diff": batch_single_max_abs_diff,
        "reversible_swap_max_abs_diff": reversible_swap_max_abs_diff,
        "randomized_smiles_cosine_similarity": randomized_similarity,
        "all_optional_metadata_missing_finite": optional_missing_finite,
    }
