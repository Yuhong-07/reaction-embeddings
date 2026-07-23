from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from rhea_embedding.chemistry.graph import segment_softmax, segment_sum
from rhea_embedding.data.reaction_dataset import ReactionBatch
from rhea_embedding.models.dmpnn import DirectedMessagePassingNetwork


@dataclass
class EncoderOutput:
    embedding: torch.Tensor
    projection: torch.Tensor


class ReactionEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_sizes: dict[str, int]) -> None:
        super().__init__()
        hidden = int(config.get("embedding_dim", 256))
        molecule_hidden = int(config.get("molecule_hidden_dim", hidden))
        dropout = float(config.get("dropout", 0.1))
        self.embedding_dim = hidden
        if bool(config.get("use_ec", False)) or bool(config.get("ec_as_input", False)):
            raise ValueError("EC input is forbidden for the main reaction embedding; use the auxiliary prediction head")
        self.use_cofactor = bool(config.get("use_cofactor", True))

        self.molecule_encoder = DirectedMessagePassingNetwork(
            hidden_dim=molecule_hidden,
            message_passing_steps=int(config.get("molecule_message_passing_steps", 4)),
            dropout=dropout,
            readout=str(config.get("molecule_readout", "attention")),
        )
        self.molecule_projection = nn.Linear(molecule_hidden, hidden) if molecule_hidden != hidden else nn.Identity()
        self.side_embedding = nn.Embedding(2, 16)
        self.stoichiometry_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32))
        self.cofactor_embedding = nn.Embedding(max(vocab_sizes["cofactor"], 1), 8)
        participant_input_dim = hidden + 16 + 32 + 8 + 1
        self.participant_projection = nn.Sequential(
            nn.Linear(participant_input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden)
        )
        self.participant_attention = nn.Linear(hidden, 1)
        self.side_set_projection = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden)
        )
        self.delta_projection = nn.Linear(hidden, hidden, bias=False)
        self.masked_molecule_token = nn.Parameter(torch.zeros(hidden))

        self.direction_embedding = nn.Embedding(5, 32)  # four values plus masked direction
        fusion_dim = hidden * 4 + 32 + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.projection_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.ec_class_count = max(int(vocab_sizes.get("ec", 1)) - 1, 1)
        self.ec_auxiliary_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, self.ec_class_count)
        )

    def encode_molecules(self, batch: ReactionBatch) -> torch.Tensor:
        return self.molecule_projection(self.molecule_encoder(batch.molecule_graphs))

    def _encode_once(
        self,
        molecule_embeddings: torch.Tensor,
        batch: ReactionBatch,
        participant_mask: torch.Tensor | None = None,
        stoichiometry_mask: torch.Tensor | None = None,
        mask_direction_input: bool = False,
        mask_optional_metadata: bool = False,
    ) -> EncoderOutput:
        participant_molecule = molecule_embeddings[batch.participant_molecule_index]
        if participant_mask is not None:
            participant_molecule = torch.where(
                participant_mask.unsqueeze(-1), self.masked_molecule_token.unsqueeze(0), participant_molecule
            )

        log_coefficient = torch.log1p(batch.coefficient.abs()).unsqueeze(-1)
        stoich_missing = torch.zeros_like(log_coefficient)
        if stoichiometry_mask is not None:
            log_coefficient = torch.where(stoichiometry_mask.unsqueeze(-1), torch.zeros_like(log_coefficient), log_coefficient)
            stoich_missing = stoichiometry_mask.float().unsqueeze(-1)
        stoich_features = self.stoichiometry_mlp(torch.cat([log_coefficient, stoich_missing], dim=-1))

        use_cofactor = self.use_cofactor and not mask_optional_metadata
        cofactor_index = batch.cofactor_index if use_cofactor else torch.zeros_like(batch.cofactor_index)
        cofactor_missing = (cofactor_index == 0).float().unsqueeze(-1)
        participant_input = torch.cat([
            participant_molecule,
            self.side_embedding(batch.side),
            stoich_features,
            self.cofactor_embedding(cofactor_index),
            cofactor_missing,
        ], dim=-1)
        tokens = self.participant_projection(participant_input)

        side_segment = batch.participant_reaction_index * 2 + batch.side
        side_scores = self.participant_attention(tokens).squeeze(-1)
        side_weights = segment_softmax(side_scores, side_segment, batch.reaction_count * 2)
        side_attended = segment_sum(tokens * side_weights.unsqueeze(-1), side_segment, batch.reaction_count * 2)
        side_summed = segment_sum(tokens, side_segment, batch.reaction_count * 2)
        side_counts = torch.bincount(side_segment, minlength=batch.reaction_count * 2).to(tokens.dtype).clamp_min(1)
        side_stoichiometry = segment_sum(
            batch.coefficient.abs().unsqueeze(-1), side_segment, batch.reaction_count * 2
        ).squeeze(-1)
        side_features = torch.cat([
            side_attended,
            side_summed / side_counts.sqrt().unsqueeze(-1),
            torch.log1p(side_counts).unsqueeze(-1),
            torch.log1p(side_stoichiometry).unsqueeze(-1),
        ], dim=-1)
        side_pooled = self.side_set_projection(side_features).view(batch.reaction_count, 2, self.embedding_dim)
        reactants = side_pooled[:, 0]
        products = side_pooled[:, 1]
        delta = segment_sum(
            self.delta_projection(participant_molecule) * batch.coefficient.unsqueeze(-1),
            batch.participant_reaction_index,
            batch.reaction_count,
        )

        direction_index = torch.full_like(batch.direction_index, 4) if mask_direction_input else batch.direction_index
        direction_missing = (batch.direction_index == 3).float().unsqueeze(-1)
        if mask_direction_input:
            direction_missing = torch.ones_like(direction_missing)

        fusion_input = torch.cat([
            reactants,
            products,
            products - reactants,
            delta,
            self.direction_embedding(direction_index),
            direction_missing,
        ], dim=-1)
        embedding = self.fusion(fusion_input)
        projection = F.normalize(self.projection_head(embedding), dim=-1)
        return EncoderOutput(embedding, projection)

    def predict_ec(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict EC labels from an embedding; EC is never consumed by the encoder."""
        return self.ec_auxiliary_head(embedding)

    def encode_canonical(
        self,
        molecule_embeddings: torch.Tensor,
        batch: ReactionBatch,
        participant_mask: torch.Tensor | None = None,
        stoichiometry_mask: torch.Tensor | None = None,
        mask_direction_input: bool = False,
        mask_optional_metadata: bool = False,
    ) -> EncoderOutput:
        forward = self._encode_once(
            molecule_embeddings, batch, participant_mask, stoichiometry_mask,
            mask_direction_input, mask_optional_metadata,
        )
        needs_reverse = (batch.direction_index == 1) | (batch.direction_index == 2)
        if not bool(needs_reverse.any()):
            return forward
        reverse = self._encode_once(
            molecule_embeddings, batch.swapped_orientation(), participant_mask, stoichiometry_mask,
            mask_direction_input, mask_optional_metadata,
        )
        embedding = forward.embedding.clone()
        projection = forward.projection.clone()
        right_to_left = batch.direction_index == 1
        reversible = batch.direction_index == 2
        embedding[right_to_left] = reverse.embedding[right_to_left]
        projection[right_to_left] = reverse.projection[right_to_left]
        embedding[reversible] = 0.5 * (forward.embedding[reversible] + reverse.embedding[reversible])
        projection[reversible] = F.normalize(
            0.5 * (forward.projection[reversible] + reverse.projection[reversible]), dim=-1
        )
        return EncoderOutput(embedding, projection)
