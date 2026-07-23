from __future__ import annotations

import torch
from torch import nn

from rhea_embedding.chemistry.graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MoleculeGraphBatch,
    segment_softmax,
    segment_sum,
)


class DirectedMessagePassingNetwork(nn.Module):
    """Shared directed-bond message-passing molecular encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        message_passing_steps: int = 4,
        dropout: float = 0.1,
        readout: str = "attention",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps
        self.readout = readout
        self.input_projection = nn.Linear(ATOM_FEATURE_DIM + BOND_FEATURE_DIM, hidden_dim)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atom_projection = nn.Linear(ATOM_FEATURE_DIM + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.attention_sum_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, batch: MoleculeGraphBatch) -> torch.Tensor:
        atom_count = batch.atom_features.shape[0]
        edge_count = batch.edge_index.shape[1]
        if edge_count:
            source = batch.edge_index[0]
            target = batch.edge_index[1]
            initial = self.activation(self.input_projection(torch.cat([batch.atom_features[source], batch.edge_features], dim=-1)))
            hidden = initial
            for _ in range(self.message_passing_steps - 1):
                incoming = segment_sum(hidden, target, atom_count)
                messages = incoming[source] - hidden[batch.reverse_edge]
                hidden = self.activation(initial + self.message_projection(messages))
                hidden = self.dropout(hidden)
            incoming_atoms = segment_sum(hidden, target, atom_count)
        else:
            incoming_atoms = batch.atom_features.new_zeros((atom_count, self.hidden_dim))

        atom_hidden = self.activation(self.atom_projection(torch.cat([batch.atom_features, incoming_atoms], dim=-1)))
        atom_hidden = self.dropout(atom_hidden)
        summed = segment_sum(atom_hidden, batch.atom_to_molecule, batch.num_molecules)
        if self.readout == "sum":
            return summed
        if self.readout not in {"attention", "attention_sum"}:
            raise ValueError(f"Unknown molecular readout: {self.readout}")
        scores = self.attention_score(atom_hidden).squeeze(-1)
        weights = segment_softmax(scores, batch.atom_to_molecule, batch.num_molecules)
        attended = segment_sum(atom_hidden * weights.unsqueeze(-1), batch.atom_to_molecule, batch.num_molecules)
        if self.readout == "attention":
            return attended
        atom_counts = torch.bincount(
            batch.atom_to_molecule, minlength=batch.num_molecules
        ).to(atom_hidden.dtype).clamp_min(1).unsqueeze(-1)
        normalized_sum = summed / atom_counts.sqrt()
        size_feature = torch.log1p(atom_counts)
        return self.attention_sum_projection(torch.cat([attended, normalized_sum, size_feature], dim=-1))
