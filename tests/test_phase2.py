import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rhea_embedding.chemistry.graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, smiles_to_graph
from rhea_embedding.data.reaction_dataset import ReactionCorpus, ReactionExample, direction_policy
from rhea_embedding.models.reaction_encoder import ReactionEncoder
from rhea_embedding.training.phase2 import covariance_loss, split_selected_indices, variance_loss


class MiniCorpus:
    def __init__(self):
        self.graphs = [smiles_to_graph("CCO"), smiles_to_graph("CC=O"), smiles_to_graph("O")]

    collate = ReactionCorpus.collate


class TestTrainingSplit(unittest.TestCase):
    def test_zero_validation_uses_every_selected_reaction_for_training(self):
        train, validation = split_selected_indices(list(range(10)), 0.0)
        self.assertEqual(train, list(range(10)))
        self.assertEqual(validation, [])

    def test_positive_validation_fraction_remains_deterministic(self):
        train, validation = split_selected_indices(list(range(20)), 0.1)
        self.assertEqual(train, list(range(2, 20)))
        self.assertEqual(validation, [0, 1])


class TestGraphCacheCompatibility(unittest.TestCase):
    def test_unreadable_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "graphs.pt"
            cache_path.write_bytes(b"not a torch cache")
            corpus = ReactionCorpus.__new__(ReactionCorpus)
            corpus.graph_cache_path = cache_path
            corpus.smiles = ["O"]
            graphs = corpus._load_or_build_graphs("fingerprint", rebuild=False)
            self.assertEqual(len(graphs), 1)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["data_sha256"], "fingerprint")


def example(direction_index=2):
    return ReactionExample(
        reaction_id="RHEA:TEST",
        molecule_indices=[0, 2, 1],
        coefficients=[-1.0, -1.0, 1.0],
        sides=[0, 0, 1],
        role_indices=[0, 0, 0],
        cofactor_indices=[0, 0, 0],
        compartment_indices=[0, 0, 0],
        ec_indices=[],
        reaction_type_index=0,
        direction_index=direction_index,
        supported_directions=["reversible"] if direction_index == 2 else [],
    )


class TestPhase2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.corpus = MiniCorpus()
        config = {
            "embedding_dim": 32,
            "molecule_hidden_dim": 32,
            "molecule_message_passing_steps": 2,
            "molecule_readout": "attention_sum",
            "dropout": 0.0,
            "use_cofactor": True,
            "ec_as_input": False,
        }
        self.model = ReactionEncoder(
            config, {"role": 1, "cofactor": 1, "compartment": 1, "ec": 1, "reaction_type": 1}
        ).eval()

    def test_graph_feature_shapes(self):
        graph = smiles_to_graph("CCO")
        self.assertEqual(graph.atom_features.shape[1], ATOM_FEATURE_DIM)
        self.assertEqual(graph.edge_features.shape[1], BOND_FEATURE_DIM)
        self.assertEqual(graph.edge_index.shape[1], graph.reverse_edge.shape[0])

    def test_participant_order_invariance(self):
        batch = self.corpus.collate([example()])
        molecules = self.model.encode_molecules(batch)
        first = self.model.encode_canonical(molecules, batch).embedding
        second = self.model.encode_canonical(molecules, batch.permuted_participants()).embedding
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=0.0))

    def test_reversible_swap_invariance(self):
        batch = self.corpus.collate([example(direction_index=2)])
        molecules = self.model.encode_molecules(batch)
        first = self.model.encode_canonical(molecules, batch).embedding
        second = self.model.encode_canonical(molecules, batch.swapped_orientation()).embedding
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=0.0))

    def test_missing_optional_metadata_runs(self):
        batch = self.corpus.collate([example(direction_index=3)])
        molecules = self.model.encode_molecules(batch)
        result = self.model.encode_canonical(molecules, batch, mask_optional_metadata=True).embedding
        self.assertEqual(tuple(result.shape), (1, 32))
        self.assertTrue(torch.isfinite(result).all())

    def test_ec_is_not_an_embedding_input(self):
        without_ec = example(direction_index=0)
        with_ec = example(direction_index=0)
        with_ec.ec_indices = [1]
        first_batch = self.corpus.collate([without_ec])
        second_batch = self.corpus.collate([with_ec])
        first = self.model.encode_canonical(self.model.encode_molecules(first_batch), first_batch).embedding
        second = self.model.encode_canonical(self.model.encode_molecules(second_batch), second_batch).embedding
        self.assertTrue(torch.equal(first, second))

    def test_direction_changes_irreversible_embedding(self):
        batch = self.corpus.collate([example(direction_index=0), example(direction_index=3)])
        embeddings = self.model.encode_canonical(self.model.encode_molecules(batch), batch).embedding
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1], atol=1e-7, rtol=0.0))

    def test_attention_sum_readout_retains_molecule_size(self):
        graphs = [smiles_to_graph("CCC"), smiles_to_graph("CCCCCCC")]
        from rhea_embedding.chemistry.graph import collate_molecule_graphs
        graph_batch = collate_molecule_graphs(graphs)
        embeddings = self.model.molecule_projection(self.model.molecule_encoder(graph_batch))
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1], atol=1e-7, rtol=0.0))

    def test_dummy_atom_isotope_labels_are_retained(self):
        unlabeled = smiles_to_graph("*C(=O)[O-]")
        labeled = smiles_to_graph("[1*]C(=O)[O-]")
        self.assertFalse(torch.equal(unlabeled.atom_features, labeled.atom_features))
        from rhea_embedding.chemistry.graph import collate_molecule_graphs
        graph_batch = collate_molecule_graphs([unlabeled, labeled])
        embeddings = self.model.molecule_projection(self.model.molecule_encoder(graph_batch))
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1], atol=1e-7, rtol=0.0))

    def test_collapse_regularizers_are_finite(self):
        collapsed = torch.zeros((8, 32))
        varied = torch.randn((8, 32))
        self.assertGreater(float(variance_loss(collapsed, collapsed)), 0.9)
        self.assertTrue(torch.isfinite(variance_loss(varied, varied)))
        self.assertTrue(torch.isfinite(covariance_loss(varied, varied)))

    def test_ec_input_configuration_is_rejected(self):
        leaking_config = {
            "embedding_dim": 32,
            "molecule_hidden_dim": 32,
            "molecule_message_passing_steps": 2,
            "molecule_readout": "attention_sum",
            "dropout": 0.0,
            "use_ec": True,
        }
        with self.assertRaisesRegex(ValueError, "EC input is forbidden"):
            ReactionEncoder(leaking_config, {"role": 1, "cofactor": 1, "compartment": 1, "ec": 1, "reaction_type": 1})

    def test_direction_policy(self):
        self.assertEqual(direction_policy(["left_to_right"]), "left_to_right")
        self.assertEqual(direction_policy(["right_to_left"]), "right_to_left")
        self.assertEqual(direction_policy(["reversible"]), "reversible")
        self.assertEqual(direction_policy(["left_to_right", "right_to_left"]), "reversible")
        self.assertEqual(direction_policy([]), "undefined")


if __name__ == "__main__":
    unittest.main()
