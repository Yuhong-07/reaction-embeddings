import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rhea_embedding.data.prepare_rhea import build_keys, reduced_coefficients


class TestCanonicalKeys(unittest.TestCase):
    def test_reduced_stoichiometry_is_scale_invariant(self):
        first = [
            {"compound_id": "CHEBI:A", "coefficient": -2.0, "compartment": None},
            {"compound_id": "CHEBI:B", "coefficient": 2.0, "compartment": None},
        ]
        second = [
            {"compound_id": "CHEBI:A", "coefficient": -1.0, "compartment": None},
            {"compound_id": "CHEBI:B", "coefficient": 1.0, "compartment": None},
        ]
        self.assertEqual(reduced_coefficients(first), reduced_coefficients(second))

    def test_participant_order_does_not_change_key(self):
        participants = [
            {"compound_id": "CHEBI:A", "coefficient": -1.0, "compartment": None},
            {"compound_id": "CHEBI:B", "coefficient": 1.0, "compartment": None},
        ]
        a = {"participants": participants, "direction": "undefined"}
        b = {"participants": list(reversed(participants)), "direction": "undefined"}
        self.assertEqual(build_keys(a, set()), build_keys(b, set()))

    def test_loose_key_ignores_water_but_strict_does_not(self):
        base = [
            {"compound_id": "CHEBI:A", "coefficient": -1.0, "compartment": None},
            {"compound_id": "CHEBI:B", "coefficient": 1.0, "compartment": None},
        ]
        hydrated = base + [{"compound_id": "CHEBI:15377", "coefficient": 1.0, "compartment": None}]
        strict_a, loose_a = build_keys({"participants": base, "direction": "undefined"}, {"CHEBI:15377"})
        strict_b, loose_b = build_keys({"participants": hydrated, "direction": "undefined"}, {"CHEBI:15377"})
        self.assertNotEqual(strict_a, strict_b)
        self.assertEqual(loose_a, loose_b)


if __name__ == "__main__":
    unittest.main()
