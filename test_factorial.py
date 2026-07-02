"""Cheap tests for the v2 factorial island study: quadruple validity and the
difference-in-differences arithmetic. Stdlib only -- no torch, no network.

    python3 test_factorial.py
"""

import json
import tempfile
import unittest
from pathlib import Path

import generate_factorial
from syntaxbench.factorial import CONDITIONS, bootstrap_ci, did, load_quadruples


class TestDiD(unittest.TestCase):
    def test_toy_interaction(self):
        totals = {"nonisland_noextraction": 10.0, "nonisland_extraction": 14.0,
                  "island_noextraction": 12.0, "island_extraction": 25.0}
        self.assertAlmostEqual(did(totals), (25.0 - 12.0) - (14.0 - 10.0))

    def test_purely_additive_costs_cancel(self):
        # extraction costs 4, the island structure costs 2, no interaction
        totals = {"nonisland_noextraction": 10.0, "nonisland_extraction": 14.0,
                  "island_noextraction": 12.0, "island_extraction": 16.0}
        self.assertAlmostEqual(did(totals), 0.0)

    def test_negative_interaction(self):
        totals = {"nonisland_noextraction": 10.0, "nonisland_extraction": 20.0,
                  "island_noextraction": 15.0, "island_extraction": 18.0}
        self.assertAlmostEqual(did(totals), -7.0)


class TestBootstrap(unittest.TestCase):
    def test_constant_values_pin_the_ci(self):
        self.assertEqual(bootstrap_ci([3.0] * 10, n_boot=500), (3.0, 3.0))

    def test_ci_brackets_the_mean_and_is_deterministic(self):
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        first = bootstrap_ci(values, n_boot=2000, seed=7)
        again = bootstrap_ci(values, n_boot=2000, seed=7)
        self.assertEqual(first, again)
        mean = sum(values) / len(values)
        self.assertLess(first[0], mean)
        self.assertGreater(first[1], mean)


class TestQuadruples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate_factorial.rng.seed(generate_factorial.SEED)
        cls.quads = generate_factorial.build()

    def test_counts(self):
        by_subtype = {}
        for q in self.quads:
            by_subtype[q["subtype"]] = by_subtype.get(q["subtype"], 0) + 1
        expected = {st: generate_factorial.N_PER_SUBTYPE for st, _ in generate_factorial.MAKERS}
        self.assertEqual(by_subtype, expected)

    def test_four_distinct_sentences_per_quadruple(self):
        for q in self.quads:
            self.assertEqual(set(q["sentences"]), set(CONDITIONS), q["id"])
            self.assertEqual(len(set(q["sentences"].values())), 4, q["id"])

    def test_lexical_content_shared_across_all_four(self):
        for q in self.quads:
            for slot, word in q["lexicon"].items():
                for cond, sent in q["sentences"].items():
                    self.assertIn(word.lower(), sent.lower(),
                                  f"{q['id']}: {slot} {word!r} missing from {cond}")

    def test_only_extraction_members_front_a_wh_phrase(self):
        extraction = {"nonisland_extraction", "island_extraction"}
        for q in self.quads:
            for cond, sent in q["sentences"].items():
                self.assertEqual(sent.startswith("Which "), cond in extraction,
                                 f"{q['id']}/{cond}: {sent!r}")

    def test_generation_is_reproducible(self):
        generate_factorial.rng.seed(generate_factorial.SEED)
        self.assertEqual(generate_factorial.build(), self.quads)

    def test_roundtrip_through_loader(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for q in self.quads:
                f.write(json.dumps(q) + "\n")
            path = f.name
        try:
            loaded = load_quadruples(path)
        finally:
            Path(path).unlink()
        self.assertEqual(len(loaded), len(self.quads))
        self.assertEqual(loaded[0].sentences, self.quads[0]["sentences"])


if __name__ == "__main__":
    unittest.main()
