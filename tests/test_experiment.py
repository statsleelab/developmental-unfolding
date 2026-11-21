import unittest
from collections import Counter
from dataclasses import replace

import numpy as np
from numpy.random import default_rng

from experiment import CONDITIONS, Config, run_seed, summarize
from model import adjacent, evolve, grow_colony, prune, recurrent_matrix, replaced_fraction, rewire


class MatchedExperimentTests(unittest.TestCase):
    def test_colony_grows_into_unoccupied_neighbors(self):
        colony = grow_colony(256, 8, default_rng(0))
        occupied = {(0, 0)}
        self.assertEqual(tuple(colony.positions[0]), (0, 0))
        for point in colony.positions[1:]:
            position = tuple(point)
            self.assertNotIn(position, occupied)
            self.assertTrue(any(neighbor in occupied for neighbor in adjacent(position)))
            occupied.add(position)

    def test_controls_preserve_their_matching_constraints(self):
        rng = default_rng(23)
        colony = grow_colony(96, 8, rng)
        precursor = recurrent_matrix(colony, colony.candidates, 0.9)
        spontaneous = evolve(precursor, rng.normal(0, 0.18, (160, 96)), 0.5)
        activity, random = prune(colony, spontaneous, 0.5, rng)
        rewired, acceptance = rewire(colony, activity, 60, rng)

        def local_quotas(edges):
            return Counter((source, colony.distance_squared[source, target]) for source, target in edges)

        def matching_statistics(edges):
            lengths_by_sign = Counter((colony.signs[source], colony.distance_squared[source, target])
                                      for source, target in edges)
            inputs_by_sign = Counter((target, colony.signs[source]) for source, target in edges)
            outgoing_degrees = Counter(edges[:, 0])
            return lengths_by_sign, inputs_by_sign, outgoing_degrees

        self.assertEqual(local_quotas(activity), local_quotas(random))
        self.assertEqual(matching_statistics(activity), matching_statistics(rewired))
        self.assertEqual(len(set(map(tuple, rewired))), len(rewired))
        self.assertTrue(np.all(rewired[:, 0] != rewired[:, 1]))
        self.assertGreater(acceptance, 0)
        self.assertGreater(replaced_fraction(activity, rewired), 0.15)
        for edges in (activity, random, rewired):
            matrix = recurrent_matrix(colony, edges, 0.9)
            self.assertLessEqual(float(np.abs(matrix).sum(axis=1).max()), 0.9 + 1e-12)

    def test_seed_reproduction_and_paired_contrast(self):
        config = replace(Config(), nodes=48, development_steps=160, split_steps=256,
                         swap_attempts_per_edge=20, bootstrap_samples=200)
        first = run_seed(config, 40)
        repeat = run_seed(config, 40)
        second = run_seed(config, 41)
        for condition in CONDITIONS:
            np.testing.assert_allclose(first.scores[condition], repeat.scores[condition], atol=1e-9)
            self.assertTrue(np.isfinite(second.scores[condition]).all())
        summary = summarize([first, second], config.bootstrap_samples)
        expected = np.mean([(run.scores["activity"] - run.scores["random"]).mean()
                            for run in (first, second)])
        self.assertAlmostEqual(summary["contrasts"]["activity_minus_random"]["mean"], expected)


if __name__ == "__main__":
    unittest.main()
