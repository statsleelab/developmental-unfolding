import unittest
from collections import Counter
from dataclasses import replace

import numpy as np
from numpy.random import default_rng

from development import Colony, DevelopmentRules, develop, endpoint_wiring, structure_excess, weights
from development_experiment import Assay, assay_ports, trials


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = DevelopmentRules()
        cls.colony = develop(cls.rules, 0, capture=True)

    def test_one_seed_and_wiring_during_growth(self):
        center = self.rules.grid_size // 2
        self.assertEqual(self.colony.frames[0], {"tick": 0, "cells": [[center, center, 0]], "edges": 0})
        self.assertTrue(any(frame["edges"] > 0 for frame in self.colony.frames[:-1]))
        lookup = dict(zip(map(tuple, self.colony.coordinates), self.colony.born))
        for (x, y), tick in lookup.items():
            if tick > 0:
                nearby = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                self.assertTrue(any(lookup.get(site, np.inf) < tick for site in nearby))
        without_signal = develop(replace(self.rules, neighbor_coupling=0.0), 0)
        np.testing.assert_array_equal(self.colony.coordinates, without_signal.coordinates)
        np.testing.assert_array_equal(self.colony.born, without_signal.born)

    def test_endpoint_controls_match_local_edge_quotas(self):
        colony = self.colony
        endpoint = endpoint_wiring(colony, self.rules, default_rng(12))

        def quotas(edges):
            return Counter((source, int(np.sum((colony.coordinates[source] - colony.coordinates[target]) ** 2)))
                           for source, target in edges)

        self.assertEqual(quotas(colony.edges), quotas(endpoint))
        self.assertEqual(len(set(map(tuple, endpoint))), len(endpoint))
        self.assertFalse(np.any(colony.fate[endpoint] == 0))
        self.assertTrue(len(set(map(tuple, colony.edges)) - set(map(tuple, endpoint))) > 0)
        self.assertLessEqual(float(np.abs(weights(colony, endpoint, self.rules)).sum(axis=1).max()), 0.9 + 1e-12)

    def test_structural_shuffle_and_assay_match_definition(self):
        toy = Colony(np.array([[0, 0], [0, 3], [0, 6], [0, 9]]), np.zeros(4, dtype=int),
                     np.array([1, 1, -1, -1]), np.empty((0, 2), dtype=int), [])
        self.assertAlmostEqual(structure_excess(toy, 20), 2 / 5 - 1 / 3)
        a, b, readout = assay_ports(self.colony)
        self.assertEqual(len(set(np.r_[a, b, readout])), len(a) + len(b) + len(readout))
        self.assertFalse(np.any(self.colony.fate[readout] == 0))
        assay = replace(Assay(), noise=0.0)
        batch = trials(len(self.colony.fate), 16, a, b, assay, default_rng(23))
        np.testing.assert_array_equal(batch.labels.reshape(-1, 2).sum(axis=1), 0)
        np.testing.assert_array_equal(batch.drive[0, a[0]] > 0, batch.labels > 0)
        np.testing.assert_array_equal(batch.drive[assay.gap, a[0]] > 0, batch.labels < 0)


if __name__ == "__main__":
    unittest.main()
