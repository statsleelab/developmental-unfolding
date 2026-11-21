"""Shared local rules for a genome-inspired synthetic developmental model."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.sparse import csr_matrix

from model import EdgeList


@dataclass(frozen=True)
class DevelopmentRules:
    grid_size: int = 31
    ticks: int = 58
    division_probability: float = 0.24
    regulatory_feedback: float = 0.24
    neighbor_coupling: float = 0.18
    regulatory_noise: float = 0.24
    commitment_threshold: float = 0.72
    wire_radius_squared: int = 8
    out_degree_cap: int = 6
    wire_probability: float = 0.65
    same_type_preference: float = 2.0
    pruning_threshold: float = 0.005


@dataclass(frozen=True)
class Colony:
    coordinates: np.ndarray
    born: np.ndarray
    fate: np.ndarray
    edges: EdgeList
    frames: list[dict]


def neighborhoods(side: int, radius_squared: int) -> np.ndarray:
    radius = int(np.sqrt(radius_squared))
    offsets = [(dx, dy) for dx in range(-radius, radius + 1)
               for dy in range(-radius, radius + 1) if 0 < dx * dx + dy * dy <= radius_squared]
    positions = np.indices((side, side)).reshape(2, -1).T
    candidates = positions[:, None, :] + np.array(offsets)
    inside = np.all((candidates >= 0) & (candidates < side), axis=2)
    x, y = np.moveaxis(candidates, -1, 0)
    return np.where(inside, x * side + y, side ** 2)


def births(
    born: np.ndarray, energy: np.ndarray, neighbors: np.ndarray, tick: int,
    rules: DevelopmentRules, rng: Generator,
) -> None:
    """Propose divisions by local vacancy and resources; resolve target collisions."""
    sites = np.flatnonzero((born[:-1] >= 0) & (tick - born[:-1] >= 3) & (energy >= 0.9))
    selected = sites[rng.random(len(sites)) < rules.division_probability]
    vacancies = (born[neighbors[selected]] < 0) & (neighbors[selected] != len(energy))
    rankings = np.where(vacancies, rng.random(vacancies.shape), np.inf)
    attempts = np.flatnonzero(vacancies.any(axis=1))
    targets = neighbors[selected[attempts], rankings[attempts].argmin(axis=1)]
    # Independent proposal priorities resolve two cells attempting the same vacancy.
    order = rng.permutation(len(targets))
    children, first = np.unique(targets[order], return_index=True)
    parents = selected[attempts[order[first]]]
    energy[parents] -= 0.45
    energy[children] = 0.45
    born[children] = tick


def differentiate(
    regulator: np.ndarray, fate: np.ndarray, born: np.ndarray,
    neighbors: np.ndarray, tick: int, rules: DevelopmentRules, rng: Generator,
) -> None:
    """Receive the average local signal and commit when an internal threshold is crossed."""
    active = np.flatnonzero((born[:-1] >= 0) & (fate[:-1] == 0))
    adjacent = neighbors[active]
    occupancy = born[adjacent] >= 0
    state = regulator[active]
    mean = np.sum(regulator[adjacent] * occupancy, axis=1) / np.maximum(1, occupancy.sum(axis=1))
    coupling = rules.neighbor_coupling * (mean - state) * occupancy.any(axis=1)
    noise = rng.standard_normal(len(born) - 1)[active]
    updated = state + rules.regulatory_feedback * state * (1 - state ** 2) + coupling + rules.regulatory_noise * noise
    regulator[active] = np.clip(updated, -1, 1)
    committed = active[(np.abs(updated) >= rules.commitment_threshold) & (tick - born[active] >= 3)]
    fate[committed] = np.sign(regulator[committed])
    regulator[committed] = fate[committed]


def develop(rules: DevelopmentRules, seed: int, capture: bool = False) -> Colony:
    side = rules.grid_size
    count = side ** 2
    neighbors = neighborhoods(side, 1)
    reach = neighborhoods(side, rules.wire_radius_squared)
    growth_rng, regulation_rng, wiring_rng, activity_rng = map(default_rng, SeedSequence(seed).spawn(4))
    born = np.full(count + 1, -1, dtype=int)
    center = (side // 2) * side + side // 2
    born[center] = 0
    regulator = np.zeros(count + 1)
    fate = np.zeros(count + 1, dtype=int)
    energy = np.zeros(count)
    energy[center] = 1.0
    activity = np.zeros(count + 1)
    formed_at = np.full(reach.shape, -1, dtype=int)
    trace = np.zeros(reach.shape)
    frames = []

    def save_frame(tick: int) -> None:
        sites = np.flatnonzero(born >= 0)
        cells = np.column_stack((sites // side, sites % side, fate[sites])).tolist()
        frames.append({"tick": tick, "cells": cells, "edges": int((formed_at >= 0).sum())})

    if capture:
        save_frame(0)
    for tick in range(1, rules.ticks + 1):
        living = born[:-1] >= 0
        energy[living] = np.minimum(1.0, energy[living] + 0.26)
        births(born, energy, neighbors, tick, rules, growth_rng)
        differentiate(regulator, fate, born, neighbors, tick, rules, regulation_rng)

        sources, slots = np.where(formed_at >= 0)
        targets = reach[sources, slots]
        weights = 0.9 * fate[sources] / reach.shape[1]
        recurrent = np.bincount(targets, weights * activity[sources], minlength=count)
        previous = activity.copy()
        spontaneous = 0.20 * activity_rng.standard_normal(count)
        activity[:-1] = (0.5 * activity[:-1] + 0.5 * np.tanh(recurrent + spontaneous)) * (fate[:-1] != 0)
        trace[sources, slots] = 0.82 * trace[sources, slots] + 0.18 * np.abs(previous[sources] * activity[targets])
        remove = (formed_at >= 0) & (tick - formed_at >= 9) & (trace < rules.pruning_threshold)
        formed_at[remove] = -1
        trace[remove] = 0

        typed = fate[:-1] != 0
        capacity = np.sum(formed_at >= 0, axis=1) < rules.out_degree_cap
        requested = typed & capacity & (wiring_rng.random(count) < rules.wire_probability)
        eligible = requested[:, None] & (fate[reach] != 0) & (formed_at < 0)
        preference = np.where(fate[:-1, None] == fate[reach], rules.same_type_preference, 1.0)
        rankings = np.where(eligible, -np.log(np.maximum(wiring_rng.random(reach.shape), 1e-15)) / preference, np.inf)
        sources = np.flatnonzero(eligible.any(axis=1))
        formed_at[sources, rankings[sources].argmin(axis=1)] = tick

        if capture and (tick % 4 == 0 or tick == rules.ticks):
            save_frame(tick)

    sites = np.flatnonzero(born >= 0)
    local_indices = np.full(count, -1, dtype=int)
    local_indices[sites] = np.arange(len(sites))
    sources, slots = np.where(formed_at >= 0)
    edges = np.column_stack((local_indices[sources], local_indices[reach[sources, slots]]))
    return Colony(np.column_stack((sites // side, sites % side)), born[sites], fate[sites], edges, frames)


def endpoint_wiring(colony: Colony, rules: DevelopmentRules, rng: Generator) -> EdgeList:
    """Sample directly at maturity using the same type preference and exact local quotas."""
    offsets = colony.coordinates[:, None, :] - colony.coordinates[None, :, :]
    distance = (offsets ** 2).sum(axis=2)
    chosen = []
    for source in range(len(colony.fate)):
        targets = colony.edges[colony.edges[:, 0] == source, 1]
        lengths, quotas = np.unique(distance[source, targets], return_counts=True)
        for length, quota in zip(lengths, quotas):
            candidates = np.flatnonzero((distance[source] == length) & (colony.fate != 0))
            probability = np.where(colony.fate[candidates] == colony.fate[source], rules.same_type_preference, 1.0)
            selected = rng.choice(candidates, size=quota, replace=False, p=probability / probability.sum())
            chosen.extend((source, target) for target in selected)
    return np.asarray(chosen, dtype=np.int64).reshape(-1, 2)


def weights(colony: Colony, edges: EdgeList, rules: DevelopmentRules) -> csr_matrix:
    radius = int(np.sqrt(rules.wire_radius_squared))
    possible = sum(0 < x * x + y * y <= rules.wire_radius_squared
                   for x in range(-radius, radius + 1) for y in range(-radius, radius + 1))
    source, target = edges.T
    return csr_matrix((0.9 * colony.fate[source] / possible, (target, source)),
                      shape=(len(colony.fate), len(colony.fate)))


def structure_excess(colony: Colony, ticks: int) -> float:
    """Compare typed pairs 3–7 steps apart against a stratified type permutation."""
    position = colony.coordinates
    distance = np.abs(position[:, None, :] - position[None, :, :]).sum(axis=2)
    boundary = (distance == 1).sum(axis=1) < 4
    strata = 2 * (colony.born // max(1, ticks // 4)) + boundary
    typed = colony.fate != 0
    source, target = np.where(np.triu((distance >= 3) & (distance <= 7) & typed[:, None] & typed, k=1))
    if not len(source):
        raise ValueError("No differentiated cell pairs in the structural assay")
    counts = np.bincount(strata[typed], minlength=strata.max() + 1)
    positive = np.bincount(strata[colony.fate > 0], minlength=len(counts))
    first, second = strata[source], strata[target]
    p = positive[first] / counts[first]
    q = positive[second] / counts[second]
    expected = p * q + (1 - p) * (1 - q)
    same = first == second
    plus, total = positive[first[same]], counts[first[same]]
    minus = total - plus
    expected[same] = (plus * (plus - 1) + minus * (minus - 1)) / (total * (total - 1))
    return float((colony.fate[source] == colony.fate[target]).mean() - expected.mean())
