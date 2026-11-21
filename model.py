"""Local lattice growth and matched controls for a signed recurrent network."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


EdgeList = NDArray[np.int64]
FloatArray = NDArray[np.float64]
NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0))


@dataclass(frozen=True)
class Colony:
    positions: NDArray[np.int64]
    signs: NDArray[np.int64]
    distance_squared: NDArray[np.int64]
    candidates: EdgeList


def grow_colony(nodes: int, radius_squared: int, rng: Generator) -> Colony:
    """Pick a growth-front cell and let it divide into an empty neighbor."""
    positions = [(0, 0)]
    occupied = set(positions)
    frontier = dict.fromkeys(positions)
    while len(positions) < nodes:
        parent = tuple(frontier)[int(rng.integers(len(frontier)))]
        vacancies = [site for site in adjacent(parent) if site not in occupied]
        child = vacancies[int(rng.integers(len(vacancies)))]
        positions.append(child)
        occupied.add(child)
        if any(point not in occupied for point in adjacent(child)):
            frontier[child] = None
        for site in adjacent(child):
            if site in frontier and all(point in occupied for point in adjacent(site)):
                del frontier[site]

    coordinates = np.asarray(positions, dtype=np.int64)
    signs = np.where(rng.random(nodes) < 0.2, -1, 1)
    offsets = coordinates[:, None, :] - coordinates[None, :, :]
    distance_squared = (offsets ** 2).sum(axis=2)
    candidates = np.argwhere((distance_squared > 0) & (distance_squared <= radius_squared))
    return Colony(coordinates, signs, distance_squared, candidates)


def adjacent(position: tuple[int, int]):
    x, y = position
    return ((x + dx, y + dy) for dx, dy in NEIGHBORS)


def recurrent_matrix(colony: Colony, edges: EdgeList, row_bound: float) -> csr_matrix:
    """Use one weight scale computed from the common pre-pruning graph."""
    strengths = np.where(colony.signs > 0, 1.0, -2.0)
    sources, targets = colony.candidates.T
    row_max = np.bincount(targets, weights=np.abs(strengths[sources])).max()
    sources, targets = edges.T
    weights = strengths[sources] * row_bound / row_max
    shape = (len(colony.signs), len(colony.signs))
    return csr_matrix((weights, (targets, sources)), shape=shape)


def evolve(
    matrix: csr_matrix,
    drive: FloatArray,
    leak: float,
    initial: FloatArray | None = None,
) -> FloatArray:
    """Run the leaky tanh dynamics under a provided time-by-node drive."""
    state = np.zeros(matrix.shape[0]) if initial is None else initial.copy()
    history = np.empty_like(drive)
    for step, signal in enumerate(drive):
        state = (1.0 - leak) * state + leak * np.tanh(matrix @ state + signal)
        history[step] = state
    return history


def prune(
    colony: Colony,
    spontaneous: FloatArray,
    kept_fraction: float,
    rng: Generator,
) -> tuple[EdgeList, EdgeList]:
    """Keep the strongest lagged correlations or sample the same local quota."""
    sources, targets = colony.candidates.T
    previous = standardized(spontaneous[:-1])
    following = standardized(spontaneous[1:])
    correlation = (previous.T @ following) / len(previous)
    scores = np.abs(correlation[sources, targets])
    lengths = colony.distance_squared[sources, targets]
    strata = np.column_stack((sources, lengths))
    # Each source ranks only its outgoing edges of the same length.
    order = np.lexsort((-scores, lengths, sources))
    sorted_strata = strata[order]
    cuts = np.flatnonzero(np.any(np.diff(sorted_strata, axis=0), axis=1)) + 1
    activity_indices = []
    random_indices = []
    for group in np.split(order, cuts):
        quota = max(1, int(np.floor(len(group) * kept_fraction)))
        activity_indices.extend(group[:quota])
        random_indices.extend(rng.choice(group, size=quota, replace=False))
    return colony.candidates[activity_indices], colony.candidates[random_indices]


def standardized(values: FloatArray) -> FloatArray:
    centered = values - values.mean(axis=0)
    return centered / np.maximum(values.std(axis=0), 1e-12)


def rewire(
    colony: Colony,
    edges: EdgeList,
    attempts_per_edge: int,
    rng: Generator,
) -> tuple[EdgeList, float]:
    """Swap directed targets while preserving degrees, signs and lengths."""
    sources, target_array = edges.T
    targets = target_array.tolist()
    node_count = len(colony.signs)
    sources_list = sources.tolist()
    edge_codes = set((sources * node_count + target_array).tolist())
    lengths = colony.distance_squared.tolist()
    max_length = int(colony.distance_squared[tuple(edges.T)].max())

    # A feasible swap has same-sign sources separated by at most two edge radii.
    partners = []
    for node in range(node_count):
        near = colony.distance_squared[node, sources] <= 4 * max_length
        same_sign = colony.signs[sources] == colony.signs[node]
        partners.append(np.flatnonzero(near & same_sign & (sources != node)).tolist())

    attempts = len(edges) * attempts_per_edge
    accepted = 0
    first_edges = rng.integers(len(edges), size=attempts).tolist()
    choices = rng.random(attempts).tolist()
    for first, choice in zip(first_edges, choices):
        source = sources_list[first]
        nearby = partners[source]
        if not nearby:
            continue
        second = nearby[int(choice * len(nearby))]
        other = sources_list[second]
        target, alternate = targets[first], targets[second]
        if target == alternate or source == alternate or other == target:
            continue
        new_first, new_second = source * node_count + alternate, other * node_count + target
        if new_first in edge_codes or new_second in edge_codes:
            continue
        before_a, before_b = lengths[source][target], lengths[other][alternate]
        after_a, after_b = lengths[source][alternate], lengths[other][target]
        if (before_a, before_b) not in ((after_a, after_b), (after_b, after_a)):
            continue
        edge_codes.remove(source * node_count + target)
        edge_codes.remove(other * node_count + alternate)
        edge_codes.update((new_first, new_second))
        targets[first], targets[second] = alternate, target
        accepted += 1

    return np.column_stack((sources, targets)), accepted / attempts


def replaced_fraction(reference: EdgeList, comparison: EdgeList) -> float:
    return 1.0 - len(set(map(tuple, reference)) & set(map(tuple, comparison))) / len(reference)
