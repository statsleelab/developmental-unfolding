"""Reproduce the published spatial connectome model for electrical synapses."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.linalg import eigh, solve
from scipy.optimize import minimize_scalar


DOI = "https://doi.org/10.5281/zenodo.4027588"
RELEASE_URL = "https://zenodo.org/api/records/4027588/files/kpisti/SCM-v1.0.zip/content"
SHA256 = "39092151a16d82f23d51371638db4871b8f24828c8f8ddd2ee20764a5c1ca92c"
ALPHA = 0.215
Z_THRESHOLD = 2.0
ITERATIONS = 100

# Match the candidate graph and rounded z scores in the text accompanying Fig. 5.
# Columns 3 and 11 have identical observed expression and cannot be distinguished
# by this fit. Column 17 does not occur in the published positive candidate graph.
LABELS = ("inx-1", "inx-10", "inx-5 or inx-11", "inx-12", "inx-13", "inx-17", "inx-18",
          "inx-19", "inx-2", "inx-3", "inx-5 or inx-11", "inx-6", "inx-7", "inx-8",
          "inx-9", "che-7", "Unnamed column 17", "unc-9")
REFERENCES = ("inx-19 / inx-19", "unc-9 / unc-9", "inx-10 / inx-11", "inx-3 / inx-3", "inx-6 / inx-6")


def open_release(path: Path) -> ZipFile:
    if not path.exists():
        with urlopen(RELEASE_URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("Downloaded archive does not match the pinned release")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Cached archive does not match the pinned release")
    return ZipFile(path)


def read_csv(archive: ZipFile, name: str) -> np.ndarray:
    with archive.open(f"kpisti-SCM-6cc8731/{name}") as stream:
        return np.loadtxt(stream, delimiter=",")


def null_probabilities(mask: np.ndarray, edges: np.ndarray, iterations: int) -> np.ndarray:
    """Author algorithm using inverse node factors and off-diagonal degrees."""
    degree = edges.sum(axis=1) - edges.diagonal()
    active = np.flatnonzero(degree > 0)
    target = degree[active]
    allowed = mask[np.ix_(active, active)].copy()
    np.fill_diagonal(allowed, False)
    inverse = 2 * target / allowed.sum(axis=1)
    for _ in range(iterations):
        inverse = target / np.sum(allowed / (inverse[:, None] + 1.0 / inverse[None, :]), axis=1)
    probabilities = np.zeros_like(edges)
    probabilities[np.ix_(active, active)] = allowed / (1.0 + 1.0 / inverse[:, None] / inverse[None, :])
    return probabilities


def ridge_gcv(design: np.ndarray, observed: np.ndarray) -> float:
    eigenvalues, vectors = eigh(design.T @ design)
    projected = vectors.T @ design.T @ observed
    baseline = observed @ observed

    def score(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        denominator = eigenvalues + alpha
        residual = baseline - np.sum(projected ** 2 * (eigenvalues + 2 * alpha) / denominator ** 2)
        residual_df = len(observed) - np.sum(eigenvalues / denominator)
        return float(residual / residual_df ** 2)

    optimum = minimize_scalar(score, bounds=(-6, 4), method="bounded")
    if not optimum.success:
        raise RuntimeError("Continuous generalized cross-validation search failed")
    return float(np.exp(optimum.x))


def scores(weights: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    return np.divide(weights - mean, np.sqrt(variance), out=np.zeros_like(weights), where=variance > 0)


def candidates(weights: np.ndarray, z: np.ndarray, size: int) -> np.ndarray:
    return np.triu(((weights > 0) & (z >= Z_THRESHOLD)).reshape(size, size))


def reproduce(archive: ZipFile) -> dict:
    mask = read_csv(archive, "ContactSubgraphMatrix.csv").astype(bool)
    edges = read_csv(archive, "GapJunctContact.csv")
    alternate = read_csv(archive, "CEO_Gap_OnlyContact.csv")
    expression = read_csv(archive, "INXExpressionJustContact.csv")
    if np.any(edges[~mask]) or not np.array_equal(mask, mask.T) or not np.array_equal(edges, edges.T):
        raise ValueError("Expected a symmetric Cook graph supported on the contact mask")
    rows, columns = np.nonzero(mask)
    size = expression.shape[1]
    design = (expression[rows, :, None] * expression[columns, None, :]).reshape(len(rows), -1)
    operator = solve(design.T @ design + ALPHA * np.eye(design.shape[1]), design.T, assume_a="pos")

    probabilities = null_probabilities(mask, edges, ITERATIONS)
    # C++'s default output precision in the author's release is six significant digits.
    null = np.asarray([float(f"{p:.6g}") for p in probabilities[mask]])
    observed = edges[mask]
    weights = operator @ observed
    expected = operator @ null
    variance = (operator ** 2) @ (null * (1.0 - null))
    z = scores(weights, expected, variance)
    positives = candidates(weights, z, size)

    # The two matrix entries for an unordered electrical contact share one draw.
    forward = np.flatnonzero(rows < columns)
    index = np.full(mask.shape, -1, dtype=int)
    index[mask] = np.arange(len(rows))
    backward = index[columns[forward], rows[forward]]
    p = null[forward]
    paired_variance = (operator[:, forward] + operator[:, backward]) ** 2 @ (p * (1.0 - p))
    symmetric_z = scores(weights, expected, paired_variance)
    symmetric_positives = candidates(weights, symmetric_z, size)

    alternate_null = null_probabilities(mask, alternate, ITERATIONS)[mask]
    alternate_weights = operator @ alternate[mask]
    alternate_z = scores(alternate_weights, operator @ alternate_null,
                         operator ** 2 @ (alternate_null * (1.0 - alternate_null)))
    alternate_positives = candidates(alternate_weights, alternate_z, size)

    references = {
        "rule_weights": (weights, read_csv(archive, "rule_matrix_0.2150.csv")),
        "null_rule_mean": (expected, read_csv(archive, "rule_matrix_0.2150rnd_av.csv")),
        "null_rule_variance": (variance, read_csv(archive, "rule_matrix_0.2150rnd_var.csv")),
        "null_edge_probability": (null, read_csv(archive, "alpha_exp_GapJunctContact.txt_ContactSubgraphMatrix.txt_100.csv")[mask]),
    }
    discrepancies = {name: float(np.max(np.abs(calculated - original.ravel())))
                     for name, (calculated, original) in references.items()}
    if max(discrepancies.values()) > 1e-9:
        raise RuntimeError(f"Numerical reproduction differs from archived outputs: {discrepancies}")

    published_rules = []
    for i, j in zip(*np.where(positives)):
        direction = i * size + j
        published_rules.append({"column_a": int(i + 1), "column_b": int(j + 1), "label_a": LABELS[i],
                                "label_b": LABELS[j], "weight": float(weights[direction]),
                                "z": float(z[direction]), "symmetric_null_z": float(symmetric_z[direction]),
                                "varshney_z_same_alpha": float(alternate_z[direction])})
    published_rules.sort(key=lambda pair: (-pair["z"], pair["column_a"], pair["column_b"]))

    return {"source": DOI, "archive_sha256": SHA256,
            "paper": "https://doi.org/10.1073/pnas.2009093117", "alpha": ALPHA, "z_threshold": Z_THRESHOLD,
            "continuous_gcv_minimum_alpha": ridge_gcv(design, observed), "null_iterations": ITERATIONS,
            "neurons": int(expression.shape[0]), "innexin_columns": size,
            "off_diagonal_contact_pairs": int(np.triu(mask, 1).sum()), "self_contacts": int(mask.diagonal().sum()),
            "off_diagonal_gap_junction_pairs": int(np.triu(edges, 1).sum()), "self_gap_junctions": int(edges.diagonal().sum()),
            "cook_positive_candidates": int(positives.sum()), "paper_prior_experimental_interactions": list(REFERENCES),
            "expression_columns_indistinguishable": [3, 11],
            "symmetric_null_positive_candidates": int(symmetric_positives.sum()),
            "varshney_positive_candidates_same_alpha": int(alternate_positives.sum()),
            "varshney_overlap_with_cook_same_alpha": int((positives & alternate_positives).sum()),
            "maximum_absolute_archive_differences": discrepancies, "candidates": published_rules}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/tmp/innexin/release.zip"))
    parser.add_argument("--output", type=Path, default=Path("docs/innexin-reproduction.json"))
    args = parser.parse_args()
    with open_release(args.archive) as archive:
        results = reproduce(archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Cook candidates: {results['cook_positive_candidates']}. "
          f"Symmetric null variance: {results['symmetric_null_positive_candidates']}.")


if __name__ == "__main__":
    main()
