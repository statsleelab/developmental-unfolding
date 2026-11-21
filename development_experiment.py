"""Measure spatial order and sample efficiency after synthetic development."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.linalg import solve
from scipy.sparse import csr_matrix

from development import Colony, DevelopmentRules, develop, endpoint_wiring, structure_excess, weights
from experiment import interval
from model import replaced_fraction


PILOT_SEEDS = tuple(range(8))
EVALUATION_SEEDS = tuple(range(2000, 2032))


@dataclass(frozen=True)
class Assay:
    budgets: tuple[int, ...] = (16, 32, 64, 128, 256)
    testing_trials: int = 512
    gap: int = 3
    delay: int = 6
    noise: float = 0.004
    ridge: float = 0.01


@dataclass(frozen=True)
class TrialBatch:
    labels: np.ndarray
    drive: np.ndarray


def assay_ports(colony: Colony) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the two input ports from geometry and record other committed cells."""
    typed = np.flatnonzero(colony.fate != 0)
    positions = colony.coordinates[typed]
    left, right = np.quantile(positions[:, 0], (0.22, 0.78))
    midline = np.median(positions[:, 1])
    a = typed[np.argsort(((positions - (left, midline)) ** 2).sum(axis=1), kind="stable")[:4]]
    available = ~np.isin(typed, a)
    candidates = typed[available]
    b = candidates[np.argsort(((positions[available] - (right, midline)) ** 2).sum(axis=1), kind="stable")[:4]]
    readout = typed[~np.isin(typed, np.r_[a, b])]
    if len(a) != 4 or len(b) != 4 or not len(readout):
        raise ValueError("Too few mature cells for two disjoint inputs and a separate readout")
    return a, b, readout


def trials(
    n_cells: int, n_trials: int, a: np.ndarray, b: np.ndarray, assay: Assay, rng: Generator,
) -> TrialBatch:
    """Use balanced independent trials and keep every even training prefix balanced."""
    labels = np.tile((-1, 1), (n_trials // 2, 1))
    reversed_pairs = rng.random(n_trials // 2) < 0.5
    labels[reversed_pairs] *= -1
    labels = labels.ravel()
    drives = assay.noise * rng.standard_normal((assay.gap + assay.delay + 1, n_cells, n_trials))
    first, second = rng.uniform(0.55, 0.90, (2, n_trials))
    drives[0, a] += first * (labels > 0)
    drives[0, b] += first * (labels < 0)
    drives[assay.gap, a] += second * (labels < 0)
    drives[assay.gap, b] += second * (labels > 0)
    return TrialBatch(labels, drives)


def record(matrix: csr_matrix, batch: TrialBatch, readout: np.ndarray) -> np.ndarray:
    state = np.zeros(batch.drive.shape[1:])
    for signal in batch.drive:
        state = 0.5 * state + 0.5 * np.tanh(matrix @ state + signal)
    return state[readout].T


def learning_curve(
    train: np.ndarray, train_labels: np.ndarray, test: np.ndarray,
    test_labels: np.ndarray, assay: Assay,
) -> np.ndarray:
    accuracy = []
    for budget in assay.budgets:
        seen = train[:budget]
        mean = seen.mean(axis=0)
        scale = np.maximum(seen.std(axis=0), 1e-10)
        design = (seen - mean) / scale
        if len(seen) <= seen.shape[1]:
            kernel = design @ design.T
            coefficients = design.T @ solve(kernel + budget * assay.ridge * np.eye(budget),
                                            train_labels[:budget], assume_a="pos")
        else:
            gram = design.T @ design
            coefficients = solve(gram + budget * assay.ridge * np.eye(seen.shape[1]),
                                 design.T @ train_labels[:budget], assume_a="pos")
        predictions = np.where(((test - mean) / scale) @ coefficients >= 0, 1, -1)
        accuracy.append(np.mean(predictions == test_labels))
    return np.asarray(accuracy)


def run_seed(rules: DevelopmentRules, assay: Assay, seed: int, capture: bool = False) -> tuple[dict, list[dict]]:
    grown = develop(rules, seed, capture)
    signal_off = develop(replace(rules, neighbor_coupling=0.0), seed)
    if not np.array_equal(grown.coordinates, signal_off.coordinates):
        raise RuntimeError("The signal ablation unexpectedly changed growth geometry")
    reference = structure_excess(grown, rules.ticks)
    ablation = structure_excess(signal_off, rules.ticks)
    wiring_rng, train_rng, test_rng = map(default_rng, SeedSequence(seed, spawn_key=(99,)).spawn(3))
    endpoint = endpoint_wiring(grown, rules, wiring_rng)
    a, b, readout = assay_ports(grown)
    train = trials(len(grown.fate), max(assay.budgets), a, b, assay, train_rng)
    test = trials(len(grown.fate), assay.testing_trials, a, b, assay, test_rng)
    graphs = {"growth": grown.edges, "endpoint": endpoint}
    curves = {}
    for key, edges in graphs.items():
        matrix = weights(grown, edges, rules)
        curves[key] = learning_curve(record(matrix, train, readout), train.labels,
                                     record(matrix, test, readout), test.labels, assay)
    uncoupled = csr_matrix((len(grown.fate), len(grown.fate)))
    blank = learning_curve(record(uncoupled, train, readout), train.labels,
                           record(uncoupled, test, readout), test.labels, assay)
    return {"seed": seed, "structure": {"growth": reference, "signal_off": ablation}, "curves": curves,
            "uncoupled": blank, "diagnostics": {"cells": len(grown.fate), "mature_fraction": (grown.fate != 0).mean(),
            "edges": len(grown.edges), "replaced_fraction": replaced_fraction(grown.edges, endpoint),
            "excitatory_fraction": (grown.fate > 0).sum() / (grown.fate != 0).sum()}}, grown.frames


def summarize(runs: list[dict], assay: Assay) -> dict:
    bootstrap = default_rng(63170).integers(len(runs), size=(10000, len(runs)))
    shape = {key: np.asarray([run["structure"][key] for run in runs]) for key in ("growth", "signal_off")}
    shape["difference"] = shape["growth"] - shape["signal_off"]
    curves = {key: np.stack([run["curves"][key] for run in runs]) for key in ("growth", "endpoint")}
    budget_axis = np.log2(assay.budgets)
    areas = np.trapezoid(curves["growth"] - curves["endpoint"], x=budget_axis, axis=1) / np.ptp(budget_axis)
    diagnostics = {key: float(np.mean([run["diagnostics"][key] for run in runs])) for key in runs[0]["diagnostics"]}
    diagnostics["uncoupled_accuracy"] = np.stack([run["uncoupled"] for run in runs]).mean(axis=0).tolist()
    return {"seeds": [run["seed"] for run in runs],
            "structure": {key: interval(values, bootstrap) for key, values in shape.items()},
            "learning": {"curves": {key: interval(values, bootstrap) for key, values in curves.items()},
                         "auc_difference": interval(areas, bootstrap)}, "diagnostics": diagnostics}


def write_table(path: Path, runs: list[dict], assay: Assay) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "label_budget", "growth_accuracy", "endpoint_accuracy", "uncoupled_accuracy",
                         "growth_type_excess", "signal_off_type_excess", *runs[0]["diagnostics"]))
        for run in runs:
            for index, labels in enumerate(assay.budgets):
                writer.writerow((run["seed"], labels, run["curves"]["growth"][index], run["curves"]["endpoint"][index],
                                 run["uncoupled"][index], *run["structure"].values(), *run["diagnostics"].values()))


def plot(path: Path, summary: dict, assay: Assay) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.1, 4.5), layout="constrained")
    for key, label, color, marker in (("growth", "Developmental wiring", "#24725e", "o"),
                                       ("endpoint", "Direct endpoint wiring", "#a0522d", "s")):
        curve = summary["learning"]["curves"][key]
        axis.plot(assay.budgets, curve["mean"], marker=marker, color=color, label=label)
        axis.fill_between(assay.budgets, curve["low"], curve["high"], color=color, alpha=0.17)
    axis.axhline(0.5, linestyle="--", color="#68746d", label="Chance")
    axis.set(title="Delayed cue-order classification", xlabel="Labeled training trials", ylabel="Test accuracy",
             xscale="log", xticks=assay.budgets, xticklabels=assay.budgets)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    rules, assay = DevelopmentRules(), Assay()
    pilots = [run_seed(rules, assay, seed)[0] for seed in PILOT_SEEDS]
    pilot = summarize(pilots, assay)
    print("Pilot", json.dumps({"structure": pilot["structure"], "learning": pilot["learning"]}), flush=True)
    if args.pilot_only:
        return
    runs, example = [], []
    for seed in EVALUATION_SEEDS:
        result, frames = run_seed(rules, assay, seed, capture=seed == EVALUATION_SEEDS[0])
        runs.append(result)
        example.extend(frames)
    evaluation = summarize(runs, assay)
    payload = {"config": {**asdict(rules), **asdict(assay)}, "pilot": pilot, "evaluation": evaluation,
               "example": {"seed": EVALUATION_SEEDS[0], "frames": example}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "development-results.json").write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    write_table(args.output / "development-seeds.csv", runs, assay)
    plot(args.output / "development-learning.png", evaluation, assay)
    print("Evaluation", json.dumps({"structure": evaluation["structure"], "learning": evaluation["learning"],
                                    "diagnostics": evaluation["diagnostics"]}), flush=True)


if __name__ == "__main__":
    main()
