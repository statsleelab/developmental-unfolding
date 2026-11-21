"""Run the paired development experiment and export the GitHub Pages data."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.sparse import csr_matrix

from model import FloatArray, evolve, grow_colony, prune, recurrent_matrix, replaced_fraction, rewire


CONDITIONS = ("activity", "random", "rewired")
PILOT_SEEDS = tuple(range(8))
EVALUATION_SEEDS = tuple(range(1000, 1032))


@dataclass(frozen=True)
class Config:
    nodes: int = 256
    radius_squared: int = 8
    kept_fraction: float = 0.5
    development_steps: int = 512
    split_steps: int = 2200
    washout: int = 96
    delays: tuple[int, ...] = (2, 4, 8, 12, 16)
    leak: float = 0.5
    input_sites: int = 16
    input_gain: float = 0.22
    recurrent_row_bound: float = 0.9
    spontaneous_noise: float = 0.18
    swap_attempts_per_edge: int = 320
    bootstrap_samples: int = 10000
    ridge_penalties: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    scores: dict[str, FloatArray]
    penalties: dict[str, float]
    diagnostics: dict[str, float]


def r_squared(expected: FloatArray, predicted: FloatArray) -> FloatArray:
    residual = ((expected - predicted) ** 2).sum(axis=0)
    total = ((expected - expected.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - residual / total


def delayed_states(
    matrix: csr_matrix, bits: FloatArray, injection: FloatArray, config: Config
) -> tuple[FloatArray, FloatArray]:
    drive = bits[:, None] * injection
    states = evolve(matrix, drive, config.leak)
    times = np.arange(config.washout, len(bits))
    targets = bits[times[:, None] - np.asarray(config.delays)]
    return states[times], targets


def fit_readout(
    train: tuple[FloatArray, FloatArray],
    validate: tuple[FloatArray, FloatArray],
    test: tuple[FloatArray, FloatArray],
    penalties: tuple[float, ...],
) -> tuple[FloatArray, float]:
    train_states, train_targets = train
    center = train_states.mean(axis=0)
    scale = np.maximum(train_states.std(axis=0), 1e-10)
    train_states = (train_states - center) / scale
    target_mean = train_targets.mean(axis=0)
    covariance = train_states.T @ train_states / len(train_states)
    eigenvalues, basis = np.linalg.eigh(covariance)
    projections = basis.T @ (train_states.T @ (train_targets - target_mean)) / len(train_states)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    validate_states, validate_targets = validate
    validate_basis = ((validate_states - center) / scale) @ basis
    best_score = -np.inf
    best_penalty = penalties[0]
    for penalty in penalties:
        weights = projections / (eigenvalues[:, None] + penalty)
        score = r_squared(validate_targets, validate_basis @ weights + target_mean).mean()
        if score > best_score:
            best_score, best_penalty = score, penalty
    weights = projections / (eigenvalues[:, None] + best_penalty)
    test_states, test_targets = test
    prediction = (((test_states - center) / scale) @ basis) @ weights + target_mean
    return r_squared(test_targets, prediction), best_penalty


def bit_streams(length: int, sequence: SeedSequence) -> tuple[FloatArray, ...]:
    return tuple(
        2.0 * default_rng(child).integers(2, size=length) - 1.0
        for child in sequence.spawn(3)
    )


def input_weights(config: Config, rng: Generator) -> FloatArray:
    injection = np.zeros(config.nodes)
    sites = rng.choice(config.nodes, size=config.input_sites, replace=False)
    injection[sites] = config.input_gain * rng.choice((-1, 1), size=len(sites))
    return injection


def run_seed(config: Config, seed: int) -> SeedResult:
    seeds = SeedSequence(seed).spawn(6)
    growth_rng, noise_rng, pruning_rng, rewire_rng, input_rng = map(default_rng, seeds[:5])
    colony = grow_colony(config.nodes, config.radius_squared, growth_rng)
    precursor = recurrent_matrix(colony, colony.candidates, config.recurrent_row_bound)
    noise = noise_rng.normal(0.0, config.spontaneous_noise, (config.development_steps, config.nodes))
    spontaneous = evolve(precursor, noise, config.leak)
    activity, random = prune(colony, spontaneous, config.kept_fraction, pruning_rng)
    rewired, acceptance = rewire(colony, activity, config.swap_attempts_per_edge, rewire_rng)
    graphs = dict(zip(CONDITIONS, (activity, random, rewired)))
    injection = input_weights(config, input_rng)
    streams = bit_streams(config.split_steps + config.washout, seeds[5])
    washout_drive = streams[0][:config.washout, None] * injection
    starts = np.ones(config.nodes)
    scores, penalties = {}, {}
    stability = []
    for condition, edges in graphs.items():
        matrix = recurrent_matrix(colony, edges, config.recurrent_row_bound)
        recordings = [delayed_states(matrix, bits, injection, config) for bits in streams]
        scores[condition], penalties[condition] = fit_readout(*recordings, config.ridge_penalties)
        end_positive = evolve(matrix, washout_drive, config.leak, starts)[-1]
        end_negative = evolve(matrix, washout_drive, config.leak, -starts)[-1]
        stability.append(float(np.max(np.abs(end_positive - end_negative)) / 2.0))

    costs = [np.sqrt(colony.distance_squared[tuple(edges.T)]).sum() for edges in graphs.values()]
    if max(costs) - min(costs) > 1e-8:
        raise RuntimeError("Control wiring lengths differ from the activity-pruned graph")
    return SeedResult(seed, scores, penalties, {
        "edge_count": float(len(activity)),
        "wiring_length": float(costs[0]),
        "random_coverage": replaced_fraction(activity, random),
        "rewire_coverage": replaced_fraction(activity, rewired),
        "rewire_acceptance": acceptance,
        "stability_max": max(stability),
    })


def interval(values: FloatArray, samples: np.ndarray) -> dict:
    centers = values.mean(axis=0)
    low, high = np.quantile(values[samples].mean(axis=1), (0.025, 0.975), axis=0)
    return {"mean": centers.tolist(), "low": low.tolist(), "high": high.tolist()}


def summarize(results: list[SeedResult], bootstrap_samples: int) -> dict:
    rng = default_rng(51019)
    indices = rng.integers(len(results), size=(bootstrap_samples, len(results)))
    scores = {condition: np.stack([run.scores[condition] for run in results]) for condition in CONDITIONS}
    scores_by_seed = {condition: curves.mean(axis=1) for condition, curves in scores.items()}
    contrasts = {
        f"activity_minus_{control}": interval(scores_by_seed["activity"] - scores_by_seed[control], indices)
        for control in ("random", "rewired")
    }
    diagnostics = {}
    for name in results[0].diagnostics:
        measurements = [run.diagnostics[name] for run in results]
        aggregate = np.max if name == "stability_max" else np.mean
        diagnostics[name] = float(aggregate(measurements))
    return {
        "seeds": [run.seed for run in results],
        "curves": {condition: interval(curve, indices) for condition, curve in scores.items()},
        "contrasts": contrasts,
        "diagnostics": diagnostics,
    }


def write_seed_results(path: Path, cohorts: dict[str, list[SeedResult]], config: Config) -> None:
    diagnostics = list(next(iter(cohorts.values()))[0].diagnostics)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["cohort", "seed", "condition", "lag", "test_r2", "ridge_penalty", *diagnostics])
        for cohort, results in cohorts.items():
            for run in results:
                for condition in CONDITIONS:
                    for lag, score in zip(config.delays, run.scores[condition]):
                        writer.writerow([cohort, run.seed, condition, lag, score, run.penalties[condition],
                                         *(run.diagnostics[key] for key in diagnostics)])


def plot_memory(path: Path, delays: tuple[int, ...], evaluation: dict) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "activity": ("Activity pruning", "#007F70", "o", "-"),
        "random": ("Random pruning", "#555B8C", "s", "--"),
        "rewired": ("Matched rewiring", "#AD5217", "^", "-."),
    }
    figure, axes = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for condition, (label, color, marker, line) in styles.items():
        curve = evaluation["curves"][condition]
        axes.fill_between(delays, curve["low"], curve["high"], color=color, alpha=0.12)
        axes.plot(delays, curve["mean"], color=color, label=label, marker=marker, linestyle=line)
    axes.set(title=f"Delayed bit reconstruction ({len(evaluation['seeds'])} held-out seeds)",
             xlabel="Delay (time steps)", ylabel="Held-out $R^2$", xticks=delays)
    axes.axhline(0, color="#A0A0A0", linewidth=0.8)
    axes.grid(axis="y", alpha=0.22)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--pilot-only", action="store_true", help="Run only the development seeds")
    args = parser.parse_args()
    config = Config()
    cohorts = {"pilot": [run_seed(config, seed) for seed in PILOT_SEEDS]}
    print(f"Finished {len(cohorts['pilot'])} pilot seeds", flush=True)
    if not args.pilot_only:
        cohorts["evaluation"] = [run_seed(config, seed) for seed in EVALUATION_SEEDS]
        print(f"Finished {len(cohorts['evaluation'])} evaluation seeds", flush=True)
    summaries = {cohort: summarize(results, config.bootstrap_samples) for cohort, results in cohorts.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    data = {"config": asdict(config), **summaries}
    if args.pilot_only:
        summary_path = args.output / "pilot-results.json"
        seed_path = args.output / "pilot-seed-results.csv"
    else:
        summary_path = args.output / "results.json"
        seed_path = args.output / "seed-results.csv"
    summary_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_seed_results(seed_path, cohorts, config)
    if not args.pilot_only:
        plot_memory(args.output / "memory-curve.png", config.delays, summaries["evaluation"])


if __name__ == "__main__":
    main()
