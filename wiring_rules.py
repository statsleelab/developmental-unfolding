"""Test compact, named gene-pair rules for directed chemical synapses."""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import pinvh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler

from connectome_experiment import contact_features, dataset, subtype_pair_folds


PANEL = (
    "syg-1", "syg-2", "unc-6", "unc-40", "clr-1", "nrx-1", "nlg-1", "madd-4", "bam-2", "casy-1",
    "sax-7", "rig-6", "sax-3", "unc-5", "rig-1", "rig-3", "ncam-1", "cdh-4", "ptp-3", "cam-1",
)
BUDGETS = (5, 10, 20)
MODELS = ("baseline", "named_main", *(f"named_pairs_{k}" for k in BUDGETS))
SCHEMES = ("held_specimen", "held_specimen_and_subtype_pair", "early_to_later")


def named_expression(cache: Path, profiles: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    symbols = pd.read_csv(cache / "genes.csv.gz").gene_short_name.to_numpy()
    counts = np.log1p(profiles / profiles.sum(axis=1, keepdims=True) * 1e6)
    indices, used, dropped = [], [], []
    for gene in PANEL:
        matching = np.flatnonzero(symbols == gene)
        if len(matching) == 1 and np.ptp(counts[:, matching[0]]) > 0:
            indices.append(int(matching[0]))
            used.append(gene)
        else:
            dropped.append(gene)
    return counts[:, indices], used, dropped


def designs(data: pd.DataFrame, expression: np.ndarray, names: list[str]) -> tuple[np.ndarray, ...]:
    _, baseline = contact_features(data)
    lookup = {name: i for i, name in enumerate(names)}
    pre = expression[data.source_subtype.map(lookup).to_numpy()]
    post = expression[data.target_subtype.map(lookup).to_numpy()]
    named_main = np.column_stack((baseline, pre, post))
    interactions = (pre[:, :, None] * post[:, None, :]).reshape(len(data), -1)
    return baseline, named_main, interactions


def fit(design: np.ndarray, train: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scale = StandardScaler().fit(design[train])
    model = LogisticRegression(C=0.1, max_iter=1000).fit(scale.transform(design[train]), y[train])
    return scale, model


def rank_interactions(main: np.ndarray, interactions: np.ndarray, y: np.ndarray,
                      p: np.ndarray) -> np.ndarray:
    """Rank interactions by the conditional score after projecting out main effects."""
    nuisance = np.column_stack((np.ones(len(main)), main))
    variance = p * (1.0 - p)
    weighted = nuisance.T * variance
    cross = weighted @ interactions
    projection = pinvh(weighted @ nuisance) @ cross
    residual = y - p
    score = residual @ interactions - (residual @ nuisance) @ projection
    total = variance @ (interactions ** 2)
    information = total - np.sum(cross * projection, axis=0)
    usable = information > np.maximum(1e-8, total * 1e-9)
    scaled = np.divide(np.abs(score), np.sqrt(np.maximum(0.0, information)),
                       out=np.full_like(score, -np.inf), where=usable)
    if usable.sum() < max(BUDGETS):
        raise ValueError("Too few variable interactions in the training fold")
    return np.argsort(-scaled, kind="stable")[:max(BUDGETS)]


def splits(animals: np.ndarray, partitions: np.ndarray, scheme: str):
    if scheme == "early_to_later":
        early = np.isin(animals, ("Dataset1", "Dataset2", "Dataset3"))
        yield early, ~early
        return
    for specimen in sorted(set(animals)):
        held = animals == specimen
        if scheme == "held_specimen":
            yield ~held, held
        else:
            for group in range(5):
                yield ~held & (partitions != group), held & (partitions == group)


def evaluate(data: pd.DataFrame, design: tuple, genes: list[str]) -> tuple[dict, list, dict]:
    baseline, main, interactions = design
    animals, y = data.specimen.to_numpy(), data.synapse.to_numpy()
    partitions = subtype_pair_folds(data)
    candidates = list(product(genes, repeat=2))
    collected = defaultdict(list)
    results, prediction_sets = {}, {}
    for scheme in SCHEMES:
        scores = {name: np.full(len(data), np.nan) for name in MODELS}
        transferred = []
        evaluated = np.zeros(len(data), dtype=bool)
        for train, test in splits(animals, partitions, scheme):
            if not test.any():
                continue
            if np.any(train & test) or np.any(evaluated & test):
                raise RuntimeError(f"Overlapping train/test assignments in {scheme}")
            evaluated |= test
            base_scale, base_model = fit(baseline, train, y)
            scores["baseline"][test] = base_model.predict_proba(base_scale.transform(baseline[test]))[:, 1]
            main_scale, main_model = fit(main, train, y)
            train_design = main_scale.transform(main[train])
            train_probability = main_model.predict_proba(train_design)[:, 1]
            scores["named_main"][test] = main_model.predict_proba(main_scale.transform(main[test]))[:, 1]
            selected = rank_interactions(train_design, interactions[train], y[train], train_probability)
            for k in BUDGETS:
                extension = np.column_stack((main, interactions[:, selected[:k]]))
                scale, model = fit(extension, train, y)
                scores[f"named_pairs_{k}"][test] = model.predict_proba(scale.transform(extension[test]))[:, 1]
                if k == 10:
                    for index, coefficient in zip(selected[:k], model.coef_[0, main.shape[1]:]):
                        source, target = candidates[index]
                        if scheme == "held_specimen":
                            collected[source, target].append(float(coefficient))
                        elif scheme == "early_to_later":
                            transferred.append({"source_gene": source, "target_gene": target,
                                                "coefficient": float(coefficient)})
        rows = []
        for specimen in sorted(set(animals[evaluated])):
            held = animals == specimen
            if any(not np.isfinite(score[held]).all() for score in scores.values()):
                raise RuntimeError(f"Missing held-out predictions for {scheme} {specimen}")
            rows.append({"specimen": specimen, "scores": {name: {
                "ap": float(average_precision_score(y[held], predicted[held])),
                "log_loss": float(log_loss(y[held], predicted[held])),
            } for name, predicted in scores.items()}})
        results[scheme] = {"specimens": rows, "mean": {name: {
            metric: float(np.mean([row["scores"][name][metric] for row in rows]))
            for metric in ("ap", "log_loss")
        } for name in MODELS}}
        if transferred:
            results[scheme]["rules"] = transferred
        prediction_sets[scheme] = scores
        print(scheme, results[scheme]["mean"], flush=True)
    rules = [{"source_gene": source, "target_gene": target, "selected_folds": len(values),
              "positive_folds": sum(value > 0 for value in values),
              "negative_folds": sum(value < 0 for value in values), "mean_coefficient": float(np.mean(values))}
             for (source, target), values in collected.items()]
    rules.sort(key=lambda rule: (-rule["selected_folds"], -abs(rule["mean_coefficient"]),
                                 rule["source_gene"], rule["target_gene"]))
    return results, rules, prediction_sets


def motifs(matrix: np.ndarray) -> dict:
    outgoing = matrix.sum(axis=1)
    incoming = matrix.sum(axis=0)
    return {
        "reciprocal": int((matrix * matrix.T).sum() // 2),
        "divergent": int((outgoing * (outgoing - 1)).sum() // 2),
        "convergent": int((incoming * (incoming - 1)).sum() // 2),
        "feedforward": int(((matrix @ matrix) * matrix).sum()),
    }


def adult_comparison(data: pd.DataFrame, scores: dict) -> tuple[dict, dict]:
    adult = data.specimen == "Dataset8"
    rows = data.loc[adult]
    nodes = sorted(set(rows.source) | set(rows.target))
    lookup = {node: i for i, node in enumerate(nodes)}
    indices = rows.source.map(lookup).to_numpy(), rows.target.map(lookup).to_numpy()
    actual = rows.synapse.to_numpy()
    edges = int(actual.sum())
    summaries, overlap = {}, {}
    for name in ("observed", "baseline", "named_pairs_10"):
        if name == "observed":
            selected = actual
        else:
            selected = np.zeros_like(actual)
            order = np.argsort(-scores[name][adult], kind="stable")
            selected[order[:edges]] = 1
        matrix = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
        matrix[indices] = selected
        summaries[name] = motifs(matrix)
        overlap[name] = int((selected * actual).sum())
    return summaries, overlap


def plot(path: Path, results: dict, adult: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(1, 2, figsize=(11.3, 4.6), layout="constrained")
    labels = ("Contact +\ncell type", "Individual\ngenes", "5 rules", "10 rules", "20 rules")
    styles = (("Held-out animal", "#24725e", "o"), ("Also held-out subtype pairs", "#a0522d", "s"),
              ("Early to later animals", "#4d6290", "^"))
    for scheme, (label, color, mark) in zip(SCHEMES, styles):
        means = results[scheme]["mean"]
        axes[0].plot(labels, [means[key]["ap"] for key in MODELS], color=color, marker=mark, label=label)
    axes[0].set(title="Prediction with a limited rule budget", ylabel="Mean average precision")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="best")

    marks = np.arange(len(adult["observed"]))
    for i, (key, label, color) in enumerate((("baseline", "Contact + cell type", "#65716a"),
                                             ("named_pairs_10", "10 interaction rules", "#24725e"))):
        relative = np.asarray(list(adult[key].values())) / np.asarray(list(adult["observed"].values())) - 1.0
        axes[1].bar(marks + (i - 0.5) * 0.34, relative, width=0.34, color=color, label=label)
    axes[1].axhline(0, color="#60716a", linewidth=0.8)
    axes[1].set(title="Adult motif counts at the observed edge density", ylabel="Difference from observed count",
                xticks=marks, xticklabels=[key.capitalize() for key in adult["observed"]])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(frameon=False, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/worm-connectome"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    data, profiles, subtypes, metadata = dataset(args.cache)
    expression, used, dropped = named_expression(args.cache, profiles)
    results, rules, scores = evaluate(data, designs(data, expression, subtypes), used)
    adult, overlap = adult_comparison(data, scores["held_specimen"])
    payload = {"panel": {"used": used, "dropped": dropped}, "candidate_count": len(data),
               "specimens": int(data.specimen.nunique()), "expression_profiles": metadata["expression_profiles"],
               "results": results, "rules": rules, "adult_motifs": adult, "adult_edge_overlap": overlap}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "wiring-rules-results.json"
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot(args.output / "wiring-rules.png", results, adult)


if __name__ == "__main__":
    main()
