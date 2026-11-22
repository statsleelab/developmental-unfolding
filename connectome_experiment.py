"""Predict chemical synapses among observed C. elegans neurite contacts."""

import argparse
import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WIRING = "https://raw.githubusercontent.com/dwitvliet/nature2021/0646af9d25896ae660f97d462eab2d67282f5625/tables/"
ATLAS = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126954/suppl/GSE126954_"
FILES = {
    "contact.xlsx": WIRING + "contact_matrices.xlsx",
    "synapses.xlsx": WIRING + "synapse_count_matrices.xlsx",
    "cell.csv.gz": ATLAS + "cell_annotation.csv.gz",
    "genes.csv.gz": ATLAS + "gene_annotation.csv.gz",
    "counts.txt.gz": ATLAS + "gene_by_cell_count_matrix.txt.gz",
}
MIN_AGE = 450
MIN_CELLS = 20
FUNCTIONAL_TYPES = {"Sensory", "Inter", "Motor", "Modulatory"}


def inputs(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = cache / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            print(f"Downloading {name}", flush=True)
            urlretrieve(url, partial)
            partial.replace(target)


def workbook_sheet(book: pd.ExcelFile, sheet: str) -> tuple[pd.DataFrame, dict]:
    raw = book.parse(sheet, header=None)
    columns = raw.iloc[2, 3:].dropna()
    rows = raw.iloc[4:, 2].dropna()
    table = raw.loc[rows.index, columns.index].astype(float).fillna(0)
    table.index, table.columns = rows.tolist(), columns.tolist()
    kinds = dict(zip(columns, raw.iloc[1, 3:].ffill().loc[columns.index]))
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"Duplicate neuron identifiers in {sheet}")
    return table, kinds


def subtype_mapping(kinds: dict, available: set) -> dict:
    matches = {}
    for name, kind in kinds.items():
        if kind not in FUNCTIONAL_TYPES:
            continue
        if name in available:
            matches[name] = name
        elif name.endswith(("L", "R")) and name[:-1] in available:
            matches[name] = name[:-1]
    return matches


def read_profiles(cache: Path, meta: pd.DataFrame, subtypes: list[str]) -> np.ndarray:
    # GEO supplies a one-based gene-by-cell Matrix Market file in annotation order.
    cached = cache / "late-embryo-profiles.npz"
    if cached.exists():
        with np.load(cached) as saved:
            if saved["names"].tolist() == subtypes:
                return saved["counts"]
    genes = pd.read_csv(cache / "genes.csv.gz")
    lookup = pd.Categorical(meta["cell.subtype"], categories=subtypes).codes.copy()
    late = meta["passed_initial_QC_or_later_whitelisted"] & (meta["embryo.time"] >= MIN_AGE)
    lookup[~late] = -1
    result = np.zeros((len(subtypes), len(genes)), dtype=np.float64)
    with gzip.open(cache / "counts.txt.gz", "rt") as matrix:
        matrix.readline()
        gene_count, cell_count, _ = map(int, matrix.readline().split())
    if (gene_count, cell_count) != (len(genes), len(meta)):
        raise ValueError("GEO annotation lengths do not match matrix dimensions")
    chunks = pd.read_csv(cache / "counts.txt.gz", sep=r"\s+", skiprows=2, names=["gene", "cell", "umi"],
                         dtype=np.int32, chunksize=1_000_000)
    for chunk in chunks:
        groups = lookup[chunk.cell.to_numpy() - 1]
        selected = (groups >= 0) & (chunk.umi.to_numpy() > 0)
        np.add.at(result, (groups[selected], chunk.gene.to_numpy()[selected] - 1), chunk.umi.to_numpy()[selected])
    if np.any(result.sum(axis=1) <= 0):
        raise ValueError("An eligible subtype has no matched RNA counts")
    np.savez_compressed(cached, names=np.asarray(subtypes), counts=result)
    return result


def contacts(cache: Path, mapping: dict) -> tuple[pd.DataFrame, list[dict]]:
    specimens, coverage = [], []
    with pd.ExcelFile(cache / "contact.xlsx") as contact_book, pd.ExcelFile(cache / "synapses.xlsx") as synapse_book:
        for sheet in contact_book.sheet_names:
            area, kinds = workbook_sheet(contact_book, sheet)
            synapses, _ = workbook_sheet(synapse_book, sheet)
            neurons = sorted(n for n, kind in kinds.items() if kind in FUNCTIONAL_TYPES
                             and n in area.index and n in synapses.index and n in synapses.columns)
            area = area.loc[neurons, neurons]
            synapses = synapses.loc[neurons, neurons]
            physical = area.to_numpy()
            if not np.array_equal(physical, physical.T):
                raise ValueError(f"Asymmetric physical-contact matrix in {sheet}")
            synapse_matrix = synapses.to_numpy()
            n_contacts = (physical > 0).sum(axis=0)
            total_area = physical.sum(axis=0)
            target_idx, source_idx = np.where((physical > 0) & ~np.eye(len(neurons), dtype=bool))
            names = np.asarray(neurons)
            source, target = names[source_idx], names[target_idx]
            positive = synapse_matrix[target_idx, source_idx] > 0
            data = pd.DataFrame({
                "specimen": sheet, "source": source, "target": target,
                "source_type": [kinds[n] for n in source], "target_type": [kinds[n] for n in target],
                "source_subtype": [mapping.get(n) for n in source], "target_subtype": [mapping.get(n) for n in target],
                "log_area": np.log1p(physical[target_idx, source_idx]),
                "log_source_area": np.log1p(total_area[source_idx]),
                "log_target_area": np.log1p(total_area[target_idx]),
                "source_contacts": n_contacts[source_idx], "target_contacts": n_contacts[target_idx],
                "development_order": int(sheet.removeprefix("Dataset")), "synapse": positive.astype(int),
            })
            retained = data.dropna(subset=["source_subtype", "target_subtype"])
            coverage.append({"specimen": sheet, "neurons": len(neurons),
                             "mapped_neurons": sum(name in mapping for name in neurons),
                             "contact_candidates": len(data), "retained_candidates": len(retained),
                             "positive_at_contacts": int(positive.sum()), "retained_positives": int(retained.synapse.sum()),
                             "positives_outside_contacts": int(((synapse_matrix > 0) & (physical == 0)
                                                                & ~np.eye(len(neurons), dtype=bool)).sum())})
            specimens.append(retained)
    return pd.concat(specimens, ignore_index=True), coverage


def contact_features(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    geometry = data[["log_area", "log_source_area", "log_target_area", "source_contacts",
                     "target_contacts", "development_order"]].to_numpy()
    pair_type = pd.get_dummies(data.source_type + ">" + data.target_type, dtype=float).to_numpy()
    return geometry, np.column_stack((geometry, pair_type))


def features(data: pd.DataFrame, profiles: np.ndarray, names: list[str]) -> dict:
    log_expression = np.log1p(1e6 * profiles / profiles.sum(axis=1, keepdims=True))
    variable = np.argsort(-log_expression.var(axis=0), kind="stable")[:256]
    pcs = PCA(n_components=6, svd_solver="full").fit_transform(log_expression[:, variable])
    indices = {name: index for index, name in enumerate(names)}
    source, target = data.source_subtype.map(indices).to_numpy(), data.target_subtype.map(indices).to_numpy()
    geometry, typed = contact_features(data)
    molecular = np.column_stack((pcs[source], pcs[target], pcs[source] * pcs[target]))
    return {"contact": geometry, "type": typed, "expression": np.column_stack((typed, molecular))}


def subtype_pair_folds(data: pd.DataFrame) -> np.ndarray:
    pairs = [tuple(sorted((a, b))) for a, b in zip(data.source_subtype, data.target_subtype)]
    unique_pairs = sorted(set(pairs))
    strata = {pair: group for pair, group in zip(unique_pairs, np.random.default_rng(2026).permutation(len(unique_pairs)) % 5)}
    return np.asarray([strata[pair] for pair in pairs])


def predict(data: pd.DataFrame, designs: dict) -> tuple[dict, dict]:
    labels = data.synapse.to_numpy()
    animals = data.specimen.to_numpy()
    fold = subtype_pair_folds(data)
    results, predictions = {}, {}
    for scheme in ("held_specimen", "held_specimen_and_subtype_pair"):
        summary, exported = [], {}
        for specimen in sorted(set(animals)):
            held = animals == specimen
            y = labels[held]
            row = {"specimen": specimen, "candidate_count": len(y), "prevalence": float(y.mean()), "scores": {}}
            for name, design in designs.items():
                score = np.full(len(data), np.nan)
                groups = (None,) if scheme == "held_specimen" else range(5)
                for group in groups:
                    train, test = ~held, held
                    if group is not None:
                        train, test = train & (fold != group), test & (fold == group)
                    if not test.any():
                        continue
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
                    model.fit(design[train], labels[train])
                    score[test] = model.predict_proba(design[test])[:, 1]
                p = score[held]
                row["scores"][name] = {"ap": float(average_precision_score(y, p)),
                                        "auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p))}
                exported.setdefault(name, np.full(len(data), np.nan))[held] = p
            summary.append(row)
        metrics = {name: {key: float(np.mean([row["scores"][name][key] for row in summary]))
                          for key in ("ap", "auc", "log_loss")} for name in designs}
        results[scheme] = {"specimens": summary, "mean": metrics}
        predictions[scheme] = exported
    return results, predictions


def plot(path: Path, data: pd.DataFrame, scores: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    adult = data["specimen"] == "Dataset8"
    observed = data.loc[adult]
    cells = sorted(set(observed.source) | set(observed.target))
    node_index = {name: index for index, name in enumerate(cells)}
    index = observed.target.map(node_index).to_numpy(), observed.source.map(node_index).to_numpy()
    binary = np.full((len(cells), len(cells)), np.nan)
    binary[index] = observed.synapse
    palette = ListedColormap(("#e8eee9", "#125e50"))
    palette.set_bad("white")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), layout="constrained")
    axes[0].imshow(binary, cmap=palette, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Observed synapses")
    for axis, name, title in zip(axes[1:], ("type", "expression"), ("Contact + cell type", "+ embryonic expression")):
        p = scores[name][adult]
        # Same density only for this side-by-side visualization. Scores use held-out probabilities.
        k = int(observed.synapse.sum())
        predicted = np.zeros(len(p))
        predicted[np.argsort(-p, kind="stable")[:k]] = 1
        grid = np.full_like(binary, np.nan)
        grid[index] = predicted
        axis.imshow(grid, cmap=palette, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
    for axis in axes:
        axis.set(xlabel="Presynaptic neuron", ylabel="Postsynaptic neuron", xticks=[], yticks=[])
    figure.suptitle(f"Held-out adult specimen · {len(cells)} neurons in the analyzed contact graph")
    figure.legend(handles=[Patch(color="#125e50", label="Synapse"), Patch(color="#e8eee9", label="Contact only"),
                           Patch(facecolor="white", edgecolor="#ddd", label="Outside assay")],
                  loc="outside lower center", ncols=3, frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def dataset(cache: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    inputs(cache)
    meta = pd.read_csv(cache / "cell.csv.gz")
    counts = meta.loc[(meta["embryo.time"] >= MIN_AGE) & meta["passed_initial_QC_or_later_whitelisted"],
                      "cell.subtype"].value_counts()
    with pd.ExcelFile(cache / "contact.xlsx") as book:
        _, kinds = workbook_sheet(book, "Dataset8")
    mapping = subtype_mapping(kinds, set(counts[counts >= MIN_CELLS].index))
    names = sorted(set(mapping.values()))
    data, coverage = contacts(cache, mapping)
    print(f"Matched {len(mapping)} neurons to {len(names)} profiles; {len(data)} candidate contacts", flush=True)
    profiles = read_profiles(cache, meta, names)
    metadata = {"coverage": coverage, "matched_neurons": len(mapping), "expression_profiles": len(names),
                "expression_cells": int(counts.loc[names].sum())}
    return data, profiles, names, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/worm-connectome"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    data, profiles, names, metadata = dataset(args.cache)
    results, predictions = predict(data, features(data, profiles, names))
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "results": results}
    (args.output / "connectome-results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for scheme, scores in predictions.items():
        for model, values in scores.items():
            data[f"{scheme}_{model}"] = values
    data.to_csv(args.output / "connectome-predictions.csv", index=False)
    plot(args.output / "connectome-comparison.png", data, predictions["held_specimen"])
    print(json.dumps({name: value["mean"] for name, value in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
