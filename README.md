# Developmental Unfolding

Can activity-dependent pruning during development improve the temporal memory of a frozen recurrent network? This synthetic experiment starts from one artificial cell and compares activity-based edge selection with two resource-matched controls.

## Reproduce the experiment

Use Python 3.12 and install the versions in `requirements.txt`. From the repository root, run the experiment and tests.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python experiment.py
python -m http.server 8080 --directory docs
```

The report is available at `http://localhost:8080`. To reproduce the pilot on its own, run `python experiment.py --pilot-only --output pilot-output`. The complete run writes the study summary, delay-level seed results, and a publication-resolution figure to [`docs/`](docs/). The static site can be served from the `main` branch and `/docs` folder on GitHub Pages.

## Experimental protocol

1. Grow a 256-node colony from one lattice location. Each division fills an empty orthogonal neighbor of a randomly selected frontier node. Assign excitatory or inhibitory signs with probabilities 0.8 and 0.2. A directed candidate edge can join any two occupied sites up to √8 lattice units apart.
2. Give the pre-pruning graph 512 steps of independent Gaussian drive at every node as a model of spontaneous activity. For each candidate connection, calculate the absolute correlation of the source state at step *t* and the target state at step *t* + 1. Each source ranks its own edges within each exact squared-distance group and retains half, rounded down with a minimum of one per group.
3. Construct the random control by sampling from each group using exactly the same quota. Construct the rewired control from the activity-pruned graph with same-sign directed target swaps. Each swap preserves incoming degree by sign at every target, outgoing degree at every source, and the exact edge-distance histogram by sign. All three graphs share the same nodes, total edge count, and total connection length.
4. Use the same signed connection strengths and global scale across the three graphs within a seed. The incoming absolute row sum is at most 0.9. At each step, the next state is the average of the current state and the `tanh` of the summed recurrent and external inputs. Freeze the connections for evaluation.
5. Use the same set of 16 input nodes and the same three independent streams of symmetric binary inputs across the paired graphs. Reset the recurrent state for each stream and discard 96 steps before recording 2,200 observations. Fit linear ridge readouts to the train stream and choose the regularization penalty by mean validation R². Evaluate at lags 2, 4, 8, 12, and 16 using the test stream.
6. Use the eight pilot seeds to check the simulator and control construction. Run the same final configuration on 32 separate evaluation seeds. Average test R² over the five delays within each seed. For each contrast, subtract the matched control score within the same seed and compute a percentile bootstrap interval by resampling seeds with replacement. The delay curves use pointwise intervals.

## Interpretation

Across the 32 evaluation seeds, activity pruning changed mean test R² by **+0.0020** relative to matched random pruning (95% bootstrap interval −0.0012 to +0.0053) and **−0.0011** relative to matched rewiring (−0.0033 to +0.0011). The activity-pruned circuit recovered recent inputs with mean R² of 0.969 at a two-step delay and 0.174 at 16 steps. Rewiring replaced 38.6% of its edges on average.

The primary contrast estimates the effect of selecting edges by spontaneous activity within this growth and pruning protocol. The rewiring contrast tests the contribution of the remaining edge arrangement conditional on the matched features. Degree- and distance-preserving swaps explore a constrained set of graphs. The report tracks the fraction of edges replaced and the proposal acceptance rate so the amount of change is visible.

The linear readout quantifies how much past input can be recovered from frozen dynamics. The experiment holds the growth process fixed and leaves the effect of growing from a single cell for a separate comparison. Biological fit, compression, task learning speed, and damage recovery remain research questions beyond this synthetic evaluation.

## Related work

- Bauer et al. (2014), *Developmental Self-Construction and -Configuration of Functional Neocortical Neuronal Networks*. [Article](https://doi.org/10.1371/journal.pcbi.1003994)
- Raghavan and Thomson (2019), *Neural networks grown and self-organized by noise*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html)
- Dambre et al. (2012), *Information Processing Capacity of Dynamical Systems*. [Article](https://doi.org/10.1038/srep00514)
- Barandiaran and Stovold (2026), *Growing Echo State Networks through Graph-Based Morphogenesis with Developmental Graph Cellular Automata*. [Accepted manuscript](https://eprints.whiterose.ac.uk/id/eprint/245447/)
