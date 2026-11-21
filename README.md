# Developmental Unfolding

Can a compact, shared developmental program grow circuitry with reproducible spatial order and useful computational priors? The project uses abstract cell states, signed connections, and local interactions on a 2D lattice. It is genome-inspired. The simulator uses synthetic data and rules throughout.

## Predicting real connectome edges

A held-out analysis asks whether late-embryonic gene expression improves the prediction of chemical synapses among physically touching *C. elegans* neurons. It compares the same regularized connection rule under contact-only, cell-type, and expression inputs. [Results](https://statsleelab.github.io/developmental-unfolding/connectome.html) · [Data and reproduction](CONNECTOME.md)

## Local differentiation and developmental wiring

One uncommitted cell expands into neighboring vacancies. A bounded internal regulator receives signals from its four immediate neighbors. Positive feedback and a commitment threshold produce two model cell types. Committed cells form nearby signed connections throughout growth under a local degree limit. Edge activity traces determine whether older weakly used edges are pruned. The simulation records the progression from one cell through commitment and wiring.

- **Spatial order.** Compare the proportion of same-type pairs at Manhattan distances 3–7 with the expectation after shuffling types within birth-cohort and boundary strata. Repeat with neighbor signaling disabled while preserving the exact growth trajectory. Pilot exploration adjusted neighbor coupling and regulator noise to permit both types to coexist. Separate evaluation seeds use the resulting fixed rule.
- **Learning efficiency.** Compare developmental wiring to direct wiring drawn at maturity on exactly the same positions and cell types. Each source has the same number of outgoing edges at each distance and the two conditions share signed weights, connection preferences, input ports, and trial streams. A frozen circuit receives two cues in sequence. A ridge readout fitted to non-input cell states predicts their order six steps after the second cue. Balanced, nested label budgets range from 16 to 256. Separate test trials evaluate every budget. The primary difference averages test accuracy over the log₂ label budget. The analysis resamples growth seeds for paired intervals.

Across 32 evaluation seeds, the neighbor signal changed same-type pair enrichment by **+0.66 percentage points** (95% bootstrap interval −0.15 to +1.52). The normalized learning-curve area changed by **−0.38 percentage points** (−1.70 to +1.00) relative to direct wiring. At 256 labels, test accuracy was **83.3%** with developmental wiring, **84.1%** with direct wiring, and **50.1%** with recurrence removed. Both primary intervals include zero. [Development experiment](https://statsleelab.github.io/developmental-unfolding/development.html)

The next research stage defines which synthetic predictions can be evaluated against processed developmental atlases, then uses held-out donors and age-aware baselines. [Biological validation roadmap](BIOLOGICAL_VALIDATION.md)

## Reproduce the experiment

Use Python 3.12 and install the versions in `requirements.txt`. From the repository root, run the experiment and tests.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python development_experiment.py
OPENBLAS_NUM_THREADS=1 python experiment.py
python -m http.server 8080 --directory docs
```

The development experiment is available at `http://localhost:8080/development.html`. The earlier memory experiment is available at `http://localhost:8080/`. Each complete run writes individual seed results, summaries, and a figure to [`docs/`](docs/). Run `python development_experiment.py --pilot-only` for the local-rule pilot or `python experiment.py --pilot-only --output pilot-output` for the original memory pilot. GitHub Pages uses `main` and `/docs`.

## Earlier activity-pruning experiment

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
