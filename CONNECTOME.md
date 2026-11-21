# Predicting synapses from embryonic expression

Among physically touching *C. elegans* neurons, does embryonic expression improve prediction of a directed chemical synapse? The model applies the same compact probabilistic rule to every candidate pair given measured anatomy.

## Reproduce

```sh
python -m pip install -r requirements.txt pandas==2.2.3 scikit-learn==1.8.0 openpyxl==3.1.5
OPENBLAS_NUM_THREADS=1 python connectome_experiment.py --cache /tmp/worm-connectome
```

The script downloads about 254 MB to the chosen cache on the first run. It processes the count matrix in chunks and caches only aggregated expression for subsequent runs. It writes the directed-pair predictions, specimen-level metrics, and adult adjacency figure to `docs/`. [Published report](https://statsleelab.github.io/developmental-unfolding/connectome.html)

## Design

- **Anatomy and targets.** Use the authors’ contact and chemical synapse matrices. Seven animals supply both matrices. Dataset7 has only the synapse matrix and is excluded. Each direction of a physical neuron-neuron contact is a candidate. A candidate is positive when its directed chemical synapse count is above zero. Muscle contacts, self-pairs, and synapses outside the measured contact mask are outside the assay.
- **Molecular inputs.** Aggregate raw UMI counts over quality-controlled atlas cells for each annotated subtype with at least 20 profiles at estimated embryonic ages of 450 minutes or later. Match neuron names directly or by dropping a terminal L/R suffix when the atlas provides a paired subtype. Side-specific differences are unresolved for these pooled entries. The match yields 87 annotated neurons, 47 expression profiles based on 5,133 atlas cells, and 18,732 specimen-specific directed contact candidates. Each model uses exactly these candidates.
- **Shared rule.** Fit three ridge-regularized logistic models. The first uses the contact area, each endpoint’s total contact area and degree, and ordinal developmental order. The second adds broad source-target type combinations. The third adds six principal components of log-normalized expression for each endpoint and their six component-wise products. Gene selection and PCA use only the independent atlas. Scaling and logistic coefficients are fitted using training contacts in each fold.
- **Validation.** Leave out each whole specimen and average its scores with equal weight. For a stricter check, also omit the unordered atlas subtype pair from training, so left-right homolog pairs grouped under a subtype cannot supply a label for that same pairing. A fixed five-fold partition assigns pairs to held-out groups within each test specimen. The same splits apply to all three models.

## Findings

| Held-out samples | Contact | Contact and type | Contact, type and expression |
|---|---:|---:|---:|
| Whole specimen | 0.510 | 0.559 | 0.608 |
| Specimen and subtype pairs | 0.506 | 0.524 | 0.554 |

Values are average precision, a score for ranking observed synapses above contact-only pairs. A chance ranking has average precision near the fraction of positive candidates, which ranges from 0.111 to 0.230 by specimen. Expression improves average precision in all seven specimens under both splits. Under the stricter split, average log loss changes from 0.3198 to 0.3174. With seven cross-sectional specimens, these values describe the benchmark and its sensitivity to the split.

The adult comparison figure displays the same number of edges in every panel. The plot uses the adult edge count to set density; benchmark scores use held-out probabilities. At that density, the contact-and-type model recovers 403 of 658 assayed adult edges, and the model with expression recovers 406. The two scores capture different aspects of the predictions.

This analysis conditions on measured neurite contacts and atlas subtype matches. It tests one statistical step of connection selection. The RNA and anatomical data come from separate sets of animals. Expression profiles can proxy finer cell identities, and they cannot establish that any particular gene causes a synapse. The formation of the contact graph from a precursor cell and the information needed to encode the entire developmental process remain open.

- [Witvliet et al. Nature (2021)](https://doi.org/10.1038/s41586-021-03778-8) · [Author matrices at the pinned revision](https://github.com/dwitvliet/nature2021/tree/0646af9d25896ae660f97d462eab2d67282f5625/tables)
- [Packer et al. Science (2019)](https://doi.org/10.1126/science.aax1971) · [GEO GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954)
