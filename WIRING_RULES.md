# Named wiring rules in the *C. elegans* connectome

This analysis tests whether a small set of source-gene × target-gene interactions adds information about directed chemical synapses beyond contact geometry, broad neuron type, and the expression of each gene on its own. It reuses the matched 18,732 directed contact observations across seven specimens and the 47 late-embryonic subtype profiles from the [previous assay](CONNECTOME.md). [Public report](https://statsleelab.github.io/developmental-unfolding/wiring-rules.html)

## Reproduce

```sh
python -m pip install -r requirements.txt pandas==2.2.3 scikit-learn==1.8.0 openpyxl==3.1.5
OPENBLAS_NUM_THREADS=1 python wiring_rules.py --cache /tmp/worm-connectome
```

The analysis reads the original public data or the cached expression profiles and writes `docs/wiring-rules-results.json` and `docs/wiring-rules.png`. The archived electrical-synapse reproduction and its null-model sensitivity have their own entry point. [Electrical-synapse reproduction](INNEXIN_REPRODUCTION.md)

## Model and evaluation

- Use a fixed panel of 20 genes with roles in cell recognition, adhesion, guidance, or synapse organization. Its rationale includes UNC-6 and UNC-40 in a partner-recognition study and several adhesion proteins studied in another neuronal circuit. All 20 were uniquely mapped and varied across the target expression profiles. The candidate list is the 400 ordered products of a presynaptic gene and a postsynaptic gene. An interaction coefficient is an association conditional on the other fitted features. [Partner recognition](https://doi.org/10.1186/1749-8104-6-28) · [Circuit adhesion](https://doi.org/10.7554/eLife.29257) · [Gene-family background](https://www.ncbi.nlm.nih.gov/books/NBK154158/)
- Start from the contact-and-type logistic baseline. Add the 40 individual log-normalized RNA expression terms. Fit the main-effects model on training observations and rank interaction products by their residual score after projecting out the main features, using weights from the fitted logistic model. Refit with the 5, 10, or 20 highest-ranked interactions. Feature scaling, candidate ranking, and fitting are performed within each training split. The rule budget counts interaction terms in addition to the main effects.
- Use the same specimen and subtype-pair partitions as the earlier assay. For the chronological check, learn the rules and coefficients once from Dataset1–3 and apply them to Dataset4, 5, 6, and 8. This check uses each later specimen’s measured anatomy and the same late-embryonic expression profiles. It measures later-stage prediction from a fixed early expression proxy. The results JSON includes the ten transferred rules and their fitted coefficients.
- Summarize selection frequency and coefficient sign for the ten-rule model across seven specimen-held-out fits. The training sets overlap, so frequency describes selection stability for this benchmark. Correlated expression can make several products interchangeable.

## Results

| Test data | Contact and type | Individual genes | Five interactions | Ten interactions | Twenty interactions |
|---|---:|---:|---:|---:|---:|
| Held-out specimen | 0.559 | 0.650 | 0.664 | 0.668 | 0.669 |
| Held-out specimen and subtype pairs | 0.524 | 0.569 | 0.568 | 0.560 | 0.550 |
| Train early, test later | 0.593 | 0.671 | 0.679 | 0.682 | 0.683 |

The scores are mean average precision across the evaluated animals. The ten-rule model improves ranking over individual genes in all seven specimen-held-out fits. Under the subtype-pair exclusion, it scores below the individual-gene model in all seven specimens. The five-rule model is close to the individual-gene model under that split. These results support individual expression as a portable predictor and attribute a narrower predictive role to interactions between familiar subtype combinations. The late-embryonic RNA data and post-hatching reconstructions come from different animals and ages. [Embryonic atlas](https://doi.org/10.1126/science.aax1971) · [Developmental connectomes](https://doi.org/10.1038/s41586-021-03778-8)

In the held-out adult, selecting the same number of predicted edges as observed gives 403 of 658 true edges for the contact-and-type baseline and 437 for the ten-rule model. At that density, the observed divergent count is 3,097, compared with 2,762 and 3,110 respectively. Reciprocal counts also move closer to the observed value. Convergent and feedforward counts move farther away. The adult motif panel is a descriptive comparison at an observed density, and the performance table uses held-out probabilities.

Prior work includes contact-constrained chemical synapse prediction from gene expression and generative connectome models with cell type, birth time, and geometry. This experiment asks how much a small, named molecular interaction component adds to an existing contact-conditioned assay. The contact graph is supplied by microscopy. A developmental mechanism that generates contacts, and the causal roles of any named genes, require distinct evidence. [Earlier chemical-synapse prediction](https://doi.org/10.1371/journal.pcbi.1000120) · [Developmental generative model](https://doi.org/10.1073/pnas.2504913122)
