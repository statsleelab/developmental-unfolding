# Reproducing the PNAS innexin rules

This checkpoint independently implements the spatial connectome model of Kovács, Barabási and Barabási from the authors’ archived input matrices. It reads CSV files from the pinned public release and computes the numerical results in Python. [Paper](https://doi.org/10.1073/pnas.2009093117) · [Author code and data](https://doi.org/10.5281/zenodo.4027588)

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python innexin_reproduction.py
```

The first run caches the verified archive at `/tmp/innexin/release.zip`. An alternative local archive path can be supplied with `--archive`. The runner writes [`docs/innexin-reproduction.json`](docs/innexin-reproduction.json).

## Cook reconstruction

The archived expression matrix covers 185 neurons and 18 binary innexin profiles. The physical-contact mask contains 5,557 unordered pairs of distinct neurons and 35 diagonal entries. The Cook target includes 594 gap junctions between distinct neurons and seven diagonal entries. The fit uses the contact mask, including the diagonal entries, as in the authors’ release.

Using the reported ridge penalty of 0.215 and 100 iterations of the degree-based contact null reproduces the archived connection probabilities to their saved precision. The rule weights, null means and null variances agree with the archived matrices within 1.1 × 10⁻¹². The continuous generalized cross-validation minimum is near 0.2173. The paper’s threshold of a positive weight and z ≥ 2 selects **19 distinct matrix entries** in the upper triangle. The paper cites prior experimental support for five named interactions.

The release does not label expression columns. The report matches the published candidate graph and rounded z scores to recover labels where that match identifies them. Columns 3 and 11 are identical and yield the same inx-10 interaction weight and z score. Their labels, inx-5 and inx-11, are interchangeable for this fit. The archive therefore identifies an observational equivalence that further expression or intervention data can resolve.

## Sensitivity checks

The release propagates the variance of each ordered matrix entry independently through its ridge operator. Propagating one Bernoulli draw per **unordered** contact pair, with the same expected-degree probabilities, gives **15** positive Cook entries at the same z threshold. This changes the null variance and leaves the fitted Cook weights fixed. The resulting thresholds describe the null models used here.

Repeating the calculation with the archived Varshney target at the same ridge penalty gives 20 positive entries, 11 shared with the Cook list. This equal-penalty check uses the archive’s variance convention. The primary reproduction uses the Cook data and the authors’ own null convention.
