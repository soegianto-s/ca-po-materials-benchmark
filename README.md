# Ca–P–O materials benchmark

Scientific code, composition-disjoint splits, input SHA-256 hashes, and aggregate results accompanying **Physical Constraints and Cross-Source Transfer in Graph Learning for Ca–P–O-Containing Materials** (research manuscript).

## Scope
The original benchmark uses 511 Materials Project structures, 421 reduced compositions, three partition seeds, five outer folds, and three initialization seeds. Physical validity, accuracy, calibration, and transfer are evaluated separately. This repository does not claim a universally superior hybrid architecture.

## Current extension
A fixed post hoc extension adds a small inner-validation graph search, composition-only MLPs, XGBoost, paired composition-cluster bootstrap intervals, and an analytic class-weight logit-offset diagnostic. The extension is complete: 90 new graph fits plus 45 cached graph fits, 135 composition MLP fits, and 180 XGBoost candidate groups (each with one classifier and two regressors). Selection uses mean inner-validation objective across three seeds within each outer split. The completion audit verifies frozen input hashes, grouped splits, checkpoint consistency, and selection arithmetic. Mean AP is 0.339 for the searched graph, 0.323 for the MLP and 0.347 for XGBoost; these are descriptive comparisons. See the complete summary and conditional bootstrap report for detailed results.

## Contents and reproducibility boundary
- `scripts/`: scientific training, analysis, and focused checks.
- `revision_ieee_access/*/experiment.json`: frozen configuration and original input hashes.
- `revision_ieee_access/*/partition*/split.json`: exact fit/validation/test IDs and composition groups.
- Summary JSON files: completed original experiments and completed statistical diagnostics.
- `SHA256SUMS.json`: hashes of this public code release.

This code/split release does **not** contain raw crystal structures, the feature/label table, checkpoint binaries, third-party full texts, or credentials. It is not a self-contained training dataset. Scripts expect the archived inputs at their manifest-relative paths; a newly downloaded database snapshot may differ. Full local data/checkpoints are preserved separately. Public data deposition and a Zenodo DOI are pending. Do not infer a dataset DOI from this repository URL.

## Run
Use a Python environment with the dependencies below and the original archived data matching the manifests. Then run the scientific experiment scripts from the repository root. `reviewer_extension_experiment.py --workers 3 --kind all` resumes completed/cached neural fits; `reviewer_xgboost.py` runs the fixed boosted-tree grid; `reviewer_statistics.py` requires the original per-sample OOF predictions, which are not part of this minimal code release.

All configuration and checkpoint decisions use inner validation; outer labels are used only for evaluation. Composition-bootstrap intervals condition on frozen models and do not estimate uncertainty from retraining or model selection. Class-weight offset correction is not equivalent to retraining an unweighted classifier.

## Attribution
Materials Project: Jain et al., APL Materials 1, 011002 (2013), https://doi.org/10.1063/1.4812323. XGBoost: Chen and Guestrin (2016), https://doi.org/10.1145/2939672.2939785. Third-party libraries retain their own licenses. No additional code-license grant is asserted by this release; the maintainer can assign a license separately.
