# v13.3 PSM spectral block comparison release

v13.3 keeps the v13.2.1 K-discovery and full-diagnostics core, then adds a
controlled comparison layer for the methods we discussed:

```text
v13.2.1 baseline posterior kmeans
PSM spectral clustering
PSM spectral + spatial prior
EB-shrink PSM spectral
BayesianGaussianMixture / truncated-DP comparison
```

The main purpose is to test whether the already-strong coassociation posterior
can improve D/block recovery when it is treated as a kernel rather than only as
a post-hoc diagnostic.

## Run from notebook

Open:

```text
v13_3_run_notebook.ipynb
```

Default notebook mode is quick.  Change `MODE = "wide"` in the second code cell
for the full diagnostic run.

## Run from terminal

Quick:

```bash
python v13_3_psm_spectral_block_discovery.py --quick --outdir v13_3_quick_outputs
```

Wide:

```bash
python v13_3_psm_spectral_block_discovery.py --wide --outdir v13_3_wide_outputs
```

Config-based wide:

```bash
python v13_3_psm_spectral_block_discovery.py --config config_wide.json --outdir v13_3_wide_outputs
```

## New v13.3 outputs

The comparison layer writes:

```text
v13_3_method_summary.csv
v13_3_method_candidates.csv
v13_3_method_assignment.csv
v13_3_affinity_psm.csv
v13_3_affinity_psm_spatial.csv
v13_3_affinity_eb_shrink_psm_spatial.csv
```

Useful plots:

```text
v13_3_method_score_comparison.png
v13_3_method_ARI_comparison.png
v13_3_spectral_candidate_surface.png
v13_3_affinity_psm.png
v13_3_affinity_psm_spatial.png
v13_3_affinity_eb_shrink_psm_spatial.png
```

The full v13.2.1 diagnostics are still present with v13.3 filenames:

```text
v13_3_report.json
v13_3_report.md
v13_3_space_diagnostics.csv
v13_3_subspace_angles.csv
v13_3_center_matching.csv
v13_3_component_alignment.csv
v13_3_latent_matrices.npz
```

## Method details

### PSM spectral

Uses the block coassociation matrix as a precomputed affinity/kernel, then
clusters components by spectral clustering.  This tests the PSM idea directly:
the posterior same-block probabilities become the graph used for block summary.

### PSM spectral + spatial prior

Multiplies the PSM affinity by a soft spatial prior:

```text
A_ij = C_ij * [floor + (1 - floor) exp(-0.5 * distance(center_i, center_j)^2 / tau^2)]
```

This is the lightweight dd-IBP-inspired part: nearby components are encouraged
to share blocks, without introducing full MCMC.

### EB-shrink PSM spectral

Downweights pairwise affinities involving components with lower posterior
reliability or boundary-pulled centers.  This borrows the EBMF/flashr spirit of
adaptive shrinkage without depending on the R package.

### BayesianGaussianMixture

Fits a truncated-DP-style Gaussian mixture to component features derived from
centers, spectral PSM embeddings, and D-localized dependency strength.  It is a
comparison route rather than the preferred default.

## Selection logic

For each spectral method, v13.3 evaluates all F values in `config["block"]["F_grid"]`.
The unsupervised score is:

```text
affinity within-between contrast
+ block-size balance
- complexity / singleton / entropy penalties
+ F posterior prior bonus
```

The F posterior prior bonus is important.  Without it, affinity-only scoring can
over-split because smaller blocks often have cleaner within-block affinity.  The
prior lets the v13.2 posterior over block count guide the spectral summary.

The default preferred method is:

```text
psm_spatial_spectral
```

but `v13_3_method_summary.csv` keeps every method side by side.

## Quick smoke result on this machine

Quick mode completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
K_eff_soft = 11.7934
F_selected = 3
F_selected_by_score = 3
v13_3_selected_method = psm_spatial_spectral
v13_3_selected_F = 3
all_data_R2 = 0.9816
localized_R2_X = 0.9816
```

In quick mode, the spectral methods mostly recover the same coarse F=3 summary
as the baseline.  The useful part is the new comparison table: it exposes when
PSM-only scoring wants to over-split and how much the F posterior prior corrects
that tendency.

## Wide baseline result on this machine

Wide mode completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
K_eff_soft = 11.9234
F_selected = 3
F_selected_by_score = 5
v13_3_selected_method = psm_spatial_spectral
v13_3_selected_F = 3
all_data_R2 = 0.9721
localized_R2_X = 0.9721
```

Important wide diagnostics:

```text
coassociation_auc_like = 0.9877
D_localized_auc_like = 0.8320
D_true_auc_like = 0.9838
center_match_max_error = 0.3000
baseline_kmeans_posterior ARI = 0.5119
psm_spatial_spectral at posterior F=3 ARI = 1.0000
psm_spatial_spectral self-scored best F=4 ARI = 0.5434
bgm_truncated_dp selected F=8 ARI = 0.4538
```

Interpretation:

```text
The PSM + spatial prior direction is useful.
When F is supplied by the block posterior, spectral clustering recovers the
toy block structure cleanly.
But affinity-only method scoring still prefers F=4, so the next scoring
improvement should target the F-selection objective rather than the spectral
partition itself.
BGM is currently over-splitting and should stay a comparison baseline, not the
main route.
```
