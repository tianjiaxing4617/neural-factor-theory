# v13.4 loss-calibrated PSM block discovery release

v13.4 keeps the v13.2.1/v13.3 K-discovery and full-diagnostics core, then
changes the block-summary selection rule.

The v13.3 result showed an important issue: the PSM/spectral partition itself
was useful, but an affinity-only score could prefer an over-split F.  v13.4
therefore evaluates the same family of candidate partitions with a posterior
clustering loss.

## Main idea

Let `C_ij` be the posterior same-block probability from the coassociation
matrix.  A proposed hard summary `z` is scored by a weighted Binder-style risk:

```text
split risk: merge posterior-same evidence is lost when z_i != z_j
merge risk: posterior-different evidence is lost when z_i == z_j
```

The final v13.4 score combines:

```text
weighted Binder risk
+ PSM cross-entropy proxy
+ weighted-SBM / MDL risk
+ small complexity and singleton penalties
- F posterior support
- affinity contrast support
```

The goal is to let the posterior clustering evidence choose F, instead of
manually forcing the v13.3 preferred method to use the posterior F.

## Candidate methods

v13.4 compares:

```text
baseline_kmeans_posterior
PSM spectral clustering
PSM spectral + spatial prior
EB-shrink PSM spectral
PSM-NMF soft block summary
spatial-constrained hierarchical clustering
BayesianGaussianMixture / truncated-DP comparison
```

BGM remains a comparison baseline, not the preferred route.

## Run from notebook

Open:

```text
v13_4_run_notebook.ipynb
```

Default notebook mode is quick.  Change `MODE = "wide"` in the second code cell
for the full diagnostic run.

## Run from terminal

Quick:

```bash
python v13_4_loss_calibrated_psm_block_discovery.py --quick --outdir v13_4_quick_outputs
```

Wide:

```bash
python v13_4_loss_calibrated_psm_block_discovery.py --wide --outdir v13_4_wide_outputs
```

On this machine, the package-complete Python was:

```text
C:\Users\tianj\anaconda3\python.exe
```

## New outputs

The v13.4 comparison layer writes:

```text
v13_4_method_summary.csv
v13_4_method_candidates.csv
v13_4_method_assignment.csv
v13_4_soft_membership_nmf.csv
v13_4_affinity_psm.csv
v13_4_affinity_psm_spatial.csv
v13_4_affinity_eb_shrink_psm_spatial.csv
```

Useful plots:

```text
v13_4_loss_score_comparison.png
v13_4_loss_candidate_surface.png
v13_4_binder_risk_surface.png
v13_4_method_score_comparison.png
v13_4_method_ARI_comparison.png
v13_4_spectral_candidate_surface.png
```

The full recovery diagnostics are still present:

```text
v13_4_report.json
v13_4_report.md
v13_4_space_diagnostics.csv
v13_4_subspace_angles.csv
v13_4_center_matching.csv
v13_4_component_alignment.csv
v13_4_latent_matrices.npz
```

## Quick smoke result on this machine

Quick mode completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
F_selected = 3
F_selected_by_score = 3
v13_4_selected_method = psm_nmf_soft
v13_4_selected_F = 3
all_data_R2 = 0.9816
localized_R2_X = 0.9816
```

Quick mode is intentionally small, so one boundary-near component can still be
ambiguous.  Use wide mode for the meaningful comparison.

## Wide result on this machine

Wide mode completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
K_eff_soft = 11.9099
F_selected = 3
F_selected_by_score = 3
v13_4_selected_method = psm_spectral
v13_4_selected_F = 3
v13_4_selected_loss_score = 0.3104
all_data_R2 = 0.9711
localized_R2_X = 0.9711
```

Important wide diagnostics:

```text
W_raw_subspace_angle_mean_deg = 6.0082
H_raw_subspace_angle_mean_deg = 2.8223
selected_block_ARI_vs_nearest_true_group = 1.0000
coassociation_auc_like = 1.0000
D_localized_auc_like = 0.9097
D_true_auc_like = 0.9838
center_match_max_error = 0.0583
```

Interpretation:

```text
The v13.4 loss no longer prefers the v13.3 over-split F=4 candidate.
The selected wide summary is F=3 and aligns with the true block structure.
D_localized improves relative to the previous stored v13.3 wide result, though
it remains below D_true, so the next bottleneck is still localized dependency
estimation rather than K discovery.
```
