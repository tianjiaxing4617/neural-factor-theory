# v13.2.1 full-diagnostics release

This release is a diagnostic freeze of v13.2.  The algorithmic core is kept
close to v13.2, but the output layer is expanded so we can inspect exactly
where recovery succeeds or fails.

Main question for this version:

```text
Are W space, H/K space, and D/block space recovering the same latent structure,
or is one of them breaking the mutual posterior loop?
```

## Run from notebook

Open:

```text
v13_2_1_run_notebook.ipynb
```

Default notebook mode is quick.  Change `MODE = "wide"` in the second code cell
for the full run.

## Run from terminal

Quick:

```bash
python v13_2_1_full_diagnostics_block_discovery.py --quick --outdir v13_2_1_quick_outputs
```

Wide:

```bash
python v13_2_1_full_diagnostics_block_discovery.py --wide --outdir v13_2_1_wide_outputs
```

Config-based wide:

```bash
python v13_2_1_full_diagnostics_block_discovery.py --config config_wide.json --outdir v13_2_1_wide_outputs
```

## Core outputs

The usual v13.2 outputs are still written:

```text
v13_2_1_report.json
v13_2_1_report.md
v13_2_1_iterative_trace.csv
v13_2_1_candidate_trace.csv
v13_2_1_component_posterior.csv
v13_2_1_gaussian_components.csv
v13_2_1_block_candidates.csv
v13_2_1_block_assignment.csv
v13_2_1_block_posterior_summary.csv
v13_2_1_block_coassociation.csv
config_used.json
```

## New full diagnostics

These files are the reason v13.2.1 exists:

```text
v13_2_1_space_diagnostics.csv
v13_2_1_subspace_angles.csv
v13_2_1_center_matching.csv
v13_2_1_component_alignment.csv
v13_2_1_latent_matrices.npz
```

How to read them:

```text
space_diagnostics
  W/H subspace principal angles, canonical correlations, center error,
  component alignment, coassociation pairwise separation, and D-space pairwise
  separation.

subspace_angles
  Per-dimension principal angles between recovered and true W/H subspaces.

center_matching
  Estimated Gaussian center matched to nearest true component center.

component_alignment
  Hungarian-matched true/recovered component correlations for W and H.

latent_matrices.npz
  Raw arrays for later inspection: W/H raw, localized, true matrices,
  D_localized, D_true, coassociation, centers, labels, and selected assignments.
```

## Important plots

```text
v13_2_1_iterative_K_trace.png
v13_2_1_iterative_gain_trace.png
v13_2_1_component_inclusion_posterior.png
v13_2_1_F_posterior_mass.png
v13_2_1_block_coassociation_heatmap.png
v13_2_1_localized_component_centers.png
```

## Quick smoke result on this machine

The quick smoke completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
K_eff_soft = 11.7934
F_selected = 3
F_selected_by_score = 3
all_data_R2 = 0.9816
localized_R2_X = 0.9816
```

Key quick diagnostic signals:

```text
W_raw_subspace_angle_mean_deg = 5.8073
W_raw_subspace_angle_max_deg = 16.9243
H_raw_subspace_angle_mean_deg = 3.2359
H_raw_subspace_angle_max_deg = 6.4611
coassociation_auc_like = 0.8600
D_localized_auc_like = 0.7088
D_true_auc_like = 0.9942
```

This means quick mode already shows a useful pattern: W/H subspaces are strong,
but localized D/block separation is much weaker than the true D separation.
That is exactly the next optimization target.

## Wide baseline result on this machine

The wide run also completed successfully with:

```text
K_selected = 12
K_eff_posterior = 12
K_eff_soft = 11.9234
F_selected = 3
F_selected_by_score = 5
all_data_R2 = 0.9721
localized_R2_X = 0.9721
```

Key wide diagnostic signals:

```text
W_raw_subspace_angle_mean_deg = 5.7528
W_raw_subspace_angle_max_deg = 15.5785
H_raw_subspace_angle_mean_deg = 2.7824
H_raw_subspace_angle_max_deg = 5.2927
selected_block_ARI_vs_nearest_true_group = 0.5119
coassociation_auc_like = 0.9877
D_localized_auc_like = 0.8320
D_true_auc_like = 0.9838
center_match_max_error = 0.3000
```

Interpretation:

```text
W/H subspaces are recovered well.
The posterior co-association matrix is already strong.
D_localized is better than quick mode but still weaker than D_true.
One localized component is pulled to the left boundary center 0.02, so the next
optimization target is the localization/componentization step rather than the
global K discovery step.
```
