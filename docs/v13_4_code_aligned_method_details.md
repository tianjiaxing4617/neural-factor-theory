# v13.4 代码对齐方法细节说明

这份文档只解释当前最新实现：

`v13_4_loss_calibrated_psm_release/v13_4_loss_calibrated_psm_block_discovery.py`

它的目的不是重新包装理论，而是把代码里真正做的事情说清楚。尤其要纠正一个容易犯的错误：当前 K component discovery 不是普通 PCA，也不是直接取残差最大方差方向，而是在残差中寻找跨 split 重复出现的可靠方向。

## 1. 总流程

当前 pipeline 顺序如下：

1. 生成 clustered Gaussian toy 数据。
2. 做 identifiability audit，检查每个真实 group 的 W/H 有效秩。
3. 做 residual split-repeat iterative K expansion。
4. 对接受的 K 维 W/H 做 all-data refit。
5. 计算 component posterior，并做可选 ARD-style pruning。
6. 把接受的 W 子空间转成 Gaussian-localized components。
7. 用 localized H 计算 nonlinear dependency matrix。
8. 用 center 和 dependency 做 baseline block discovery。
9. 对 block discovery 加扰动，得到 F posterior 和 coassociation matrix。
10. 用 PSM、spatial prior、EB shrinkage、NMF、hierarchical、BGM 生成候选 partition。
11. 用 v13.4 posterior loss 选择最终 block summary。
12. 输出完整诊断表、图和 report。

## 2. Toy 生成

toy 中有 `n_groups` 个功能组，每组有 `comps_per_group` 个 component。因此真实 component 数为：

```math
K_{\mathrm{true}} = G M.
```

真实 block 数为：

```math
F_{\mathrm{true}} = G.
```

每个 component 的神经元侧 loading 是一个 Gaussian footprint。第 g 组第 m 个 component 的中心大致为：

```math
\mu_{g,m} = c_g + \delta_m + \epsilon_{g,m}.
```

其中 `c_g` 是 group center，`\delta_m` 是组内 offset，`\epsilon_{g,m}` 是小 jitter。代码中还给 width 和 amplitude 加 jitter，并允许 random sign。

每个 loading 会做去均值、标准化、单位范数归一化：

```math
w_{g,m} \leftarrow {w_{g,m} - \bar w_{g,m} \over \|w_{g,m} - \bar w_{g,m}\|}.
```

H 侧不是简单复制同一个时间模式。每个 group 先生成一个 trial-specific latent trace `q(t)`，然后组内第 m 个 component 使用不同幂次：

```math
q(t),\quad q(t)^2,\quad q(t)^3,\quad q(t)^4.
```

随后代码对这些 powers 做 whitening。这个设计很关键：它避免同组 components 在 H 侧完全坍缩成 rank-1。也就是说，toy 不是只靠 W 中心差异来制造 K，而是同时让 H 侧也有足够可识别的维度。

最后信号为：

```math
X_{\mathrm{signal}}(r,t,n) = \sum_k H(r,t,k) W(n,k).
```

再加 Gaussian noise，并按 neuron 标准化。为了避免算法偷看 neuron order，代码还会 shuffle neuron。

## 3. Identifiability audit

在讨论 recovery 前，代码先检查 toy 本身是否可识别。对每个真实 group，取该 group 内的 W 子矩阵和 H 子矩阵，计算奇异值平方谱：

```math
s_1^2,\ldots,s_m^2.
```

effective rank 使用 participation ratio：

```math
r_{\mathrm{eff}} =
{(\sum_i s_i^2)^2 \over \sum_i s_i^4}.
```

如果一个 group 的 W 或 H effective rank 太低，那么即使 toy 名义上有多个 components，观测层面也可能只支持少数方向。这一步是为了区分“算法没恢复”和“目标本来不可识别”。

## 4. ALS refit 不是 K discovery 本身

代码使用 ALS 是为了在给定 W 初值和 K 的情况下做 global refit。它不是用来单独定义 component 的理论来源。

给定当前 W，先更新 H：

```math
H = X W (W^\top W + \lambda_H I)^{-1}.
```

再更新 W：

```math
W = X^\top H (H^\top H + \lambda_W I)^{-1}.
```

每轮更新后，W 会重新中心化并 QR 正交化。最后用：

```math
\widehat X = H W^\top
```

计算训练 R-squared。

注意：在 validation 上，代码使用的是 projection reconstruction：

```math
\widehat X_{\mathrm{val}} = X_{\mathrm{val}} Q Q^\top,
```

其中 Q 是 W 的正交基。这是在检验当前 W 子空间能否解释 held-out data。

## 5. K discovery：不是 PCA，而是 residual split-repeat reliability

这是当前代码最重要的部分。

假设当前已经接受 q 个 components，对训练数据做 refit 后，代码把训练数据投影到已接受 W 子空间的正交补上：

```math
R_q = X_{\mathrm{train}} - X_{\mathrm{train}} Q_q Q_q^\top.
```

这里 Q_q 是当前 W 的正交基。也就是说，新 component 必须来自旧 W 子空间之外的残差。

### 5.1 split templates

代码反复把训练 trial 随机分成 A/B 两半。每次 split 中：

1. A 半 trials 取平均，得到一个 T by N 的残差视图。
2. B 半 trials 也取平均，得到另一个 T by N 的残差视图。
3. 每个视图按 neuron 方向做 z-score。
4. 多次 split 的结果沿 time 维堆叠。

最终得到：

```math
X_A \in \mathbb{R}^{S \times N},
\qquad
X_B \in \mathbb{R}^{S \times N}.
```

这里 S 等于 `n_residual_splits` 乘以 time points。

### 5.2 cross-split repeat matrix

普通 PCA 会看单个矩阵的方差，但这里代码看 A/B 之间的重复性：

```math
C_{\mathrm{rel}}
=
{X_A^\top X_B + X_B^\top X_A \over 2(S-1)}.
```

这个矩阵强调的是：某个神经元 loading 方向是否同时出现在 A 和 B 两个独立片段中。

如果一个方向只是在某个 split 里偶然出现，它不会在 cross-split matrix 中稳定变强。如果它是真实 residual component，它应该在 A 和 B 里都出现。

### 5.3 投影掉已接受子空间

如果已经有 accepted W，代码构造：

```math
P = I - Q_q Q_q^\top.
```

然后做：

```math
C_{\mathrm{rel}} \leftarrow P C_{\mathrm{rel}} P.
```

这一步防止算法再次发现已经接受过的 component。

### 5.4 候选方向

代码对对称化后的 C_rel 做 eigen decomposition。前几个 eigenvectors 是候选 W 方向。

候选方向 v 还会再次从旧 W 子空间中正交化，并归一化：

```math
v \leftarrow {v - Q_q Q_q^\top v \over \|v - Q_q Q_q^\top v\|}.
```

然后计算它与旧 components 的最大相似度：

```math
d(v) = \max_j |\langle q_j, v\rangle|.
```

如果 `d(v)` 超过 `max_duplicate_corr`，这个候选会被标记为 duplicate 并跳过。

这里要特别区分两个词：

1. split-repeat reliable：同一个新方向在 A/B split 里重复出现，这是好事。
2. duplicate with existing W：新方向和已经接受的旧方向重复，这是坏事。

当前代码寻找的是第一种重复，避免的是第二种重复。

### 5.5 null calibration

为了判断 repeat strength 是否只是噪声，代码构造 null。它会打乱 XB 的行，破坏 A/B 之间的重复关系，再计算 null cross-split matrix 的最大 eigenvalue。

重复多次后得到 null top eigenvalue 分布，并取分位数：

```math
Q_{\mathrm{null}} = \mathrm{quantile}_{0.95}(\lambda_{\max}^{\mathrm{null}}).
```

候选方向的 residual reliability evidence 为：

```math
E_{\mathrm{rel}} = {\lambda_{\mathrm{cand}} \over Q_{\mathrm{null}}}.
```

wide config 中要求：

```math
E_{\mathrm{rel}} \ge 1.02.
```

### 5.6 held-out acceptance

每个非 duplicate candidate 都会被加入当前 W，形成 W_try。然后代码用 ALS 在 train trials 上 global refit，得到 W_try_fit。

接着在 validation trials 上计算 projection R-squared：

```math
R^2_{\mathrm{val}}(K+1).
```

相对于当前模型的 validation gain 为：

```math
\Delta R^2_{\mathrm{val}}
=
R^2_{\mathrm{val}}(K+1)
-
R^2_{\mathrm{val}}(K).
```

候选还会记录 train gain：

```math
\Delta R^2_{\mathrm{train}}.
```

代码对候选做一个临时排序分数：

```math
\mathrm{score}
=
\Delta R^2_{\mathrm{val}}
+ 0.25\Delta R^2_{\mathrm{train}}
+ 0.0001 E_{\mathrm{rel}}
- 0.001 d(v).
```

这个 score 只是用于在 top candidates 里选择 best candidate。最终是否接受还要满足硬条件：

```math
\Delta R^2_{\mathrm{val}} \ge \tau_{\mathrm{val}},
```

```math
\Delta R^2_{\mathrm{train}} \ge \tau_{\mathrm{train}},
```

```math
E_{\mathrm{rel}} \ge \tau_{\mathrm{rel}}.
```

如果接受，就更新 W，并再次 global refit。若连续拒绝次数达到 `patience`，K expansion 停止。

最后代码用所有数据对接受的 K 维 W/H 做 final all-data refit。

所以，当前 K 的准确定义是：

> K 是 residual 中仍能跨 split 重复出现、强于 null、并且能提升 held-out reconstruction 的可靠方向数量。

它不是传统 PCA rank，也不是简单最大方差方向。

## 6. Component posterior 与 pruning

K expansion 结束后，代码给每个 accepted component 一个 empirical posterior-style inclusion probability。

首先计算 residual noise variance：

```math
\sigma^2_{\mathrm{res}} = \mathrm{var}(X - \widehat X).
```

第 k 个 component 的能量 proxy 为：

```math
E_k = \|h_k\|^2 \|w_k\|^2.
```

每个 component 从接受 trace 中取出：

1. validation gain。
2. train gain。
3. eig-over-null。
4. component energy。

代码构造 logit：

```math
\ell_k
=
b
+ \alpha_g \log(1+\Delta R^2_{\mathrm{val},k}/s_g)
+ \alpha_r \log(E_{\mathrm{rel},k})
+ \alpha_e \log(1+E_k/\mathrm{median}(E)).
```

然后：

```math
\pi_k = {1 \over 1+\exp(-\ell_k)}.
```

如果 `pi_k` 小于 `prune_threshold`，该 component 会被剪枝。剪枝后代码会用剩余 W 重新做 all-data ALS refit。

注意：这不是严格完整 Bayesian posterior。它是 empirical-Bayes / variational-style reliability score，用来表达 component active probability 的近似。

代码还计算 posterior variance proxy：

```math
\mathrm{Var}(w_k) \approx {1 \over \beta \|h_k\|^2+\lambda_w},
```

```math
\mathrm{Var}(h_k) \approx {1 \over \beta \|w_k\|^2+\lambda_h},
```

其中：

```math
\beta = 1/\sigma^2_{\mathrm{res}}.
```

## 7. Localization：解释旋转，不是重新发现 K

accepted W 是一个 K 维子空间。由于因子分解有旋转不唯一性，单个 raw component 不一定可解释。代码因此构造 Gaussian dictionary：

```math
D = [a_1,\ldots,a_J].
```

每个 atom 是一个按 neuron coordinate z 定义的 Gaussian footprint，并做去均值和单位范数归一化。

对 accepted W 子空间取正交基 Q。每个 atom 在子空间内的能量为：

```math
s_j = \|Q^\top a_j\|^2.
```

代码用 greedy OMP-like 方式选 K 个 atoms。选择时加入 center redundancy penalty，避免多个 atom 挤在同一个位置：

```math
\mathrm{score}_j
=
\|Q_{\mathrm{res}}^\top a_j\|^2
-
\rho \exp[-(d_j/\delta)^2].
```

选出 atoms 后，并不是直接把 Gaussian atoms 当作 W，而是先投影回 accepted W 子空间：

```math
W_{\mathrm{loc}}
=
\mathrm{orth}\{ Q Q^\top D_{\mathrm{selected}}\}.
```

再得到 localized H：

```math
H_{\mathrm{loc}} = X W_{\mathrm{loc}}.
```

所以 localization 的数学含义是：

> 在已经被 split-repeat K discovery 支撑的 W 子空间里，选择一个空间上可解释的 Gaussian rotation。

它不是传统 PCA，也不是重新估计 K。

## 8. H-side nonlinear dependency

当前 block 不能只看 W center。代码还用 localized H 计算 component 之间的 nonlinear dependency。

对 H 的每一列，代码构造 polynomial transforms：

```math
h_k,\quad h_k^2,\quad h_k^3.
```

然后对 component i 和 j，计算所有 transform 之间的最大绝对相关：

```math
D_{ij}
=
\max_{a,b\in\{1,2,3\}}
|\mathrm{corr}(h_i^a,h_j^b)|.
```

这是一个轻量 dependency proxy。它能捕捉 q 与 q-squared、q-cubed 的关系，但还不是完整 Bayesian dependency posterior。

## 9. Baseline block discovery

baseline block discovery 先对 localized component centers 做 one-dimensional kmeans。对每个 F，得到 labels。

然后统计：

1. 同 block 的 center distance 均值。
2. 不同 block 的 center distance 均值。
3. 同 block 的 H dependency 均值。
4. 不同 block 的 H dependency 均值。
5. block size balance。
6. singleton penalty。

空间分离分数：

```math
S_{\mathrm{space}}
=
\tanh\left(
{d_{\mathrm{between}} \over 3(d_{\mathrm{within}}+\epsilon)}
\right).
```

H dependency contrast：

```math
\Delta_D
=
D_{\mathrm{within}} - D_{\mathrm{between}}.
```

dependency 分数：

```math
S_D = \tanh(3\Delta_D).
```

baseline block score 为：

```math
S_{\mathrm{block}}
=
0.45S_{\mathrm{space}}
+0.35S_D
+0.10S_{\mathrm{balance}}
-0.05F
-P_{\mathrm{small}}.
```

这个 baseline score 后面不是最终 v13.4 选择标准，而是用于生成和诊断 block surface。

## 10. Block posterior 和 PSM

代码接着用 perturbation 近似 block posterior。

每次 run：

1. 给 component centers 加 Gaussian jitter。
2. 给 dependency matrix 加对称噪声。
3. 对每个 F 重新做 center kmeans。
4. 重新计算 block score。
5. 加额外 complexity penalty、balance reward、singleton penalty。

得到每个 partition 的 posterior log score：

```math
S^{(s)}_{\mathrm{post}}
=
S^{(s)}_{\mathrm{block}}
- \lambda_F F
+ \lambda_B S_{\mathrm{balance}}
- P_{\mathrm{singleton}}.
```

然后 softmax 成 partition weights：

```math
\omega_s
=
{\exp(S^{(s)}_{\mathrm{post}}/T)
\over
\sum_r \exp(S^{(r)}_{\mathrm{post}}/T)}.
```

PSM/coassociation matrix 为：

```math
C_{ij}
=
\sum_s \omega_s I(z_i^{(s)} = z_j^{(s)}).
```

同时得到 F posterior：

```math
P(F=f)
=
\sum_{s:F_s=f}\omega_s.
```

这一层的含义是：不要相信某一次 hard clustering，而是总结 perturbation posterior 下两个 components 同 block 的概率。

## 11. v13.4 候选 partition 生成

v13.4 不只用一个 partition 方法，而是生成一批候选，然后统一评分。

### 11.1 PSM affinity

基础 affinity 是 PSM：

```math
A_{\mathrm{PSM}} = C.
```

### 11.2 spatial prior affinity

代码根据 localized center 距离构造 spatial prior：

```math
P_{ij}
=
p_0 + (1-p_0)
\exp\left[-{(c_i-c_j)^2 \over 2\tau^2}\right].
```

然后：

```math
A_{\mathrm{spatial}} = \mathrm{normalize}(C \odot P).
```

这相当于轻量 dd-IBP-style locality prior：空间近的 components 更倾向同 block，但不会完全压死远距离 evidence。

### 11.3 EB shrink affinity

代码为每个 component 计算 reliability：

```math
\rho_k
=
0.50\pi_k
+0.30e_k
+0.20b_k.
```

其中 `pi_k` 是 inclusion probability，`e_k` 是 subspace energy proxy，`b_k` 是远离边界的程度。然后：

```math
A^{\mathrm{EB}}_{ij}
=
(1-\eta)A_{ij}
+\eta A_{ij}\sqrt{\rho_i\rho_j}.
```

低可靠 component 的 affinity 会被 shrink。

### 11.4 candidate methods

代码比较以下候选：

1. `baseline_kmeans_posterior`：上一层 perturbation posterior 选出的 baseline partition。
2. `psm_spectral`：直接对 PSM 做 spectral clustering。
3. `psm_spatial_spectral`：对 spatial-prior PSM 做 spectral clustering。
4. `eb_shrink_psm_spatial_spectral`：对 EB-shrunk spatial PSM 做 spectral clustering。
5. `psm_nmf_soft`：用 NMF 对 spatial PSM 得到 soft block membership，再取 argmax。
6. `spatial_hierarchical_psm`：把 PSM distance 和 center distance 混合后做 hierarchical clustering。
7. `bgm_truncated_dp`：BayesianGaussianMixture 作为 truncated-DP-style baseline。

BGM 只是比较路线，不在默认 selection pool 里作为最终偏好路线。

## 12. 两套评分：method_score 和 v13.4 loss

代码保留了 affinity-based method_score，但最终 v13.4 主要看 loss-calibrated score。

### 12.1 method_score

对一个候选 partition，先计算 affinity 的 within-between contrast：

```math
\Delta_A
=
A_{\mathrm{within}} - A_{\mathrm{between}}.
```

method_score 大致为：

```math
S_{\mathrm{method}}
=
\Delta_A
+\lambda_B B
-\lambda_F F
-P_{\mathrm{singleton}}
-\lambda_H H(A).
```

然后加 F posterior prior bonus：

```math
\mathrm{bonus}(F)
=
\lambda_{\mathrm{prior}}
\log {P(F) \over P_0(F)}.
```

method_score 用于保留每个方法自己的最佳结果，但它不是最终最重要的选择逻辑。

### 12.2 weighted Binder risk

v13.4 loss 从 PSM 的 decision loss 开始。若 candidate 把 i,j 分开，却有很高 posterior same-block probability C_ij，就付 split risk。若 candidate 把 i,j 合并，但 C_ij 很低，就付 merge risk。

```math
R_B(z)
=
\mathrm{mean}_{i \lt j}
\left[
a C_{ij} I(z_i \neq z_j)
+ b(1-C_{ij})I(z_i=z_j)
\right].
```

wide config 中：

```math
a=2.25,\qquad b=1.00.
```

因此 split 错误比 merge 错误更贵，用来抑制 v13.3 暴露出的 over-split 倾向。

### 12.3 PSM cross entropy

```math
R_{\mathrm{CE}}(z)
=
-\mathrm{mean}_{i \lt j}
\left[
I(z_i=z_j)\log C_{ij}
+ I(z_i \neq z_j)\log(1-C_{ij})
\right].
```

### 12.4 weighted SBM/MDL

对每个 block pair，代码估计一个平均 PSM probability：

```math
p_{ab}
=
\mathrm{mean}(C_{ij}:z_i=a,z_j=b).
```

然后用 Bernoulli-style weighted log likelihood：

```math
\ell_{\mathrm{SBM}}
=
\sum_{i \lt j}
\left[
C_{ij}\log p_{z_i,z_j}
+(1-C_{ij})\log(1-p_{z_i,z_j})
\right].
```

MDL risk 为：

```math
R_{\mathrm{MDL}}
=
{-\ell_{\mathrm{SBM}}+\lambda q\log N_p \over N_p}.
```

其中 q 是有效 block-pair 参数数，N_p 是 pair 数。

### 12.5 最终 v13.4 loss

最终 loss 是：

```math
L(z)
=
R_B(z)
+\lambda_{\mathrm{CE}}R_{\mathrm{CE}}(z)
+\lambda_{\mathrm{MDL}}R_{\mathrm{MDL}}(z)
+\lambda_F F
+P_{\mathrm{singleton}}
-\lambda_A\Delta_A
-\lambda_B B
-\lambda_P\mathrm{bonus}(F).
```

代码选择 selection pool 中 v13.4 loss 最小的 candidate：

```math
\widehat z
=
\arg\min_z L(z).
```

也就是选择 `v13_4_loss_score = -L(z)` 最大的候选。

## 13. 输出文件对应关系

核心输出如下：

| 文件 | 含义 |
|---|---|
| `config_used.json` | 实际使用的完整配置 |
| `v13_4_identifiability_audit.csv` | 每个真实 group 的 W/H effective rank |
| `v13_4_iterative_trace.csv` | K expansion 每轮是否接受、原因、gain、null evidence |
| `v13_4_candidate_trace.csv` | 每轮 top residual split-repeat candidates |
| `v13_4_component_posterior.csv` | component inclusion probability、ARD proxy、variance proxy |
| `v13_4_gaussian_components.csv` | localized Gaussian center、width、subspace energy |
| `v13_4_block_candidates.csv` | baseline block score surface 与 F posterior |
| `v13_4_block_assignment.csv` | baseline posterior selected assignments |
| `v13_4_block_coassociation.csv` | PSM/coassociation matrix |
| `v13_4_block_posterior_summary.csv` | F posterior |
| `v13_4_method_candidates.csv` | 所有 v13.4 candidate partitions 与 scores |
| `v13_4_method_summary.csv` | 每种方法的代表性 selected rows |
| `v13_4_method_assignment.csv` | v13.4 每个 candidate 的 component assignment |
| `v13_4_affinity_psm.csv` | 原始 PSM affinity |
| `v13_4_affinity_psm_spatial.csv` | 加 spatial prior 的 affinity |
| `v13_4_affinity_eb_shrink_psm_spatial.csv` | EB shrink 后的 affinity |
| `v13_4_space_diagnostics.csv` | W/H 子空间、D、coassociation、center matching 等总结 |
| `v13_4_subspace_angles.csv` | raw/localized W/H 与真值的 principal angles |
| `v13_4_center_matching.csv` | localized centers 与 true component centers 的匹配 |
| `v13_4_component_alignment.csv` | W/H component-level alignment 诊断 |
| `v13_4_latent_matrices.npz` | raw/localized/true latent matrices 与诊断矩阵 |
| `v13_4_report.json` | 机器可读总报告 |
| `v13_4_report.md` | 人类可读总报告 |

## 14. 目前实现中需要小心的地方

第一，component posterior 是 empirical posterior-style score，不是严格从完整 generative prior 和 likelihood 推出的 Bayesian posterior。

第二，K discovery 证明的是 split-repeat reliable residual direction，不是每个 biological latent factor 的唯一恢复。

第三，如果 artifact 在 split A/B 中也稳定重复，它可能被当作 reliable direction。因此 null 和 validation gate 很关键。

第四，如果两个真实 components 的 W/H 都高度共线，代码可能只恢复一个有效方向。这是可识别性限制，不一定是算法错误。

第五，`Cavg` 在 residual candidate search 中被计算并投影，但当前没有进入候选评分。这不是 bug，但说明当前 K candidate 主要由 cross-split repeat matrix C_rel 驱动。

第六，H dependency 当前只用了 powers 1、2、3 的轻量 proxy。toy H 生成可以包含到 q^4，所以 dependency proxy 不是完整充分统计量。

第七，`F_selected`、`F_selected_by_score` 和 `v13_4_selected_F` 不是同一个概念。前两个来自 baseline block posterior/hard score，最后一个来自 advanced candidate pool 的 v13.4 loss selection。报告时必须区分。

第八，BGM 是 comparison baseline，不是当前主路线。

## 15. 一句话总结

当前 v13.4 方法不是传统 PCA 加 kmeans。更准确地说，它是：

> residual split-repeat reliable K discovery + global ALS refit + empirical component posterior + Gaussian localization rotation + perturbation PSM posterior + loss-calibrated block summary selection.

这句话才和最新代码一致。
