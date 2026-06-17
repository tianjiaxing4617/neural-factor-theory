# v13.2 迭代贝叶斯 W/K 互证优化方案

这份文档整理当前项目从 v12 到 v13.1 暴露出的主要问题，并把新的优化方向形式化为一个 W/H/K/block 后验互证的迭代贝叶斯框架。

核心判断：v13.1 已经把逐步发现 component K 这一步做得比较接近正确答案；真正需要加强的是 K 被接受之后，W 空间、H 空间、block/F 结构之间还没有形成后验闭环。

---

## 1. 当前代码的状态判断

以 `v13_1_true_iterative_release` 为当前最接近正确方向的版本，它的流程是：

1. 从残差中寻找 split-repeat reliable 的候选方向。
2. 每次只加入一个 candidate。
3. 加入后对所有 H/W 做 global ALS refit。
4. 用 train gain、held-out validation gain、residual eig/null evidence 决定是否接受。
5. K 确定后，再做 Gaussian localization 和 H-side dependency block discovery。

本地 quick trace 显示，前 12 个 component 都被接受；第 13 到 16 次尝试因为 held-out gain 约为 0.00053 而连续拒绝。这说明 K 扩展机制已经有比较强的 evidence gate。

但同一份报告里：

```text
K_selected = 12
F_selected = 5
true F = 3
```

所以现在最值得优化的不是简单调紧 K 阈值，而是让 block/F 结构也进入 W/H/K 的后验互证过程。

---

## 2. 新想法的数学化表述

你提出的方向是对的：如果 W 空间推断准确，它应该强约束 K 和 block 的可能结构；如果 K/block 推断准确，它又应该反过来帮助 W 空间收缩到更稳定、更可解释的位置。

建议把变量分清：

- W 空间：神经元 loading / spatial loading / component footprint。
- H 空间：trial-time expression / latent coefficient。
- K：有效 component 数，应该是后验意义下的 active components。
- B 或 F：component 到 block/group 的分配结构。
- z：神经元排序或低维空间坐标，用于 W 的局部性和 block localization。

真正要估计的是联合后验：

$$
p(W,H,K,B,z,θ | X)
$$

当前 v13.1 更像是单向流程：

$$
K → W,H → B
$$

v13.2 应该变成互证循环：

$$
W,H,K,B,z → W,H,K,B,z → ⋯
$$

直到 held-out evidence、active K、block posterior 都稳定。

---

## 3. 基础生成模型

把 trial-time 展平后，令观测矩阵为：

$$
X ∈ R^{S × N}
$$

其中 S 是 trial-time 样本数，N 是神经元数。基础分解为：

$$
X = H W^T + E
$$

噪声项先用高斯近似：

$$
E_{s,n} ∼ N(0, σ²)
$$

第 k 个 component 包含：

$$
h_k ∈ R^S,   w_k ∈ R^N
$$

v13.2 不应该只保留点估计，而要保留每个 component 的后验均值和不确定性：

$$
q(w_k),   q(h_k),   q(γ_k),   q(b_k)
$$

其中 γ_k 是第 k 个 component 是否 active 的 inclusion variable，b_k 是它属于哪个 block 的分配变量。

---

## 4. W 与 H 的互证更新

### 4.1 给定 H 更新 W

如果暂时固定 H，W 的更新不应该只是最小二乘点估计，而应该保留后验不确定性。第一版可以使用 Gaussian ridge posterior：

$$
q(W | X,H,B,z) ≈ N(M_W, Σ_W)
$$

令噪声精度为：

$$
β = σ^{-2}
$$

给 W 加入结构先验：

$$
w_k ∼ N(μ_{W,k}(B,z), Λ_{W,k}^{-1})
$$

则可以用如下形式更新：

$$
Σ_{W,k} = (β h_k^T h_k I + Λ_{W,k})^{-1}
$$

$$
m_{W,k} = Σ_{W,k}(β X^T h_k + Λ_{W,k} μ_{W,k})
$$

直觉是：H 解释出来的信号越稳定，W 的后验越尖；block/locality prior 越强，W 越倾向落回可解释的神经元局部结构。

### 4.2 给定 W 更新 H

反过来，固定 W 时更新 H：

$$
q(H | X,W,B) ≈ N(M_H, Σ_H)
$$

给 H 加入 trial-time smoothness、label dependency 或 block-level dependency prior：

$$
h_k ∼ N(μ_{H,k}(B), Λ_{H,k}^{-1})
$$

更新形式为：

$$
Σ_{H,k} = (β w_k^T w_k I + Λ_{H,k})^{-1}
$$

$$
m_{H,k} = Σ_{H,k}(β X w_k + Λ_{H,k} μ_{H,k})
$$

这样，W 的后验会成为 H 的新证据，H 的后验又会成为 W 的新证据。

### 4.3 damping 避免震荡

由于 W/H/K/B 是耦合的，直接全量替换可能震荡。建议使用阻尼更新：

$$
M_W^{t+1} = (1-ρ)M_W^t + ρ M̃_W^{t+1}
$$

$$
M_H^{t+1} = (1-ρ)M_H^t + ρ M̃_H^{t+1}
$$

其中：

$$
0 < ρ ≤ 1
$$

第一版可取 `ρ = 0.3` 到 `0.7`，再由 evidence trace 判断是否调节。

---

## 5. K 应该是 active posterior

当前 v13.1 是逐个加入 candidate，然后判断是否接受。这个思路保留，但建议把 K 变成 active posterior。

每个 component 有一个 inclusion variable：

$$
γ_k ∈ {0,1}
$$

对应后验为：

$$
π_k = P(γ_k=1 | X)
$$

有效 K 定义为：

$$
K_eff = Σ_k I[π_k > τ]
$$

v13.2 可以先不用完整 MCMC，而用 empirical Bayes / variational ARD 思路。给每个 component 一个自动相关性精度：

$$
w_k ∼ N(0, α_k^{-1} I)
$$

如果某个 component 解释力弱、split reliability 弱、held-out evidence 弱，则：

$$
α_k ↑,   π_k ↓
$$

当某个残差 candidate 的 posterior evidence 足够强，则加入新 component：

$$
ΔL_heldout > ε,   π_new > τ_add
$$

这样 K 的选择就不再是一次性数字，而是 active components 的后验稳定状态。

---

## 6. block/F 也要做 posterior

当前 v13.1 的核心症状是：

```text
K selected correctly, but F can be over-split.
```

这说明 block discovery 不能只作为 K 之后的后处理。它应该参与 W/H 的后验更新。

对每个 component k，设 block label 为：

$$
b_k ∈ {1,…,F}
$$

我们关心 component i 和 j 是否属于同一 block 的概率：

$$
C_{i,j} = P(b_i=b_j | X)
$$

这个 co-association posterior 由三类证据共同决定：

1. W/locality evidence：两个 component 的 W center、width、overlap 是否支持同一 block。
2. H dependency evidence：两个 component 的 H 是否具有稳定的功能依赖。
3. split-repeat evidence：不同 split、seed、perturbation 下，它们是否反复被放在一起。

可以先用一个 logistic edge model：

$$
logit C_{i,j} = a_0 + a_W S^W_{i,j} + a_H S^H_{i,j} + a_R S^R_{i,j} - a_D D_{i,j} - a_C C_complexity
$$

最终 F 不应该由一次 k-means 或一次 spectral cut 决定，而应该由 partition posterior 决定：

$$
q(B) ∝ exp(S_block(B) - λ_F F)
$$

这一步直接针对 v13.1 中的 F 过度拆分问题。

---

## 7. v13.2 推荐迭代流程

建议新版本命名为：

```text
v13_2_bayesian_coupled_release
```

主循环：

```text
initialize W,H from v13.1 residual candidates
initialize γ_k = 1 for accepted components
initialize block posterior q(B) from W centers + H dependency

repeat until convergence:
    update q(W | X,H,K,B,z)
    update q(H | X,W,K,B)
    update q(γ | W,H,heldout,split)
    propose add/drop/split/merge moves for K
    update q(B | W,H,z,split)
    update hyperparameters θ
    record ELBO, held-out likelihood, active K, block stability

final:
    refit posterior means on all data
    report active K posterior
    report block posterior and consensus F
    run toy oracle diagnostics when truth exists
```

收敛条件建议同时看三件事：

$$
|L^{t+1}-L^t| < ε_L
$$

$$
K_eff^{t+1} = K_eff^t
$$

$$
||C^{t+1}-C^t||_F < ε_C
$$

其中 C 是 block co-association posterior matrix。

---

## 8. 和当前代码的对应关系

可以保留：

- `residual_reliability_candidates`：继续作为新 component proposal 的来源。
- `true_iterative_K_expansion`：保留 train/validation split 和 candidate acceptance 的骨架。
- `refit_HW_als`：改造成 posterior refit 的第一版实现。
- `gaussian_dictionary` 和 `localize_subspace_gaussian_omp`：继续作为 W locality evidence。
- `discover_blocks`：从 hard block discovery 改造成 block posterior update。

建议新增数据结构：

```python
@dataclass
class ComponentPosterior:
    mean_w: np.ndarray
    var_w: np.ndarray
    mean_h: np.ndarray
    var_h: np.ndarray
    inclusion_prob: float
    block_prob: np.ndarray
    reliability: float
    heldout_gain: float
```

建议新增核心函数：

```python
def posterior_refit_WH(X, posterior, cfg):
    ...

def update_inclusion_prob(posterior, evidence, cfg):
    ...

def update_block_posterior(posterior, z, split_runs, cfg):
    ...

def propose_component_moves(residual, posterior, cfg):
    ...

def convergence_check(trace, cfg):
    ...
```

第一版不需要一步到位写完整贝叶斯采样。最现实的 v13.2 是：

```text
deterministic v13.1
+ Gaussian posterior uncertainty
+ ARD-style inclusion probability
+ block co-association posterior
+ nested held-out evidence
```

---

## 9. 当前最需要修的点

### P1. F/block 需要进入闭环

v13.1 的 K 结果已经很好，但 F 可过拆。这不是单纯调阈值的问题，而是 block 结构没有反过来约束 W/H。

优化方向：

```text
hard F selection -> q(B) posterior
single run block graph -> split-repeat co-association matrix
downstream interpretation -> participates in W/H prior
```

### P2. K 应该报告 posterior active set

不要只报告：

```text
K_selected = 12
```

还要报告：

```text
P(γ_k = 1 | X)
K_eff mean
K_eff credible interval
components pruned by ARD
components proposed but rejected
```

### P3. validation evidence 需要更严格

下一版建议区分：

- candidate selection split
- posterior update split
- final reporting split

否则 candidate 被 residual search 看到过，validation evidence 会偏乐观。

### P4. residual normalization 前后都要记录

残差方向搜索可以 normalize，但 evidence 计算应保留 raw residual norm，否则会高估弱残差上的结构显著性。

### P5. z/order 方向需要符号一致化

v12.7 的 consensus z 思路是对的，但有些 split 会出现 mirror sign。v13.2 需要显式记录 orientation alignment：

$$
z ← s z,   s ∈ {-1,1}
$$

选择 s 使其和 consensus z 的相关为正。

---

## 10. 可参考的公开方法

这个项目不是直接套一个现成包就能解决，因为它同时包含 matrix factorization、component number discovery、W locality、H dependency、block graph discovery 和 split-repeat reliability。

但有几类公开方法可以作为设计参照：

- Factor Analysis / probabilistic PCA：低维 latent factors 线性生成观测，并加入 Gaussian noise。参考 scikit-learn FactorAnalysis。
- Bayesian Gaussian Mixture / Dirichlet process：设置最大 component 数，但让模型只激活其中一部分。参考 scikit-learn BayesianGaussianMixture。
- Empirical Bayes Matrix Factorization：从数据中估计 factor/loading 的 prior，并用 variational fitting 做矩阵分解。参考 EBMF / flashr。
- PyMC / ADVI：适合做小 toy 的 sanity check，但不适合作为当前大规模主实现。

参考链接：

- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
- https://arxiv.org/abs/1802.06931
- https://github.com/stephenslab/flashr
- https://www.pymc.io/
- https://arxiv.org/abs/1603.00788

---

## 11. 推荐实施顺序

第一阶段：稳定复制 v13.1。

```text
copy v13_1_true_iterative_release
rename to v13_2_bayesian_coupled_release
keep existing toy generator and outputs
confirm K still reaches 12
```

第二阶段：把 `refit_HW_als` 改成 posterior refit。

```text
return mean_w, var_w, mean_h, var_h
add ridge prior
add damping
record posterior uncertainty
```

第三阶段：加入 ARD/inclusion probability。

```text
initialize K_max
track γ_k
allow low-posterior components to shrink
add residual proposals only when posterior evidence passes threshold
```

第四阶段：替换 block discovery。

```text
multiple split/seed block runs
build co-association matrix C
estimate q(B)
report consensus F and uncertainty
feed q(B) back into W/H prior
```

第五阶段：建立诊断报告。

必须输出：

```text
elbo_trace.csv
heldout_loglik_trace.csv
component_posterior.csv
block_coassociation.csv
block_posterior_summary.csv
v13_2_report.json
```

toy 数据有真值时，还要输出：

```text
K posterior vs Ktrue
F posterior vs Ftrue
component center error
block ARI
W/H reconstruction R2
```

---

## 12. 最终目标

v13.2 的目标不是让分数更漂亮，而是让模型的每个层次都能彼此解释：

```text
W explains H.
H explains W.
W/H explain K.
K explains residual stopping.
W/H/K explain block structure.
block structure feeds back to W/H priors.
```

如果这个闭环成立，模型结果应该表现为：

1. K 不靠 PCA variance，而靠 active posterior 收敛。
2. F 不靠一次聚类，而靠 block posterior 收敛。
3. W 空间稳定，split-repeat 下 center/order 不乱跳。
4. H 空间稳定，trial-time expression 有可重复的功能依赖。
5. held-out evidence、ELBO、K_eff、block co-association 同时收敛。

这会比当前 v13.1 更接近我们真正想要的“可发现、可验证、可解释”的 neural factor discovery。
