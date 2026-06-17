# v12.6 神经轨迹 Factor / Block 模型：数学思路、推导过程与问题审查

## 0. 先给结论

目前这条模型线已经形成了一个相当清楚的数学结构：

1. 先不直接恢复“真实 latent factor”，而是恢复神经数据中可重复、可验证的 reliable subspace。
2. 再在这个 reliable subspace 内寻找可解释 component。
3. 最后把 components 组织成功能 block。

也就是说，当前模型不是：

```text
X -> 直接恢复真实 factor
```

而是：

```text
X
-> split-repeat reliable subspace, K
-> localized components
-> H/W dependency graph
-> functional blocks, F
```

这个思路是对的，尤其是把 `K discovery` 和 `F interpretation` 分开，是目前最稳的理论内核。

但现在模型的主要问题也很清楚：**K 的可靠方向发现比较稳，F/block 解释仍然高度依赖 localization、z-order 和 graph score。** 换句话说，当前最危险的地方不是“能不能重构”，而是“从可靠 subspace 旋转成哪些 components，再把它们分成哪些 blocks”。

---

## 1. 数据与符号

设有多 trial 神经数据：

```text
X ∈ R^{R × T × N}
```

其中：

- `R` 是 trial 数；
- `T` 是每个 trial 的时间点数；
- `N` 是神经元数；
- 展平后记为：

```text
M = R T
X_flat ∈ R^{M × N}
```

当前 toy 中假设有：

```text
F = 3                 functional groups
C = 4                 components per group
K_true = F C = 12
```

第 `g` 个功能组内有 `C` 个 component。整体信号写作：

```text
X = H W^T + ε
```

其中：

```text
H ∈ R^{M × K}
W ∈ R^{N × K}
ε ∈ R^{M × N}
```

第 `k` 个 component 的贡献为：

```text
E_k = h_k w_k^T
```

第 `g` 个 group 的贡献为：

```text
X_g = Σ_{m=1}^C h_{g,m} w_{g,m}^T
```

从 v1 到 v12.6 的核心思想可以压缩成一句话：

> 我们不能把每一个 recovered component 直接当成真实 latent factor；我们只能先找可靠表达子空间，再在可识别条件允许的范围内解释 component 和 block。

---

## 2. Toy 生成模型：为什么 v12.5/v12.6 要改 toy

### 2.1 原 toy 的不可识别性

旧 toy 的问题是，同一个功能组内的多个 W loading 近似共线：

```text
w_{g,m}(n) ≈ a_{g,m} u_g(n)
```

于是第 `g` 个 group 的信号：

```text
X_g(t,n)
= Σ_m h_{g,m}(t) w_{g,m}(n)
≈ Σ_m h_{g,m}(t) a_{g,m} u_g(n)
= h̃_g(t) u_g(n)
```

其中：

```text
h̃_g(t) = Σ_m a_{g,m} h_{g,m}(t)
```

这说明：虽然名义上 group 内有 4 个 H component，但观测数据中主要只看到一个合成 timecourse 乘以一个 spatial profile。

也就是：

```text
rank(W_g) ≈ 1
rank(X_g) ≈ 1
```

因此，在这种 toy 上要求模型恢复：

```text
K = 12
F = 3
每组 4 个 component
```

是不公平的。因为数据本身并没有给出 12 个独立方向的可观测证据。

这就是 v12.5/v12.6 的第一个重要修正：**先让 toy 本身进入 block-identifiable regime。**

### 2.2 v12.6 clustered-Gaussian W

现在每个 component 都有自己的 Gaussian loading：

```text
w_{g,m}(z_n)
= a_{g,m} exp(-(z_n - μ_{g,m})^2 / (2 σ_{g,m}^2))
  + η_{g,m}(n)
```

同一 group 内 component 的中心形成局部 cluster：

```text
μ_{g,m} = μ_g + δ_{g,m}
δ_{g,m} ∈ [-d_g, d_g]
```

所以同组 component：

- 空间上相近；
- 但不完全共线；
- 因此有机会被恢复为多个可分辨 component；
- 再由 block graph 合并成功能 group。

这一步的数学意义是：

```text
rank(W_g) 不再约等于 1
```

而是希望：

```text
effective_rank(W_g) 接近 C
```

### 2.3 H-side 也必须可识别

只让 W-side 可识别还不够。若 H-side 的内部维度也高度相关，例如：

```text
q, q^2, q^3, q^4
```

在实际时间分布上接近低秩，那么 group 仍然不可识别。

因此 v12.6 使用 whitened powers：

```text
Φ_g = [q_g, q_g^2, q_g^3, q_g^4]
H_g = QR(zscore(Φ_g))
```

或者等价地理解为：

```text
H_g = Φ_g (Φ_g^T Φ_g + ηI)^(-1/2)
```

目标是让：

```text
effective_rank(H_g) 接近 C
effective_rank(W_g) 接近 C
```

这就是 identifiability audit 的数学意义。

---

## 3. 可识别性审计

对每个 true group，计算 H/W 的奇异值：

```text
s_1, ..., s_C
```

用 entropy effective rank：

```text
p_i = s_i / Σ_j s_j
effective_rank = exp(-Σ_i p_i log p_i)
```

如果：

```text
effective_rank(H_g) ≈ C
effective_rank(W_g) ≈ C
```

说明这个 group 在 toy 中确实有 C 个可观测内部方向。

如果其中任何一边接近 1：

```text
effective_rank(H_g) ≈ 1
或
effective_rank(W_g) ≈ 1
```

那么要求模型恢复 4 个 component 就不合理。

### 这里的潜在问题

1. 这个 audit 在 toy 中可以用 true H/W；真实数据中没有 true H/W。
2. 因此真实数据需要替代指标，例如 split subspace rank、condition number、subspace survival、held-out perturbation sensitivity。
3. 当前 audit 只能说明 toy 是否公平，不能直接证明真实数据中的 K/F 可识别。

---

## 4. K discovery：从 PCA variance 到 split-repeat reliability

### 4.1 为什么不是 PCA

PCA 找的是最大方差方向：

```text
maximize_w Var(Xw)
```

但我们真正关心的是可重复方向：

```text
同一神经 loading direction 是否在不同 trial split 中产生一致表达？
```

所以当前模型构造 split views。

### 4.2 split views

对每次 split，把 trials 分成两半：

```text
A_s(t,n) = mean_{r in split A} X_r(t,n)
B_s(t,n) = mean_{r in split B} X_r(t,n)
```

把所有 split/time 拼接：

```text
A ∈ R^{S T × N}
B ∈ R^{S T × N}
```

其中 `S` 是 split 数。

构造可靠协方差：

```text
C_rel = (A^T B + B^T A) / (2 S T)
```

求特征分解：

```text
C_rel w_k = λ_k w_k
```

如果 `w_k` 是真实可重复方向，则：

```text
h_A = A w_k
h_B = B w_k
```

应该高度一致。

### 4.3 null threshold

为了判断 `λ_k` 是否超过偶然水平，构造 null：

```text
B_null = B with permuted neuron identity
```

得到 null eigenvalue distribution。

可靠方向条件：

```text
λ_k > quantile_0.95(λ_k^null)
```

### 4.4 residual-driven iterative K

当前不是一次性取 top-K，而是迭代式发现：

```text
W_0 = empty
R_0 = X
```

第 `K+1` 步：

```text
R_K = X - projection_X_on_span(W_K)
```

在 residual 上重复 split-repeat reliability：

```text
w_{K+1} = top eigenvector of C_rel(R_K)
```

再把它投影出已有子空间：

```text
w_{K+1} <- (I - Q_K Q_K^T) w_{K+1}
```

其中：

```text
Q_K = orth(W_K)
```

候选方向被接受需要满足：

```text
λ_resid > λ_null × min_lambda_ratio
ΔR2_val > min_gain   or   relative gain > min_rel_gain
split_corr > min_split_corr
```

接受后：

```text
W_{K+1} = orth([W_K, w_{K+1}])
```

重构使用：

```text
X_hat_K = X Q_K Q_K^T
```

因此这个阶段本质上是在找：

```text
可重复、能提升 held-out reconstruction 的 neural loading subspace
```

### 4.5 posthoc K

由于 residual tail 也可能有微弱可靠性，模型允许先接受较多 candidate，再用 held-out gain elbow 回退：

```text
若某个 accepted step 的 gain < posthoc_gain_floor
则 K_posthoc = previous K
```

这对应 v12.5 的思想：

```text
先让 residual process 暴露 tail，再用 gain floor 判断 tail/noise regime
```

### 4.6 v12.6 K evidence proxy

v12.6 又加入一个 Bayesian-evidence-style stopping。

给定 K 维子空间 `Q_K`，假设：

```text
X ~ Normal(0, Σ_K)
Σ_K = Q_K diag(σ_1^2, ..., σ_K^2) Q_K^T
      + σ_res^2 (I - Q_K Q_K^T)
```

训练集估计：

```text
S_train = X_train Q_K
σ_k^2 = Var(S_train[:,k])
σ_res^2 = mean((X_train - S_train Q_K^T)^2)
```

验证集 log likelihood：

```text
LL_K =
-1/2 [
  M_val N log(2π)
  + M_val logdet(Σ_K)
  + quadratic_term
]
```

其中：

```text
logdet(Σ_K)
= Σ_k log σ_k^2 + (N-K) log σ_res^2
```

复杂度惩罚：

```text
Occam_K = 1/2 × complexity_weight × p_eff(K) × log(M_val)
```

其中当前代码使用：

```text
p_eff(K) = K N - K(K+1)/2 + K + 1
```

最终：

```text
FreeEnergy_K = LL_K - Occam_K
K_evidence = argmax_K FreeEnergy_K
```

### K 阶段目前的主要问题

#### 问题 K1：validation 被二次使用

candidate 接受已经用过：

```text
ΔR2_val
```

随后 evidence 又在同一批 validation trials 上算：

```text
FreeEnergy_K
```

这会造成 double dipping。严格来说，现在的 evidence 不能被称为独立 held-out model comparison。

更稳的做法：

```text
train split:   发现 candidate
validation:    调 K / acceptance
test split:    最后 evidence comparison
```

或者使用 nested cross-validation / cross-fitting。

#### 问题 K2：evidence 是 subspace covariance proxy，不是完整 factor model evidence

当前 evidence 评价的是：

```text
low-rank covariance subspace
```

而不是：

```text
H/W factor expression model
```

所以它可以支持“这个 K 维 subspace 是否有预测价值”，但不能直接证明“这 K 个 components 是真实 factor components”。

#### 问题 K3：complexity_weight 是经验常数

`complexity_weight` 不是由真正的 marginal likelihood 推出来的，而是一个 tuning knob。

因此不同数据规模下：

```text
quick / full / wide
```

这个惩罚是否可比，需要额外 calibration。

#### 问题 K4：报告中的 final R2 不是 selected K 的 R2

当前报告里的 final train/val R2 来自最大 accepted K，而不是最终 selected K。

这会导致：

```text
K_selected_final = 12
但 reported final_val_R2 是 K=20 或 K=24 的结果
```

所以报告层面需要重新计算：

```text
R2(W_final)
```

---

## 5. 从 reliable subspace 到 localized components

K discovery 给出的是一个子空间：

```text
span(W_final)
```

但功能解释需要 component。

这一步的本质问题是 rotation/gauge：

```text
X ≈ H W^T = (H A)(W A^{-T})^T
```

只要 `A` 可逆，重构不变。

所以，K discovery 只能稳定得到 subspace；component 解释需要额外约束。

当前约束是 Gaussian localization。

### 5.1 hidden neuron coordinate z

给定可靠子空间基 `Q`，看每个 neuron 在子空间中的 row embedding：

```text
Q[n,:]
```

构造神经元相似度：

```text
S_ij = (cos(Q[i,:], Q[j,:]) + 1) / 2
```

建 kNN graph，计算 normalized Laplacian：

```text
L = I - D^{-1/2} A D^{-1/2}
```

取第二小特征向量作为 hidden coordinate：

```text
z_hat = Fiedler vector
```

再归一化到：

```text
z_hat ∈ [0, 1]
```

### 5.2 Gaussian dictionary

在 `z_hat` 上建立 Gaussian dictionary：

```text
d_{μ,σ}(n)
= exp(-(z_hat_n - μ)^2 / (2σ^2))
```

把 dictionary atom 投影到可靠子空间：

```text
P d = Q Q^T d
```

atom score：

```text
score(d) = cos^2(d, Q Q^T d)
```

如果一个 Gaussian atom 能被可靠子空间很好表达，它就被认为是一个可解释 component 候选。

### 5.3 greedy localization

当前代码贪心选择 K 个 Gaussian atoms：

1. 按 subspace score 排序；
2. 避免中心完全重复；
3. 将新 atom 对已选择 components 做残差化；
4. 得到 localized W columns：

```text
W_loc = [w_loc,1, ..., w_loc,K]
```

然后：

```text
H_loc = X W_loc
```

### localization 阶段目前的主要问题

#### 问题 L1：z 的镜像方向校正有数学 bug

当前逻辑比较：

```text
abs(corr(1-z, target)) > abs(corr(z, target))
```

但：

```text
corr(1-z, target) = -corr(z, target)
```

所以绝对值必然相等。

这意味着该条件无法决定是否翻转。

正确逻辑应该是：

```text
if corr(z, target) < 0:
    z = 1 - z
```

或者在无 true target 时，对 base z 使用 orientation-invariant metric：

```text
max(corr(z, base), corr(1-z, base))
```

#### 问题 L2：diversity penalty 实际失效

当前 `project_out()` 返回的是归一化后的 residual：

```text
resid = project_out(w, Q_selected)
```

然后：

```text
diversity = norm(resid)
```

由于 `resid` 已被归一化，`diversity` 基本恒为 1。

所以：

```text
selection_diversity_penalty
```

几乎不起作用。

应改为在归一化前计算：

```text
raw_resid = (I - Q_selected Q_selected^T) w
diversity = norm(raw_resid) / norm(w)
```

#### 问题 L3：记录的 μ/σ 与实际 W_loc 不再完全对应

当前 row 中保存的是原 dictionary atom 的：

```text
μ, σ
```

但实际用于后续 block graph 的 `W_loc` 是 residualized atom：

```text
w_use = project_out(Pd, Q_selected)
```

这个 `w_use` 可能已经不再是原始 Gaussian atom。

于是后续 spatial graph 使用的：

```text
μ_i, μ_j
```

未必对应真实的 `W_loc` center。

更稳的做法是 residualization 后重新估计：

```text
μ_hat, σ_hat = argmax_{μ,σ} corr^2(w_use, gaussian(z; μ,σ))
```

#### 问题 L4：当前 localization 仍是 greedy dictionary selection

理论上，我们真正想做的是：

```text
在可靠 subspace 内找一个旋转 R，使 W_K R 更像 localized Gaussian components
```

即：

```text
W_loc = Q R
maximize_R Σ_k max_{μ_k,σ_k} corr^2(W_loc,k, g_{μ_k,σ_k})
          - λ Ω(R)
```

当前 greedy selection 是近似版，容易受 z-order 和 dictionary grid 影响。

这也是为什么 quick/full 容易选出 F=6，而 wide 才能稳定到 F=3。

---

## 6. H-side dependency

有了 localized components 后，计算：

```text
H_loc = X W_loc
```

然后对每对 components 计算 H-side nonlinear dependency。

当前默认使用 distance correlation：

```text
dCor^2(X,Y)
= dCov^2(X,Y) / sqrt(dVar(X) dVar(Y))
```

它比线性相关更适合检测：

```text
q, q^2, q^3, q^4
```

之间的非线性依赖。

数学直觉是：

```text
如果 components 属于同一个 latent q_g 的不同内部维度，
它们的 H expression 应有非线性依赖。
```

### H dependency 阶段的问题

#### 问题 H1：H_loc = X W_loc 不是严格 demixing

如果 `W_loc` 不是正交基，那么：

```text
H = X W_loc
```

只是 projection score，不是 least-squares coefficient。

更严格的估计应为：

```text
H = X W_loc (W_loc^T W_loc + λI)^(-1)
```

否则 H dependency 可能混入 loading overlap 的影响。

#### 问题 H2：H dependency 没有 null

distance correlation 在高自相关时间序列和 trial-structured data 中可能偏高。

应加入：

```text
phase-shift null
trial-shuffle null
within-condition permutation
```

否则 H-side dependency 可能把共同 trial/time structure 当成同一功能组证据。

---

## 7. Block graph：从 K components 到 F functional blocks

当前 block graph 融合三类证据。

### 7.1 W-center proximity

若两个 components 的 Gaussian center 接近：

```text
P^W_{ij}
= exp(-(μ_i - μ_j)^2 / (2 τ_μ^2))
```

说明它们可能属于同一个 spatial cluster。

### 7.2 H-side dependency

记：

```text
P^H_{ij} = dCor(h_i, h_j)
```

说明它们可能属于同一个 latent process 的不同内部维度。

### 7.3 split reliability

当前使用每个 component 的 split correlation：

```text
s_i = split_corr(component i)
```

构造 pair reliability：

```text
S^split_{ij} = sqrt(s_i s_j)
```

### 7.4 最终 graph

```text
G_{ij}
= P^W_{ij}
  × (P^H_{ij})^α
  × (S^split_{ij})^β
```

对 `F = 1,...,Fmax` 做 spectral clustering。

### 7.5 F score

当前 score 是：

```text
Score(F)
= mean(G_within)
  - mean(G_between)
  + 0.35 [mean(H_within) - mean(H_between)]
  - λ_F F
  - λ_singleton N_singleton
```

选择：

```text
F_selected = argmax_F Score(F)
```

### Block graph 阶段的问题

#### 问题 B1：score 容易偏向过分裂

当 F 增大时，cluster 变小，within pair 更容易变得相似：

```text
mean(G_within) 上升
```

如果复杂度惩罚不够强，就会过选 F。

现有结果正好表现为：

```text
quick: F=6
full:  F=6
wide:  F=3
```

说明 F selection 对 z/localization 稳定性很敏感。

#### 问题 B2：当前 split survival 不是真正的 block survival

当前使用的是：

```text
sqrt(split_corr_i split_corr_j)
```

这只是 component-level reliability。

真正的 block survival 应该是：

```text
C_ij = (1/S) Σ_s 1[component i and j assigned to same block in split s]
```

然后：

```text
G_ij = P^W_ij × P^H_ij × C_ij
```

也就是说，要看一对 components 是否在多个 split 的独立 block discovery 中反复被分到一起。

#### 问题 B3：没有报告 toy-only block recovery 指标

toy 中我们知道 true group assignment。

因此应报告：

```text
ARI
NMI
pairwise same-block precision/recall
confusion matrix
```

否则 F=3 即使出现，也不知道是不是正确 grouping。

---

## 8. F evidence proxy：当前最不可靠的一块

v12.6 增加了 post-hoc Bayesian model reduction proxy for F。

当前写法大致是：

```text
accuracy_F
= n_edges × [
    graph_within_mean - graph_between_mean
    + 0.35(h_within_mean - h_between_mean)
  ]
```

复杂度：

```text
complexity_F
= 0.5 × (2F + F-1) × log(n_edges)
```

再加 soft penalty：

```text
free_energy_F = accuracy_F - complexity_F - soft_penalty
```

### 这里的问题

#### 问题 E1：F=1 退化

当 `F=1` 时，没有 between-block edge。

当前逻辑相当于：

```text
graph_between_mean = 0
h_between_mean = 0
```

所以 F=1 会得到不公平的 accuracy：

```text
accuracy_1 = n_edges × graph_within_mean
```

但这不是“one-block model 的解释力”，只是因为没有 between 对照。

这导致现有 quick/full/wide 中：

```text
F_selected_by_evidence = 1
posterior_prob_proxy ≈ 0.997 到 0.999
```

因此目前这个 F evidence proxy 不能作为诊断证据。

#### 问题 E2：accuracy 与 complexity 不在同一个概率模型内

`accuracy` 是 graph contrast，`complexity` 是 BIC-like penalty。

但没有明确假设：

```text
G_ij | same block
G_ij | different block
```

的概率分布。

如果要做真正的 evidence，应建立类似 weighted stochastic block model：

```text
G_ij ~ distribution(θ_same) if c_i = c_j
G_ij ~ distribution(θ_diff) if c_i != c_j
```

然后比较不同 F 的 likelihood 和 complexity。

---

## 9. Noise-probe z/order stability

v12.6 另一个新增诊断是 noise probe。

给定 `W_final` 的正交基：

```text
Q = orth(W_final)
```

先算 base z：

```text
z_base = spectral_z(Q)
```

然后重复加小扰动：

```text
Q' = Q + noise
z' = spectral_z(Q')
```

检查：

```text
Spearman(z', z_base)
neighbor_preservation(z', z_base)
```

直觉是：

```text
真实稳定的 z/order 应该对小扰动稳定；
gauge artifact 会大幅波动。
```

### noise probe 阶段的问题

#### 问题 N1：镜像方向 bug 影响 Spearman

如前所述：

```text
z 和 1-z
```

是同一个 ordering 的镜像。

若不正确对齐，Spearman 可能变负或接近 0。

当前 quick 中出现：

```text
z_spearman_to_base_mean ≈ 0.0015
z_neighbor_preservation_mean ≈ 0.72
```

这说明局部邻域有一定保持，但全局方向/排序指标没有被正确处理或本身很不稳。

#### 问题 N2：扰动 Q 不等于 split-level stability

向 Q 加噪声测试的是：

```text
spectral z 对小数值扰动的敏感性
```

但更重要的是：

```text
重新抽 trial split 后，W_final 和 z 是否稳定？
```

因此更强的 probe 应该是：

```text
for each bootstrap/split:
    rerun K discovery
    estimate z
    align z
    compare order/neighborhood/block graph
```

---

## 10. 当前模型的层级解释

现在最合理的解释层级是：

### Level 1: reliable subspace

这是目前最可信的输出。

```text
span(W_final)
```

如果 K=12 在 reliability、held-out gain 和 evidence 中都稳定出现，可以说：

> 数据支持约 12 个可重复 neural loading dimensions。

### Level 2: localized components

这一步需要更谨慎。

```text
W_loc, H_loc
```

它依赖：

- z/order 是否稳定；
- Gaussian dictionary 是否合理；
- greedy localization 是否找到正确 rotation；
- residualization 后 μ/σ 是否仍可解释。

所以目前只能说：

> 在可靠 subspace 内，我们尝试寻找 Gaussian-localized component basis。

不能过早说：

> 已恢复真实 components。

### Level 3: functional blocks

这是最不稳定的一层。

```text
F_selected
block assignment
```

目前 wide 可以选 F=3，但 quick/full 仍容易 F=6。

因此现在更准确的说法是：

> block graph route 有希望，但 F discovery 仍受 localization 和 score calibration 限制。

---

## 11. 我认为当前成果最强的部分

### 11.1 从不可识别 toy 到可识别 toy

这是很重要的理论进步。

你们没有继续调参硬救旧 toy，而是发现了问题在：

```text
rank(W_g) 与 rank(H_g) 不支持目标 K/F
```

于是先修改数据生成条件，让恢复目标变得公平。

这是很正确的建模态度。

### 11.2 K discovery 主线已经比较成熟

核心不是 PCA variance，而是：

```text
split-repeat reliability
+ residual iteration
+ held-out gain
+ evidence-style tail penalty
```

这条路线与最初 v1 的 representation/interpretation 分离是一致的。

### 11.3 F 不再由 W-atlas reconstruction 决定

从 v12.2-v12.4 到 v12.6，已经避免了一个大坑：

```text
用 W-atlas fit 直接决定 F
```

现在改成：

```text
稳定 K 后，再看 W-center + H-dependency + split evidence
```

方向是对的。

---

## 12. 当前最需要修的地方

### 优先级 1：修正 z orientation 与 noise probe

这是小改动，但会直接影响诊断解释。

应改为：

```text
if corr(z, target) < 0:
    z = 1 - z
```

或者报告：

```text
orientation_invariant_spearman
= max(spearman(z, base), spearman(1-z, base))
```

### 优先级 2：修正 localization 的 diversity 与 μ/σ

需要把 residualization 前后的量区分清楚：

```text
raw_atom
projected_atom
residualized_atom
final_W_component
```

并对 final component 重新估计 center：

```text
μ_final, σ_final
```

否则 block graph 的 spatial term 不可靠。

### 优先级 3：F evidence proxy 暂时不要使用

目前它系统性选 F=1，应在报告里标注：

```text
F evidence proxy is diagnostic only and currently not calibrated.
```

修法：

1. F=1 单独作为 null baseline，不参与同一 accuracy 公式；
2. 或建立 weighted SBM likelihood；
3. 或用 held-out edge prediction / split co-association likelihood。

### 优先级 4：加入真正的 split block survival

应该实现：

```text
for each split:
    localize components
    compute graph
    cluster F
    record same-block matrix

C_ij = mean_s 1[c_i^s = c_j^s]
```

然后 block graph 使用：

```text
G_ij = P^W_ij × P^H_ij × C_ij
```

这会比当前 `sqrt(split_corr_i split_corr_j)` 更接近理论。

### 优先级 5：做 toy-only recovery metrics

既然 toy 中有 true group，应直接报告：

```text
K_selected
subspace R2 to W_true
z Spearman / Kendall / neighbor preservation
component center error
ARI for block assignment
pairwise same-block precision/recall
```

这样可以分清：

```text
K 错了？
z 错了？
localization 错了？
block graph score 错了？
```

---

## 13. 建议下一轮 debug 实验

### 实验 A：oracle z

使用 true z，只测试 localization + block graph。

如果 oracle z 下仍不能稳定 F=3，问题在：

```text
localization / H-dependency / F score
```

如果 oracle z 下 F=3 稳定，问题主要在：

```text
z/order recovery
```

### 实验 B：oracle component centers

用 true component centers 构造 spatial graph，只测试 H-dependency 和 F score。

如果仍失败，说明 block graph objective 本身有问题。

### 实验 C：rank-1 negative control

rank1 toy 中 group 内 W 接近共线，理论上不应支持 K=12 和 F=3。

当前 rank1_debug 结果：

```text
K_final = 7
F_score = 5
```

这说明模型确实不会盲目给 K=12，但 F score 仍会从不充分可识别结构中分出多个 blocks。

这应作为 negative-control failure signal。

### 实验 D：nested evidence

把 trials 分成：

```text
train / validation / test
```

其中：

- train 用来发现 directions；
- validation 用来调 acceptance；
- test 用来计算 K evidence。

如果 K=12 仍然稳定，K evidence 才更可信。

### 实验 E：split co-association block survival

对每个 split 独立跑 localization + block graph，计算：

```text
C_ij = P(i,j same block)
```

如果 true group 内的 `C_ij` 明显高于组间，F discovery 才真正稳定。

---

## 14. 最简数学故事

如果要把现在的模型写成论文方法的核心数学故事，可以这样说：

1. 神经数据被视为低秩 expression 的叠加：

```text
X = H W^T + ε
```

2. 由于 H/W 分解有旋转不辨识性，我们不先解释单个 component，而先寻找可重复 loading subspace：

```text
C_rel w = λ w
```

3. 可靠 K 由 residual-driven split-repeat eigen-directions 和 held-out gain 共同决定：

```text
K = arg stop where new reliable direction no longer improves held-out fit
```

4. 在稳定 K 维 subspace 内，再加入 localization prior，寻找 Gaussian-like component basis：

```text
W_loc = Q R
```

5. 功能 block 不由 W 空间单独决定，而由 spatial proximity、H-side nonlinear dependency 和 split stability 共同决定：

```text
G_ij = P^W_ij (P^H_ij)^α (S^split_ij)^β
```

6. 最终 block 是解释层对象，不等同于原始 recovered component：

```text
components -> functional blocks
```

### 这套故事成立的前提

1. 目标 K/F 在数据中可识别；
2. split-repeat reliability 真能区分 signal 与 noise；
3. localized component basis 能稳定从 reliable subspace 中恢复；
4. H dependency 不是共同时间结构造成的假相关；
5. block graph score 不系统性过分裂或欠分裂；
6. evidence proxy 不重复使用验证集，也不退化到 trivial model。

目前第 1、2 条比较强；第 3、4、5、6 条还需要继续修。

---

## 15. 当前审查判断

我目前会这样评价 v12.6：

```text
K discovery:        基本方向正确，已有较强证据，但 evidence 需要 nested test 修正。
z/order recovery:   当前瓶颈之一，orientation 和 stability 指标需要修。
localization:       方向正确，但 greedy/residualization 细节会污染 μ/σ。
F block graph:      有希望，但 quick/full 过分裂，split survival 尚未真正实现。
F evidence proxy:   当前不可用，系统性退化到 F=1。
negative control:   rank1 能降低 K，但 F 仍过分裂，需要加强拒绝机制。
```

所以，当前最稳的科学结论应该写成：

> 我们可以比较稳定地从可识别 toy 中发现 K≈12 的 reliable neural subspace；但从 reliable subspace 到 localized components，再到 F=3 functional blocks 的解释链仍需修正 z/order、localization、split block survival 与 F evidence calibration。

这句话保守，但准确。
