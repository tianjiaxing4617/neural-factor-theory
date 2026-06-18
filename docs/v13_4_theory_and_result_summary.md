# v13.4 神经因子理论路线阶段性总结

本文总结截至 v13.4 的核心数学思路、模型推导、方法选择理由与实验诊断结果。

当前阶段的结论可以先说在前面：这条路线已经从“直接寻找 hard components / hard blocks”转向了“先恢复可解释的因子子空间，再用 posterior coassociation 总结 block 结构”。这个转向是合理的，因为神经因子分解天然存在旋转不唯一性，直接要求每一个 component 完全可识别会让模型过于脆弱；而 block posterior、PSM kernel、以及 loss-calibrated posterior partition summary 更接近这个问题真正可稳定识别的对象。

v13.4 的作用不是重新发明整个模型，而是修正 v13.3 的一个关键问题：PSM/spectral 方法本身有效，但 affinity-only 的候选评分有时会偏向 over-split。v13.4 用 weighted Binder risk、PSM cross-entropy、weighted-SBM/MDL 和 F posterior support 构成一个 posterior-loss calibrated objective，让最终的 block summary 不再靠人为指定 posterior F，而是由 posterior clustering loss 自动选择。

## 1. 基本观测模型

我们把数据写成三维张量：

```math
X \in \mathbb{R}^{R \times T \times N},
```

其中 $R$ 是 trial 数，$T$ 是时间点数，$N$ 是 neuron 数。把 trial 和 time 合并后，得到矩阵

```math
Y \in \mathbb{R}^{M \times N}, \qquad M = R T.
```

核心低秩因子模型是：

```math
Y = H W^\top + E.
```

其中

```math
H \in \mathbb{R}^{M \times K}, \qquad
W \in \mathbb{R}^{N \times K}.
```

$W$ 表示 neuron 空间上的 loading pattern，$H$ 表示 trial-time 空间上的 latent expression。第 $k$ 个 component 的贡献是

```math
Y_k = h_k w_k^\top.
```

因此整体信号为

```math
\widehat Y_K = \sum_{k=1}^{K} h_k w_k^\top.
```

模型还假设 component 可以进一步聚成 block 或 functional group。令

```math
z_k \in \{1,\ldots,F\}
```

表示第 $k$ 个 component 所属 block。当前 toy 设定中真实结构为：

```math
K_{\mathrm{true}} = 12, \qquad F_{\mathrm{true}} = 3,
```

每个 group 内有 4 个 component。

## 2. 为什么不能只做普通矩阵分解

普通因子分解有一个核心不唯一性。对任意可逆矩阵

```math
A \in \mathbb{R}^{K \times K},
```

都有

```math
H W^\top
= (H A) (W A^{-\top})^\top.
```

这说明 component-level 的 $h_k,w_k$ 不是绝对唯一的。可稳定识别的通常不是某一列本身，而是由这些列张成的子空间：

```math
\mathrm{span}(W), \qquad \mathrm{span}(H).
```

这正是旧版本反复遇到的问题：如果一个 functional group 内多个 $W$ loading 近似共线，例如

```math
w_{g,m}(n) \approx a_{g,m} u_g(n),
```

那么该 group 的总贡献为

```math
X_g(t,n)
= \sum_m h_{g,m}(t) w_{g,m}(n)
\approx
\left( \sum_m a_{g,m} h_{g,m}(t) \right) u_g(n).
```

这会在观测层面坍缩成近似 rank-1 group。也就是说，即便真实生成过程有多个 component，观测数据也可能只强烈支持一个共同方向。此时如果模型强行恢复每一个 component，就会出现 component 合并、边界漂移、或者过拟合式分裂。

因此我们后来的路线改成：

1. 先恢复 $W/H$ 的有效子空间和 component 数 $K$。
2. 再把 component localization 作为解释旋转，而不是当作唯一真相。
3. 最后用 posterior coassociation 总结 block 结构。

这个思路比直接 hard kmeans 更稳，因为它承认了旋转不唯一性，同时把可识别的 block posterior 保留下来。

## 3. K 的发现：residual-driven iterative expansion

给定一个候选 $K$，我们通过 ridge ALS 拟合：

```math
(\widehat H_K,\widehat W_K)
=
\arg\min_{H,W}
\|Y - H W^\top\|_F^2
+ \lambda_H \|H\|_F^2
+ \lambda_W \|W\|_F^2.
```

对应残差为

```math
R_K = Y - \widehat H_K \widehat W_K^\top.
```

每一步从残差中寻找新的可靠方向。直觉上，如果 $R_K$ 中仍然存在可重复、非噪声的结构，那么 $K$ 还不够。候选方向需要同时满足两类证据：

```math
\Delta R^2_{\mathrm{val}} > \tau_R,
```

以及 residual reliability 大于 null：

```math
\frac{s_1(R_K)}{Q_{\mathrm{null}}(s_1)} > \tau_s.
```

其中 $s_1(R_K)$ 表示残差主方向强度，$Q_{\mathrm{null}}(s_1)$ 表示打乱或 null 过程下主方向强度的参考分位数。

接受新 component 后，不是只追加一列，而是全局 refit：

```math
(\widehat H_{K+1},\widehat W_{K+1})
=
\arg\min_{H,W}
\|Y - H W^\top\|_F^2
+ \lambda_H \|H\|_F^2
+ \lambda_W \|W\|_F^2.
```

这一步很重要，因为新方向加入后，旧方向也应该重新分配解释量。v13.4 wide 中该过程恢复：

```math
K_{\mathrm{selected}} = 12.
```

同时 posterior active K 也为：

```math
K_{\mathrm{eff,posterior}} = 12.
```

## 4. Component posterior 与 ARD-style pruning

每个 component 都有一个经验后验可靠性分数。抽象地说，我们把 component 的解释增益、残差可靠性、能量等证据组合成一个 logit：

```math
\ell_k
= b
+ \alpha_g g_k
+ \alpha_r r_k
+ \alpha_e e_k.
```

然后得到 inclusion probability：

```math
\pi_k = \sigma(\ell_k)
= \frac{1}{1+\exp(-\ell_k)}.
```

若

```math
\pi_k < \tau_{\mathrm{prune}},
```

则 component 可以被剪枝。v13.4 wide 中没有 component 被剪枝：

```math
\min_k \pi_k = 0.9770,
\qquad
\frac{1}{K}\sum_k \pi_k = 0.9925.
```

这说明当前 12 个 component 都有很强支持。

## 5. Localization：把可恢复子空间转成可解释 components

由于 $W$ 的列向量存在旋转不唯一性，直接解释 $\widehat W$ 的每一列并不稳定。v13.4 的 localization 思路是：先取 $\widehat W$ 的子空间，再用 Gaussian dictionary 在 neuron 坐标上寻找可解释的局部 atoms。

令

```math
Q_W = \mathrm{orth}(\widehat W)
```

表示 $W$ 子空间的正交基。定义 Gaussian atom：

```math
a_{\mu,\sigma}(z_n)
=
\exp\left(
-\frac{(z_n-\mu)^2}{2\sigma^2}
\right),
```

并标准化为单位范数。对 dictionary 中每个 atom，计算其落在 $W$ 子空间内的能量：

```math
E(\mu,\sigma)
=
\|Q_W^\top a_{\mu,\sigma}\|_2^2.
```

然后以 greedy OMP-like 方式选择 $K$ 个 atoms，同时加入中心距离的 redundancy penalty，避免多个 atoms 重复挤在同一个位置：

```math
j^\star
=
\arg\max_j
\left[
\|Q_{\mathrm{res}}^\top a_j\|_2^2
- P_{\mathrm{red}}(j)
\right].
```

选出 atoms 后，把它们投影回 $\widehat W$ 子空间：

```math
W_{\mathrm{loc}}
=
\mathrm{orth}
\left(
Q_W Q_W^\top A_S
\right),
```

再得到 localized expression：

```math
H_{\mathrm{loc}} = Y W_{\mathrm{loc}}.
```

这里 $W_{\mathrm{loc}},H_{\mathrm{loc}}$ 是一个解释旋转：它不否定旋转不唯一性，而是选择一个空间上更可读的代表。

## 6. H-side dependency matrix：D 空间

block 不应该只由 neuron center 决定，也应该由 $H$ 空间的共同变化支持。当前版本使用一个轻量 nonlinear dependency proxy。

给定 localized expression $H_{\mathrm{loc}}$，定义第 $i,j$ 个 component 的 dependency：

```math
D_{ij}
=
\max_{p,q \in \{1,2,3\}}
\left|
\mathrm{corr}
\left(
h_i^p,\,
h_j^q
\right)
\right|.
```

这里的 $p,q$ 表示简单 polynomial transforms。这个设计的目的，是让模型能捕捉 $q,q^2,q^3$ 这类非线性相关，而不是只看线性 correlation。

当前版本的 $D$ 仍然是轻量 proxy。v13.4 wide 中：

```math
D_{\mathrm{localized,AUC}} = 0.9097,
\qquad
D_{\mathrm{true,AUC}} = 0.9838.
```

这说明 localized $D$ 已经能区分大部分 within-block 与 between-block 关系，但还没有达到真实 latent $D$ 的上限。也就是说，当前路线的主要剩余瓶颈不是 $K$，而是 $D_{\mathrm{localized}}$ 的估计精度。

## 7. Block posterior：从 hard partitions 到 PSM

早期 block discovery 会对 component centers 做 hard clustering，并加入 $D$ 的 within-between contrast。对一个 partition $z$，可以定义：

```math
\Delta_{\mathrm{space}}(z)
=
\frac{
\mathrm{between\ center\ distance}
}{
\mathrm{within\ center\ distance}+\epsilon
},
```

以及

```math
\Delta_D(z)
=
\mathrm{mean}(D_{ij}:z_i=z_j)
-
\mathrm{mean}(D_{ij}:z_i\ne z_j).
```

再组合成 block score：

```math
S_{\mathrm{block}}(z)
=
\beta_s \tanh(\Delta_{\mathrm{space}}/3)
+ \beta_D \tanh(3\Delta_D)
+ \beta_b B(z)
- \beta_F F
- P_{\mathrm{small}}(z).
```

但是单次 hard clustering 对扰动很敏感。于是 v13.2.1 以后引入 posterior perturbation：对 centers 和 $D$ 加扰动，多次运行 block discovery，得到一组 partitions：

```math
z^{(1)},z^{(2)},\ldots,z^{(S)}.
```

每个 partition 有权重

```math
\omega_s
=
\frac{
\exp(S_s / T)
}{
\sum_{r=1}^{S}\exp(S_r / T)
}.
```

于是得到 posterior similarity matrix，也就是 PSM：

```math
C_{ij}
=
\sum_{s=1}^{S}
\omega_s\,
\mathbf{1}
\left[
z_i^{(s)} = z_j^{(s)}
\right].
```

$C_{ij}$ 的含义非常直接：

```math
C_{ij}
=
P(z_i=z_j \mid \mathrm{evidence}).
```

同时也得到 block count posterior：

```math
P(F=f)
=
\sum_{s:F_s=f}\omega_s.
```

v13.4 wide 中：

```math
P(F=3)=0.7388,
\qquad
P(F=2)=0.1434,
\qquad
P(F=4)=0.0951.
```

这说明 posterior 主要支持 $F=3$。

## 8. 为什么 PSM 是正确路线

PSM 有一个关键好处：它是 posterior partition evidence 的稳定总结，而不是某一次 hard clustering 的偶然结果。

对每个 partition $z^{(s)}$，定义同块矩阵：

```math
B^{(s)}_{ij}
=
\mathbf{1}
\left[
z_i^{(s)}=z_j^{(s)}
\right].
```

则 PSM 是这些同块矩阵的加权平均：

```math
C
=
\sum_s \omega_s B^{(s)}.
```

每个 $B^{(s)}$ 都是 block-membership kernel，可写成

```math
B^{(s)} = M^{(s)} {M^{(s)}}^\top,
```

其中 $M^{(s)}$ 是 component-block membership matrix。因此 $B^{(s)}$ 是 positive semidefinite。加权平均仍然是 positive semidefinite：

```math
C \succeq 0.
```

所以 PSM 不只是一个表格，它可以自然作为 kernel 或 affinity 使用。v13.3 正是基于这个点，尝试了：

1. PSM spectral clustering。
2. PSM + spatial prior。
3. EB-shrink PSM。
4. BayesianGaussianMixture comparison。

v13.3 的发现是：PSM 方向有效，但 affinity-only score 会偏向 over-split。也就是说，partition 的候选生成已经对了，真正需要修的是候选选择准则。

## 9. v13.4：loss-calibrated PSM selection

v13.4 的核心变化是用 posterior clustering decision loss 选择最终 summary partition。

令候选 partition 为 $z$。对于每一对 component $i<j$，若 candidate 把它们分开，则损失 posterior same-block evidence；若 candidate 把它们合并，则损失 posterior different-block evidence。

weighted Binder risk 定义为：

```math
R_{\mathrm{Binder}}(z)
=
\frac{1}{|\mathcal P|}
\sum_{i<j}
\left[
a C_{ij}\mathbf{1}(z_i\ne z_j)
+ b(1-C_{ij})\mathbf{1}(z_i=z_j)
\right],
```

其中

```math
\mathcal P=\{(i,j):i<j\}.
```

v13.4 使用：

```math
a=2.25,
\qquad
b=1.00.
```

这意味着 false split 比 false merge 更贵。这个选择不是任意的，而是针对 v13.3 暴露出的 over-split 倾向：如果模型倾向于把一个真实 block 切成多个小 block，我们就应该提高 split risk。

pairwise decision threshold 可以由两种选择的局部风险推出。若把 $i,j$ 合并，风险为

```math
b(1-C_{ij}).
```

若把 $i,j$ 分开，风险为

```math
aC_{ij}.
```

合并更优当且仅当

```math
b(1-C_{ij}) < a C_{ij}.
```

因此

```math
C_{ij} > \frac{b}{a+b}.
```

在 v13.4 中：

```math
\frac{b}{a+b}
=
\frac{1}{3.25}
\approx 0.3077.
```

这表示只要 posterior 有足够 evidence 认为两者同块，就不要轻易 split。这正是防止 F=4 over-split 的数学机制。

v13.4 还加入 PSM cross-entropy：

```math
R_{\mathrm{CE}}(z)
=
-\frac{1}{|\mathcal P|}
\sum_{i<j}
\left[
\mathbf{1}(z_i=z_j)\log C_{ij}
+
\mathbf{1}(z_i\ne z_j)\log(1-C_{ij})
\right].
```

同时用 weighted-SBM/MDL 检查 partition 是否需要过多参数解释 PSM。对 block pair $(r,s)$，定义：

```math
\widehat p_{rs}
=
\mathrm{mean}
\left(
C_{ij}: z_i=r, z_j=s
\right).
```

其 Bernoulli-style weighted log likelihood 为：

```math
\ell_{\mathrm{SBM}}(z)
=
\sum_{i<j}
\left[
C_{ij}\log \widehat p_{z_i z_j}
+
(1-C_{ij})
\log(1-\widehat p_{z_i z_j})
\right].
```

MDL risk 为：

```math
R_{\mathrm{MDL}}(z)
=
\frac{
-\ell_{\mathrm{SBM}}(z)
+ \frac{1}{2} q(z)\log |\mathcal P|
}{
|\mathcal P|
},
```

其中 $q(z)$ 是有效 block-pair 参数数。

最终 v13.4 loss 可以写成：

```math
L_{13.4}(z)
=
R_{\mathrm{Binder}}(z)
+ \lambda_{\mathrm{CE}}R_{\mathrm{CE}}(z)
+ \lambda_{\mathrm{MDL}}R_{\mathrm{MDL}}(z)
+ \lambda_F F(z)
+ P_{\mathrm{singleton}}(z)
- \lambda_A \Delta_A(z)
- \lambda_B B(z)
- \lambda_P \log\frac{P(F(z))}{P_0(F(z))}.
```

其中 $\Delta_A(z)$ 是 affinity within-between contrast，$B(z)$ 是 block-size balance，$P(F)$ 是前面得到的 block-count posterior，$P_0(F)$ 是 uniform baseline prior。

最终选择：

```math
\widehat z
=
\arg\min_{z\in\mathcal Z}
L_{13.4}(z).
```

这里的候选集合 $\mathcal Z$ 包括：

1. PSM spectral。
2. PSM spatial spectral。
3. EB-shrink PSM spatial spectral。
4. PSM-NMF soft summary。
5. spatial hierarchical PSM。

baseline kmeans posterior 和 BayesianGaussianMixture 保留为诊断对照，但不作为最终主路线。

## 10. 为什么选择这条路线

### 10.1 它尊重因子分解的不唯一性

如果模型本身存在

```math
H W^\top
=
(HA)(WA^{-\top})^\top,
```

那么 hard component identity 本来就不应该被当作唯一真相。v13.4 的路线先恢复子空间，再用 posterior partition evidence 总结 block，是更符合问题结构的。

### 10.2 它把 uncertainty 放进了核心对象

PSM 的每个元素都是：

```math
C_{ij}=P(z_i=z_j\mid\mathrm{evidence}).
```

这比一次 kmeans label 更丰富。它保留了 pairwise uncertainty，也能暴露哪些 component pair 处于边界。

### 10.3 它把 F-selection 从 heuristic 变成 decision problem

v13.3 中 affinity-only score 容易喜欢更小、更干净的 block，因此出现 over-split 倾向。v13.4 把选择问题改成：

```math
\mathrm{choose\ the\ partition\ with\ minimum\ posterior\ decision\ loss}.
```

这和我们真正想做的事情一致：不是让图切得最漂亮，而是在 posterior evidence 下犯错成本最小。

### 10.4 它和现有结果对齐

v13.3 已经证明：

1. PSM evidence 很强。
2. spectral partition 在正确 F 下可以恢复 true block。
3. BGM 容易 over-split。
4. 问题主要出在候选评分，而不是候选生成。

因此 v13.4 只修评分层，是最小而正确的改动。

## 11. v13.4 wide 结果

v13.4 wide 的核心结果为：

| 指标 | 数值 |
|---|---:|
| $K_{\mathrm{true}}$ | 12 |
| $K_{\mathrm{selected}}$ | 12 |
| $K_{\mathrm{eff,posterior}}$ | 12 |
| $K_{\mathrm{eff,soft}}$ | 11.9099 |
| $F_{\mathrm{true}}$ | 3 |
| $F_{\mathrm{selected}}$ | 3 |
| v13.4 selected method | psm_spectral |
| v13.4 selected F | 3 |
| all-data $R^2$ | 0.9711 |
| localized $R^2$ | 0.9711 |

空间恢复诊断：

| 指标 | 数值 |
|---|---:|
| $W$ subspace angle mean | 6.0082 deg |
| $W$ subspace angle max | 16.9800 deg |
| $H$ subspace angle mean | 2.8223 deg |
| $H$ subspace angle max | 5.3814 deg |
| $W$ min canonical correlation | 0.9564 |
| $H$ min canonical correlation | 0.9956 |
| center match MAE | 0.0233 |
| center match max error | 0.0583 |

block / posterior 诊断：

| 指标 | 数值 |
|---|---:|
| selected block ARI vs nearest true group | 1.0000 |
| coassociation AUC-like | 1.0000 |
| coassociation within mean | 0.8721 |
| coassociation between mean | 0.0733 |
| coassociation mean diff | 0.7988 |

$D$ 空间诊断：

| 指标 | 数值 |
|---|---:|
| $D_{\mathrm{localized}}$ AUC-like | 0.9097 |
| $D_{\mathrm{true}}$ AUC-like | 0.9838 |
| $D_{\mathrm{localized}}$ within mean | 0.3223 |
| $D_{\mathrm{localized}}$ between mean | 0.1434 |
| $D_{\mathrm{localized}}$ mean diff | 0.1790 |
| $D_{\mathrm{true}}$ mean diff | 0.3016 |

v13.4 loss-selected candidate：

| method | F | loss score | Binder risk | ARI |
|---|---:|---:|---:|---:|
| psm_spectral | 3 | 0.3104 | 0.1548 | 1.0000 |

对比 v13.3 stored wide：

| 项目 | v13.3 stored wide | v13.4 wide |
|---|---:|---:|
| selected F by score | 5 | 3 |
| selected block ARI | 0.5119 | 1.0000 |
| coassociation AUC-like | 0.9877 | 1.0000 |
| $D_{\mathrm{localized}}$ AUC-like | 0.8320 | 0.9097 |
| center max error | 0.3000 | 0.0583 |

注意：v13.3 中 `psm_spatial_spectral` 在 posterior F=3 条件下也能达到 ARI=1.0；问题是它自己的 affinity-only score 会倾向 F=4。v13.4 的意义就是不再需要手工指定 F，而是让 posterior loss 自动选择 F=3。

## 12. 当前结果说明了什么

### 12.1 K 恢复已经足够稳定

v13.4 wide 恢复：

```math
K_{\mathrm{selected}}=K_{\mathrm{true}}=12.
```

inclusion probability 也显示 12 个 component 都有强支持。这说明 residual-driven iterative expansion 这条 K 路线目前有效。

### 12.2 W/H 子空间恢复良好

$H$ 子空间尤其稳定：

```math
\rho_{\min,H}=0.9956.
```

$W$ 子空间也达到可用水平：

```math
\rho_{\min,W}=0.9564.
```

这说明当前模型恢复的是正确的整体 latent subspace。

### 12.3 block posterior 已经非常强

coassociation AUC-like 为 1.0，说明 PSM 对 within-block 与 between-block pair 的区分非常清楚。

这也是为什么 PSM 路线成立：它不是在弱证据上硬切，而是把强 posterior evidence 变成一个稳定 kernel。

### 12.4 D 空间仍是剩余瓶颈

$D_{\mathrm{localized}}$ 已经达到 0.9097，但仍低于 $D_{\mathrm{true}}$ 的 0.9838。这个差距说明：

1. localized rotation 会损失一部分 H-side dependency information。
2. polynomial correlation proxy 还比较轻量。
3. 未来如果继续优化，最值得做的是改进 $D$ 的估计，而不是继续改 K 或 PSM selection。

## 13. 当前方法的边界

这条路线目前足够好，但边界也应该写清楚。

第一，component identity 不是绝对对象。由于旋转不唯一性，我们应该更相信 subspace、PSM、block posterior，而不是单个 component 编号。

第二，PSM 的质量依赖 candidate partition distribution。如果 perturbation 过程完全没有覆盖某类真实 partition，PSM 自身也无法凭空恢复它。

第三，当前 $D$ 是 polynomial dependency proxy，不是完整 Bayesian dependency posterior。它快、稳定、可解释，但不是最终形式的 H-side dependence model。

第四，当前结果主要来自 controlled toy。真实数据没有 true labels，因此需要用 posterior mass、PAC、stability、held-out reconstruction、跨 session consistency 等无监督诊断替代 ARI。

## 14. 阶段性结论

截至 v13.4，我们可以把当前路线固定为一个阶段性稳定版本：

```math
Y
\rightarrow
\widehat K
\rightarrow
(\widehat W,\widehat H)
\rightarrow
(W_{\mathrm{loc}},H_{\mathrm{loc}})
\rightarrow
D_{\mathrm{loc}}
\rightarrow
C_{\mathrm{PSM}}
\rightarrow
\widehat z_{\mathrm{loss}}.
```

其中最关键的理论转变是：

```math
\mathrm{hard\ component\ recovery}
\quad\longrightarrow\quad
\mathrm{subspace\ recovery + posterior\ block\ summary}.
```

v13.4 的 loss-calibrated PSM selection 解决了 v13.3 的主要缺陷：候选 partition 可以由 PSM/spectral 生成，但最终 F 和 block summary 应该由 posterior decision loss 选择。

因此这条线目前可以视为“效果足够好”的稳定基线。后续如果继续优化，重点应放在：

1. 更强的 $D_{\mathrm{localized}}$ 估计。
2. 真实数据上的无监督稳定性诊断。
3. 多 session / 多条件下 block posterior 的一致性。

而不是继续反复修改 K discovery 或 hard clustering 规则。
