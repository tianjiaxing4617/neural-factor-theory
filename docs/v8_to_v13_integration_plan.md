# v8 大框架与 v13 方法内核的整合设计

本文记录当前对 v8.8 框架与 v13.4 方法路线的结构性理解，并给出下一阶段整合成 v14 的建议。

核心判断：

```text
v8.8 = 科学任务框架
v13.4 = 无监督 posterior discovery 方法内核
v14   = 把 v13.4 作为 discovery engine 接入 v8.8 的 FactorSpec / DiagnosticSpec / audit 框架
```

也就是说，v13 不应该替代 v8。v13 应该成为 v8 里 discovery factor 的新生成机制，让整个任务从“方法验证”回到“科学问题验证”。

## 1. v8.8 的框架角色

v8.8 已经具备一个比较完整的真实数据分析框架。它的结构是：

```text
SessionData / NeuralDataset
    -> external FactorSpec declarations
    -> ModelSpec fit
    -> FitResult
    -> DiagnosticSpec / AuditSpec
    -> standard diagnostics and status taxonomy
```

### 1.1 数据层

v8 用 `SessionData` 表示单个 recording/session：

```text
X: trials x time x neurons
trials: trial metadata
time: time coordinate
metadata: session metadata
```

多个 session 组成：

```text
NeuralDataset(sessions=[...])
```

这说明 v8 的核心已经适合真实数据，不是 toy-only。

### 1.2 因子声明层

v8 的关键抽象是 `FactorSpec`：

```text
name
H_by_session
regime
kind
role
metadata
discovery
block_rank
window
init_mode
update_mode
```

这个设计很好，因为模型 core 不需要知道 cue/reward/toy 等具体名字。科学假设通过 adapter 或 factor builder 声明，而不是写死在 fitting 函数里。

### 1.3 拟合层

v8 的模型是：

```text
single_session_independent
same_region_unaligned
matched_hard_shared
matched_hierarchical_shared
```

对应数学形式可以理解为：

$$
X_s(t,n)
=
b_s(n)
+
\sum_g H_{s,g}(t) W_{s,g}(n)
+
\epsilon_s(t,n).
$$

在 matched hierarchical shared-W 模式下：

$$
W_{s,g}
=
W_g^{\mathrm{global}}
+
\Delta W_{s,g}.
$$

并用 factor-specific ridge / EB lambda 控制：

$$
\|\Delta W_{s,g}\|_2^2
$$

的复杂度。

### 1.4 v8 Stage-2 discovery

v8 已经有 Stage-2 discovery block：

```text
FactorSpec(discovery=True, block_rank>1)
```

当前逻辑是：

1. 用 template/window + local PCA 初始化 $H$ block。
2. 拟合 $W$。
3. 用 candidate-specific partial residual 更新 $H$：

$$
R_{\mathrm{without\ parent}}
=
R_{\mathrm{full}}
+
E_{\mathrm{parent}}.
$$

4. 从 partial residual 投影回该 parent block：

$$
H_{\mathrm{new}}
\leftarrow
R_{\mathrm{without\ parent}} W_{\mathrm{parent}}.
$$

5. 重复若干轮，再把 parent block 展开成 atomic factors。

这个机制是 v13 可以接入的天然入口。

### 1.5 诊断层

v8 的诊断体系非常完整：

```text
fit_quality
factor_blocks
discovery_blocks
stage2_trace
stage2_optimality
candidate_ablation
structured_nulls
free_residual_pc
split_half_W_stability
heldout_session
deltaW_diagnostics
stage1_workspace
residual_leakage
core_component_atlas
core_anchor_W_overlap
core_anchor_block_W_overlap
condition_traces
candidate_status
```

这说明 v8 的目标不是只找一个分解，而是回答科学问题：

```text
这个 factor/block 是否解释真实数据？
是否超过 null？
是否稳定？
是否可跨 session 泛化？
是否只是 free PC 可以替代的残差结构？
是否和任务 label / condition trace 有关系？
```

这正是我们最终需要保留的科学框架。

## 2. v13.4 的方法角色

v13.4 解决的是另一个层面的问题：如何在不依赖人工标签的情况下，从神经数据中稳定恢复 latent component / block posterior。

其路线是：

```text
Y
    -> residual-driven iterative K
    -> W/H subspace
    -> Gaussian localization
    -> H-side dependency D
    -> block posterior
    -> PSM
    -> loss-calibrated partition summary
```

数学核心是：

$$
Y = H W^\top + E.
$$

先通过 residual-driven expansion 找 $K$：

$$
R_K
=
Y-\widehat H_K \widehat W_K^\top.
$$

若 residual 中仍有可靠方向，则接受新 component。最终得到：

$$
\widehat K.
$$

然后用 posterior perturbation 得到 partition samples：

$$
z^{(1)},\ldots,z^{(S)}.
$$

形成 PSM：

$$
C_{ij}
=
\sum_s
\omega_s
\mathbf{1}
\left[
z_i^{(s)}=z_j^{(s)}
\right].
$$

最后用 loss-calibrated posterior risk 选择 summary partition：

$$
\widehat z
=
\arg\min_z
L_{13.4}(z).
$$

其中最重要的损失项是 weighted Binder risk：

$$
R_{\mathrm{Binder}}(z)
=
\frac{1}{|\mathcal P|}
\sum_{i<j}
\left[
a C_{ij}\mathbf{1}(z_i\ne z_j)
+
b(1-C_{ij})\mathbf{1}(z_i=z_j)
\right].
$$

v13.4 的意义是：它把 discovery 从 hard clustering 变成 posterior decision problem。

## 3. v8 与 v13 的关系

两者不是竞争关系，而是上下游关系。

| 层级 | v8.8 | v13.4 |
|---|---|---|
| 任务定位 | 真实数据科学框架 | 无监督 discovery engine |
| 数据结构 | multi-session `NeuralDataset` | 当前主要是 toy / matrix-level pipeline |
| 因子入口 | `FactorSpec` | discovered $H,W,K,z,C$ |
| block 表达 | parent factor + atomic basis | PSM posterior block |
| 选择准则 | likelihood / null / stability audit | posterior clustering loss |
| 验证方式 | ablation, null, heldout, split-half | true-label toy diagnostics + posterior risk |
| 最终作用 | 科学结论 | 生成更可靠的 candidate factors |

因此整合目标应该是：

```text
用 v13.4 生成更好的 discovery factors；
用 v8.8 检验这些 factors 是否科学成立。
```

## 4. 最小整合策略：v14.0

v14.0 不建议重写 v8 core。最小可行方案是新增一个 discovery adapter：

```text
neural_factor_v14/
    discovery_v13.py
    psm_blocks.py
    integration.py
```

或者在 v8 package 内新增：

```text
neural_factor_v88_final/v13_discovery.py
```

### 4.1 新增 DiscoverySpec

建议新增：

```python
@dataclass
class V13DiscoverySpec:
    max_K: int = 24
    K_accept_policy: str = "residual_gain_plus_null"
    localization: str = "gaussian_omp"
    block_selection: str = "loss_calibrated_psm"
    binder_split_cost: float = 2.25
    binder_merge_cost: float = 1.0
    train_only: bool = True
    per_session: bool = True
    pool_sessions: bool = False
```

关键是 `train_only=True`。真实数据中不能在全数据上 discovery 后再声称 held-out 有效，否则会有 circularity。

### 4.2 v13 输出对象

v13 discovery engine 应返回一个结构化对象：

```python
@dataclass
class V13DiscoveryResult:
    H_by_session: list[np.ndarray]
    W_by_session: list[np.ndarray]
    component_table: pd.DataFrame
    block_assignment: pd.DataFrame
    block_psm: pd.DataFrame
    block_posterior: pd.DataFrame
    method_candidates: pd.DataFrame
    diagnostics: dict[str, pd.DataFrame]
```

其中 `H_by_session` 和 `W_by_session` 是真正接入 v8 的对象；其他表进入 `FitResult.model_artifacts` 和 diagnostics。

### 4.3 映射为 FactorSpec

把 v13 的 selected blocks 映射成 v8 parent block factors：

```text
v13_block_01
v13_block_02
...
v13_block_F
```

每个 block 内包含若干 atomic components。若 block $b$ 包含 component 集合：

$$
\mathcal K_b=\{k:z_k=b\},
$$

则对每个 session 构造：

$$
H_{s,b}
\in
\mathbb{R}^{R_s \times T_s \times |\mathcal K_b|}.
$$

对应 `FactorSpec`：

```python
FactorSpec(
    name=f"v13_block_{b:02d}",
    H_by_session=H_block_by_session,
    regime="discovered",
    kind="posterior_block",
    role="candidate",
    discovery=False,
    block_rank=len(K_b),
    metadata={
        "engine": "v13.4_loss_calibrated_psm",
        "psm_posterior_mass": ...,
        "binder_risk": ...,
        "components": ...
    },
)
```

注意：接入 v8 时可以先设 `discovery=False`，因为 v13 已经完成 discovery。v8 负责 fit/audit，而不是再次 discovery。

## 5. v14.0 pipeline 建议

最小完整流程：

```text
1. load_raworderfct(paths)
2. build external known factors:
   q_common_time, time_ramp, cue_identity, reward_outcome, interaction
3. run v13 discovery on train data or residualized train data
4. convert v13 blocks -> FactorSpec posterior_block candidates
5. combine known factors + v13 discovered factors
6. fit_model(...)
7. run_standard_diagnostics(...)
8. output v13-specific diagnostic tables into same result directory
```

更明确地：

$$
X_s
\rightarrow
\mathrm{known\ factor\ residualization}
\rightarrow
R_s^{\mathrm{known}}
\rightarrow
\mathrm{v13\ discovery}
\rightarrow
\{H_{s,b}^{(v13)}\}
\rightarrow
\mathrm{v8\ audit}.
$$

这个版本的核心科学问题是：

```text
在已知 cue/reward/task regressors 之外，v13 posterior blocks 是否解释了稳定、可泛化、非 null 的神经结构？
```

## 6. 必须防止的 circularity

整合时最重要的风险是 data leakage。

如果 v13 在全数据上发现 $H,W$，然后 v8 再用同一批数据评估 ablation / held-out gain，结果会偏乐观。

因此 v14.0 应该采用：

```text
discovery split != evaluation split
```

最小做法：

1. v13 discovery 只在 train mask 上运行。
2. 得到 $W$ 或 localized atoms 后，把 test 上的 $H$ 通过 projection 得到：

$$
H_{\mathrm{test}}
=
Y_{\mathrm{test}} W
\left(
W^\top W+\lambda I
\right)^{-1}.
$$

3. v8 的 candidate ablation / structured null / free PC 继续使用 test mask。

这样 scientific audit 才成立。

## 7. v14.1：multi-session consensus

v8 的强项是 multi-session。v13 当前主要验证了单数据矩阵上的 discovery。下一步应该做 session-level consensus。

### 7.1 per-session discovery

对每个 session 单独运行 v13：

$$
X_s
\rightarrow
\left(
K_s,
H_s,
W_s,
C_s,
z_s
\right).
$$

然后跨 session 对齐 blocks。可以用 block-level W subspace similarity：

$$
S_{bb'}^{(s,s')}
=
\frac{1}{r}
\left\|
Q_{s,b}^{\top}Q_{s',b'}
\right\|_F^2.
$$

也可以结合 condition trace / time profile：

$$
S_{\mathrm{total}}
=
\eta_W S_W
+
\eta_H S_H
+
\eta_C S_{\mathrm{condition}}.
$$

最终得到 cross-session block consensus。

### 7.2 pooled same-region discovery

如果 neuron 不匹配，不能强行 shared-W；但可以在每个 session 的 latent block posterior 上做 consensus。

如果 neuron 匹配，则可以进入 v8 的 matched hierarchical shared-W：

$$
W_{s,b}
=
W_b^{\mathrm{global}}
+
\Delta W_{s,b}.
$$

这会让 v13 的 discovery 和 v8 的 shared-W EB 框架真正结合。

## 8. v14.2：label-aware posterior，而不是 label-forced clustering

v8 有外部 task labels，v13 有无监督 posterior。整合后不建议直接用 label 强行定义 blocks，而是让 label 成为 posterior evidence。

可以令 block partition posterior 为：

$$
P(z\mid X,L)
\propto
P(X\mid z)
P(L\mid z)
P(z).
$$

其中 $L$ 是 task label / condition / cue / reward。一个轻量版本是给 v13 的 candidate score 加 label readout evidence：

$$
L_{14}(z)
=
L_{13.4}(z)
-
\lambda_L
\Delta_{\mathrm{label}}(z).
$$

但这一步要谨慎，因为它可能把无监督 discovery 变成 label confirmation。建议 v14.2 才做，v14.0 先保持 v13 discovery label-free。

## 9. 新增诊断表建议

整合后，v8 输出目录里应新增：

```text
v13_discovery_summary.csv
v13_component_posterior.csv
v13_block_posterior.csv
v13_block_psm.csv
v13_loss_candidates.csv
v13_block_assignment.csv
v13_block_to_task_overlap.csv
v13_cross_session_block_alignment.csv
v13_train_test_projection.csv
```

其中最关键的是：

### 9.1 train-test projection

检查 discovery components 在 held-out test 上是否仍然有效：

$$
\Delta R^2_{\mathrm{test}}
=
R^2_{\mathrm{with\ v13}}
-
R^2_{\mathrm{without\ v13}}.
$$

### 9.2 block-to-task overlap

不是用 label 定义 block，而是事后检查：

$$
\mathrm{overlap}(b,\ell)
=
\mathrm{corr}
\left(
H_b,
L_\ell
\right).
$$

### 9.3 residual leakage

如果移除 v13 block 后，残差仍能 decode 该 block 的 H，则说明 block 没有被充分建模：

$$
R^2_{\mathrm{decode}}
\left(
H_b
\leftarrow
R_{\mathrm{without}\ b}
\right).
$$

这可以直接接入 v8 的 `residual_leakage_table` 思路。

## 10. 科学叙事如何更完整

整合后，我们的任务叙事可以变成：

```text
我们不是只问“有没有 cue/reward factor”，
而是问：

1. 神经数据中是否存在稳定的低维 latent factor subspace？
2. 这些 latent factors 是否形成 posterior-supported functional blocks？
3. 这些 blocks 是否超越已知 task labels，解释 residual neural structure？
4. 它们是否通过 structured null、split-half、held-out session、free residual PC 检验？
5. 它们和 cue/reward/condition trace 的关系是怎样的？
```

这样整个项目会比单纯方法优化更像一个科学研究闭环。

## 11. 推荐下一步

我建议下一版命名为：

```text
v14_0_integrated_v13_discovery_in_v8_framework
```

第一版只做最小整合，不做过度理想化：

1. 从 v8 package 复制出 v14 package。
2. 新增 `v13_discovery.py`，封装 v13.4 的核心函数。
3. 新增 `build_v13_discovered_factor_set(dataset, spec)`。
4. 先支持 `same_region_unaligned`，每个 session 独立 discovery。
5. discovery 只用 train mask。
6. 把 v13 blocks 转成 `FactorSpec(kind="posterior_block")`。
7. 复用 v8 的 `run_standard_diagnostics`。
8. 输出 v13-specific diagnostic CSV。

如果 v14.0 跑通，并且 v8 audit 显示 v13 blocks 通过 null/stability/held-out 检验，再进入 v14.1 的 cross-session consensus。

## 12. 总结

v8 和 v13 的关系可以压缩成一句话：

```text
v13 负责更科学地发现 latent blocks；
v8 负责更科学地证明这些 blocks 不是幻觉。
```

所以最终整合方向不是“把 v8 改成 v13”，而是：

```text
v8 scientific framework
    + v13 posterior discovery engine
    + train/test anti-leakage
    + structured null and stability audit
    = v14 integrated scientific pipeline
```

这会让项目从方法论验证，推进到真正可以写成完整科学分析的整体框架。
