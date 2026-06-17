# 神经轨迹 Factor 理论演化梳理：v1 到 v12.5

## 一句话总览

这套理论并不是不断更换研究目标，而是在逐步收紧“什么对象可以被恢复、凭什么说它被恢复、在什么条件下它可识别”：从过完备 component 的两阶段解释，发展为 overlapping-W 下的条件贡献，再发展为 label-aware block，最后加入 identifiability-first 的可靠维数与功能 block graph。

## 版本主线

| 版本 | 核心问题 | 主要理论变化 | 保留下来的内容 |
|---|---|---|---|
| v1 | 如何避免把降维 component 直接当作真实 latent factor | 分离 representation problem 与 interpretation problem；允许 `K_model > K_true`；先得到过完备表示，再合并解释 | 两阶段结构成为全系列稳定内核 |
| v2 | 真实神经 mixed selectivity 下，“独立 factor”是什么意思 | 放弃专属神经元/unique-W 强假设；允许 W 高度重叠；把独立性改写为已有解释之外的 conditional contribution | 主解释对象从单独 H/W 转为 `E_g = H_g W_g` |
| v3 | 如何摆脱 q + fixed window toy，处理弱标签、错位标签和无标签数据 | anchor 降级为可选 EvidenceSource；建立 candidate generation、expression fitting、evidence audit、cross-session structural alignment | 候选可以被提出，也必须允许被拒绝 |
| v3.3 | H→W→H 的反馈如何获得正规的统计解释 | 用固定先验下的 alternating MAP 解释 dual-space refinement；避免反复把 posterior 当 prior 导致 evidence double counting | fast search、selected-model refinement、final audit 三段式流程 |
| v7 | 不同标签强度下，究竟能合理恢复什么 | 建立 label regime ladder；从 scalar-first 转为 block-first；弱标签只支持 anchor-compatible block，不支持 exact H | block 是主结果，scalar gauge 是摘要；multi-session 是辨识信息来源 |
| v8.8 | 如何冻结一个可迁移真实数据的 clean core | 固定 hierarchical shared-W 模型；用 factor-specific empirical-Bayes penalty、ΔNLL、strict null、held-out session、W stability 做正式审计 | 形成真实数据 package 主线，不再把探索性搜索塞入核心模型 |
| v12.5 | 为什么 K/F 恢复失败，以及失败究竟来自优化还是不可识别 | 先做 W/H effective-rank identifiability audit；把 K recovery 与 F interpretation 分离；K 由 split reliability + held-out gain 决定，F 由 W/H block graph 决定 | reliability-first、recovery/interpretation separation 被进一步严格化 |

## 真正发生的四次范式转向

### 1. 从“恢复真实 factor”到“先恢复可用表示”

v1 最关键的决定是承认矩阵分解的旋转不辨识性。重构好不等于恢复了真实生成因子，因此第一阶段只回答需要多少 model-level components，第二阶段才回答这些 components 如何组成科学上可解释的 factor group。

### 2. 从“空间互斥”到“条件不可替代”

v2 把 factor 独立性的定义从 neuron-level exclusivity 改为 population-level conditional contribution。不同 factors 的 W 可以高度重叠；真正需要证明的是，candidate 在 no-candidate/refit baseline 之外仍有 held-out gain、residual leakage reduction 和跨 session 稳定性。

### 3. 从“唯一 scalar”到“标签决定可恢复对象”

v7 明确了一个重要认识论边界：观测信息有多强，结论就只能有多强。window label 只能支持 anchor-compatible block；known-shape 才能支持 template-aligned H；oracle H 只属于 toy diagnostic。因而 block 是弱监督任务的主要发现对象，scalar 只是 gauge-dependent summary。

### 4. 从“拟合后解释失败”到“拟合前审计可识别性”

v12.5 把理论推进到更前面：如果组内 H 或 W 的有效秩不足，目标 K/F 在观测上就不存在唯一证据。此时继续调优化器没有意义。只有先确认目标在 H-side 和 W-side 都具有足够 effective rank，才应讨论 recovery accuracy。

## 当前稳定内核

截至 v12.5，以下主张在各版本中持续保留，并可视为理论的稳定核心：

1. `representation != interpretation`，重构 component 不自动等于真实 factor。
2. 主要解释对象是 time × neuron 空间中的 expression，而不是孤立的 H 或 W。
3. W 重叠是默认情形，独立性应通过条件增量和不可替代性定义。
4. factor discovery 必须允许 null 结论，不能因为给了 label/window 就强制发现 factor。
5. 多 session、trial split 和重复结构不仅用于验证，也提供可识别性信息。
6. recovery 与 interpretation 必须分离：先确定可靠 K/subspace，再讨论 component localization 和功能分组 F。
7. 评价标准必须与 label regime 和可识别条件匹配，toy oracle 指标不能直接迁移到真实数据。

## 当前两条互补路线

### v8.8：真实数据核心模型

负责给定 candidate/FactorSpec 后的拟合与严格证据审计：

```text
X_s(t,n) ≈ baseline_s(n)
          + Σ_g H_s,g(t) [W_g(n) + ΔW_s,g(n)]
          + noise_s(t,n)
```

它回答：一个指定或生成的 candidate 是否有条件贡献、是否强于 strict null、是否跨 split/session 稳定。

### v12.5：可靠结构发现模块

负责在可识别 toy 中回答：数据支持多少个可靠 components，以及这些 components 如何形成 functional blocks。它的新增模块包括 identifiability audit、iterative K discovery、localization 和 block graph。

二者不是替代关系。合理整合方式是：v12.5 作为 discovery/audit 模块接入 v8.8 package，而 v8.8 保持 real-data model、数据接口和证据审计核心。

## 仍需解决的理论接口

1. **真实数据 identifiability audit**：toy 中有 true group 和 effective rank；真实数据中需要用 split rank、condition number、perturbation sensitivity、subspace survival 等可观测替代量。
2. **跨 session W 的定义**：v3 允许 permutation、partial observation 和 structural identity；v8.8 的 `W_global + ΔW_s` 默认 neuron 维度可对齐。真实数据必须明确是 matched-neuron、registered population，还是 alignment-invariant comparison。
3. **K 与 F 的接口**：K stopping 应在 discovery 阶段自动完成，不能先过度接受再 posthoc 回退；F 必须报告 split survival，而非依赖一次 localization。
4. **toy-specific localization 的边界**：Gaussian center 与 hidden z 是验证 block discovery 的 toy 工具，不能自动上升为一般神经编码假设。
5. **术语统一**：建议固定 component、candidate expression、factor block、functional factor 四个层级，避免把不同层级都称为 factor。

## 当前理论的最简表述

神经数据中的可解释结构不应被预设为一组唯一、互斥、逐个对应真实原因的 scalar factors。更合理的目标是：先检查数据是否支持目标维度；再利用跨 trial/session 的重复结构发现可靠 expression subspace；随后依据标签强度、条件增量、严格 null、稳定性以及 H/W 依赖，将可靠 components 组织为具有相应证据等级的 factor blocks。最终解释的强度不能超过数据所提供的可识别性与监督信息。
