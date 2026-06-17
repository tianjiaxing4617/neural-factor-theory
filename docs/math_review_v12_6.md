# v12.6 神经轨迹 Factor / Block 模型：数学思路、推导与问题审查

> 这一版专门把数学部分改成 GitHub 可渲染的 LaTeX 公式。阅读目标不是“像代码一样跑一遍”，而是像论文推导一样看清楚：模型在估计什么、哪些量是可识别的、哪些步骤目前仍有数学或统计风险。

---

## 0. 总览

我们目前的理论主线已经从早期的“找一组神经 latent factors”，逐渐变成了更稳健的三层结构：

1. 先从重复 split 中寻找可靠的低维子空间；
2. 再在这个可靠子空间里寻找可解释的 localized components；
3. 最后把 components 聚合成 function blocks，并用图结构描述 block 之间的关系。

用最简洁的数学语言说，当前版本不是直接声称每一个 recovered component 都等于一个真实 latent factor，而是声称：

$$
\text{data}
\longrightarrow
\text{reliable subspace}
\longrightarrow
\text{localized components}
\longrightarrow
\text{block graph}.
$$

这一步很关键。它把“单个 component 是否真实”这个高风险命题，改成了“哪些子空间、哪些局部模式、哪些 block 关系在 split 下稳定”这个更可检验的问题。

---

## 1. 数据与基础分解

设神经数据为

$$
X \in \mathbb{R}^{R \times T \times N},
$$

其中 $R$ 是 trial 数，$T$ 是每个 trial 的时间点数，$N$ 是神经元数。将 trial 与 time 展平成样本维度：

$$
M = R T,
\qquad
X_{\mathrm{flat}} \in \mathbb{R}^{M \times N}.
$$

基础 factor 模型写成

$$
X_{\mathrm{flat}}
= H W^\top + \varepsilon,
$$

其中

$$
H \in \mathbb{R}^{M \times K},
\qquad
W \in \mathbb{R}^{N \times K},
\qquad
\varepsilon \in \mathbb{R}^{M \times N}.
$$

第 $k$ 个 component 对数据的 rank-1 贡献为

$$
E_k = h_k w_k^\top.
$$

如果第 $g$ 个功能组包含 $C$ 个 components，则该组贡献为

$$
X_g
= \sum_{m=1}^{C} h_{g,m} w_{g,m}^{\top}.
$$

这里的核心辨析是：$K$ 是算法恢复的 component 数，不一定等于真实功能组数 $F$。真实功能结构更可能出现在 components 的组合、子空间和 graph level 上。

---

## 2. Toy 模型的可识别性问题

### 2.1 旧 toy 的退化结构

旧 toy 的主要问题是，同一功能组内多个 $W$ loading 近似共线：

$$
w_{g,m}(n) \approx a_{g,m} u_g(n).
$$

于是第 $g$ 个 group 的信号为

$$
\begin{aligned}
X_g(t,n)
&= \sum_m h_{g,m}(t) w_{g,m}(n) \\
&\approx \sum_m h_{g,m}(t) a_{g,m} u_g(n) \\
&= \widetilde{h}_g(t) u_g(n),
\end{aligned}
$$

其中

$$
\widetilde{h}_g(t)
= \sum_m a_{g,m} h_{g,m}(t).
$$

这说明原来的 $C$ 个 components 在观测层面坍缩成了一个 rank-1 group：

$$
\operatorname{rank}(X_g) \approx 1.
$$

因此旧 toy 虽然设定了

$$
K_{\mathrm{true}} = F C,
$$

但从数据角度真正可稳定识别的更接近

$$
K_{\mathrm{identifiable}} \approx F.
$$

这就是旧版本中 $K$ 经常回到 $F$ 附近的数学原因：算法不是没有能力，而是 toy 本身没有给出足够的独立方向。

### 2.2 v12.5/v12.6 的改进

新 toy 把每个 functional group 放在连续的 latent coordinate $z_n$ 上，并让同组内不同 component 的中心位置、宽度、时间 pattern 和 label 调制有所区别。

一个典型的 spatial loading 可以写成

$$
w_{g,m}(n)
= A_{g,m}
\exp\!\left(
-\frac{(z_n-\mu_{g,m})^2}{2\sigma_{g,m}^2}
\right)
+ \eta_{g,m}(n).
$$

其中 $\mu_{g,m}$ 和 $\sigma_{g,m}$ 控制空间位置与宽度，$\eta_{g,m}$ 是扰动或噪声。为了避免同组 components 完全共线，需要满足近似的分离条件：

$$
|\mu_{g,m} - \mu_{g,m'}|
\gtrsim c \cdot \min(\sigma_{g,m}, \sigma_{g,m'}).
$$

同时，$H$ 端也需要有足够独立的时间/label 变化。若只是在 $W$ 端制造局部差异，而 $H$ 端仍高度共线，整体 rank 仍会偏低。

### 2.3 有效秩

对某一组的 loading 矩阵

$$
W_g = [w_{g,1},\ldots,w_{g,C}]
\in \mathbb{R}^{N \times C},
$$

计算奇异值

$$
s_1 \ge s_2 \ge \cdots \ge s_C.
$$

归一化后

$$
p_i = \frac{s_i}{\sum_j s_j},
$$

有效秩可以写成

$$
r_{\mathrm{eff}}(W_g)
= \exp\!\left(
-\sum_i p_i \log p_i
\right).
$$

如果

$$
r_{\mathrm{eff}}(W_g) \ll C,
$$

那么 toy 名义上有 $C$ 个 components，但数据实际只支持更少的独立方向。这个指标应该作为 toy 生成阶段的诊断量，而不是只在 recovery 后解释结果。

**当前问题：** v12.6 已经比旧 toy 好很多，但代码仍需要明确报告每个 group 的 $r_{\mathrm{eff}}(W_g)$、$r_{\mathrm{eff}}(H_g)$ 和 $r_{\mathrm{eff}}(X_g)$，否则我们无法判断失败来自算法还是生成模型本身。

---

## 3. K discovery：从 split 稳定性到 evidence

### 3.1 split 视角

对不同 split $s$，算法得到一个估计的 component 子空间：

$$
\widehat{U}^{(s)}_K
\in \mathbb{R}^{N \times K}.
$$

两个 split 之间的子空间相似度可以用主角度表示。若奇异值为

$$
\sigma_i
\left(
(\widehat{U}^{(s_1)}_K)^\top
\widehat{U}^{(s_2)}_K
\right),
$$

则可以定义 reliability：

$$
C_{\mathrm{rel}}(K)
= \frac{1}{K}
\sum_{i=1}^{K}
\sigma_i^2.
$$

直觉上，如果 $K$ 太小，模型欠拟合；如果 $K$ 太大，额外方向主要是噪声，split 之间不稳定。

### 3.2 residual discovery

当前代码采用逐步 residual 的想法。第 $q$ 步残差为

$$
R_q
= X_{\mathrm{flat}}
- \sum_{k=1}^{q}
\widehat{h}_k \widehat{w}_k^\top.
$$

从残差中继续寻找新方向：

$$
(\widehat{h}_{q+1}, \widehat{w}_{q+1})
= \operatorname{Factorize}(R_q).
$$

这是合理方向，因为它避免了一次性分解把强信号吞掉弱信号。但 residual 方法有一个隐含风险：后续 component 的质量依赖前面 component 的估计误差。如果早期方向稍微偏了，残差会携带结构性伪影。

### 3.3 evidence / free-energy 直觉

理想情况下，对每个候选 $K$，我们想比较：

$$
\log p(X \mid K).
$$

如果采用高斯噪声近似：

$$
X = H_K W_K^\top + \varepsilon,
\qquad
\varepsilon \sim \mathcal{N}(0,\sigma^2 I),
$$

则负对数似然的主要部分为

$$
-\log p(X \mid H_K,W_K,\sigma^2)
\propto
\frac{1}{2\sigma^2}
\left\|X-H_K W_K^\top\right\|_F^2
+ \frac{MN}{2}\log\sigma^2.
$$

如果加入复杂度惩罚，可以得到类似 BIC 的 score：

$$
\operatorname{BIC}(K)
= -2 \log p(X \mid \widehat{\theta}_K)
+ d_K \log(MN),
$$

其中 $d_K$ 是模型自由度。

当前 v12.6 的 evidence 更像是启发式组合：

$$
S_K
= \alpha \cdot \operatorname{Reliability}(K)
+ \beta \cdot \operatorname{ExplainedVar}(K)
- \gamma \cdot \operatorname{Penalty}(K).
$$

这个方向可以工作，但目前最大的问题是：某些 evidence 量同时参与了搜索和验证，导致 validation 不再是独立检验。

### 3.4 K discovery 的核心风险

**K1. 验证集双重使用。**  
如果同一个 validation split 既用于选择 residual candidate，又用于评价最终 $K$，那么得到的 $K$ evidence 会偏乐观。严格做法应该至少分三层：

$$
\text{train}
\longrightarrow
\text{select candidates}
\longrightarrow
\text{held-out validation}.
$$

**K2. reliability 只能证明子空间稳定，不能直接证明 component 唯一。**  
如果存在任意正交旋转 $Q$：

$$
H W^\top
= (H Q)(W Q)^\top,
$$

则数据重构不变。split 稳定性若只在 subspace 上成立，并不能自动推出每个 column 的解释是唯一的。

**K3. 最大接受 $K$ 与最终选择 $K$ 的统计量混淆。**  
报告时必须区分：

$$
K_{\mathrm{accepted,max}}
\quad \text{和} \quad
K_{\mathrm{selected}}.
$$

如果最终 $R^2$ 来自最大接受 $K$，却被解释成 selected $K$ 的性能，就会高估模型。

**K4. 需要 null model。**  
每个 $K$ 的 score 应该与 permutation 或 phase-shuffle null 比较：

$$
Z_K
=
\frac{
S_K - \mathbb{E}[S_K^{\mathrm{null}}]
}{
\operatorname{sd}(S_K^{\mathrm{null}})
}.
$$

没有 null，稳定性和 explained variance 的绝对值很难解释。

---

## 4. Localization：从可靠子空间到局部 components

### 4.1 神经元坐标与 Fiedler ordering

当前模型把神经元之间的相似度写成图权重：

$$
A_{ij} = \operatorname{sim}(x_i,x_j).
$$

图 Laplacian 为

$$
L = D - A,
\qquad
D_{ii} = \sum_j A_{ij}.
$$

Fiedler vector 是

$$
z
= \arg\min_{v \perp \mathbf{1},\,\|v\|=1}
v^\top L v.
$$

这个 $z$ 给出一维 ordering，用来描述 loading 是否局部集中。

### 4.2 局部 dictionary

一个局部 bump basis 可以写成

$$
\phi_j(n)
=
\exp\!\left(
-\frac{(z_n-c_j)^2}{2\sigma_j^2}
\right).
$$

将所有 basis 组成

$$
\Phi
=
[\phi_1,\ldots,\phi_J]
\in \mathbb{R}^{N \times J}.
$$

如果某个 loading $w_k$ 可以被少数局部 basis 表示：

$$
w_k \approx \Phi a_k,
\qquad
\|a_k\|_0 \ll J,
$$

则该 component 具有 localization 解释。

### 4.3 rotation 的本质

从 factorization 得到的 $W$ 通常只确定到旋转：

$$
W \sim WQ,
\qquad
Q^\top Q = I.
$$

因此 localization 步骤实际是在可靠子空间内寻找一个旋转 $Q$，使得

$$
\widehat{W}_{\mathrm{loc}}
= \widehat{W}_{\mathrm{sub}} Q
$$

具有更强的局部性。可以把目标写成：

$$
Q^\star
=
\arg\max_{Q^\top Q=I}
\sum_{k=1}^{K}
\operatorname{Locality}
\left(
\widehat{w}_k(Q)
\right).
$$

这一步是整个模型的解释核心：subspace 由 split reliability 支撑，component 解释由 localization criterion 支撑。

### 4.4 localization 的当前问题

**L1. mirror orientation bug。**  
当前代码用类似下面的规则比较 $z$ 与 $1-z$：

$$
\left|
\operatorname{corr}(1-z,y)
\right|
\quad \text{vs.} \quad
\left|
\operatorname{corr}(z,y)
\right|.
$$

但因为

$$
\operatorname{corr}(1-z,y)
=
-\operatorname{corr}(z,y),
$$

所以两边取绝对值后永远相等：

$$
\left|
\operatorname{corr}(1-z,y)
\right|
=
\left|
\operatorname{corr}(z,y)
\right|.
$$

这意味着 mirror orientation 实际上没有被正确选择。应该改用带符号的 correlation，或引入明确的 anchor。

**L2. residualized loading 的元数据不一致。**  
如果先将 $w_k$ 投影掉已解释方向：

$$
w_k^{\perp}
=
w_k
-
P_{\mathcal{S}_{k-1}} w_k,
$$

那么后续报告的 $\mu_k,\sigma_k$ 应该基于 $w_k^{\perp}$ 重新计算，而不能继续使用原始 $w_k$ 的 metadata。

**L3. normalization 可能破坏 diversity penalty。**  
如果 diversity penalty 用的是 residual norm：

$$
\left\|w_k^{\perp}\right\|_2,
$$

但代码在计算前后强制归一化：

$$
\frac{w_k^{\perp}}{\|w_k^{\perp}\|_2},
$$

那么 norm 本身携带的“是否真的有新方向”的信息会被抹掉。

**L4. localization 不等于真实功能组。**  
局部性只能说明 component 在 $z$ ordering 上集中，不能单独证明它对应一个功能模块。还需要 label dependency、split stability 和 graph evidence 共同支撑。

---

## 5. H dependency：时间与 label 的解释层

对每个 component，时间/label embedding 为 $h_k$。若行为标签或任务变量为 $Y$，可以定义依赖强度：

$$
D_k
=
\operatorname{Dep}(h_k,Y).
$$

若用 distance correlation，可以写成：

$$
\operatorname{dCor}^2(h_k,Y)
=
\frac{
\operatorname{dCov}^2(h_k,Y)
}{
\operatorname{dVar}(h_k)\operatorname{dVar}(Y)
}.
$$

对两个 components 的 H-side 依赖可以写成：

$$
P^H_{ij}
=
\operatorname{Dep}(h_i,h_j).
$$

关键问题是：如果 $h_k$ 来自同一数据的 factorization，然后再用同一数据评估 dependency，就可能出现过拟合解释。更稳妥的做法是 cross-fit：

$$
\widehat{w}_k^{\mathrm{train}}
\longrightarrow
\widehat{h}_k^{\mathrm{test}}
=
X_{\mathrm{test}}
\widehat{w}_k^{\mathrm{train}}.
$$

然后只在 test side 评估 dependency。

---

## 6. Block graph：components 到 function blocks

### 6.1 W-side 相似性

component $i$ 与 $j$ 的 loading 相似性可以写为

$$
P^W_{ij}
=
\left|
\operatorname{corr}(w_i,w_j)
\right|.
$$

也可以加入 spatial overlap：

$$
O^W_{ij}
=
\frac{
\sum_n |w_i(n)| |w_j(n)|
}{
\sqrt{
\sum_n w_i(n)^2
}
\sqrt{
\sum_n w_j(n)^2
}
}.
$$

### 6.2 H-side 依赖

H-side 依赖可以写成

$$
P^H_{ij}
=
\operatorname{Dep}(h_i,h_j).
$$

如果希望强调 label-aware block，可以进一步写成条件依赖：

$$
P^{H \mid Y}_{ij}
=
\operatorname{Dep}(h_i,h_j \mid Y).
$$

### 6.3 split survival

如果 block 在多个 split 中稳定出现，可以定义 survival：

$$
S_{ij}^{\mathrm{split}}
=
\mathbb{P}_{s}
\left[
i \sim j
\text{ in split } s
\right].
$$

理想的 co-association matrix 应该是：

$$
C_{ij}
=
\frac{1}{S}
\sum_{s=1}^{S}
\mathbf{1}
\left[
c_i^{(s)} = c_j^{(s)}
\right].
$$

当前代码里所谓的 split survival 更像是 component-level 稳定性，还没有完全达到 pairwise block co-association 的定义。

### 6.4 综合 graph score

一个合理的 block graph 边权可以写成：

$$
G_{ij}
=
\lambda_W P^W_{ij}
+ \lambda_H P^H_{ij}
+ \lambda_S S_{ij}^{\mathrm{split}}
+ \lambda_Y P^{H \mid Y}_{ij}.
$$

然后对 $G$ 做 community detection：

$$
\widehat{B}
=
\operatorname{CommunityDetect}(G).
$$

最终 block 数为

$$
\widehat{F}
=
|\widehat{B}|.
$$

---

## 7. F evidence 与 noise probe

### 7.1 F evidence

给定 graph partition $B$，可以定义 block-level score：

$$
\operatorname{Score}(B)
=
\operatorname{Within}(B)
- \operatorname{Between}(B)
- \lambda \operatorname{Complexity}(B).
$$

其中

$$
\operatorname{Within}(B)
=
\sum_{b \in B}
\frac{1}{|b|(|b|-1)}
\sum_{i,j \in b,\,i\ne j}
G_{ij},
$$

而

$$
\operatorname{Between}(B)
=
\frac{1}{|\mathcal{P}_{\mathrm{out}}|}
\sum_{(i,j)\in \mathcal{P}_{\mathrm{out}}}
G_{ij}.
$$

当前 v12.6 的问题是，实际输出中 $F_{\mathrm{evidence}}$ 经常退化为 1。这通常意味着 graph score 或 partition penalty 过度偏向“合并所有节点”。需要检查：

$$
\Delta \operatorname{Score}(F)
=
\operatorname{Score}(F)
- \operatorname{Score}(F-1)
$$

是否在所有 $F>1$ 时都被惩罚项压住。

### 7.2 noise probe

noise probe 的目标是判断发现的 components 是否超过噪声 baseline。可以构造：

$$
X^{\mathrm{null}}
=
\operatorname{Shuffle}(X),
$$

并重复完整 pipeline 得到 null score：

$$
S^{\mathrm{null}}_K
=
\operatorname{PipelineScore}(X^{\mathrm{null}},K).
$$

真实 score 的显著性为

$$
p_K
=
\frac{
1 + \sum_{b=1}^{B}
\mathbf{1}
\left[
S_{K,b}^{\mathrm{null}} \ge S_K
\right]
}{
B+1
}.
$$

noise probe 必须和真实 pipeline 使用同样的搜索自由度，否则会低估 false positive。

---

## 8. 当前 v12.6 代码层面的主要问题

### 8.1 K evidence 双重使用 validation

当前实现中，validation 信息既参与 candidate 的接受，又参与最终 evidence 的解释。这会导致：

$$
\mathbb{E}
\left[
S_{\mathrm{val}}(\widehat{K})
\right]
>
\mathbb{E}
\left[
S_{\mathrm{test}}(\widehat{K})
\right].
$$

建议：加入真正 held-out test split，或者使用 nested cross-validation。

### 8.2 z mirror 方向选择失效

如上所述，绝对相关会让 $z$ 与 $1-z$ 无法区分。这个问题应该优先修，因为它会影响所有 spatial metadata。

### 8.3 localization diversity penalty 被 normalization 削弱

如果算法希望惩罚重复方向，就必须保留 residual norm 或 overlap 的原始尺度。归一化后再计算 norm，等价于把关键证据抹平。

### 8.4 F evidence 退化

quick 与 rank1 debug 结果都显示 $F_{\mathrm{evidence}}=1$。这说明 block graph 目前还不能可靠恢复 toy 中设定的功能组数。

### 8.5 final $R^2$ 归属不清

需要明确报告：

$$
R^2(K_{\mathrm{selected}})
\quad \text{而不是} \quad
R^2(K_{\mathrm{accepted,max}}).
$$

否则会把“探索过程中最宽模型的拟合能力”误当成“最终模型的泛化性能”。

---

## 9. 我目前认为最稳的理论表述

当前模型最安全、最有说服力的数学表述是：

$$
\boxed{
\text{v12.6 recovers reliable neural subspaces and localized block candidates,}
}
$$

而不是：

$$
\boxed{
\text{v12.6 uniquely identifies all true latent factors.}
}
$$

更完整地说：

$$
\begin{aligned}
X
&\xrightarrow{\text{split reliability}}
\widehat{\mathcal{S}}_K \\
&\xrightarrow{\text{localized rotation}}
\{\widehat{w}_k,\widehat{h}_k\}_{k=1}^{K} \\
&\xrightarrow{\text{W/H/split graph}}
\widehat{B}_1,\ldots,\widehat{B}_{\widehat{F}}.
\end{aligned}
$$

其中真正被当前 evidence 支撑的是：

1. $\widehat{\mathcal{S}}_K$ 是否可靠；
2. localized components 是否在 split 下稳定；
3. block graph 是否显著优于 null；
4. label-aware dependency 是否能在 held-out 数据上复现。

---

## 10. 下一步最该修的数学与代码接口

我建议按优先级处理：

1. 修正 $z$ mirror orientation，避免绝对相关导致方向选择无效。
2. 把 K discovery 改成 nested 或 held-out validation，避免 evidence 双重使用。
3. 为 toy 生成器报告 $r_{\mathrm{eff}}(W_g)$、$r_{\mathrm{eff}}(H_g)$ 和 $r_{\mathrm{eff}}(X_g)$。
4. 重写 split survival 为真正的 pairwise co-association matrix。
5. 重新校准 F evidence 的 merge/split penalty，避免 $F_{\mathrm{evidence}}$ 总是塌缩到 1。
6. 区分并分别报告 $K_{\mathrm{selected}}$ 与 $K_{\mathrm{accepted,max}}$ 的 $R^2$。

如果这六点修完，模型叙事会从“有一个很有想象力的 pipeline”变成“有清楚可检验边界的数学模型”。这会更利于后面写成正式论文或技术报告。
