# FYP — COOM EWC 基线改进笔记

2026-04-22

---

## 一、任务目标

COOM (Tomilin 2023) 是基于 ViZDoom 的 Continual RL benchmark，包含 8 个场景和多种任务序列（CD / CO / COC 等），在 SAC 上实现了 9 种 CL 方法作为 baseline。

FYP 要做三件事：

1. 挑一个 baseline（已选 EWC），用论文的配置复现出接近论文的数字，作为对照组
2. 对 EWC 做改进
3. 把改进版和原版并排跑，分析是否真的有提升；如果数字对不上论文，要能说清楚为什么

评估用论文里的三个指标：

- **Average Performance**：训练结束后在所有任务上的平均 success
- **Forgetting**：任务 i 训练结束时的性能减去整个序列结束时任务 i 的性能
- **Forward Transfer**：相对 SAC-from-scratch 的学习效率增益

后两个指标都需要训练过程中对过去任务做 eval（即 `--test=True`），目前的 run 关了 test，所以只能看 Average Performance 的粗略折算。

### 目前完成了什么

已完成：

- Python 3.10 + TF 2.11 (CPU-only) + ViZDoom 环境搭好
- EWC pipeline 端到端验证
- CO4 × steps_per_env=20k × seed=1 跑完一次，有 training curves
- 绘图脚本和对比分析（见 `ewc_co4_20k_seed1.png` 和 `comparison_ewc_co4_vs_paper.md`）

还没做：

- 没开 eval，没有 Forgetting / FT 数字
- 没跑 Fine-Tuning 对照组
- 没按论文配置跑完整 2e5 步

---

## 二、EWC 在这个仓库的实现

### 原始 EWC (Kirkpatrick 2017)

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{current}}(\theta) + \lambda \sum_i F_i (\theta_i - \theta^*_i)^2$$

$F_i$ 是 Fisher 信息矩阵对角元素，衡量参数 i 对旧任务的重要性；$\theta^*$ 是旧任务训练结束时的参数快照。

### COOM 里的 RL 适配（`CL/methods/ewc.py`）

几个关键点跟监督版不太一样：

- **Fisher 怎么算**：对 actor logits 之和、critic Q1/Q2 输出各自求 Jacobian，平方后按 batch 平均。严格讲这是 empirical Fisher 的变体，不是数学上完整的 expected Fisher
- **从哪采样**：从当前任务末尾的 replay buffer 里采 10 batch × 32 样本
- **哪些参数受约束**：只正则 `common_variables`（多头架构下的共享骨干），每个任务的输出头不受约束
- **多任务怎么合并**：逐元素相加（`_merge_weights`），Fisher 会随任务数单调增长
- **下限**：Fisher clip 到 `1e-5`，避免某些参数被彻底放弃
- **critic 要不要正则**：默认 `regularize_critic=False`，只约束 actor

论文推荐配置：

```bash
python run_cl.py --cl_method ewc --cl_reg_coef 250 \
                 --sequence CO8 --seed {1..10}
```

其余超参走 `config.py` 默认：lr=1e-3, γ=0.99, α=auto, replay_size=5e4, batch_size=128, steps_per_env=2e5。

### EWC 在 COOM 分数一般（0.54）的几个可能原因

论文 baseline 表里 EWC 排第 5（PackNet 0.74 > ClonEx 0.73 > L2 0.64 > MAS 0.56 > EWC 0.54），也就是说 EWC 在 COOM 上其实不强。可能的原因：

1. Fisher 近似不严格（jacobian-of-sum 不等于 expected Fisher）
2. Fisher 累加后量级失控，长序列上 λ 要么压死策略，要么被淹没
3. critic 不受约束，Q 的变化会反过来通过 actor loss 扰动 actor
4. SAC 的 α 自适应在任务切换时会重新学，引入额外参数漂移
5. replay buffer 默认在任务切换时重置，Fisher 只能用新任务末尾那一段 rollout 估计

这几点都是改进的切入点。

---

## 三、改进方向

把想到的 8 个方向放一起，大致按 "写出来好不好看" 和 "改动成本" 排：

| 方案 | 动机 | 改动量 | 对论文的增量 |
|---|---|---|---|
| Online EWC（Fisher EMA 代替累加） | 累加在长序列上量级失控 | 小 | 清晰，有先例 |
| Per-layer λ（CNN / MLP / head 分别调） | CNN 特征跨任务共享多，MLP 头更任务相关，应区别对待 | 小 | 清晰，容易写故事 |
| True Fisher（log-prob 梯度替代 logits-sum） | 现实现数学上不完全是 Fisher | 中 | 偏技术细节 |
| Fisher 分位数裁剪（只正则 top-k 重要参数） | 大部分参数 Fisher 接近 0，约束它们浪费算力 | 小 | 一般 |
| 动态 λ（随任务难度或 Fisher 大小调） | 固定 λ=250 跨任务不公平 | 中 | 一般 |
| regularize_critic on/off | 默认关，打开看看 | 极小 | 消融用 |
| EWC + 小型 reservoir replay 混合 | 纯正则方法普遍弱于 memory 方法 | 大 | 有野心，风险高 |
| Uncertainty-aware Fisher（MC Dropout 估参数不确定性） | 给出更鲁棒的重要性度量 | 大 | 学术味强 |

推荐从 **Online EWC** + **Per-layer λ** 这两个组合入手，理由：

- 两个方向正交，可以独立消融也可以组合
- 都只改 `regularization.py` 和 `ewc.py`，不动 SAC 主体
- 故事完整："EWC 在长序列失效 → Fisher 管理（EMA）+ 空间维度管理（per-layer）"

### Online EWC 的改动

改 `regularization.py` 里的 `_merge_weights`：

```python
# 原版：Fisher 累加
merged = F_old + F_current_task

# 改版：Fisher EMA
merged = γ * F_old + (1 - γ) * F_current_task    # γ ∈ [0.9, 0.99]
```

消融扫 γ = 0.9 / 0.95 / 0.99。预期 Forgetting 会略差（对远古任务约束变弱），但新任务学习效率会好；总分看 trade-off。

### Per-layer λ 的改动

把 `reg_weights` 按 layer 分桶（CNN 编码器 / 共享 MLP / actor head / critic head），每桶一个 λ：

```python
λ_cnn   = 500    # 强约束：特征跨任务共享
λ_mlp   = 100    # 中约束
λ_head  = 0      # 不约束：任务头本来就该变
```

先手设几组试，再视情况上网格搜索。

### 能叠加的消融

- 原版 `regularize_critic=False` vs `True`
- `reset_buffer_on_task_change=True` vs `False`
- Fisher 样本数从 10×32 扩到 50×64 看稳定性

---

## 四、路线图

### 已完成
环境 + pipeline 验证 + 首次 20k 步 run + 可视化。

### 下一步（本机能做，2–3 天）
1. 跑 Fine-Tuning 基线（同 CO4 × 20k × seed=1，`cl_method=None`），和 EWC 画在一起。这是判断 EWC 到底有没有用的最直接方法
2. 开着 eval 重跑一次 EWC，拿到 Forgetting 数字
3. 跑一次 `regularize_critic=True` 的 EWC 做消融

产出：一张图对比 Fine-Tuning / EWC / EWC+critic 三条曲线，外加 Forgetting 表。

### 实现改进（3–5 天）
4. 挑 Online EWC 和 Per-layer λ 中的一个（或两个都做）作为主改进
5. 新建 `CL/methods/ewc_online.py`（或 `ewc_layerwise.py`），继承 `Regularization_SAC`，只 override 需要的函数
6. 在 `run_cl.py` 的 `CLMethod` enum 里注册
7. 本机 CO4 × 20k 跑通，确认没崩

### 云端规模化（3–5 天，约 $15–30）
8. RunPod 租 RTX 3090
9. 按论文 `steps_per_env=2e5` 跑：原版 EWC × 3 seed、改进版 × 3 seed、Fine-Tuning × 3 seed，都在 CO4 上
10. 预算够就扩到 CO8

### 分析和写作（3–5 天）
11. 三个指标（Avg Perf / Forgetting / FT）全部对比
12. 讨论改进版能否 reach 论文 0.54；如果低于，是 seed 方差还是实现差异；假说成不成立
13. 出终版图表，更新 comparison.md

---

## 五、需要你先决定的几件事

1. **改进方向**：是按我推荐的 Online EWC + Per-layer λ，还是你更想试别的（比如 EWC + replay 混合）？
2. **硬件**：Phase 3 愿意上 RunPod（$15–30 / 一套完整对比）吗？有没有学校 GPU 资源可以用？
3. **交付物**：最终要的是论文（LaTeX）、报告（Markdown / Word）、代码仓库，还是加 PPT？
4. **时间节点**：FYP deadline 是什么时候？有没有中期检查？

这几件答了我再把下一步细化成具体的实验脚本。

---

## 附：文件位置

```
COOM/
├── CL/methods/ewc.py                 要读的核心
├── CL/methods/regularization.py      Fisher 合并在这
├── CL/rl/sac.py                      SAC 主循环
├── CL/run_cl.py                      实验入口
├── CL/config.py                      CLI 超参
├── CL/scripts/cl.sh                  论文运行命令
├── CL/logs/ewc_co4_20k_seed1/        本次 run 的日志
├── COOM/env/builder.py               环境构造
├── COOM/utils/config.py              序列 / 任务定义
├── results/ewc_co4_20k_seed1.png     训练曲线
├── results/comparison_ewc_co4_vs_paper.md
└── results/fyp_ewc_plan.md           本文件
```
