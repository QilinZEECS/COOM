# EWC on CO4 — 本次运行 vs COOM 论文（首次对比）

> 日期：2026-04-22
> 运行目录：`CL/logs/ewc_co4_20k_seed1/2026_04_21__19_17_33_OeSi3e/`
> 训练曲线：`results/ewc_co4_20k_seed1.png`

---

## 1. 实验设置对照

| 维度 | 本次运行 | COOM 论文 (EWC baseline) | 差距说明 |
|---|---|---|---|
| 序列 | CO4 | CO4 / CO8 / CD4 / CD8 / CD16 / CO16 / COC | 本次只跑 CO4 |
| 每任务步数 | **20,000** | **200,000** | **1/10** |
| 总训练步数 (CO4) | 80,000 | 800,000 | 1/10 |
| 随机种子数 | **1** | **10** | 本次无均值/方差 |
| `cl_reg_coef` (λ) | 250 | 250 | ✅ 一致 |
| `regularize_critic` | False | False | ✅ 一致 |
| 其余超参 (lr, γ, α, buffer) | 论文默认 | 论文默认 | ✅ 一致 |
| 硬件 | Apple Silicon CPU-only | NVIDIA GPU (CUDA) | 约 10–30× 慢 |
| 训练时 Eval | **关闭** (`--no_test`) | 开启 (test_episodes=3) | 本次无 Forgetting/FT 指标 |

---

## 2. 核心指标对比

### 2.1 论文 EWC 综合分数

根据 README baseline 表：

| 方法 | 类型 | Score |
|---|---|---|
| PackNet | Structure | 0.74 |
| ClonEx-SAC | Memory | 0.73 |
| L2 | Regularization | 0.64 |
| MAS | Regularization | 0.56 |
| **EWC** | **Regularization** | **0.54** |
| Fine-Tuning | Naïve | 0.40 |
| VCL | Regularization | 0.33 |
| AGEM | Memory | 0.28 |

**说明**：论文 0.54 是 Average Performance（训练结束时对序列里所有任务 success rate 平均），跨多个序列、10 seed 平均后报告。

### 2.2 本次运行的原始训练曲线数字

| # | 任务 | 峰值 success | 末期 success | 峰值 return |
|---|---|---|---|---|
| 0 | Chainsaw | 16.7% | 0% | 99 |
| 1 | Raise the Roof | 7.5% | 6.2% | 30 |
| 2 | Run and Gun | 0% | 0% | 28 |
| 3 | Health Gathering | 34.5% | 1.9% | 162 |

**粗略折算 Average Performance（训练中的末期值）**
$$\text{Avg Perf} \approx \frac{0 + 0.062 + 0 + 0.019}{4} \approx 0.020$$

---

## 3. 差距分析

### 3.1 为什么本次数字远低于论文 0.54？

**直接原因（预期、不是 bug）**

1. **训练步数只有 1/10**
   - 论文 2e5 步/任务，才能让 SAC + CNN 在像素观测上收敛
   - 20k 步：Run and Gun 0% success 已经说明策略尚未起步
   - 训练曲线呈"上升 + 未收敛"，远未到 plateau

2. **没有 Eval 阶段**
   - 论文的"末期 success" = 训练结束后用 `test_episodes=3` 无探索地评估，可重复
   - 本次"末期 success" = 训练过程中带 alpha 探索的 rollout，一个 epoch 1 个 episode 左右，**噪声极大**
   - 比如 Chainsaw "末期 0%" 其实是训练最后 epoch 恰好没成功 → 不等于策略不会

3. **只有 1 个 seed**
   - 论文 10 seed 平均后数字更稳定；单 seed 可能落在分布下尾

**间接原因**

4. **CO4 是 CO8 的后 4 个任务** (`CO_scenarios[4:]`)，本次序列是：
   Chainsaw → Raise the Roof → Run and Gun → Health Gathering
   
   这 4 个任务目标差异大（射击 / 按开关 / 射击 / 收集），对 CL 更难。论文分数可能混合 CD / CO 序列平均。

5. **CPU 训练可能引入随机性差异**
   - TF 2.11 在 CPU 上和 GPU 上的浮点累加顺序不同，即使同 seed 也会有细微轨迹差
   - 但这是 seed 级别的差异，不该解释 0.02 vs 0.54 的量级差距

### 3.2 哪些结果"符合论文预期"？

✅ **EWC 正则机制的定性行为与 EWC 原论文/COOM 论文一致**
- 任务切换时 `loss_reg` 冲到 200–470 然后快速衰减 → Fisher 累积、新参数偏移旧参数被惩罚、优化又把它拉回
- Fisher 随任务累加，约束越来越强（task 1 峰值 25 < task 2 峰值 378 < task 3 峰值 470）

✅ **任务难度排序粗略对齐**
- Health Gathering 学习最快、return 最高 → 与论文 "简单任务" 的定性一致
- Run and Gun 最难学 → 与论文匹配

✅ **Fine-Tuning 的退化特征**（下一步验证）
- 当前观察：Chainsaw 训练末期 success 掉到 0，可能是 EWC 约束过强，也可能是训练不稳
- 需跑 Fine-Tuning 基线对照：如果 Fine-Tuning 也掉到 0，说明是 SAC 通病；如果 Fine-Tuning 更平稳而 EWC 退化，说明 λ=250 过大

### 3.3 哪些问题无法从当前数据判断

❌ **Forgetting（灾难性遗忘程度）**
- 需要训练过程中持续在过去任务上 eval → 本次 `--no_test` 没做

❌ **Forward Transfer**
- 需要与 SAC-only（从头在单任务训）的 AUC 对比 → 本次未跑

❌ **EWC 相对 Fine-Tuning 的真实效果**
- 需要同配置跑一次 `cl_method=None` 的 Fine-Tuning

---

## 4. 结论

**本次运行是一个"pipeline 级"的验证，不是"数字级"的复现。**

| 问题 | 回答 |
|---|---|
| EWC 代码能不能正确跑？ | ✅ 能，曲线定性符合 EWC 机制 |
| 本次 Average Performance (~0.02) 能跟论文 0.54 直接对比吗？ | ❌ 不能，步数、seed 数、eval 方式全部不同 |
| 如果按论文配置跑是否能复现 0.54？ | 仍待验证——需要云端 GPU 跑完整 2e5 × 8 任务 × 10 seed |
| 本次结果能告诉我什么？ | EWC 正则机制激活正常；CO4 任务难度梯度合理；训练 loss 稳定不发散 |

---

## 5. 下一步建议

按对 FYP 价值排序：

1. **[高价值, 低成本]** 同配置跑一次 Fine-Tuning 基线 (`cl_method=None`, 其余参数相同)，在同一张图上对比，观察 EWC 是否真的在缓解 task 切换时的性能退化
2. **[中价值, 中成本]** 重跑本实验但开启 `--test`，拿到 Forgetting 数字（耗时 ~3–4h CPU）
3. **[高价值, 高成本]** 上云端 GPU 跑论文完整配置（2e5 × 8 任务 × 3–5 seed），这一步才真正能对比 0.54
4. **[准备改进]** 确定改进方向后，先在 CO4 / 20k 规模上做消融，再上云 final run

---

## 6. 附：当前已有产出

- `CL/logs/ewc_co4_20k_seed1/2026_04_21__19_17_33_OeSi3e/progress.tsv` — 训练日志（80 个 epoch）
- `CL/logs/ewc_co4_20k_seed1/2026_04_21__19_17_33_OeSi3e/config.json` — 完整超参记录
- `checkpoints/ewc/20260421_191733_ContinualLearningEnv/` — 最终模型
- `checkpoints/ewc/20260421_191733_ContinualLearningEnv_task{0,1,2,3}/` — 每任务结束时快照
- `results/ewc_co4_20k_seed1.png` — 4 面板训练曲线图
