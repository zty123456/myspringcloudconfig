# 《Squeezing Operator Performance Potential for the Ascend Architecture》逐章节讲解

> 论文：Yuhang Zhou et al., **Squeezing Operator Performance Potential for the Ascend Architecture**, ASPLOS 2025。  
> 原文 PDF：<https://grace-liu.github.io/static/papers/25-ASPLOS-ascend.pdf>  
> 本文按论文原始章节顺序整理，适合直接作为组内分享讲稿。

## Abstract：摘要在讲什么

摘要先点出问题：Ascend 这类 DSA 架构因为有专用计算单元、复杂片上存储和显式数据搬运，所以传统 GPU/CPU 性能分析方法很难准确定位算子瓶颈。

论文提出的方法是：

- 把 Ascend AICore 拆成多个 component；
- 对每个 component 建立 component-based roofline；
- 判断 bottleneck 是 component bound 还是 underutilization；
- 对 underutilization 继续区分并行度不足、计算效率低、搬运效率低；
- 用真实算子和真实模型验证这套方法能指导优化。

摘要中最重要的一句话可以概括成：

```text
要想优化 Ascend 算子，必须知道到底是哪个硬件 component 慢，以及为什么慢。
```

## 1. Introduction：为什么需要这篇论文

第 1 章介绍背景和动机。

### 1.1 DSA 加速器越来越重要

论文从深度学习发展讲起：越来越多厂商开发专用 AI 加速器，例如 TPU 和 Ascend。Ascend 已经支持很多真实模型和大模型任务，因此算子性能分析和优化非常重要。

### 1.2 Ascend 与 GPU 的关键不同

论文强调 Ascend 的几个特点：

- **Dedicated compute units**：Scalar、Vector、Cube 分别负责不同计算任务。
- **Customized on-chip memory**：L0A、L0B、L0C、L1、UB、GM 等存储层级更细。
- **Explicit transfer and pipeline**：数据搬运由 MTE 和指令显式控制。
- **Manual scheduling matters**：开发者/编译器需要安排指令顺序、同步和 buffer 使用。

这些特征使 Ascend 优化空间更大，但也让瓶颈更难判断。

### 1.3 传统分析方法太粗

传统 Roofline 可以判断 compute-bound 或 memory-bound，但很难回答：

- 是 Cube 慢，还是 Vector 慢？
- 是 MTE-GM 慢，还是 MTE-UB 慢？
- 是带宽达到上限，还是 component 没有并行起来？
- 是忙时效率低，还是活跃时间太短？

因此论文提出 component-based roofline。

### 1.4 论文贡献

第 1 章列出的贡献主要有：

1. 提出 Ascend component 抽象。
2. 提出 component-based roofline。
3. 提出 underutilization 分析方法。
4. 用 Add_ReLU、Depthwise、AvgPool 等案例说明优化过程。
5. 在 11 个真实模型上验证方法有效。

## 2. Background and Motivation：背景与动机

第 2 章解释 Ascend 架构、为什么算子优化困难，以及传统 Roofline 为什么会分析错。

### 2.1 Ascend Architecture

论文首先给出 AICore 的结构图。

![Figure 1. The architecture of AICore in Ascend.](assets/ascend_paper_figures/figures/figure_01_aicore_architecture.png)

图中可以看到三类计算单元：

- Cube
- Vector
- Scalar

以及多级存储和搬运路径：

- GM
- L1
- L0A / L0B / L0C
- UB
- MTE-GM / MTE-L1 / MTE-UB

论文强调两个设计特点。

第一，Ascend 有专门面向 AI 工作负载的计算单元：

- Cube 适合矩阵密集计算；
- Vector 适合向量类计算；
- Scalar 适合控制与标量逻辑。

第二，Ascend 的数据搬运更灵活：

- 支持跨层搬运，例如 GM 可以直接到 L0A/B；
- 不同 transfer path 的带宽可能不同；
- 开发者需要根据算子特点选择路径。

### 2.2 Operator Optimization on Ascend Is Non-trivial

这一节解释：Ascend 提供了灵活性，但优化很难。

论文举卷积类算子说明，性能差可能由多种原因造成：

- 计算单元配置不合适；
- 数据应该进入 L0A，却被放入较慢路径；
- L0A 和 L0B 分配不合理；
- component 之间没有形成流水；
- 同步指令过多；
- buffer 读写冲突。

核心观点是：**Ascend 算子性能不是单一因素决定，而是 compute、memory transfer、instruction schedule 共同决定。**

### 2.3 Limitations of Performance Analysis

论文接着回顾传统 Roofline。

![Figure 2. Existing roofline models.](assets/ascend_paper_figures/figures/figure_02_existing_roofline_models.png)

Figure 2a 是传统 DRAM Roofline：

- 横轴是 arithmetic intensity；
- 纵轴是 performance；
- 斜线代表 memory bandwidth ceiling；
- 横线代表 compute ceiling。

Figure 2b 是 hierarchical roofline：

- 考虑多层 memory；
- 考虑不同 precision / compute unit；
- 但仍不适合 Ascend 的 component 和 MTE 并行特征。

论文进一步给出 naive roofline 在 Ascend 上的错误分析案例。

![Figure 3. Incorrect analysis cases of the naive roofline.](assets/ascend_paper_figures/figures/figure_03_naive_roofline_wrong_cases.png)

Figure 3a 说明：如果把 `GM->L0A` 和 `GM->L0B` 分别看成独立 transfer，就会误判带宽利用率。但它们其实共享 MTE-GM 队列，需要从 MTE-GM 这个 component 角度看。

Figure 3b 说明：如果一个 Cube 中混合 FP16 和 INT8，简单按最大峰值或平均峰值计算，都会错误估计利用率。需要根据算子真实 precision 组合计算 operator-aware ideal performance。

第 2 章的结论是：

```text
传统 Roofline 对 Ascend 太粗；naive 扩展又会因为忽略 component 队列和混合 precision 而误判。
```

## 3. Overview：论文方法总览

第 3 章提出两个关键观察，并展示分析系统工作流。

### 3.1 Key Observation：Component 抽象

论文用矩阵乘法 `A × B` 解释 component 执行关系。

![Figure 4. The execution of matrix multiplication A × B.](assets/ascend_paper_figures/figures/figure_04_matmul_component_execution.png)

图中有三个步骤：

1. Tensor A 从 GM 搬到 L1；
2. Tensor A 从 L1 搬到 L0A，同时 Tensor B 从 GM 搬到 L0B；
3. Cube 执行矩阵乘法。

这个例子得出论文最核心的执行规律：

```text
同一个 component 内部顺序执行；
不同 component 之间可以并行执行。
```

论文把 component 定义为具有物理 instruction queue 的硬件单元，例如：

- Cube
- Vector
- Scalar
- MTE-GM
- MTE-L1
- MTE-UB

### 3.2 Workflow：分析系统流程

![Figure 5. The workflow of analysis system.](assets/ascend_paper_figures/figures/figure_05_analysis_workflow.png)

论文系统流程包括四步：

1. **Profiling**
   - 收集每个 component 的 operations、bytes、execution time 和算子总时间。

2. **Modeling**
   - 建立 component-based roofline。
   - 判断 component bound 或 underutilization。

3. **Underutilization Analysis**
   - 将 utilization 拆成 time ratio 和 efficiency。
   - 判断是并行度不足还是 component 自身低效。

4. **Optimization**
   - 根据瓶颈类型选择优化策略。

第 3 章的价值是把论文方法串起来：从 profiling 数据进入模型，再从模型导出优化动作。

## 4. Bottleneck Analysis：瓶颈分析

第 4 章是论文理论核心，提出公式和诊断规则。

### 4.1 Component-Based Roofline Model

论文以 Cube 为例，先定义实际性能：

```text
A_cube = O_cube / T_total
```

其中：

- `O_cube` 是 Cube 执行的总操作数(总计算量)；
- `T_total` 是整个 operator 的执行时间；
- `A_cube` 是从整个 operator 视角看到的 Cube 实际性能。

但是Cube 的理想性能不能简单用硬件最大峰值。

因为 Cube 内可能混合了不同精度指令，例如 INT8 峰值可能是 FP16 的 2 倍。如果直接用最大峰值，会不公平；如果用所有精度的简单平均，也不符合算子真实指令占比

因此提出 operator-aware ideal performance(算子感知理论峰值性能)：

```text
I_cube = O_cube / T_ideal
```

其中：

```text
T_ideal = Σ_prec O_prec / P_prec
```

也就是每种 precision 的理想执行时间相加。

例如Cube 里有两类计算：
```text
O_fp16 = 100 TFLOPs
P_fp16 = 200 TFLOP/s

O_int8 = 80 TOPs
P_int8 = 400 TOP/s

T_ideal_fp16 = 100 / 200 = 0.5 s
T_ideal_int8 = 80 / 400 = 0.2 s
T_ideal = 0.5 + 0.2 = 0.7 s
```
如果 Cube 对每种 precision 都跑在对应峰值性能上，
完成当前算子里这些混合 precision 操作，最少需要 0.7 秒。

所以这个公式最后为：

```text
I_cube = Σ_prec O_prec / Σ_prec(O_prec / P_prec)
```

这是一个按操作数(计算量)加权的调和平均。它比最大峰值和简单平均更符合当前算子真实负载(workload)。

最后定义 Cube utilization：
A_cube：component实际性能， I_cube：compo理论峰值性能， U_cube：component利用率

```text
U_cube = A_cube / I_cube
```

其他 component 也类似：

- Compute component 用 operations / peak performance；
- MTE component 用 bytes / path bandwidth。

### 4.2 Underutilization Analysis

如果某个 component 的 utilization 超过阈值，它就是 bound。否则，如果所有 component utilization 都低，就说明硬件 underutilized。

需要进一步分析为什么component没有bound，但是算子还是慢。

提出以下公式：

```text
U_cube = A_cube / I_cube
A_cube = O_cube / T_total
两者推导出
U_cube = O_cube / (T_total × I_cube)
分子分母都乘以T_cube就得到
U_cube =
O_cube / (T_cube × I_cube)
×
T_cube / T_total
```

定义：

```text
E_cube = O_cube / (T_cube × I_cube)
R_cube = T_cube / T_total
```

所以：

```text
U_cube = E_cube × R_cube
```

含义：

- `E`：component 忙时效率；
- `R`：component 活跃时间占比；
- `U`：整体利用率。

论文用这个分解区分两类 underutilization：

1. 如果所有 component 的 `R` 都低：

说明没有任何 component 在算子期间占据主要执行时间， 也就是说，各个 component 没有形成充分 overlap / pipeline。判断为
Insufficient Parallelism。理论上如果算子流水做得好，总会有一个或多个 component 长时间保持活跃；
如果所有 component 的活跃占比都低，说明 pipeline 没搭起来。 说明 component 之间没有充分并行。

2. 如果存在 component 的 `R` 高但 `U` 仍低：

说明`E`低
根据公式U = E x R，假如R = 0.9， U却只有0.3， 那说明E = U / R = 0.33

这种情况就要细分是Compute component 低效还是MTE component 低效

例如MTE 忙很久，但效率低

```text
R_MTE-UB = 0.9
U_MTE-UB = 0.35
那E_MTE-UB = 0.39， 即内存带宽效率只有39%
```
这可能是：

每次 transfer 太小；
搬运粒度太碎；
有很多零散写回；
数据重复搬运；
transfer path 不合适；
MTE 队列里一直有活，但每个活都跑不满带宽。

如果是Compute component 低效
例如Vector 忙很久，但效率低
```text
R_Vector = 0.85
U_Vector = 0.25
那E_Vector = 0.3， 即Vector计算效率只有30%
```
可能是：

repeat 参数太小；
mask 设置不合理；
每次 vector 指令处理的数据太少；
tile 太碎；
需要很多额外 loop；
指令调度开销占比太高。

### 4.3 Pruning, Visualization, and Analysis

如果把所有 compute component 和 memory component 直接组合，点太多，不适合可视化。论文进行剪枝：

- 删除对瓶颈分析无帮助的 memory component；
- 删除不可能的 compute-memory 组合。

剪枝后最多只需考虑 7 个 performance points。 一个performance point代表一种 compute component 和 MTE component 的组合，
例如Cube + MTE-GM、Vector + MTE-UB

所以，如果某个 performance point 靠近 Cube 水平线，说明 Cube 可能接近满载。

如果某个 performance point 靠近 Vector 水平线，说明 Vector 可能成为瓶颈。

performance point 靠近某条 MTE 斜线，说明对应 MTE component 可能成为瓶颈

![img.png](img.png)

Figure 6 是 component-based roofline 的示意。

第 4 章结论：

```text
先判断 component 是否 bound；
如果不 bound，再用 U = E × R 判断是并行度不足还是 component 低效。
```

## 5. Optimization Experience：优化经验

第 5 章用三个算子案例说明如何根据第 4 章的分析结果选择优化。

### 5.1 Add_ReLU

Add_ReLU 的计算形式是：

```text
Add_ReLU(x) = ReLU(x + c)
```

它涉及：

- MTE-GM：把输入从 GM 搬到 UB；
- Vector：执行 Add 和 ReLU；
- MTE-UB：把结果从 UB 写回 GM。
![img_1.png](img_1.png)

论文对 Add_ReLU 的优化过程：

1. 初始分析：最高 utilization 只有 38.42%，判断为 insufficient parallelism。
![img_2.png](img_2.png)
2. RSD：Reducing Spatial Dependency，减少 MTE-GM 和 MTE-UB 访问同一 buffer 地址导致的依赖。
![img_3.png](img_3.png)
![img_4.png](img_4.png)
3. MRT：Minimizing Redundant Transfer，把循环内重复搬运的常量移到循环外。
![img_5.png](img_5.png)
![img_6.png](img_6.png)

经过两次优化， Add_ReLU算子的效率从38.24%提高到了70.52%，执行时间从98.673us降低到了57.157us，算子
加速了1.72x

### 5.2 Depthwise

Depthwise 涉及输入从 GM 到 L1，再从 L1 到 L0A/B，最后 Cube 执行乘加。

![Figure 11-12. Code of AIS optimization and adjusting instruction sequence.](assets/ascend_paper_figures/figures/figure_11_12_depthwise_ais.png)

Depthwise深度卷积算子，公式为Y_i,j = sum(X_window ⊙ W)，也就是输入 feature map 的一个窗口
和 depthwise filter 做逐元素乘加得到输出，在Ascend上可以拆分为：
```text
1. MTE-GM：输入数据从 GM 搬到 L1
2. MTE-L1：数据从 L1 搬到 L0A / L0B
3. Cube：执行乘加计算
```
所以它关心的点为：
GM -> L1、 L1 -> L0A/B、 Cube MAD


论文对 Depthwise 做了多轮优化：

1. 第一次分析发现，MTE_utilization = 35.45%，说明没有component 接近自己的 ceiling， 
进一步看 component time ratio，最耗时的是MTE-GM，占总时间 46.66%，根据4.2节分析U 低，R 也低
=> component 没有充分并行 => Insufficient Parallelism， 所以第一轮优化目标不是提高某个 component 的忙时效率，而是
让 MTE-GM、MTE-L1、Cube 更好地重叠执行

**AIS：Adjusting Instruction Sequence**
   - 调整指令顺序，让 MTE-GM 更早发起下一轮搬运。

2. 发现代码中有很多pipe_barrier(PIPE_ALL)，这类同步会强制所有 component 等待。也就是说，即使：
MTE-GM 可以继续搬下一块数据，MTE-L1 可以继续搬到 L0 ，Cube 可以继续计算，一旦遇到 PIPE_ALL，大家都必须停下来同步。
这会让原本可以并行的指令变成顺序执行。

**RUS：Removing Unnecessary Synchronization**
   - 删除过多 `pipe_barrier(PIPE_ALL)`，避免所有 component 被迫同步。

3.以上优化做完后发现 Depthwise 没有使用 ping-pong。这样会导致一个 buffer 被占用时，读写不能并行
以L1为例，GM -> L1 ，L1 -> L0A，如果 L1 只有一块区域，当前 L1 正在被 L1 -> L0A 读取时，下一轮 GM -> L1 写入就必须等待。
结果就是MTE-GM 等 MTE-L1，MTE-L1 等 Cube，Cube 等数据
**PP：Ping-pong Policy**
   - 把 buffer 分成两块，一块计算，一块搬运，减少读写冲突。

4. 
经过 AIS、RUS、PP 三个并行度优化后，论文重新分析，发现MTE_utilization 从 35.45% 提升到 71.56%
这说明前面针对 insufficient parallelism 的优化有效。但是还有问题：MTE-GM 的 component_time_ratio = 94.18%。
这表示：MTE-GM 几乎一直在忙，如果一个 component 忙了这么久，但整体仍有 underutilization，就符合第 4.2 的逻辑：
R 高，但 U 还不够理想 => E 低 => Inefficient Component，由于这个 component 是 MTE-GM，所以诊断为：Inefficient MTE-GM
瓶颈从Insufficient Parallelism(并行度不足)到了Inefficient MTE(MTE低效)。

Inefficient MTE 最常见原因之一是transfer granularity 太小，也就是每次搬运的数据块太小。
如果每次 transfer 都很小，会有几个问题：调度开销占比高 、带宽打不满 、MTE 队列一直忙，但有效吞吐低
检查 MTE-UB transfer 后发现：每次 UB -> GM 只有 30 KB，这个粒度低于充分利用带宽的阈值

**ITG：Increasing Transfer Granularity**
   - 增大 transfer 粒度，提升 MTE 忙时效率。

### 5.3 AvgPool

```text
AvgPool 初始实现
  ↓
component-based roofline 分析
  ↓
utilization = 13.54%
  ↓
说明 underutilization
  ↓
进一步看 component time ratio
  ↓
Vector time ratio = 83.98%
  ↓
说明 Vector 大部分时间都在忙
  ↓
U 低但 R 高
  ↓
判断为 Inefficient Compute in Vector
  ↓
检查 Vector instruction 参数
  ↓
发现 Add 的 repeat = 1，导致 98 个 loops
  ↓
应用 AIP：repeat 从 1 调到 98
  ↓
Vector utilization 提升到 59.07%
  ↓
执行时间 69.821 μs -> 16.206 μs
```

AvgPool 的主要问题是 Vector utilization 低，论文判断为 inefficient compute。

原因是 instruction 参数不合理：

- repeat 参数太低；
- mask 设置不佳；
- 导致需要额外 loop 和搬运补偿。

论文通过 AIP：Adjusting Instruction Parameters，将 repeat 参数调大，减少额外循环，显著降低执行时间。

### 5.4 Summary

论文最后总结了不同瓶颈对应的优化方法。

![Table 1. Optimization and speedup of operators.](assets/ascend_paper_figures/figures/table_01_operator_optimization.png)

几个重要缩写：

| 缩写 | 含义 |
| --- | --- |
| OP | Operator Fusion |
| MRT | Minimizing Redundant Transfer |
| RSD | Reducing Spatial Dependency |
| AIS | Adjusting Instruction Sequence |
| RUS | Removing Unnecessary Synchronization |
| PP | Ping-pong Policy |
| ITG | Increasing Transfer Granularity |
| AIP | Adjusting Instruction Parameters |
| EA | Enhanced Algorithm |
| LC | Low-precision Calculation |
| CT | Computation Transformation |

第 5 章的关键思想是：**优化不是随机尝试，而是根据 component-based roofline 的诊断结果选择策略。**

## 6. Evaluation：实验评估

第 6 章验证论文方法在真实模型上的效果。

### 6.1 Experimental Setting

论文覆盖的模型包括：

- Vision：MobileNetV3、ResNet50、ViT、VGG16
- NLP：BERT、GPT2
- Recommendation：DeepFM、Wide & Deep、DLRM
- LLM：Llama 2、PanGu-α

硬件包括 Ascend inference chip 和 training chip。

### 6.2 End-to-End Optimization

论文展示两个端到端案例：

- PanGu-α training；
- MobileNetV3 inference。

![Figure 13. Optimization results.](assets/ascend_paper_figures/figures/figure_13_optimization_results.png)

| 缩写 | 含义 |
| --- | --- |
| CB | Compute Bound |
| MB | MTE Bound |
| IP | Insufficient Parallelism |
| IM | Inefficient MTE |
| IC | Inefficient Compute |

Figure 13 展示优化前后的瓶颈原因和执行时间。

对 PanGu-α：

- 优化前大量算子是 insufficient parallelism；
- MTE-bound 也占较大比例；
- 优化后计算时间明显降低。

对 MobileNetV3：

- 优化也降低了 compute time；
- 说明方法不仅适用于大模型训练，也适用于推理模型。

### 6.3 Insights from Comprehensive Experiments

论文进一步分析不同模型、不同框架、训练/推理下的瓶颈分布。

![Figure 14. The distribution of performance impediments.](assets/ascend_paper_figures/figures/figure_14_impediment_distribution.png)

Figure 14 中五类瓶颈：

| 缩写 | 含义 |
| --- | --- |
| CB | Compute Bound |
| MB | MTE Bound |
| IP | Insufficient Parallelism |
| IM | Inefficient MTE |
| IC | Inefficient Compute |

论文观察：

- 不同模型瓶颈分布差异很大；
- 不同框架也会改变瓶颈分布；
- 训练更容易出现 MTE-bound，因为数据搬运更多；
- 推理中 inefficient compute / MTE 更常见。

整体加速效果如下：

![Figure 15. Time speedup with optimization.](assets/ascend_paper_figures/figures/figure_15_speedup.png)

论文总结：

- 算子计算时间加速 1.08× 到 2.70×；
- 端到端整体加速 1.07× 到 2.15×；
- 已有 41 个优化算子集成进 Ascend operator library。

