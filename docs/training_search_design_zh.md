# 训练寻优设计文档

本文档基于 `python/zrt/training/search/training_search_util.py` 及相关 search / estimator / report / metric_filter 代码，总结训练并行策略寻优的功能域规格、实现流程、策略设计和接口设计。

该寻优链路面向大模型训练配置搜索，核心输入是参数网格 `param_grid`，核心输出是排序后的结果表、最佳配置分析 Excel 和单配置详细 Excel。它覆盖 TP、CP、PP、EP、DP、ZeRO、重计算、PP schedule、optimizer、量化预设、硬件与序列长度等维度。

总体架构图：

![diagram 01 - training search architecture](assets/training_search_design/diagram_01_search_architecture.png)

## 3. 训练寻优功能域规格设计

### 3.1 功能定位

训练寻优功能域负责把用户给定的搜索空间转换成可评估的训练配置集合，并对每个配置调用训练性能估算器，最后输出可排序、可过滤、可对比、可导出的结果。

它在系统中的位置如下：

- 上游：模型 YAML、硬件 YAML、用户 `param_grid`、可选 Excel 模板。
- 核心：`TrainingConfigManager` 生成合法配置，`run_training_search_parallel()` 调度并行评估。
- 中游：`run_training_task_wrapper()` 构造 `ModelSpec`、`SystemSpec`、`Strategy`，调用 `estimate()`。
- 下游：`format_results()` 扁平化 `TrainingReport`，再导出 CSV、最佳分析 Excel 和详细估算 Excel。

### 3.2 输入规格

主要输入是 `param_grid`。每个字段可以是单值或列表，内部会统一转成列表。

| 输入字段 | 示例 | 说明 |
| --- | --- | --- |
| `model` | `deepseek_v4_pro` | 对应 `python/zrt/training/configs/models/*.yaml` |
| `hw` | `nvidia_b300`、`ascend_910c` | 通过硬件 registry 加载硬件规格 |
| `world_size` | `8192` | 当前寻优路径只支持单个 world_size |
| `tp/cp/pp/ep/dp` | `[1,2,4]` 或 `"auto"` | 并行维度搜索空间 |
| `micro_batch` | `1` | 单 micro-batch 大小 |
| `global_batch` | `1024` | 可由 `total_token / seq_len` 精确推导 |
| `seq_len` | `262144` | 覆盖模型默认 seq_len |
| `total_token` | `536870912` | exact-token 模式，要求能被 seq_len 整除 |
| `zero_stage` | `0/1/2/3` | ZeRO 级别 |
| `pp_schedule` | `1f1b`、`dualpipev` | PP 调度策略 |
| `vpp_chunks` | `1/2/4` | 仅 interleaved / dualpipev 保留 |
| `recompute` | `none/mhc/full/selective` | 重计算策略 |
| `optimizer` | `adam/muon` | 优化器类型 |
| `quant_preset` | `deepseek_v4_fp8_fp4` | 模型量化预设 |
| `filters` | `[{metric, op, value}]` | 额外指标过滤 |
| `sort_by` | `tokens_per_sec` | 排序指标 |

配置生成流程：

![diagram 02 - config generation](assets/training_search_design/diagram_02_config_generation.png)

### 3.3 输出规格

核心输出是一个 `pandas.DataFrame`，每行代表一个可行训练配置及其性能指标。

主要输出文件：

| 输出 | 生成函数 | 说明 |
| --- | --- | --- |
| `results_summary.csv` | `save_results()` | 过滤和排序后的全量结果摘要 |
| `best_config_analysis.xlsx` | `export_best_analysis_excel()` | 按硬件组和 seq_len 选择最佳吞吐配置，并生成对比分析 |
| `<model>_<hw>_seq*_ws*_best.xlsx` | `export_best_configs_excel()` | 每个 model/hw/seq/world_size 的 step time 最优详细报告 |

主要指标：

- 端到端：`step_time_ms`、`pipeline_time_ms`、`tokens_per_sec`。
- 利用率：`mfu`、`mfu_native`、`hfu`。
- 计算：`compute_time_ms`、`fwd_compute_ms`、`bwd_compute_ms`。
- 通信：`tp/cp/ep/pp/dp_total_ms` 与 `*_exposed_ms`。
- Pipeline：`bubble_fraction`、`bubble_time_ms`。
- 内存：`weights_gb`、`grads_gb`、`opt_state_gb`、`activations_gb`、`comm_buffers_gb`、`memory_gb`。
- 拓扑：`tp/cp/ep/pp/dp_comm_domain`。

### 3.4 搜索约束规格

寻优不是简单笛卡尔积枚举，而是在生成阶段就做结构化剪枝。

合法性剪枝规则：

![diagram 03 - pruning rules](assets/training_search_design/diagram_03_pruning_rules.png)

关键约束：

- `world_size = tp * cp * pp * dp`，当前代码中 EP 不占额外 rank。
- TP 要求 `num_heads`、`hidden`、`ffn` 等维度可整除。
- Ulysses CP 要求 `num_heads % cp == 0`。
- PP 不能超过模型 layer 数。
- `global_batch` 必须能被 `micro_batch * dp` 整除。
- ZeRO stage 大于等于 1 时要求 `dp > 1`。
- EP 要求 `dp >= ep` 且 `dp % ep == 0`，MoE 模型还要求 `num_experts % ep == 0`，并且 `ep <= dp * pp * cp`。
- pod packing 会基于硬件拓扑和 `pod_packing_axes` 判断当前并行组合是否能合理落在物理资源上。

## 4. 训练寻优功能实现设计

## 4.1 训练寻优功能实现

### 4.1.1 功能概述

训练寻优实现可以拆成五个阶段。

第一阶段是搜索空间展开：`TrainingConfigManager` 将 `param_grid` 规范化，处理 `auto`、`total_token`、world size 和硬件拓扑。

第二阶段是合法配置枚举：通过 `_enumerate_valid_parallel_configs()` 对 TP/CP/PP/EP/DP 做整除性、batch、ZeRO、EP、拓扑剪枝。

第三阶段是并行评估：`run_training_search_parallel()` 使用 `ProcessPoolExecutor` 按 batch 投递配置，worker 内部缓存 model/system/graph。

第四阶段是单点估算：`run_training_task_wrapper()` 构造 `ModelSpec`、`SystemSpec`、`Strategy`，先做 validation 和内存预过滤，再调用 `estimate()` 得到 `TrainingReport`。

第五阶段是结果处理：主进程聚合 success/skipped/error，做指标过滤、HBM 过滤、排序，并导出 CSV 和 Excel。

### 4.1.2 现有代码位置

| 文件 | 说明 |
| --- | --- |
| `python/zrt/training/search/training_search_util.py` | 并行寻优主实现、配置生成、导出 |
| `python/zrt/training/search/estimator.py` | 单配置训练性能估算入口 |
| `python/zrt/training/search/metric_filters.py` | 指标过滤和排序公共逻辑 |
| `python/zrt/training/search/report.py` | `TrainingReport` 转 dict/json/summary |
| `python/zrt/training/search/space.py` | 较早的 `SearchSpace` 策略枚举模型 |
| `python/zrt/training/io/config_loader.py` | YAML 到 `ModelSpec/SystemSpec/Strategy` 的解析 |
| `python/zrt/training/models/memory.py` | 内存估算与搜索内存过滤 |
| `python/zrt/training/compose/schedules.py` | pipeline step time 和 schedule composer |
| `python/zrt/training/ir/opgraph_builder.py` | 显式训练图构建 |
| `python/zrt/training/io/excel_exporter.py` | 单配置详细 Excel 导出 |

### 4.1.3 实现思路

#### 4.1.3.1 整体流程

整体流程：

```text
1. 用户传入 param_grid。
2. TrainingConfigManager 计算输出路径和总配置数。
3. generate_static_configs_stream() 流式生成合法配置。
4. 主进程按 batch_size 将配置提交给 ProcessPoolExecutor。
5. worker 初始化并缓存基础 ModelSpec。
6. 单配置评估时复用 model/system/graph cache。
7. validation + memory_breakdown 预过滤不可行配置。
8. build_explicit_graph + estimate() 得到 TrainingReport。
9. 主进程收集 success/skipped/error。
10. 对成功结果做 metric filters、HBM 0.8x 过滤、MFU threshold 过滤。
11. format_results() 生成 DataFrame。
12. save_results() 输出 summary CSV。
13. select_best_configs_by_tokens() 按硬件组与 seq_len 选最佳。
14. export_best_analysis_excel() / export_best_configs_excel() 输出报告。
```

并行执行时序：

![diagram 04 - parallel execution](assets/training_search_design/diagram_04_parallel_execution.png)

单配置评估流程：

![diagram 05 - single task evaluation](assets/training_search_design/diagram_05_single_task.png)

#### 4.1.3.2 策略模式

当前代码没有单独的 `SearchPolicy` 抽象类，但实际包含多类可配置策略点。

| 策略点 | 控制参数 | 当前行为 |
| --- | --- | --- |
| 搜索空间策略 | `param_grid` | 用户指定显式列表或 `"auto"` |
| Auto 展开策略 | `_expand_auto_values_optimized()` | 基于 world_size、固定维度最小乘积、模型专家数收紧候选 |
| 合法性剪枝策略 | `_enumerate_valid_parallel_configs()` | TP/CP/PP/DP/EP/ZeRO/batch/topology 约束 |
| 内存过滤策略 | `memory_limit_ratio` / `max_memory_gb` | 默认 `0.8 * HBM`，先 worker 预过滤，后主进程复核 |
| 指标过滤策略 | `filters` | `report_passes_filters()`，未知 metric/op fail closed |
| 排序策略 | `sort_by` / `sort_ascending` | 默认按 `tokens_per_sec` 降序 |
| 最佳配置选择策略 | `comparison_hw_groups` / `seq_lens` | 每个对比组、硬件、seq_len 选 tokens/s 最高 |
| 导出策略 | `export_best_excel` / `export_analysis_excel` | 控制详细 best Excel 和对比分析 Excel |

这套设计的优点是调用入口稳定，但策略可通过参数改变；缺点是策略逻辑目前分散在多个函数里，后续如果搜索规则继续扩展，可以考虑抽象成 `SearchPolicy`、`PruningPolicy`、`RankingPolicy`、`ExportPolicy`。

#### 4.1.3.3 对接训练估算器与报告导出链路

寻优本身不重新实现性能模型，而是对接 `estimate()`。

`estimate()` 有两条路径：

- 传入 `graph` 时走 `_estimate_legacy()`，即当前搜索主路径：`build_explicit_graph()` 后调用 `pipeline_step_time()` 和手工成本模型。
- 未传入 `graph` 或使用 capture 时走 `estimate_via_pipeline()`，即抓图建模 + Transform Pipeline 路径。

估算器对接路径：

![diagram 06 - estimator paths](assets/training_search_design/diagram_06_estimator_paths.png)

结果过滤、排序和导出路径：

![diagram 07 - filter rank export](assets/training_search_design/diagram_07_filter_rank_export.png)

Excel 导出路径：

![diagram 08 - excel exports](assets/training_search_design/diagram_08_excel_exports.png)

### 4.1.4 实现设计

#### 4.1.4.1 配置规范化与输出目录

`TrainingConfigManager.__post_init__()` 根据 `model` 和最大 `world_size` 生成输出目录：

```text
output/training_search/<model>_ws_<max_world_size>
```

`run_training_search_parallel()` 启动前会清理该目录下旧文件，保证本次输出不会混入历史结果。

#### 4.1.4.2 Auto 搜索空间展开

`_expand_auto_values_optimized()` 是搜索空间收缩的关键优化。

对于 TP/CP/PP/DP：

```text
max_allowed_val = world_size // min_explicit_prod
candidate = divisors(world_size) 中 <= max_allowed_val 的值
```

对于 EP：

- 如果模型有 `num_experts`，候选来自 `num_experts` 的约数。
- 如果没有模型信息，则退化使用 world size 约数。
- EP 不参与 `world_size = tp * cp * pp * dp` 的 rank 乘积计算。

这种设计避免把 `"auto"` 直接扩展成全量 world size 约数，显著减少无效组合。

#### 4.1.4.3 合法配置枚举

`generate_static_configs_stream()` 以 generator 方式流式产出配置，避免一次性构建超大列表。

枚举顺序是：

```text
other dimensions product
  -> base_config
  -> apply total_token/global_batch rule
  -> enumerate TP/CP/PP/EP/DP
  -> yield full config
```

其中 `other_keys` 会排除并行维度和 `world_size`。如果启用 `total_token`，则 `global_batch` 会从 other_keys 移除，避免同时枚举旧值。

#### 4.1.4.4 Worker 缓存设计

worker 进程内有三类缓存：

| 缓存 | key | 作用 |
| --- | --- | --- |
| `_WORKER_MODEL_CACHE` | `(model_name, quant_preset)` | 避免重复加载模型 YAML |
| `_WORKER_HW_CACHE` | `(hw_name, world_size)` | 避免重复构造 `SystemSpec` |
| `_WORKER_GRAPH_CACHE` | graph key | 避免重复构建显式训练图 |

graph key 包含：

```text
model, quant_preset, seq_len, micro_batch, tp, cp, ep, cp_kind, cp_ulysses, cp_ring
```

它不包含 DP、global batch、optimizer、ZeRO，因为这些主要被后续估算和内存公式消费，不决定训练 IR 图结构。

#### 4.1.4.5 内存过滤设计

内存过滤有两层。

第一层在 worker 内：

```text
memory_breakdown(None, model, system, strategy)
memory_gb <= max_memory_gb 或 memory_limit_ratio * HBM
```

如果不满足，直接返回 `status="skipped"`，避免继续构图和估算。

第二层在主进程：

```text
rep.memory.total / 1e9 <= 0.8 * GPU HBM
```

同时还会应用用户传入的 `filters`。这保证最终 CSV 和 Excel 不会选择 OOM 风险配置。

#### 4.1.4.6 结果表构建

`format_results()` 将 `TrainingReport` 和原始 config 合并为一行。

它补充：

- 通信域列：`ep_comm_domain`、`pp_comm_domain`、`dp_comm_domain`、`tp_comm_domain`、`cp_comm_domain`。
- 计算/通信/优化器/重计算/pipeline 指标。
- MFU/HFU/tokens/s。
- per-rank memory breakdown。

最终按 `sort_by` 排序，默认 `tokens_per_sec` 降序。

#### 4.1.4.7 最佳配置选择与对比组

`select_best_configs_by_tokens()` 不是全局只选一个 best，而是按：

```text
comparison_hw_groups -> seq_len -> hw
```

逐组选择对应硬件和序列长度下 `tokens_per_sec` 最高的配置。

这样同一硬件可以出现在多个对比组中，例如：

```python
[
    ["nvidia_b300", "nvidia_gb300_nvl576"],
    ["nvidia_b300", "ascend_910c"],
]
```

`nvidia_b300` 会作为两个组各自的 baseline 出现，便于生成归一化对比。

#### 4.1.4.8 Excel 导出设计

`export_best_analysis_excel()` 生成适合横向对比的 Excel：

- `raw_data`：完整配置和指标。
- `analysis`：硬件+seq、吞吐归一化、计算/通信/空泡/优化器占比。
- 支持模板文件。
- 支持组号合并单元格和颜色样式。

`export_best_configs_excel()` 生成单配置详细报告：

- 按 `model/hw/seq_len/world_size` 分组。
- 每组选择 `step_time_ms` 最小的配置。
- 重新构建 graph 和 op_costs。
- 调用 `export_estimate_excel()` 输出详细算子/阶段/硬件/策略报告。

### 4.1.5 接口设计

接口分层图：

![diagram 09 - interface layers](assets/training_search_design/diagram_09_interface_layers.png)

#### 4.1.5.1 主入口接口

```python
def run_training_search_parallel(
    param_grid: Dict[str, List[Any]],
    workers: int = 8,
    mfu_threshold: float = 0.0,
    batch_size: int = 32,
    export_best_excel: bool = True,
    export_analysis_excel: bool = True,
    analysis_excel_template: Optional[str] = None,
    analysis_excel_name: Optional[str] = None,
    comparison_hw_groups: Optional[List[List[str]]] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    sort_by: str = "tokens_per_sec",
    sort_ascending: bool = False,
) -> pd.DataFrame:
    ...
```

该接口负责端到端寻优，适合 CLI、脚本或上层服务调用。

#### 4.1.5.2 配置管理接口

```python
class TrainingConfigManager:
    def count_total_configs(self) -> int: ...
    def generate_static_configs_stream(self) -> Generator[Dict[str, Any], None, None]: ...
```

职责：

- 规范化 `param_grid`。
- 扩展 auto。
- 生成合法训练配置。
- 估算总配置数用于进度条。

#### 4.1.5.3 单配置评估接口

```python
def run_training_task_wrapper(config: Dict) -> Optional[Dict]:
    ...
```

返回结果有三种：

```text
success: {status, config, report, model_name, hw_name}
skipped: {status, config, type="memory", memory_gb, memory_limit_gb}
error:   {status, config, type, message?}
```

该接口是 worker 进程内的最小执行单元。

#### 4.1.5.4 过滤与排序接口

```python
def report_passes_filters(report, filters) -> bool: ...
def format_results(reports, configs, sort_by, ascending) -> pd.DataFrame: ...
def select_best_configs_by_tokens(df, comparison_hw_groups, seq_lens) -> pd.DataFrame: ...
```

支持的过滤 metric：

- `step_time_ms`
- `tokens_per_sec`
- `mfu`
- `hfu`
- `bubble_fraction`
- `memory_gb`

过滤器采用 fail-closed 策略：未知 metric、未知 op、缺失值、非法阈值都会排除配置。

#### 4.1.5.5 导出接口

```python
def save_results(df: pd.DataFrame, output_path: str): ...
def export_best_analysis_excel(best_df, output_path, comparison_hw_groups, ...): ...
def export_best_configs_excel(all_results, output_path): ...
```

导出接口分两类：

- 面向搜索汇总：CSV + best analysis Excel。
- 面向单配置深挖：best detailed Excel。

#### 4.1.5.6 验收标准

功能验收：

- 能从 `param_grid` 流式生成合法配置。
- `auto` 能正确展开并避免明显无效组合。
- 非法并行组合、OOM 风险配置、validation error 不会进入最终结果。
- 并行 worker 能正确复用 model/system/graph cache。
- 输出结果可按 `tokens_per_sec`、`step_time_ms`、`mfu` 等指标排序。
- best analysis Excel 能按硬件组和 seq_len 生成对比视图。

质量验收：

- 搜索结果中不会包含超过默认 `0.8 * HBM` 的配置。
- `total_token` 模式下 `global_batch` 精确等于 `total_token // seq_len`。
- 同一硬件可以在多个 comparison group 中重复出现，归一化基线互不干扰。
- `git diff --check` 无空白问题。
- 文档中的架构图/流程图以 PNG 图片直接展示，不依赖 Markdown Mermaid 渲染。

## 5. 相关代码索引

| 文件 | 说明 |
| --- | --- |
| `python/zrt/training/search/training_search_util.py` | 训练寻优主链路 |
| `python/zrt/training/search/estimator.py` | 单配置性能估算入口 |
| `python/zrt/training/search/metric_filters.py` | 指标过滤和排序 |
| `python/zrt/training/search/report.py` | 报告格式化 |
| `python/zrt/training/search/space.py` | 搜索空间 dataclass |
| `python/zrt/training/models/memory.py` | 内存估算 |
| `python/zrt/training/compose/schedules.py` | pipeline schedule 估算 |
| `python/zrt/training/ir/opgraph_builder.py` | 显式训练图构建 |
| `python/zrt/training/io/excel_exporter.py` | 单配置详细 Excel 导出 |
