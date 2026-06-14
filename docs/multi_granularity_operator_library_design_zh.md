# 多粒度算子库对接与算子预测设计文档

## 1. 背景与目标

ZRT-Sim 的算子预测链路当前以 `OpGraph` 中的 `OpNode` 为输入，通过 `SimulatorHub` 分发到不同预测后端，最终生成统一的 `SimResult`。历史实现中已经预留并接入了外部多粒度算子库能力，主要体现在 `LookupSimulator` 对 `cost_model.infer.performance_predict.PerformancePredict` 的调用。

本文档描述“多粒度算子库”在算子预测功能域中的定位、规格、实现方式和接口设计，便于后续补齐外部算子库能力、统一多后端策略，并让算子级、融合算子级、模块级预测结果具备一致的数据结构。

设计目标：

- 支持多粒度算子预测：原子算子、融合算子、模块/子结构聚合算子。
- 支持多预测后端：多粒度算子库、TileSim、Roofline 兜底。
- 统一输入输出协议：`OpNode + HardwareSpec -> SimResult`。
- 支持策略选择：按优先级、置信度、场景策略选择后端。
- 保持可降级：外部算子库不可用时自动回落到解析模型。

图表阅读建议：

- 先看“总体架构视图”和“功能域总览图”，理解多粒度算子库在 ZRT-Sim 中的位置。
- 再看“端到端预测流程图”和“策略模式结构图”，理解请求如何被分发到不同后端。
- 最后看“外部算子库对接流程图”“降级流程图”和“多粒度决策图”，理解实现细节和后续优化点。

总体架构视图：

![diagram 01 - overall architecture](assets/multi_granularity_operator_library_design/diagram_01.png)

## 2. 多粒度算子库功能域概述

### 2.1 功能定位

多粒度算子库是算子预测功能域中的高精度预测后端，主要负责对已知算子模式进行经验库/模型库查询，输出更接近真实硬件执行的 latency。

在现有代码中，它对应：

- `python/zrt/simulator/backends/lookup.py`
- `LookupSimulator`
- 外部依赖：`cost_model.infer.performance_predict.PerformancePredict`

整体调用链：

```text
OpGraph / OpNode
  -> SimulatorHub
  -> PolicyModelManager
  -> PolicyBaseModel / PriorityModel
  -> LookupSimulator / TilesimSimulator / RooflineSimulator
  -> SimResult
```

功能域总览图：

![diagram 02 - functional domain overview](assets/multi_granularity_operator_library_design/diagram_02.png)

### 2.2 功能边界

多粒度算子库负责：

- 根据算子类型、输入输出 Tensor、属性参数、硬件环境预测算子时延。
- 支持硬件版本差异，如 Ascend 910B / 910C / 910D。
- 返回预测耗时和置信度。

多粒度算子库不负责：

- 构建模型图。
- 插入通信算子。
- 执行并行策略变换。
- 负责训练全局 step time 排程。

这些能力由 graph/transform/training/simulator/report 等模块负责。

### 2.3 多粒度含义

本文中的“多粒度”包括三个层次。

| 粒度 | 输入来源 | 示例 | 预测方式 |
| --- | --- | --- | --- |
| 原子算子 | eager trace / aten op | `mm`, `bmm`, `softmax`, `copy` | 直接调用算子库或 Roofline |
| 融合算子 | fusion pipeline | `rms_norm`, `flash_attn`, `fused_mlp` | 算子库命中优先，否则拆分/解析 |
| 模块级算子 | hierarchy / report 聚合 | `self_attn`, `mlp`, `moe` | 子算子结果聚合或模块级库命中 |

当前代码已具备原子算子和融合算子元数据：

- `OpNode.op_type`
- `OpNode.op_short`
- `OpNode.fused_from`
- `OpNode.num_sub_ops`
- `OpNode.fusion_level`
- `OpNode.scope`
- `GraphHierarchy`

多粒度数据关系图：

![diagram 03 - granularity relationship](assets/multi_granularity_operator_library_design/diagram_03.png)

三种粒度不是三套割裂的数据结构，而是同一批 `OpNode` 在不同视图下的组织方式：原子粒度来自 trace，融合粒度来自 fusion pass，模块粒度来自 `scope` 和 `GraphHierarchy` 的聚合视图。

## 3. 多粒度算子库功能域规格设计

### 3.1 输入规格

多粒度算子库预测输入由以下对象构成。

#### OpNode

核心字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 节点 ID |
| `op_type` | 标准算子名，如 `aten.mm.default` / `fused.xxx` |
| `op_short` | 短算子名，如 `mm` / `bmm` |
| `inputs` | 输入 TensorMeta 列表 |
| `outputs` | 输出 TensorMeta 列表 |
| `attrs` | 算子属性 |
| `category` | `compute` / `memory` / `communication` |
| `component` | 语义组件标签 |
| `fused_from` | 融合节点的原始子算子列表 |
| `annotations` | transform/analysis 阶段写入的 FLOPs、bytes 等信息 |

#### TensorMeta

算子库调用时需要转换成外部库 tensor dict：

```python
{
    "dtype": "torch.float16",
    "format": "",
    "name": "",
    "origin_dtype": "torch.float16",
    "origin_format": "",
    "origin_shape": [M, K],
    "shape": [M, K],
    "size": M * K,
}
```

#### HardwareSpec

主要使用：

- `hw.name`
- `hw.peak_flops(dtype)`
- `hw.memory`
- 硬件名称到外部算子库环境参数的映射。

当前外部库环境配置位于 `_HW_ENV_DICT`：

```python
{
    "Ascend 910B": {
        "soc_version": "Ascend910B3",
        "cann_version": "7.0.0",
        "pta_version": "1.11.0",
        "predict_type": "task_duration"
    },
    ...
}
```

输入对象关系图：

![diagram 04 - input object relationship](assets/multi_granularity_operator_library_design/diagram_04.png)

### 3.2 输出规格

所有预测后端统一输出 `SimResult`。

| 字段 | 含义 |
| --- | --- |
| `op_node_id` | 对应 `OpNode.id` |
| `latency_us` | 总预测耗时 |
| `compute_us` | 计算耗时 |
| `memory_us` | 访存耗时 |
| `flops` | 操作数 |
| `read_bytes` | 读字节数 |
| `write_bytes` | 写字节数 |
| `arithmetic_intensity` | 算术强度 |
| `bound` | `compute` / `memory` / `latency` |
| `hw_utilization` | 硬件利用率 |
| `backend` | 产生结果的后端 |
| `confidence` | 置信度 |

多粒度算子库命中成功时：

```text
backend = "lookup"
confidence = 0.8
latency_us = result["predict_time"]
```

失败或不可用时回落：

```text
backend = "roofline" 或后续可用后端
confidence = 0.3
```

### 3.3 后端能力规格

| 后端 | 类 | 优先级 | 当前状态 | 作用 |
| --- | --- | ---: | --- | --- |
| TileSim | `TilesimSimulator` | 2 | stub | 预留 tile 级精细模拟 |
| 多粒度算子库 | `LookupSimulator` | 1 | 已对接外部库 | 查询外部预测库 |
| Roofline | `RooflineSimulator` | 0 | 可用 | 通用兜底 |

说明：当前 `TilesimSimulator.can_simulate()` 恒为 `False`，所以真实执行路径通常是 `Lookup -> Roofline`。

### 3.4 策略规格

现有策略枚举：

```python
class PolicyType(Enum):
    PRIORITY = 'priority'
    OOTB_PERFORMANCE = 'ootb_performance'
    OPERATOR_OPTIMIZATION = 'operator_optimization'
    SYSTEM_DESIGN = 'system_design'
```

当前可用策略：

- `PriorityModel`：按后端优先级选择第一个 `can_simulate=True` 的后端。

预留策略：

- `OpenBoxModel`
- `OperatorOptimizationModel`
- `SystemDesignModel`

上述三个目前是 stub，后续可扩展为针对不同分析场景的后端选择策略。

策略分发视图：

![diagram 05 - strategy dispatch](assets/multi_granularity_operator_library_design/diagram_05.png)

策略模式的关键点是：后端只回答“我能不能预测”和“我怎么预测”，策略负责回答“这个场景应该先用谁”。这样后续新增外部库、TileSim、profile database 或新的解析模型时，不需要改 `SimulatorHub` 的主入口。

## 4. 多粒度算子库功能实现设计

## 4.1 多粒度算子库功能实现

### 4.1.1 功能概述

本功能在算子预测阶段引入外部多粒度算子库，对输入 `OpNode` 做格式转换、算子类型映射、硬件环境初始化，并调用外部 `PerformancePredict` 获取预测耗时。

核心实现类：

```text
LookupSimulator
```

核心职责：

- 判断外部算子库是否可用。
- 根据硬件环境懒加载 `PerformancePredict`。
- 将 ZRT 内部 `TensorMeta` 转换为外部库输入格式。
- 将 ZRT 算子名转换为外部库算子名。
- 调用 `op_performance_predict()`。
- 将结果包装为 `SimResult`。

### 4.1.2 现有代码位置

| 模块 | 作用 |
| --- | --- |
| `python/zrt/simulator/hub.py` | 预测入口和缓存 |
| `python/zrt/policy_model/policy_model_manager.py` | 策略模型管理 |
| `python/zrt/policy_model/priority_model.py` | 按优先级选择后端 |
| `python/zrt/simulator/backends/lookup.py` | 多粒度算子库对接 |
| `python/zrt/simulator/backends/roofline.py` | 解析模型兜底 |
| `python/zrt/simulator/cache.py` | 基于内容 hash 的预测缓存 |

### 4.1.3 实现思路

#### 4.1.3.1 整体流程

```text
1. 上层传入 OpNode + HardwareSpec
2. SimulatorHub 查询 SimCache
3. 未命中则进入 PolicyModelManager
4. PriorityModel 按优先级遍历后端
5. LookupSimulator.can_simulate() 判断是否可使用外部算子库
6. LookupSimulator.simulate() 初始化 PerformancePredict
7. 转换 inputs / outputs / attrs / block_dim
8. 调用 op_performance_predict()
9. 将 predict_time 包装为 SimResult
10. 写入缓存并返回
```

时序图：

```text
Caller
  -> SimulatorHub.simulate(node, hw)
  -> SimCache.get(node, hw)
  -> PolicyModelManager.simulate(node, hw, policy)
  -> PriorityModel.predict(node, hw)
  -> LookupSimulator.can_simulate(node, hw)
  -> LookupSimulator.simulate(node, hw)
  -> PerformancePredict.op_performance_predict(op_type, params, {})
  -> SimResult
```

端到端预测流程图：

![diagram 06 - end to end prediction flow](assets/multi_granularity_operator_library_design/diagram_06.png)

端到端时序图：

![diagram 07 - end to end sequence](assets/multi_granularity_operator_library_design/diagram_07.png)

#### 4.1.3.2 策略模式

当前系统将“选择哪个预测后端”的逻辑从后端本身中解耦，使用策略模式统一管理。

抽象类：

```python
class PolicyBaseModel(ABC):
    @abstractmethod
    def predict(self, node: OpNode, hw: HardwareSpec) -> SimResult:
        pass
```

默认策略：

```python
class PriorityModel(PolicyBaseModel):
    def predict(self, node, hw):
        for backend in self._backends:
            if backend.can_simulate(node, hw):
                return backend.simulate(node, hw)
        return None
```

优点：

- 新增后端不影响调用入口。
- 可按不同业务目标切换策略。
- 支持后续按置信度、速度、硬件场景选择后端。

后续建议：

- `OOTB_PERFORMANCE`：优先使用厂商库/实测库。
- `OPERATOR_OPTIMIZATION`：优先使用可解释 component/tile 模型，便于瓶颈分析。
- `SYSTEM_DESIGN`：优先使用参数化解析模型，便于硬件假设扫描。

#### 4.1.3.3 对接多粒度算子库

当前对接逻辑在 `LookupSimulator` 中。

外部依赖导入：

```python
try:
    from cost_model.infer.performance_predict import PerformancePredict
    _COST_MODEL_AVAILABLE = True
except ImportError:
    PerformancePredict = None
    _COST_MODEL_AVAILABLE = False
```

能力判断：

```python
def can_simulate(self, node, hw):
    if not _COST_MODEL_AVAILABLE:
        return False
    if hw.name not in _HW_ENV_DICT:
        return False
    if node.category == "communication":
        return False
    return True
```

初始化：

```python
self.caller = PerformancePredict(_HW_ENV_DICT.get(hw.name, {}))
```

调用外部库：

```python
result_code, msg, result, _ = self.caller.op_performance_predict(
    cost_model_op_type,
    params,
    {}
)
```

其中：

```python
params = (inputs, outputs, node.attrs, _DEFAULT_BLOCK_DIM)
```

返回成功时：

```python
latency_us = result["predict_time"]
confidence = 0.8
```

返回失败时：

```python
latency_us = node.annotations.get("latency_us", 0)
confidence = 0.3
```

外部算子库对接流程图：

![diagram 08 - external library integration](assets/multi_granularity_operator_library_design/diagram_08.png)

该流程建议在后续实现中拆成多个小函数：`map_op_type()`、`build_params()`、`call_library()`、`parse_result()`。这样外部算子库协议变化时，只需要替换适配层，不影响策略层和缓存层。

### 4.1.4 实现设计

#### 4.1.4.1 算子名映射

内部 `op_short` 与外部算子库算子名可能不一致，需要映射。

当前特殊映射：

```python
_TO_COST_MODEL_OP_TYPE = {
    "mm": "Matmul",
    "bmm": "BatchMatMul",
    "flash_attn": "FlashAttention",
    "sdpa": "FlashAttention",
    "floor_divide": "FloorDiv",
}
```

默认映射规则：

```python
"_".join 拆词后 title 化
```

例如：

```text
rms_norm -> RmsNorm
softmax -> Softmax
```

设计建议：

- 将映射表外置为 YAML/JSON，避免硬编码。
- 支持按 backend / CANN 版本区分映射。
- 记录未命中映射的 op_type，用于扩展算子库覆盖率。

#### 4.1.4.2 Tensor 转换

内部 `TensorMeta` 转换为外部库 dict：

```python
{
    "dtype": _DTYPE_TO_TORCH_STR[tensor.dtype],
    "origin_dtype": _DTYPE_TO_TORCH_STR[tensor.dtype],
    "shape": list(tensor.shape),
    "origin_shape": list(tensor.shape),
    "size": prod(tensor.shape),
    "format": "",
    "origin_format": "",
    "name": "",
}
```

设计建议：

- 增加 format/layout 信息，如 ND / NZ / FRACTAL_NZ。
- 对动态 shape 使用符号维度或 shape range。
- 对量化 dtype 增加 scale/zero_point 元信息。

#### 4.1.4.3 硬件环境映射

外部库需要硬件环境参数：

```text
soc_version
cann_version
pta_version
predict_type
```

设计建议：

- 从 `HardwareSpec` 或配置文件中读取，而不是写死在 `_HW_ENV_DICT`。
- 支持不同 CANN 版本共存。
- 将 `predict_type` 抽象为可配置策略，如 `task_duration` / `kernel_duration`。

#### 4.1.4.4 缓存设计

`SimulatorHub` 使用 `SimCache`，cache key 来自：

```text
op_type
input shapes
input dtypes
attrs
hw.name
fused_from
```

优点：

- 同形状同硬件重复算子只预测一次。
- 与 node id 解耦。

设计建议：

- 对外部库结果缓存可增加版本字段：CANN、soc、算子库版本。
- 对失败结果设置短 TTL 或不缓存，避免临时外部库错误长期污染。

#### 4.1.4.5 异常与降级

降级路径：

```text
外部库不可导入
  -> LookupSimulator.can_simulate=False
  -> RooflineSimulator

外部库初始化失败
  -> caller=None
  -> 低置信度 SimResult

外部库返回错误
  -> latency_us 使用 annotations 中已有值或 0
  -> confidence=0.3
```

设计建议：

- 外部库调用失败时不应标记 `backend="roofline"`，除非真正调用了 Roofline；建议标记为 `lookup_failed` 或直接交给下一个后端。
- 当前 `LookupSimulator.simulate()` 内部失败后返回低置信度结果，会阻止 `PriorityModel` 继续尝试 Roofline；后续可改为抛出 `BackendUnavailable` 或返回 `None` 让策略继续降级。

推荐降级流程图：

![diagram 09 - fallback flow](assets/multi_granularity_operator_library_design/diagram_09.png)

注意：当前实现中 `LookupSimulator` 失败后可能直接返回低置信度 `SimResult`。为了让上图成立，需要策略层能够识别“后端失败”和“后端成功但置信度低”的区别，避免低质量结果提前终止后端遍历。

#### 4.1.4.6 多粒度支持设计

当前输入单位是单个 `OpNode`。多粒度扩展建议：

| 粒度 | 识别依据 | 处理方式 |
| --- | --- | --- |
| 原子算子 | `not node.is_fused` | 直接调用外部库 |
| 融合算子 | `node.is_fused` / `fused_from` | 优先查融合算子库，失败后拆子算子 |
| 模块级 | `GraphHierarchy` scope | 聚合子算子结果，或查模块模板库 |

融合算子降级策略：

```text
1. fused op 直接命中外部库
2. 命中失败则用 fused_from 拆分为子算子估计
3. 子算子缺失则 Roofline 兜底
```

模块级降级策略：

```text
1. 模块模板库命中
2. scope subtree 子节点累加
3. 仅报告聚合，不作为单独预测后端
```

多粒度决策流程图：

![diagram 10 - granularity decision flow](assets/multi_granularity_operator_library_design/diagram_10.png)

建议的聚合规则：

- `latency_us`：默认求和；如果后续引入 stream overlap，则由 scheduler 给出重叠后的 exposed latency。
- `confidence`：可取子结果的加权平均，权重建议使用 latency 或 FLOPs。
- `backend`：模块级结果可标记为 `aggregate`，并在明细中保留每个子算子的实际 backend。
- `warnings`：聚合所有子结果的降级原因，便于报告中定位覆盖率缺口。

### 4.1.5 接口设计

接口分层图：

![diagram 11 - interface layering](assets/multi_granularity_operator_library_design/diagram_11.png)

#### 4.1.5.1 后端接口

所有预测后端实现：

```python
class OpSimulator(ABC):
    name: str
    priority: int

    @abstractmethod
    def can_simulate(self, node: OpNode, hw: HardwareSpec) -> bool:
        ...

    @abstractmethod
    def simulate(self, node: OpNode, hw: HardwareSpec) -> SimResult:
        ...
```

#### 4.1.5.2 多粒度算子库后端接口

当前实现：

```python
class LookupSimulator(OpSimulator):
    name = "lookup"
    priority = 1

    def can_simulate(self, node, hw) -> bool:
        ...

    def simulate(self, node, hw) -> SimResult:
        ...
```

建议扩展接口：

```python
class LookupSimulator(OpSimulator):
    def supports_granularity(self, granularity: str) -> bool:
        ...

    def map_op_type(self, node: OpNode) -> str:
        ...

    def build_params(self, node: OpNode) -> tuple:
        ...

    def call_library(self, op_type: str, params: tuple, env: dict) -> dict:
        ...
```

#### 4.1.5.3 外部算子库调用接口

当前外部接口约定：

```python
op_performance_predict(
    op_type: str,
    params: tuple[list[dict], list[dict], dict, int],
    options: dict,
) -> tuple[int, str, dict, Any]
```

输入：

```text
op_type: 外部算子库算子名
params:
  inputs:  输入 Tensor dict 列表
  outputs: 输出 Tensor dict 列表
  attrs:   算子属性
  block_dim: 默认 64
options: 预留扩展参数
```

输出：

```text
result_code: 0 表示成功
msg: 错误信息或警告
result["predict_time"]: 预测耗时，单位 us
```

#### 4.1.5.4 策略接口

当前策略接口：

```python
class PolicyBaseModel(ABC):
    def predict(self, node: OpNode, hw: HardwareSpec) -> SimResult:
        ...
```

建议扩展：

```python
class PolicyBaseModel(ABC):
    def predict(
        self,
        node: OpNode,
        hw: HardwareSpec,
        context: PredictionContext | None = None,
    ) -> SimResult:
        ...
```

`PredictionContext` 可包含：

```python
@dataclass
class PredictionContext:
    granularity: str = "op"  # op / fused / module
    phase: str = "forward"   # forward / backward / training
    prefer_explainability: bool = False
    min_confidence: float = 0.0
    allow_decompose: bool = True
```

#### 4.1.5.5 统一输出接口

预测结果统一使用：

```python
SimResult
```

建议后续扩展字段：

```python
source_op_type: str           # 外部库实际查询的 op_type
granularity: str              # op / fused / module
decomposed: bool              # 是否由子算子聚合
warnings: list[str]           # 外部库告警或降级原因
library_version: str          # 外部算子库版本
```

## 5. 关键流程示例

### 5.1 原子 MatMul

```text
OpNode(op_short="mm")
  -> _TO_COST_MODEL_OP_TYPE["mm"] = "Matmul"
  -> inputs/outputs 转 tensor dict
  -> PerformancePredict.op_performance_predict("Matmul", params, {})
  -> predict_time
  -> SimResult(backend="lookup", confidence=0.8)
```

![diagram 12 - matmul flow](assets/multi_granularity_operator_library_design/diagram_12.png)

### 5.2 FlashAttention

```text
OpNode(op_short="sdpa")
  -> "FlashAttention"
  -> 外部库预测
  -> 成功返回 lookup
  -> 失败则建议降级 roofline
```

![diagram 13 - flashattention flow](assets/multi_granularity_operator_library_design/diagram_13.png)

### 5.3 融合算子

```text
OpNode(op_type="fused.rms_norm", fused_from=[...])
  -> 优先 map_op_type("RmsNorm")
  -> 外部库命中则直接返回
  -> 未命中则按 fused_from 子算子拆分估计
  -> 聚合 latency
```

![diagram 14 - fused operator flow](assets/multi_granularity_operator_library_design/diagram_14.png)

## 6. 现状问题与改进建议

### 6.1 当前问题

| 问题 | 影响 |
| --- | --- |
| `TilesimSimulator` 尚未实现 | 高优先级后端不可用 |
| `OpenBoxModel` / `OperatorOptimizationModel` / `SystemDesignModel` 是 stub | 策略枚举存在但不可切换 |
| `LookupSimulator` 内部失败后直接返回低置信度结果 | 可能阻止 Roofline 兜底 |
| 硬件环境写死在 `_HW_ENV_DICT` | CANN/硬件版本扩展不方便 |
| 算子名映射硬编码 | 算子库覆盖扩展不方便 |
| tensor format 为空 | 无法表达 Ascend layout 差异 |

### 6.2 建议演进

第一阶段：

- 将 op mapping 和 hw env 外置配置化。
- `LookupSimulator` 失败时允许策略继续尝试 Roofline。
- 为外部库调用增加 warnings 和 library_version。

第二阶段：

- 支持 fused op 直接查询和拆分降级。
- 增加 `PredictionContext`。
- 完善 `OperatorOptimizationModel` 和 `SystemDesignModel`。

第三阶段：

- 实现 TileSim 后端。
- 支持模块级模板库。
- 报告中展示多粒度预测覆盖率、命中率、置信度分布。

## 7. 验收标准

功能验收：

- 外部算子库可用时，Ascend 910B/910C 的支持算子走 `lookup`。
- 外部算子库不可用时，预测链路不失败，可回落 `roofline`。
- `SimResult` 字段完整，单位统一。
- 同形状同硬件重复算子命中缓存。
- communication 类算子不进入 lookup。

质量验收：

- 日志可定位外部库失败原因。
- 报告可统计后端命中率和平均置信度。
- 新增 op mapping 不需要修改核心逻辑。
- 新增硬件版本不需要修改 `LookupSimulator` 主流程。

## 8. 相关代码索引

| 文件 | 说明 |
| --- | --- |
| `python/zrt/simulator/base.py` | 后端抽象接口 |
| `python/zrt/simulator/hub.py` | 预测入口、缓存、策略委托 |
| `python/zrt/simulator/result.py` | 统一预测结果 |
| `python/zrt/simulator/cache.py` | 预测缓存 |
| `python/zrt/simulator/backends/lookup.py` | 多粒度算子库对接 |
| `python/zrt/simulator/backends/tilesim.py` | TileSim 预留 |
| `python/zrt/simulator/backends/roofline.py` | Roofline 兜底 |
| `python/zrt/policy_model/policy_base_model.py` | 策略模型抽象 |
| `python/zrt/policy_model/priority_model.py` | 默认优先级策略 |
| `python/zrt/policy_model/policy_model_manager.py` | 策略管理器 |
| `python/zrt/ir/node.py` | 算子节点输入结构 |
| `python/zrt/ir/hierarchy.py` | 多粒度 scope 层级视图 |
