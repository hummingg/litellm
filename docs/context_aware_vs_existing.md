# Context-Aware Routing vs 现有 LiteLLM 架构对比

## 发现：LiteLLM 已有上下文窗口过滤

经过源码分析，发现 LiteLLM 的 `Router._pre_call_checks()` 函数已经在**所有路由策略之前**自动过滤掉上下文窗口不足的部署。

### 现有架构流程

```python
# litellm/router.py:6686-6700
def _pre_call_checks(self, model, healthy_deployments, messages, request_kwargs):
    """
    Filter out model in model group, if:
    - model context window < message length  ← 已有此功能！
    - filter models above rpm limits
    """

    input_tokens = token_counter(messages=messages, text=input)

    for deployment in healthy_deployments:
        if input_tokens > model_info["max_input_tokens"]:
            invalid_model_indices.append(idx)  # 过滤掉
            continue

    # 返回过滤后的部署给路由策略
    return filtered_deployments
```

### 执行顺序

```
1. _pre_call_checks()
   ↓ 过滤上下文窗口不足的部署
   ↓ 过滤 RPM 超限的部署

2. 路由策略 (lowest_cost / least_busy / lowest_latency)
   ↓ 从已过滤的部署中选择最优的

3. 返回选中的部署
```

## Context-Aware Routing 还有必要吗？

### ❌ 不必要的原因

1. **功能重复**
   - `_pre_call_checks` 已经过滤上下文窗口
   - 所有路由策略都自动受益

2. **Lowest-Cost 已经很智能**
   ```python
   # lowest_cost.py:232
   input_tokens = token_counter(messages=messages, text=input)

   # L318-322: 检查 TPM/RPM
   if item_tpm + input_tokens > _deployment_tpm:
       continue  # 过滤掉

   # L329: 从剩余部署中选最便宜的
   potential_deployments = sorted(potential_deployments, key=lambda x: x[1])
   ```

3. **现有架构更灵活**
   - 过滤逻辑集中在 `_pre_call_checks`
   - 路由策略只需关注选择逻辑
   - 符合单一职责原则

### ✅ 仍有价值的场景

虽然功能重复，但 Context-Aware 在以下场景仍有独特优势：

#### 场景 1: 优选最小上下文模型（成本优化）

**问题**：`_pre_call_checks` 只是过滤，`lowest_cost` 选价格最低的，不考虑上下文大小。

**示例**：
```python
model_list = [
    {"model": "gpt-4o", "max_tokens": 128K, "cost": 0.0000025},      # 最便宜
    {"model": "gpt-4-turbo", "max_tokens": 128K, "cost": 0.00001},
    {"model": "gpt-4", "max_tokens": 8K, "cost": 0.00003},
]

# 请求: 5K tokens

# 现有架构:
# - _pre_call_checks: 三个都通过（5K < 8K/128K/128K）
# - lowest_cost: 选 gpt-4o (最便宜)
# 结果: 用 128K 模型处理 5K 请求

# Context-Aware:
# - 选 gpt-4 (8K, 最小满足需求)
# 优势:
#   1. 更快的响应时间（小模型）
#   2. 节省大模型配额给真正需要的请求
#   3. 更均衡的负载分布
```

**是否值得**：❓ 存疑
- 价格差距很大时（0.0000025 vs 0.00003 = 12x），lowest_cost 的选择更合理
- 只有在价格接近、追求极致延迟时才有意义

#### 场景 2: 梯度定价模型

**问题**：某些提供商对不同上下文长度有不同定价。

**示例**：
```python
model_list = [
    {
        "model": "claude-3-opus",
        "max_tokens": 200000,
        "input_cost": 0.000015,
        "context_tier": "large"
    },
    {
        "model": "claude-3-opus",  # 同一个模型
        "max_tokens": 20000,       # 限制上下文
        "input_cost": 0.000010,    # 更便宜的价格
        "context_tier": "medium"
    }
]

# 如果 API 提供商支持通过参数限制上下文以获得折扣
# Context-Aware 可以自动选择刚好够用的配置
```

**是否值得**：❌ 不值得
- 这种定价模式很少见
- 可以通过配置多个部署 + lowest_cost 实现

#### 场景 3: 多租户配额隔离

**问题**：希望将大请求和小请求分配到不同账号，避免互相影响。

**示例**：
```python
model_list = [
    {
        "model": "gpt-4",
        "max_tokens": 8000,        # 人为限制
        "api_key": "sk-small-requests-xxx",  # 小请求专用账号
        "tpm": 100000,
    },
    {
        "model": "gpt-4-turbo",
        "max_tokens": 128000,
        "api_key": "sk-large-requests-xxx",  # 大请求专用账号
        "tpm": 50000,
    }
]

# Context-Aware 会自动路由：
# - < 8K 请求 → 小请求账号（保护大账号配额）
# - > 8K 请求 → 大请求账号（必须用）
```

**是否值得**：✅ 有价值
- 配额管理更精细
- 防止小请求挤占大请求资源
- 但也可以通过 tag-based routing 实现

#### 场景 4: 降级策略优化

**问题**：在部分部署不可用时，选择性能最接近的备选。

**示例**：
```python
model_list = [
    {"model": "gpt-4", "max_tokens": 8K, "priority": 1},
    {"model": "gpt-3.5-turbo", "max_tokens": 16K, "priority": 2},
    {"model": "gpt-4-turbo", "max_tokens": 128K, "priority": 3},
]

# 8K 请求，gpt-4 不可用
# Context-Aware: 选 gpt-3.5-turbo (最小满足需求的备选)
# Lowest-Cost: 看价格，可能选 gpt-4-turbo
```

**是否值得**：⚠️ 边缘价值
- 可以通过调整 cost 配置实现类似效果
- 健康检查 + fallback 已经处理降级

## 实际测试对比

### 测试场景：混合请求负载

```python
# 部署配置
model_list = [
    {"model": "gpt-4o", "max_tokens": 128000, "input_cost": 0.0000025, "tpm": 100000},
    {"model": "gpt-4", "max_tokens": 8192, "input_cost": 0.00003, "tpm": 50000},
]

# 请求分布
requests = [
    {"tokens": 1000} * 70,    # 70% 小请求
    {"tokens": 5000} * 20,    # 20% 中请求
    {"tokens": 50000} * 10,   # 10% 大请求
]
```

#### 现有架构 (Lowest-Cost)

```
所有请求 → gpt-4o (最便宜)

成本: 90 * 0.0000025 = $0.000225
配额使用:
  - gpt-4o: 100% (可能超限)
  - gpt-4: 0% (浪费)
```

#### Context-Aware

```
1000 tokens → gpt-4 (8K 足够)
5000 tokens → gpt-4 (8K 足够)
50000 tokens → gpt-4o (必须用)

成本:
  - 70 * 1000 * 0.00003 = $2.1
  - 20 * 5000 * 0.00003 = $3.0
  - 10 * 50000 * 0.0000025 = $1.25
  总计: $6.35

配额使用:
  - gpt-4o: 10% (节省配额)
  - gpt-4: 90% (充分利用)
```

**结果**：❌ Context-Aware **成本更高** ($6.35 vs $0.000225)！

### 修正后的价值场景

唯一合理的使用场景是**价格相近但性能不同**的模型：

```python
model_list = [
    {"model": "claude-3-opus", "max_tokens": 200000, "cost": 0.000015, "latency": "high"},
    {"model": "claude-3-sonnet", "max_tokens": 200000, "cost": 0.000015, "latency": "medium"},
    {"model": "claude-3-haiku", "max_tokens": 200000, "cost": 0.000015, "latency": "low"},
]

# 价格相同时，Context-Aware 可以优先选择延迟更低的模型
# 但这应该用 lowest_latency 策略，而不是 context_aware
```

## 最终结论

### Context-Aware Routing 的定位

1. **不是必需功能**
   - LiteLLM 的 `_pre_call_checks` 已经处理上下文过滤
   - 现有路由策略组合可以覆盖大多数场景

2. **可能的使用场景**
   - ✅ 多租户配额隔离（但 tag-based 更好）
   - ✅ 性能优化（优先用小模型，降低延迟）
   - ❌ 成本优化（通常适得其反）

3. **推荐替代方案**
   ```python
   # 方案 1: 使用 tag-based routing 实现配额隔离
   model_list = [
       {"model": "gpt-4", "tags": ["small-context"], "max_tokens": 8000},
       {"model": "gpt-4-turbo", "tags": ["large-context"], "max_tokens": 128000},
   ]

   # 客户端根据需求打标签
   if len(messages) < 1000:
       metadata = {"tags": ["small-context"]}

   # 方案 2: 调整 lowest_cost 的成本配置
   model_list = [
       {
           "model": "gpt-4",
           "input_cost_per_token": 0.00003 * 0.8,  # 人为降低成本权重
       },
       {
           "model": "gpt-4o",
           "input_cost_per_token": 0.0000025,
       },
   ]
   # 通过调整成本权重，引导路由器选择小模型

   # 方案 3: 组合使用多种策略
   # 对小请求用 lowest_latency
   # 对大请求用 lowest_cost
   ```

### 我的实现建议

1. **保留代码作为参考实现**
   - 展示了如何实现自定义路由策略
   - 完整的测试用例和文档
   - 可以作为学习材料

2. **不建议合并到 LiteLLM**
   - 功能与现有架构重复
   - 成本优化效果通常为负
   - 增加维护复杂度

3. **可以提交 Issue/Discussion**
   - 讨论是否有其他人有类似需求
   - 收集真实使用场景
   - 可能发现我未考虑到的价值

## 总结

经过深入分析，**Context-Aware Routing 在当前 LiteLLM 架构下不是必需的**。现有的 `_pre_call_checks` + `lowest_cost` 组合已经能很好地处理上下文窗口和成本优化。

如果有特殊需求（如配额隔离、性能优先），建议使用：
- Tag-based routing（配额隔离）
- Lowest-latency routing（性能优先）
- 自定义成本权重（引导选择）

Context-Aware 更适合作为**教学示例**，展示如何扩展 LiteLLM 的路由策略。
