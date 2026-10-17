# Context-Aware Routing Strategy

## Overview

Context-Aware Routing is a smart routing strategy that automatically selects the most cost-effective LLM deployment based on the input token count and model context window sizes.

## How It Works

1. **Token Calculation**: Calculates the number of input tokens from the request
2. **Deployment Filtering**: Filters deployments that can handle the token count
3. **Optimal Selection**: Selects the deployment with the smallest sufficient context window (most cost-effective)
4. **Fallback**: If token count exceeds all limits, selects the deployment with the largest context window

## Key Features

- ✅ Automatic token counting
- ✅ Cost optimization (prefer smaller context models when possible)
- ✅ Configurable routing preferences
- ✅ Fallback to largest context when needed
- ✅ Support for both sync and async operations

## Installation

The context-aware routing strategy is included in the LiteLLM router strategies.

```python
from litellm import Router
from litellm.router_strategy.context_aware_routing import ContextAwareLoggingHandler
```

## Configuration

### Basic Setup

```python
from litellm import Router

model_list = [
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "gpt-4",
            "max_input_tokens": 8192,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
        }
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "gpt-4-turbo",
            "max_input_tokens": 128000,
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00003,
        }
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "max_input_tokens": 16385,
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        }
    }
]

router = Router(
    model_list=model_list,
    routing_strategy="context-aware",  # Use context-aware routing
)
```

### Advanced Configuration

```python
router = Router(
    model_list=model_list,
    routing_strategy="context-aware",
    routing_strategy_args={
        "prefer_smallest_context": True,  # Default: True - prefer smallest sufficient context
        "cost_weight": 1.0,  # Cost weight for optimization
    }
)
```

## Usage Examples

### Example 1: Short Conversation (< 8K tokens)

```python
import asyncio
from litellm import Router

router = Router(model_list=model_list, routing_strategy="context-aware")

async def main():
    response = await router.acompletion(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response)

asyncio.run(main())
```

**Result**: Routes to `gpt-4` (8K context) as it's the smallest model that fits the request.

### Example 2: Medium Conversation (8K - 16K tokens)

```python
async def main():
    # Long conversation with ~10K tokens
    long_messages = [
        {"role": "user", "content": "Please analyze this long document..." * 1000}
    ]

    response = await router.acompletion(
        model="gpt-4",
        messages=long_messages
    )
    print(response)

asyncio.run(main())
```

**Result**: Routes to `gpt-3.5-turbo` (16K context) as it's the smallest model that can handle 10K tokens.

### Example 3: Long Conversation (> 16K tokens)

```python
async def main():
    # Very long conversation with ~50K tokens
    very_long_messages = [
        {"role": "user", "content": "Analyze these documents..." * 5000}
    ]

    response = await router.acompletion(
        model="gpt-4",
        messages=very_long_messages
    )
    print(response)

asyncio.run(main())
```

**Result**: Routes to `gpt-4-turbo` (128K context) as it's the only model that can handle 50K tokens.

### Example 4: Exceeding All Limits

```python
async def main():
    # Extremely long input exceeding all limits
    extreme_messages = [
        {"role": "user", "content": "Process this massive dataset..." * 20000}
    ]

    response = await router.acompletion(
        model="gpt-4",
        messages=extreme_messages
    )
    # Routes to gpt-4-turbo (largest context window available)

asyncio.run(main())
```

**Result**: Routes to `gpt-4-turbo` (128K context) as it has the largest context window, even though input exceeds limit.

## Deployment Configuration

### Setting Max Input Tokens

You can specify `max_input_tokens` in three ways (in order of priority):

1. **In `litellm_params`** (Recommended)
```python
{
    "litellm_params": {
        "model": "gpt-4-turbo",
        "max_input_tokens": 128000
    }
}
```

2. **In `model_info`**
```python
{
    "model_info": {
        "id": "my-deployment",
        "max_input_tokens": 128000
    },
    "litellm_params": {
        "model": "gpt-4-turbo"
    }
}
```

3. **Auto-detected from LiteLLM's model cost map**
```python
# LiteLLM will automatically use known context limits for common models
{
    "litellm_params": {
        "model": "gpt-4-turbo"  # Automatically uses 128K limit
    }
}
```

### Setting Costs

Specify costs for accurate routing decisions:

```python
{
    "litellm_params": {
        "model": "gpt-4",
        "max_input_tokens": 8192,
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
    }
}
```

## Routing Strategies

### Prefer Smallest Context (Default)

When `prefer_smallest_context=True` (default), the router selects the deployment with the smallest context window that can handle the request. This optimizes for cost, as smaller context models are typically cheaper.

```python
routing_strategy_args = {
    "prefer_smallest_context": True
}
```

**Example**: For a 5K token request with deployments of 8K, 16K, and 128K:
- ✅ Selects 8K deployment (smallest that fits)
- ❌ Skips 16K and 128K (unnecessarily large)

### Lowest Cost

When `prefer_smallest_context=False`, the router selects based purely on cost, regardless of context window size.

```python
routing_strategy_args = {
    "prefer_smallest_context": False
}
```

**Example**: For a 5K token request:
- ✅ Selects the deployment with lowest `input_cost_per_token + output_cost_per_token`

## Integration with Other Strategies

You can combine context-aware routing with health checks and fallbacks:

```python
router = Router(
    model_list=model_list,
    routing_strategy="context-aware",
    set_verbose=True,  # Enable logging
    num_retries=3,  # Retry failed requests
    timeout=30,  # Request timeout
)
```

## Monitoring and Debugging

Enable verbose logging to see routing decisions:

```python
import litellm

litellm.set_verbose = True

router = Router(
    model_list=model_list,
    routing_strategy="context-aware",
    set_verbose=True
)
```

**Log Output Example**:
```
Context-aware routing for gpt-4: 5234 input tokens
Deployment gpt-4-small (gpt-4) can handle 5234 tokens (max: 8192)
Deployment gpt-3.5 (gpt-3.5-turbo) can handle 5234 tokens (max: 16385)
Deployment gpt-4-large (gpt-4-turbo) can handle 5234 tokens (max: 128000)
Selected gpt-4 with max_tokens=8192 for 5234 input tokens
```

## Testing

Run the test suite:

```bash
# Run all context-aware routing tests
pytest tests/test_litellm/router_strategy/test_context_aware_routing.py -v

# Run specific test
pytest tests/test_litellm/router_strategy/test_context_aware_routing.py::TestContextAwareRouting::test_async_get_available_deployments_normal_case -v
```

## Best Practices

1. **Always specify `max_input_tokens`**: While LiteLLM can auto-detect for common models, explicit configuration is more reliable.

2. **Set accurate costs**: Proper cost configuration ensures optimal routing decisions.

3. **Use async methods**: For production workloads, use `router.acompletion()` for better performance.

4. **Monitor routing decisions**: Enable verbose logging in development to verify routing behavior.

5. **Combine with health checks**: Use context-aware routing alongside LiteLLM's health check features for robust production systems.

## Comparison with Other Strategies

| Strategy | Use Case | Selection Criteria |
|----------|----------|-------------------|
| **context-aware** | Variable input sizes | Token count + context window |
| **least-busy** | Load balancing | Current request count |
| **lowest-cost** | Cost optimization | Cost per token |
| **lowest-latency** | Speed optimization | Response time |
| **tag-based** | Team/environment routing | Static tags |

## Troubleshooting

### Issue: Router selects expensive model for small requests

**Solution**: Verify `prefer_smallest_context=True` and check that smaller models have correct `max_input_tokens`.

### Issue: Token count calculation is incorrect

**Solution**: Ensure you're passing either `messages` or `input` parameter to the completion call.

### Issue: All requests route to the same model

**Solution**: Check that deployments have different `max_input_tokens` values and vary your request sizes.

## API Reference

### ContextAwareLoggingHandler

Main class implementing the context-aware routing strategy.

#### Constructor Parameters

- `router_cache` (DualCache): Cache instance for routing state
- `routing_args` (Dict, optional): Configuration options
  - `prefer_smallest_context` (bool): Prefer smallest sufficient context (default: True)
  - `cost_weight` (float): Weight for cost optimization (default: 1.0)

#### Methods

##### `async_get_available_deployments`

```python
async def async_get_available_deployments(
    model_group: str,
    healthy_deployments: List[Dict[str, Any]],
    messages: Optional[List[Dict[str, str]]] = None,
    input: Optional[Union[str, List]] = None,
    request_kwargs: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]
```

Returns the optimal deployment for the request.

##### `get_available_deployments`

Synchronous version of `async_get_available_deployments`.

## Contributing

Contributions are welcome! Please:

1. Add tests for new features
2. Update documentation
3. Follow the existing code style
4. Ensure all tests pass

## License

This is part of the LiteLLM project and follows the same license.
