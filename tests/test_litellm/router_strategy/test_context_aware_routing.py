"""
Test Context-Aware Routing Strategy
"""

import pytest
from unittest.mock import Mock, patch
from litellm.router_strategy.context_aware_routing import ContextAwareLoggingHandler
from litellm.caching.caching import DualCache


@pytest.fixture
def router_cache():
    """Create a mock router cache"""
    return Mock(spec=DualCache)


@pytest.fixture
def handler(router_cache):
    """Create a ContextAwareLoggingHandler instance"""
    return ContextAwareLoggingHandler(router_cache=router_cache)


@pytest.fixture
def sample_deployments():
    """Sample deployment configurations with different context windows"""
    return [
        {
            "model_info": {"id": "gpt-4-small"},
            "litellm_params": {
                "model": "gpt-4",
                "max_input_tokens": 8192,
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
            }
        },
        {
            "model_info": {"id": "gpt-4-large"},
            "litellm_params": {
                "model": "gpt-4-turbo",
                "max_input_tokens": 128000,
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00003,
            }
        },
        {
            "model_info": {"id": "gpt-3.5"},
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "max_input_tokens": 16385,
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            }
        },
    ]


class TestContextAwareRouting:
    """Test suite for context-aware routing strategy"""

    def test_calculate_input_tokens(self, handler):
        """Test token calculation"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]

        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=10):
            tokens = handler._calculate_input_tokens(messages=messages)
            assert tokens == 10

    def test_calculate_input_tokens_error_handling(self, handler):
        """Test token calculation error handling"""
        with patch('litellm.router_strategy.context_aware_routing.token_counter', side_effect=Exception("Error")):
            tokens = handler._calculate_input_tokens(messages=[])
            assert tokens == 0

    def test_get_deployment_max_tokens_from_litellm_params(self, handler):
        """Test extracting max tokens from litellm_params"""
        deployment = {
            "litellm_params": {"max_input_tokens": 8192}
        }
        max_tokens = handler._get_deployment_max_tokens(deployment)
        assert max_tokens == 8192

    def test_get_deployment_max_tokens_from_model_info(self, handler):
        """Test extracting max tokens from model_info"""
        deployment = {
            "model_info": {"max_input_tokens": 16000}
        }
        max_tokens = handler._get_deployment_max_tokens(deployment)
        assert max_tokens == 16000

    def test_get_deployment_max_tokens_default(self, handler):
        """Test default max tokens when not specified"""
        deployment = {
            "litellm_params": {"model": "unknown-model"}
        }
        max_tokens = handler._get_deployment_max_tokens(deployment)
        assert max_tokens == float('inf')

    def test_get_deployment_cost(self, handler):
        """Test cost calculation"""
        deployment = {
            "litellm_params": {
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00002,
            }
        }
        cost = handler._get_deployment_cost(deployment)
        assert cost == 0.00003

    def test_select_optimal_deployment_smallest_context(self, handler, sample_deployments):
        """Test selecting deployment with smallest sufficient context"""
        # For 5000 tokens, should select gpt-4 (8192) over gpt-4-turbo (128000)
        selected = handler._select_optimal_deployment(
            valid_deployments=sample_deployments,
            token_count=5000
        )
        assert selected["model_info"]["id"] == "gpt-4-small"

    def test_select_optimal_deployment_by_cost(self, handler, sample_deployments):
        """Test selecting deployment by cost when prefer_smallest_context=False"""
        handler.prefer_smallest_context = False
        selected = handler._select_optimal_deployment(
            valid_deployments=sample_deployments,
            token_count=5000
        )
        # Should select gpt-3.5 as it has lowest cost
        assert selected["model_info"]["id"] == "gpt-3.5"

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_normal_case(self, handler, sample_deployments):
        """Test async routing with normal token count"""
        messages = [{"role": "user", "content": "Hello"}]

        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=5000):
            result = await handler.async_get_available_deployments(
                model_group="gpt-4",
                healthy_deployments=sample_deployments,
                messages=messages
            )

            # Should select gpt-4-small (smallest sufficient context)
            assert result is not None
            assert result["model_info"]["id"] == "gpt-4-small"

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_exceeds_all_limits(self, handler, sample_deployments):
        """Test routing when token count exceeds all deployment limits"""
        messages = [{"role": "user", "content": "Hello"}]

        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=200000):
            result = await handler.async_get_available_deployments(
                model_group="gpt-4",
                healthy_deployments=sample_deployments,
                messages=messages
            )

            # Should select deployment with largest context window
            assert result is not None
            assert result["model_info"]["id"] == "gpt-4-large"

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_empty_list(self, handler):
        """Test routing with empty deployment list"""
        result = await handler.async_get_available_deployments(
            model_group="gpt-4",
            healthy_deployments=[],
            messages=[]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_get_available_deployments_small_token_count(self, handler, sample_deployments):
        """Test routing with very small token count"""
        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=100):
            result = await handler.async_get_available_deployments(
                model_group="gpt-4",
                healthy_deployments=sample_deployments,
                messages=[{"role": "user", "content": "Hi"}]
            )

            # Should select gpt-3.5 (smallest context and lowest cost)
            assert result is not None
            assert result["model_info"]["id"] == "gpt-3.5"

    def test_sync_get_available_deployments(self, handler, sample_deployments):
        """Test sync version of routing"""
        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=5000):
            result = handler.get_available_deployments(
                model_group="gpt-4",
                healthy_deployments=sample_deployments,
                messages=[{"role": "user", "content": "Hello"}]
            )

            assert result is not None
            assert result["model_info"]["id"] == "gpt-4-small"

    def test_routing_with_custom_weights(self, router_cache):
        """Test routing with custom cost weight"""
        handler = ContextAwareLoggingHandler(
            router_cache=router_cache,
            routing_args={"cost_weight": 2.0, "prefer_smallest_context": False}
        )

        assert handler.cost_weight == 2.0
        assert handler.prefer_smallest_context is False

    @pytest.mark.asyncio
    async def test_context_boundary_conditions(self, handler):
        """Test routing at exact context boundaries"""
        deployments = [
            {
                "model_info": {"id": "model-8k"},
                "litellm_params": {
                    "model": "test-8k",
                    "max_input_tokens": 8192,
                    "input_cost_per_token": 0.00001,
                    "output_cost_per_token": 0.00001,
                }
            },
            {
                "model_info": {"id": "model-16k"},
                "litellm_params": {
                    "model": "test-16k",
                    "max_input_tokens": 16384,
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00002,
                }
            }
        ]

        # Test with exactly 8192 tokens (should select 8k model)
        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=8192):
            result = await handler.async_get_available_deployments(
                model_group="test",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "test"}]
            )
            assert result["model_info"]["id"] == "model-8k"

        # Test with 8193 tokens (should select 16k model)
        with patch('litellm.router_strategy.context_aware_routing.token_counter', return_value=8193):
            result = await handler.async_get_available_deployments(
                model_group="test",
                healthy_deployments=deployments,
                messages=[{"role": "user", "content": "test"}]
            )
            assert result["model_info"]["id"] == "model-16k"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
