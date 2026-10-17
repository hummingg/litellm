"""
Context-Aware Routing Strategy

Routes requests to deployments based on token count and model context window size.
Selects the most cost-effective deployment that can handle the request's token count.

How it works:
- Calculates input token count from request messages/input
- Filters deployments that have sufficient context window (max_input_tokens)
- Selects the deployment with the smallest sufficient context window (cost optimization)
- Falls back to highest context deployment if token count exceeds all limits
"""

from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import token_counter, verbose_logger
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class ContextAwareLoggingHandler(CustomLogger):
    """Handler for context-aware routing strategy"""

    test_flag: bool = False

    def __init__(self, router_cache: DualCache, routing_args: Optional[Dict] = None):
        self.router_cache = router_cache
        self.routing_args = routing_args or {}
        # Configuration
        self.cost_weight = self.routing_args.get("cost_weight", 1.0)
        self.prefer_smallest_context = self.routing_args.get("prefer_smallest_context", True)

    def _calculate_input_tokens(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        model: Optional[str] = None,
    ) -> int:
        """
        Calculate the number of input tokens for the request.

        Args:
            messages: Chat messages
            input: Text input or embedding input
            model: Model name for accurate token counting

        Returns:
            Number of input tokens
        """
        try:
            token_count = token_counter(
                messages=messages,
                text=input,
                model=model
            )
            verbose_router_logger.debug(
                f"Calculated input tokens: {token_count} for model: {model}"
            )
            return token_count
        except Exception as e:
            verbose_logger.warning(
                f"Error calculating tokens, defaulting to 0: {str(e)}"
            )
            return 0

    def _get_deployment_max_tokens(self, deployment: Dict[str, Any]) -> int:
        """
        Extract max input tokens from deployment configuration.

        Checks in order:
        1. deployment.litellm_params.max_input_tokens
        2. deployment.model_info.max_input_tokens
        3. litellm.model_cost[model].max_input_tokens
        4. Default to infinity (no limit)

        Args:
            deployment: Deployment configuration dict

        Returns:
            Maximum input tokens for this deployment
        """
        # Check litellm_params first
        max_tokens = deployment.get("litellm_params", {}).get("max_input_tokens")
        if max_tokens is not None:
            return int(max_tokens)

        # Check model_info
        max_tokens = deployment.get("model_info", {}).get("max_input_tokens")
        if max_tokens is not None:
            return int(max_tokens)

        # Check litellm model_cost map
        model_name = deployment.get("litellm_params", {}).get("model")
        if model_name:
            model_info = litellm.model_cost.get(model_name, {})
            max_tokens = model_info.get("max_input_tokens")
            if max_tokens is not None:
                return int(max_tokens)

        # Default to infinity (no limit)
        return float('inf')

    def _get_deployment_cost(self, deployment: Dict[str, Any]) -> float:
        """
        Calculate the cost per token for a deployment.

        Args:
            deployment: Deployment configuration dict

        Returns:
            Total cost per token (input + output)
        """
        litellm_params = deployment.get("litellm_params", {})
        model_name = litellm_params.get("model")

        # Get costs from deployment config or model_cost map
        input_cost = (
            litellm_params.get("input_cost_per_token")
            or litellm.model_cost.get(model_name, {}).get("input_cost_per_token", 0)
        )

        output_cost = (
            litellm_params.get("output_cost_per_token")
            or litellm.model_cost.get(model_name, {}).get("output_cost_per_token", 0)
        )

        return input_cost + output_cost

    def _select_optimal_deployment(
        self,
        valid_deployments: List[Dict[str, Any]],
        token_count: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the optimal deployment from valid candidates.

        Strategy:
        1. If prefer_smallest_context=True: Select deployment with smallest context that fits
        2. Otherwise: Select by lowest cost

        Args:
            valid_deployments: List of deployments that can handle the request
            token_count: Number of input tokens

        Returns:
            Selected deployment or None
        """
        if not valid_deployments:
            return None

        if self.prefer_smallest_context:
            # Sort by max_tokens (ascending), then by cost
            scored_deployments = []
            for dep in valid_deployments:
                max_tokens = self._get_deployment_max_tokens(dep)
                cost = self._get_deployment_cost(dep)
                scored_deployments.append((dep, max_tokens, cost))

            # Sort by max_tokens first, then cost
            scored_deployments.sort(key=lambda x: (x[1], x[2]))
            selected = scored_deployments[0][0]

            verbose_router_logger.debug(
                f"Selected deployment with max_tokens={scored_deployments[0][1]}, "
                f"cost={scored_deployments[0][2]}, "
                f"model={selected.get('litellm_params', {}).get('model')}"
            )
        else:
            # Sort by cost only
            scored_deployments = []
            for dep in valid_deployments:
                cost = self._get_deployment_cost(dep)
                scored_deployments.append((dep, cost))

            scored_deployments.sort(key=lambda x: x[1])
            selected = scored_deployments[0][0]

            verbose_router_logger.debug(
                f"Selected deployment with cost={scored_deployments[0][1]}, "
                f"model={selected.get('litellm_params', {}).get('model')}"
            )

        return selected

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best deployment based on context window requirements.

        Args:
            model_group: Model group name
            healthy_deployments: List of healthy deployment configurations
            messages: Chat messages for token counting
            input: Text/embedding input for token counting
            request_kwargs: Additional request parameters

        Returns:
            Selected deployment or None if no suitable deployment found
        """
        if not healthy_deployments:
            verbose_router_logger.warning(
                f"No healthy deployments available for model_group: {model_group}"
            )
            return None

        # Calculate input token count
        sample_model = healthy_deployments[0].get("litellm_params", {}).get("model")
        token_count = self._calculate_input_tokens(
            messages=messages,
            input=input,
            model=sample_model
        )

        verbose_router_logger.info(
            f"Context-aware routing for {model_group}: {token_count} input tokens"
        )

        # Filter deployments that can handle the token count
        valid_deployments = []
        for deployment in healthy_deployments:
            max_tokens = self._get_deployment_max_tokens(deployment)
            deployment_id = deployment.get("model_info", {}).get("id")
            model_name = deployment.get("litellm_params", {}).get("model")

            if token_count <= max_tokens:
                valid_deployments.append(deployment)
                verbose_router_logger.debug(
                    f"Deployment {deployment_id} ({model_name}) can handle "
                    f"{token_count} tokens (max: {max_tokens})"
                )
            else:
                verbose_router_logger.debug(
                    f"Deployment {deployment_id} ({model_name}) cannot handle "
                    f"{token_count} tokens (max: {max_tokens})"
                )

        # If no deployments can handle it, try to select the one with largest context
        if not valid_deployments:
            verbose_router_logger.warning(
                f"No deployment can handle {token_count} tokens. "
                f"Selecting deployment with largest context window."
            )
            largest_context_deployment = max(
                healthy_deployments,
                key=lambda d: self._get_deployment_max_tokens(d)
            )
            return largest_context_deployment

        # Select optimal deployment from valid candidates
        selected_deployment = self._select_optimal_deployment(
            valid_deployments, token_count
        )

        if selected_deployment:
            model_name = selected_deployment.get("litellm_params", {}).get("model")
            max_tokens = self._get_deployment_max_tokens(selected_deployment)
            verbose_router_logger.info(
                f"Selected {model_name} with max_tokens={max_tokens} "
                f"for {token_count} input tokens"
            )

        return selected_deployment

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Sync version of get_available_deployments.

        Note: This is a simplified sync wrapper. For production use,
        consider implementing full sync logic or use async version.
        """
        if not healthy_deployments:
            return None

        # Calculate input token count
        sample_model = healthy_deployments[0].get("litellm_params", {}).get("model")
        token_count = self._calculate_input_tokens(
            messages=messages,
            input=input,
            model=sample_model
        )

        # Filter valid deployments
        valid_deployments = [
            d for d in healthy_deployments
            if token_count <= self._get_deployment_max_tokens(d)
        ]

        # Fallback to largest context if no valid deployments
        if not valid_deployments:
            return max(
                healthy_deployments,
                key=lambda d: self._get_deployment_max_tokens(d)
            )

        # Select optimal deployment
        return self._select_optimal_deployment(valid_deployments, token_count)
