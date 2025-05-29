import os
from typing import Dict, List, Optional

import requests

from src.common.logging import AppLogger, LoggerConfig

from .models import ModelListResponse, ModelMetadata, ModelRequirements

log = AppLogger(LoggerConfig(log_level="INFO", minimal_console=True)).get_logger()


class OpenRouterClient:
    """
    Client for interacting with the OpenRouter.ai API to fetch and select AI models
    based on specified requirements.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    MODELS_ENDPOINT = "/models"

    def __init__(self, api_key: str = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, will attempt to read from
                    OPENROUTER_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either directly or via OPENROUTER_API_KEY environment variable"
            )

        self._models_cache: Optional[List[ModelMetadata]] = None

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers required for API requests.

        Returns:
            Dictionary of headers including authorization.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def fetch_models(self, force_refresh: bool = False) -> List[ModelMetadata]:
        """
        Fetch available models from OpenRouter.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of model metadata objects.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the response cannot be parsed.
        """
        if self._models_cache is not None and not force_refresh:
            return self._models_cache

        url = f"{self.BASE_URL}{self.MODELS_ENDPOINT}"

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()

            # Parse the response using Pydantic model
            model_list = ModelListResponse.model_validate(response.json())
            self._models_cache = model_list.data

            return self._models_cache
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch models: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse model data: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None

    def select_model(self, requirements: ModelRequirements) -> Optional[ModelMetadata]:
        """
        Select the best model that meets the specified requirements.

        Args:
            requirements: Requirements for model selection.

        Returns:
            The best matching model, or None if no model meets the requirements.
        """
        models = self.fetch_models(force_refresh=requirements.force_refresh)

        # Filter models based on requirements
        filtered_models = self._filter_models(models, requirements)

        # If no models meet the requirements, return None
        if not filtered_models:
            return None

        # Sort models by cost (lowest first) as a default priority
        sorted_models = sorted(
            filtered_models,
            key=lambda m: float(m.pricing.prompt) + float(m.pricing.completion),
        )

        # Return the best match (lowest cost model that meets all requirements)
        return sorted_models[0] if sorted_models else None

    def _filter_models(
        self, models: List[ModelMetadata], requirements: ModelRequirements
    ) -> List[ModelMetadata]:
        """
        Filter models based on the specified requirements.

        Args:
            models: List of models to filter.
            requirements: Requirements for filtering.

        Returns:
            List of models that meet the requirements.
        """
        filtered_models = []

        for model in models:
            # Check if model meets all requirements
            if self._model_meets_requirements(model, requirements):
                filtered_models.append(model)

        return filtered_models

    def _model_meets_requirements(
        self, model: ModelMetadata, requirements: ModelRequirements
    ) -> bool:
        """
        Check if a model meets the specified requirements.

        Args:
            model: Model to check.
            requirements: Requirements to check against.

        Returns:
            True if the model meets all requirements, False otherwise.
        """
        # Check if model is in exclude list
        if requirements.exclude_models and model.id in requirements.exclude_models:
            return False

        # Check cost requirements
        if requirements.max_cost_per_token is not None:
            log.debug(
                f"Checking cost at most {requirements.max_cost_per_token} for {model.name}"
            )
            prompt_cost = float(model.pricing.prompt)
            completion_cost = float(model.pricing.completion)
            avg_cost = (prompt_cost + completion_cost) / 2

            if avg_cost > requirements.max_cost_per_token:
                return False

        # Check context length requirements
        if requirements.min_context_length is not None:
            log.debug(
                f"Checking context length at least {requirements.min_context_length} for {model.name}"
            )
            if model.context_length < requirements.min_context_length:
                return False

        # Check required features/parameters
        if requirements.required_features:
            log.debug(
                f"Checking required features {requirements.required_features} for {model.name}"
            )
            if not all(
                feature in model.supported_parameters
                for feature in requirements.required_features
            ):
                return False

        # Check input modalities
        if requirements.input_modalities:
            log.debug(
                f"Checking input modalities {requirements.input_modalities} for {model.name}"
            )
            if not all(
                modality in model.architecture.input_modalities
                for modality in requirements.input_modalities
            ):
                return False

        # Check output modalities
        if requirements.output_modalities:
            log.debug(
                f"Checking output modalities {requirements.output_modalities} for {model.name}"
            )
            if not all(
                modality in model.architecture.output_modalities
                for modality in requirements.output_modalities
            ):
                return False

        # Check moderation preference
        if requirements.prefer_unmoderated is not None:
            log.debug(
                f"Checking moderation preference {requirements.prefer_unmoderated} for {model.name}"
            )
            if requirements.prefer_unmoderated and model.top_provider.is_moderated:
                return False

        # If we've passed all checks, the model meets the requirements
        return True

    def select_models(
        self, requirements: ModelRequirements, limit: int = 5
    ) -> List[ModelMetadata]:
        """
        Select multiple models that meet the specified requirements.

        Args:
            requirements: Requirements for model selection.
            limit: Maximum number of models to return.

        Returns:
            List of models that meet the requirements, sorted by cost.
        """
        models = self.fetch_models()

        # Filter models based on requirements
        filtered_models = self._filter_models(models, requirements)

        # Sort models by cost (lowest first)
        sorted_models = sorted(
            filtered_models,
            key=lambda m: float(m.pricing.prompt) + float(m.pricing.completion),
        )

        # Return up to the specified limit
        if limit > 0:
            return sorted_models[:limit]
        else:
            return sorted_models
