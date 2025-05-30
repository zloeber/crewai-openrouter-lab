import os
from typing import Dict, List, Optional

import requests

from src.utils.logging import UnifiedLogger

from .models import ModelListResponse, ModelMetadata, ModelRequirements


class OpenRouterClient:
    """
    Client for interacting with the OpenRouter.ai API to fetch and select AI models
    based on specified requirements.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    MODELS_ENDPOINT = "/models"

    def __init__(self, logger: UnifiedLogger, api_key: str = None):
        """
        Initialize the OpenRouter client.

        Args:
            logger: UnifiedLogger instance for logging.
            api_key: OpenRouter API key. If not provided, will attempt to read from
                    OPENROUTER_API_KEY environment variable.
        """
        self.logger = logger
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            self.logger.print_error(
                "API key must be provided either directly or via OPENROUTER_API_KEY environment variable"
            )
            raise ValueError(
                "API key must be provided either directly or via OPENROUTER_API_KEY environment variable"
            )

        self._models_cache: Optional[List[ModelMetadata]] = None
        self.logger.print_debug("OpenRouter client initialized")

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
            self.logger.debug("Returning cached models")
            return self._models_cache

        url = f"{self.BASE_URL}{self.MODELS_ENDPOINT}"
        self.logger.print_debug(f"Fetching models from {url}")

        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()

            # Parse the response using Pydantic model
            model_list = ModelListResponse.model_validate(response.json())
            self._models_cache = model_list.data
            self.logger.print_debug(
                f"Successfully fetched {len(self._models_cache)} models"
            )

            return self._models_cache
        except requests.RequestException as e:
            error_msg = f"Failed to fetch models: {str(e)}"
            self.logger.print_error(error_msg)
            raise requests.RequestException(error_msg)
        except Exception as e:
            error_msg = f"Failed to parse model data: {str(e)}"
            self.logger.print_error(error_msg)
            raise ValueError(error_msg)

    def clear_cache(self) -> None:
        """Clear the models cache."""
        self._models_cache = None
        self.logger.print_info("Model cache cleared")

    def select_model(self, requirements: ModelRequirements) -> Optional[ModelMetadata]:
        """
        Select the best model that meets the specified requirements.

        Args:
            requirements: Requirements for model selection.

        Returns:
            The best matching model, or None if no model meets the requirements.
        """
        self.logger.print_info("Selecting model based on requirements")
        self.logger.print_json(requirements.model_dump(), "Model Requirements")

        models = self.fetch_models(force_refresh=requirements.force_refresh)

        # Filter models based on requirements
        filtered_models = self._filter_models(models, requirements)

        # If no models meet the requirements, return None
        if not filtered_models:
            self.logger.print_warning("No models found matching the requirements")
            return None

        # Sort models by cost (lowest first) as a default priority
        sorted_models = sorted(
            filtered_models,
            key=lambda m: float(m.pricing.prompt) + float(m.pricing.completion),
        )

        # Return the best match (lowest cost model that meets all requirements)
        selected_model = sorted_models[0] if sorted_models else None
        if selected_model:
            self.logger.print_debug(
                f"Selected model: {selected_model.name} (ID: {selected_model.id})"
            )
        return selected_model

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

        self.logger.print_debug(
            f"Found {len(filtered_models)} models matching requirements"
        )
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
            self.logger.debug(f"Model {model.name} excluded by requirements")
            return False

        # Check cost requirements
        if requirements.max_cost_per_token is not None:
            self.logger.debug(
                f"Checking cost at most {requirements.max_cost_per_token} for {model.name}"
            )
            prompt_cost = float(model.pricing.prompt)
            completion_cost = float(model.pricing.completion)
            avg_cost = (prompt_cost + completion_cost) / 2

            if avg_cost > requirements.max_cost_per_token:
                self.logger.debug(f"Model {model.name} exceeds cost requirements")
                return False

        # Check context length requirements
        if requirements.min_context_length is not None:
            self.logger.debug(
                f"Checking context length at least {requirements.min_context_length} for {model.name}"
            )
            if model.context_length < requirements.min_context_length:
                self.logger.debug(
                    f"Model {model.name} does not meet context length requirements"
                )
                return False

        # Check required features/parameters
        if requirements.required_features:
            self.logger.debug(
                f"Checking required features {requirements.required_features} for {model.name}"
            )
            if not all(
                feature in model.supported_parameters
                for feature in requirements.required_features
            ):
                self.logger.debug(
                    f"Model {model.name} does not support all required features"
                )
                return False

        # Check input modalities
        if requirements.input_modalities:
            self.logger.debug(
                f"Checking input modalities {requirements.input_modalities} for {model.name}"
            )
            if not all(
                modality in model.architecture.input_modalities
                for modality in requirements.input_modalities
            ):
                self.logger.debug(
                    f"Model {model.name} does not support all required input modalities"
                )
                return False

        # Check output modalities
        if requirements.output_modalities:
            self.logger.debug(
                f"Checking output modalities {requirements.output_modalities} for {model.name}"
            )
            if not all(
                modality in model.architecture.output_modalities
                for modality in requirements.output_modalities
            ):
                self.logger.debug(
                    f"Model {model.name} does not support all required output modalities"
                )
                return False

        # Check moderation preference
        if requirements.prefer_unmoderated is not None:
            self.logger.debug(
                f"Checking moderation preference {requirements.prefer_unmoderated} for {model.name}"
            )
            if requirements.prefer_unmoderated and model.top_provider.is_moderated:
                self.logger.debug(
                    f"Model {model.name} does not meet moderation requirements"
                )
                return False

        # If we've passed all checks, the model meets the requirements
        self.logger.debug(f"Model {model.name} meets all requirements")
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
        self.logger.print_debug(f"Selecting up to {limit} models based on requirements")
        self.logger.print_debug_json(requirements.model_dump(), "Model Requirements")

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
            selected_models = sorted_models[:limit]
        else:
            selected_models = sorted_models

        self.logger.print_debug(f"Selected {len(selected_models)} models")
        return selected_models
        return selected_models
