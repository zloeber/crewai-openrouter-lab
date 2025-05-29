import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client import OpenRouterClient
from src.models import (
    Architecture,
    ModelMetadata,
    ModelRequirements,
    Pricing,
    TopProvider,
)


class TestOpenRouterClient(unittest.TestCase):
    """Test cases for the OpenRouterClient class."""

    def setUp(self):
        """Set up test environment."""
        # Mock environment variable
        os.environ["OPENROUTER_API_KEY"] = "test_api_key"
        self.client = OpenRouterClient()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        self.assertEqual(self.client.api_key, "test_api_key")

    def test_init_with_direct_key(self):
        """Test initialization with direct API key."""
        client = OpenRouterClient(api_key="direct_key")
        self.assertEqual(client.api_key, "direct_key")

    @patch("requests.get")
    def test_fetch_models(self, mock_get):
        """Test fetching models from the API."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "model1",
                    "name": "Test Model 1",
                    "created": 1741818122,
                    "description": "A test model",
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                        "tokenizer": "GPT",
                    },
                    "top_provider": {"is_moderated": True},
                    "pricing": {
                        "prompt": "0.0000007",
                        "completion": "0.0000007",
                        "image": "0",
                        "request": "0",
                        "input_cache_read": "0",
                        "input_cache_write": "0",
                        "web_search": "0",
                        "internal_reasoning": "0",
                    },
                    "context_length": 8192,
                    "supported_parameters": ["temperature", "top_p"],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Call the method
        models = self.client.fetch_models(force_refresh=True)

        # Assertions
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "model1")
        self.assertEqual(models[0].name, "Test Model 1")
        self.assertEqual(models[0].context_length, 8192)
        mock_get.assert_called_once_with(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": "Bearer test_api_key",
                "Content-Type": "application/json",
            },
        )

    def test_model_selection_cost_filter(self):
        """Test model selection based on cost requirements."""
        # Create test models
        model1 = ModelMetadata(
            id="model1",
            name="Cheap Model",
            created=1741818122,
            description="A cheap model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0000001",
                completion="0.0000001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=4096,
            supported_parameters=["temperature", "top_p"],
        )

        model2 = ModelMetadata(
            id="model2",
            name="Expensive Model",
            created=1741818122,
            description="An expensive model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0001",
                completion="0.0001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=8192,
            supported_parameters=["temperature", "top_p", "frequency_penalty"],
        )

        # Mock fetch_models to return our test models
        self.client.fetch_models = MagicMock(return_value=[model1, model2])

        # Test with cost requirement that only model1 meets
        requirements = ModelRequirements(max_cost_per_token=0.00001)
        selected_model = self.client.select_model(requirements)
        self.assertEqual(selected_model.id, "model1")

        # Test with cost requirement that both models meet
        requirements = ModelRequirements(max_cost_per_token=0.001)
        selected_model = self.client.select_model(requirements)
        self.assertEqual(selected_model.id, "model1")  # Should select the cheaper one

    def test_model_selection_feature_filter(self):
        """Test model selection based on feature requirements."""
        # Create test models
        model1 = ModelMetadata(
            id="model1",
            name="Basic Model",
            created=1741818122,
            description="A basic model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0000007",
                completion="0.0000007",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=4096,
            supported_parameters=["temperature", "top_p"],
        )

        model2 = ModelMetadata(
            id="model2",
            name="Advanced Model",
            created=1741818122,
            description="An advanced model",
            architecture=Architecture(
                input_modalities=["text", "image"],
                output_modalities=["text"],
                tokenizer="GPT",
            ),
            top_provider=TopProvider(is_moderated=False),
            pricing=Pricing(
                prompt="0.000001",
                completion="0.000001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=8192,
            supported_parameters=[
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ],
        )

        # Mock fetch_models to return our test models
        self.client.fetch_models = MagicMock(return_value=[model1, model2])

        # Test with feature requirement that only model2 meets
        requirements = ModelRequirements(required_features=["frequency_penalty"])
        selected_model = self.client.select_model(requirements)
        self.assertEqual(selected_model.id, "model2")

        # Test with modality requirement that only model2 meets
        requirements = ModelRequirements(input_modalities=["image"])
        selected_model = self.client.select_model(requirements)
        self.assertEqual(selected_model.id, "model2")

        # Test with moderation preference
        requirements = ModelRequirements(prefer_unmoderated=True)
        selected_model = self.client.select_model(requirements)
        self.assertEqual(selected_model.id, "model2")

    def test_no_matching_models(self):
        """Test behavior when no models match the requirements."""
        # Create test model
        model = ModelMetadata(
            id="model1",
            name="Test Model",
            created=1741818122,
            description="A test model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0001",
                completion="0.0001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=4096,
            supported_parameters=["temperature", "top_p"],
        )

        # Mock fetch_models to return our test model
        self.client.fetch_models = MagicMock(return_value=[model])

        # Test with impossible requirements
        requirements = ModelRequirements(
            max_cost_per_token=0.00000001,
            min_context_length=16384,
            required_features=["non_existent_feature"],
        )
        selected_model = self.client.select_model(requirements)
        self.assertIsNone(selected_model)

    def test_select_multiple_models(self):
        """Test selecting multiple models that meet requirements."""
        # Create test models
        model1 = ModelMetadata(
            id="model1",
            name="Cheap Model",
            created=1741818122,
            description="A cheap model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0000001",
                completion="0.0000001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=4096,
            supported_parameters=["temperature", "top_p"],
        )

        model2 = ModelMetadata(
            id="model2",
            name="Medium Model",
            created=1741818122,
            description="A medium-priced model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.000001",
                completion="0.000001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=8192,
            supported_parameters=["temperature", "top_p"],
        )

        model3 = ModelMetadata(
            id="model3",
            name="Expensive Model",
            created=1741818122,
            description="An expensive model",
            architecture=Architecture(
                input_modalities=["text"], output_modalities=["text"], tokenizer="GPT"
            ),
            top_provider=TopProvider(is_moderated=True),
            pricing=Pricing(
                prompt="0.0001",
                completion="0.0001",
                image="0",
                request="0",
                input_cache_read="0",
                input_cache_write="0",
                web_search="0",
                internal_reasoning="0",
            ),
            context_length=16384,
            supported_parameters=["temperature", "top_p"],
        )

        # Mock fetch_models to return our test models
        self.client.fetch_models = MagicMock(return_value=[model1, model2, model3])

        # Test selecting multiple models
        requirements = ModelRequirements(max_cost_per_token=0.001)
        selected_models = self.client.select_models(requirements, limit=2)

        # Should return the 2 cheapest models
        self.assertEqual(len(selected_models), 2)
        self.assertEqual(selected_models[0].id, "model1")
        self.assertEqual(selected_models[1].id, "model2")


if __name__ == "__main__":
    unittest.main()
