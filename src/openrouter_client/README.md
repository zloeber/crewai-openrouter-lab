# OpenRouter.ai Client

A Python client for connecting to OpenRouter.ai and selecting AI models based on requirements like cost and features.

## Features

- Connect to OpenRouter.ai API with authentication
- Fetch available models with metadata
- Select models based on requirements:
  - Maximum cost per token
  - Minimum context length
  - Required features/parameters
  - Input/output modalities
  - Moderation preferences
- Comprehensive test coverage

## Dependencies

- Python 3.6+
- pydantic
- requests

## Usage

```python
import os
from src.models import ModelRequirements
from src.client import OpenRouterClient

# Set your API key (in a real application, use environment variables)
os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

# Initialize the client
client = OpenRouterClient()

# Define requirements for model selection
requirements = ModelRequirements(
    max_cost_per_token=0.00001,
    min_context_length=8000,
    required_features=["temperature", "top_p"],
    input_modalities=["text"],
    output_modalities=["text"],
    prefer_unmoderated=False
)

# Select a model based on requirements
model = client.select_model(requirements)

if model:
    print(f"Selected model: {model.name} (ID: {model.id})")
    print(f"Context length: {model.context_length}")
    print(f"Pricing: {model.pricing.prompt} per prompt token, {model.pricing.completion} per completion token")
else:
    print("No model found matching the requirements")

# Get multiple models that meet requirements
models = client.select_models(requirements, limit=3)
print(f"\nFound {len(models)} models matching requirements:")
for i, model in enumerate(models, 1):
    print(f"{i}. {model.name} (ID: {model.id})")
```

## API Reference

### `OpenRouterClient`

The main client class for interacting with OpenRouter.ai.

#### Methods

- `__init__(api_key=None)`: Initialize the client with an API key (or read from environment variable)
- `fetch_models(force_refresh=False)`: Fetch available models from OpenRouter
- `select_model(requirements)`: Select the best model that meets the specified requirements
- `select_models(requirements, limit=5)`: Select multiple models that meet the requirements
- `clear_cache()`: Clear the models cache

### `ModelRequirements`

Pydantic model for specifying model selection requirements.

#### Fields

- `max_cost_per_token`: Maximum cost per token
- `min_context_length`: Minimum required context length
- `required_features`: Required features or parameters
- `input_modalities`: Required input modalities
- `output_modalities`: Required output modalities
- `prefer_unmoderated`: Prefer unmoderated providers

## Testing

Run the tests with:

```bash
python -m unittest tests/test_client.py
```

## License

MIT
