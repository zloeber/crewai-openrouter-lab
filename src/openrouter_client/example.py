#!/usr/bin/env python

from client import OpenRouterClient
from dotenv import load_dotenv
from models import ModelRequirements


# Example usage of the OpenRouterClient
def main():
    load_dotenv()
    # Set your API key (in a real application, use environment variables)
    # os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"

    # Initialize the client
    client = OpenRouterClient()

    # Define requirements for model selection
    requirements = ModelRequirements(
        max_cost_per_token=0,
        min_context_length=8000,
        required_features=["temperature", "top_p", "tools"],
        input_modalities=["text"],
        output_modalities=["text"],
        prefer_unmoderated=False,
    )

    # Select a model based on requirements
    model = client.select_model(requirements)

    if model:
        print(f"Selected model: {model.name} (ID: {model.id})")
        print(f"Context length: {model.context_length}")
        print(
            f"Pricing: {model.pricing.prompt} per prompt token, {model.pricing.completion} per completion token"
        )
        print(f"Supported parameters: {', '.join(model.supported_parameters)}")
    else:
        print("No model found matching the requirements")

    # Get multiple models that meet requirements
    models = client.select_models(requirements, limit=3)
    print(f"\nFound {len(models)} models matching requirements:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.name} (ID: {model.id})")


if __name__ == "__main__":
    main()
