#!/usr/bin/env python

"""
Select and display OpenRouter AI models based on specified requirements

Example:

```bash
python -m src.select-openrouter-model \
  --max-cost 0.00001 \
  --min-context 8000 \
  --features "tools" \
  --input-mods "text" \
  --output-mods "text" \
  --prefer-unmoderated \
  --limit 10 \
  --log-level INFO \
  --output brief \
  --no-auto \
  --name-filter "llama"
```

"""

import re

import click
from dotenv import load_dotenv

from src.openrouter_client.client import OpenRouterClient
from src.openrouter_client.models import ModelRequirements
from src.utils.logging import LoggerConfig, UnifiedLogger


@click.command()
@click.option(
    "--max-cost",
    type=float,
    default=0.0,
    help="Maximum cost per token (0 for no limit)",
)
@click.option(
    "--min-context", type=int, default=8000, help="Minimum context length required"
)
@click.option(
    "--features",
    type=str,
    default=None,
    help="Comma-separated list of required features",
)
@click.option(
    "--input-mods",
    type=str,
    default="text",
    help="Comma-separated list of input modalities",
)
@click.option(
    "--output-mods",
    type=str,
    default="text",
    help="Comma-separated list of output modalities",
)
@click.option(
    "--prefer-unmoderated",
    is_flag=True,
    default=False,
    help="Prefer unmoderated models",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Maximum number of models to return when listing",
)
@click.option("--log-level", type=str, default="INFO", help="Log level")
@click.option("--output", type=str, default="json", help="Output format (json, text)")
@click.option(
    "--no-auto",
    is_flag=True,
    default=True,
    help="Filter out openrouter auto-router model (openrouter/auto)",
)
@click.option("--name-filter", type=str, default=None, help="Filter models by name")
def main(
    max_cost: float,
    min_context: int,
    features: str,
    input_mods: str,
    output_mods: str,
    prefer_unmoderated: bool,
    limit: int,
    log_level: str,
    output: str,
    no_auto: bool,
    name_filter: str,
) -> None:
    """Select and display AI models based on specified requirements using OpenRouter."""
    # Initialize unified logger
    logger = UnifiedLogger(
        LoggerConfig(
            log_level=log_level,
            log_to_file=True,
            log_file_path="openrouter_model_selection.log",
            use_rich_console=True,
            minimal_console=output == "brief",
            terse=True,
        )
    )

    load_dotenv(override=True)
    # Initialize the client with the logger
    client = OpenRouterClient(logger=logger)

    # Parse comma-separated strings into lists
    feature_list = (
        None if features is None else [f.strip() for f in features.split(",")]
    )
    input_mod_list = [m.strip() for m in input_mods.split(",")]
    output_mod_list = [m.strip() for m in output_mods.split(",")]

    # Define requirements for model selection
    requirements = ModelRequirements(
        max_cost_per_token=max_cost,
        min_context_length=min_context,
        required_features=feature_list,
        input_modalities=input_mod_list,
        output_modalities=output_mod_list,
        prefer_unmoderated=prefer_unmoderated,
        force_refresh=True,
        exclude_models=["openrouter/auto"] if no_auto else None,
    )

    # Select a model based on requirements
    models = client.select_models(requirements, limit=limit)
    if name_filter:
        models = [
            model for model in models if name_filter.lower() in model.name.lower()
        ]

    if output == "text":
        logger.debug(f"Model selection requirements: {requirements}")
        if len(models) > 0:
            for i, model in enumerate(models, 1):
                # Extract model size parameter using regex
                parameters = re.search(r"\s(\d+(?:\.\d+)?B)\s", model.name)
                param_size = parameters.group(1) if parameters else "N/A"

                logger.info(f"{i}. {model.name} (ID: {model.id})")
                logger.info(f"  - Model size: {param_size}")
                logger.info(f"  - Context length: {model.context_length}")
                logger.info(
                    f"  - Pricing: {model.pricing.prompt} per prompt token, {model.pricing.completion} per completion token"
                )
                logger.info(
                    f"  - Supported parameters: {', '.join(model.supported_parameters)}"
                )
        else:
            logger.info("No model found matching the requirements")

    elif output == "json":
        model_json = [model.model_dump() for model in models]
        logger.print_json(model_json, "Selected Models")
    elif output == "brief":
        logger.debug(f"Model selection requirements: {requirements}")
        if len(models) > 0:
            for i, model in enumerate(models, 1):
                logger.info(f"{i}. {model.name} (ID: {model.id})")
        else:
            logger.info("No model found matching the requirements")


if __name__ == "__main__":
    main()
