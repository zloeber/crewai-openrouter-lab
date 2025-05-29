from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Architecture(BaseModel):
    """Model architecture information."""

    input_modalities: List[str] = Field(
        default_factory=list,
        description="Input modalities supported by the model (e.g., text, image)",
    )
    output_modalities: List[str] = Field(
        default_factory=list,
        description="Output modalities supported by the model (e.g., text)",
    )
    tokenizer: Optional[str] = Field(
        None, description="Tokenizer type used by the model"
    )


class TopProvider(BaseModel):
    """Information about the top provider for this model."""

    is_moderated: bool = Field(False, description="Whether the provider is moderated")


class Pricing(BaseModel):
    """Pricing information for the model."""

    prompt: str = Field(..., description="Cost per token for prompt")
    completion: str = Field(..., description="Cost per token for completion")
    image: str = Field("0", description="Cost for image processing")
    request: str = Field("0", description="Cost per request")
    input_cache_read: str = Field("0", description="Cost for input cache read")
    input_cache_write: str = Field("0", description="Cost for input cache write")
    web_search: str = Field("0", description="Cost for web search")
    internal_reasoning: str = Field("0", description="Cost for internal reasoning")


class PerRequestLimits(BaseModel):
    """Per-request limits for the model."""

    key: Optional[str] = Field(None, description="Limit key")
    value: Optional[Any] = Field(None, description="Limit value")


class ModelMetadata(BaseModel):
    """Metadata for an OpenRouter AI model."""

    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Display name of the model")
    created: int = Field(..., description="Creation timestamp")
    description: str = Field(..., description="Model description")
    architecture: Architecture = Field(
        default_factory=Architecture, description="Model architecture information"
    )
    top_provider: TopProvider = Field(
        default_factory=TopProvider, description="Information about the top provider"
    )
    pricing: Pricing = Field(..., description="Pricing information for the model")
    context_length: int = Field(..., description="Maximum context length in tokens")
    hugging_face_id: Optional[str] = Field(
        None, description="Hugging Face model ID if applicable"
    )
    per_request_limits: Optional[Dict[str, Any]] = Field(
        None, description="Per-request limits"
    )
    supported_parameters: List[str] = Field(
        default_factory=list, description="Parameters supported by the model"
    )


class ModelListResponse(BaseModel):
    """Response from the OpenRouter model listing endpoint."""

    data: List[ModelMetadata] = Field(..., description="List of available models")


class ModelRequirements(BaseModel):
    """Requirements for model selection."""

    max_cost_per_token: Optional[float] = Field(
        None, description="Maximum cost per token"
    )
    min_context_length: Optional[int] = Field(
        None, description="Minimum required context length"
    )
    required_features: Optional[List[str]] = Field(
        None, description="Required features or parameters"
    )
    input_modalities: Optional[List[str]] = Field(
        None, description="Required input modalities"
    )
    output_modalities: Optional[List[str]] = Field(
        None, description="Required output modalities"
    )
    prefer_unmoderated: Optional[bool] = Field(
        False, description="Prefer unmoderated providers"
    )
    force_refresh: Optional[bool] = Field(
        True, description="Force refresh of model list and skip cache"
    )
    exclude_models: Optional[List[str]] = Field(
        None, description="Exclude models from the list"
    )
