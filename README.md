# CrewAI + OpenRouter.ai + Ollama == Free Agent Development

OpenRouter.ai is an API service that acts like an umbrella to several dozen LLM providers. A good deal of these LLMs are free for training purposes. You can use these to develop AI at no cost. Unfortunately OpenRouter does not include any form of embedding endpoints to use. Embedding is the conversion of knowledge/data for later retrieval. This is essentially how AI memory works so without this essential component your development efforts will be crippled.

This project works around that issue by using the ollama endpoint locally for embedding data and provides examples on how to use it with [CrewAI](https://www.crewai.com).

## Requirements

Create your own local .env file from the `.env_example` included and update the OpenRouter API key to be your own key. Other dependencies can be installed in macos/Linux using the included configuration script, `./configure.sh`.

> **NOTE** I use [mise](https://mise.jdx.dev/) for installing the required binaries here and recommend you install and use it if you do not already

Then start ollama locally and pull down and embedder to use. The full workflow is as follows:

```bash
# Install requirements
./configure.sh

# Configure ollama to run as a server and pull in the embedder used for storing 'memories'
ollama serve &
ollama pull mxbai-embed-large
```

## Finding a Free LLM

I've included a script you can use to query OpenRouter for LLMs of any sort. In our case we are looking for LLMs with zero cost but also include tools as a feature.

```bash
source ./.venv/bin/activate
python -m src.select-openrouter-model --max-cost 0 --limit 10 --output brief --features 'tools'
```

This should provide a list of models you can use for free that support tools capabilities.

```text
1. Mistral: Devstral Small (free) (ID: mistralai/devstral-small:free)
2. Meta: Llama 3.3 8B Instruct (free) (ID: meta-llama/llama-3.3-8b-instruct:free)
3. Meta: Llama 4 Maverick (free) (ID: meta-llama/llama-4-maverick:free)
4. Meta: Llama 4 Scout (free) (ID: meta-llama/llama-4-scout:free)
5. Google: Gemini 2.5 Pro Experimental (ID: google/gemini-2.5-pro-exp-03-25)
6. Mistral: Mistral Small 3.1 24B (free) (ID: mistralai/mistral-small-3.1-24b-instruct:free)
7. Meta: Llama 3.3 70B Instruct (free) (ID: meta-llama/llama-3.3-70b-instruct:free)
8. Mistral: Mistral 7B Instruct (free) (ID: mistralai/mistral-7b-instruct:free)
```

# Demo: Interactive Human-AI Chat with CrewAI

This demonstrates using the ollama embedder and openrouter.ai LLM with chainlit and crewai to prompt a user for more information.

## Overview

This script creates a conversational AI assistant that collects personal information through natural dialogue. Using CrewAI's agent framework and Chainlit's user interface, it demonstrates how to build interactive AI systems that gather specific information while maintaining a natural conversation flow.

## How It Works

When a user sends a message, two specialized AI agents work together:

- **Information Collector**: Asks follow-up questions to gather name and location details
- **Information Summarizer**: Transforms collected data into a natural, friendly summary

## Key Features

- Natural back-and-forth conversation with AI
- Dynamic follow-up questions when more context is needed
- Friendly web interface using Chainlit
- Structured information collection in a conversational format

## Running the Demo

```bash
source ./.venv/bin/activate
chainlit run ./src/human_input/crewai_chainlit_human_input.py
```

This example demonstrates how AI systems can be made more interactive by combining structured task workflows with natural human conversation.


# Good Take Away Knowledge

Some things to understand about all of this.

## CrewAI uses LiteLLM

CrewAI uses LiteLLM to proxy most connection requests to various LLM providers. This can lead to some confusing results when you go to run your crew and find that your manually passed information for models, endpoints, and keys to not work. This is due to how LiteLLM will source in environment variables and use them instead. In our case, we load a `.env` file into the current environment variables list of the current process which means they should align with the expected names of the LiteLLM provider. This means the variable names are not fungible. You must define them as the following for CrewAI to function properly for OpenRouter.

```
...
OPENROUTER_API_KEY="<your-key>"
OPENAI_MODEL_NAME="<model>"
OPENAI_API_BASE="https://openrouter.ai/api/v1"
...
```

## CrewAI Memory

CrewAI [Memory](https://docs.crewai.com/concepts/memory) is stored locally but still requires an embedding API endpoint to function. By default this is OpenAI's endpoint. Without modification you will be left with many errors about invalid credentials in your logs for OpenAI even when you aren't using it for your LLM calls.

In my examples I overwrite the memory target from a default location in your home directory to the local project, you can use some additional scripts to retrieve them.

```python

```