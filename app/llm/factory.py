"""OpenAI model factory.

Notes:
- `api_key` is passed explicitly: pydantic-settings does NOT populate os.environ.
- gpt-6-luna requires `reasoning_effort="none"` to accept temperature=0 and tool calling.
  `with_structured_output` relies on tool calling / json_schema under the hood.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings


def _chat(settings: Settings, model: str) -> ChatOpenAI:
    kwargs: dict = {
        "model": model,
        "api_key": settings.openai_api_key,
        "temperature": 0,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.max_retries,
    }
    if settings.llm_reasoning_effort:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    return ChatOpenAI(**kwargs)


def get_chat_model(settings: Settings) -> ChatOpenAI:
    return _chat(settings, settings.llm_model)


def get_vision_model(settings: Settings) -> ChatOpenAI:
    return _chat(settings, settings.vision_model)


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embeddings_model, api_key=settings.openai_api_key)
