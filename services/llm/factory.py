import os

from services.llm.openai_client import OpenAICompatibleClient


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} не задан в переменных окружения")
    return value


def create_llm_client() -> OpenAICompatibleClient:
    """
    Создаёт LLM-клиент из переменных окружения:
      LLM_API_URL          — полный URL chat/completions, напр. http://10.0.0.5:8080/v1/chat/completions
      LLM_MODEL            — имя модели (как отдаёт GET /v1/models)
      LLM_API_KEY          — Bearer-токен (опционально)
      LLM_ENABLE_THINKING  — true/false, режим размышлений для thinking-моделей (по умолчанию false)
      LLM_MAX_TOKENS       — лимит токенов ответа (по умолчанию 4096)
      LLM_TEMPERATURE      — температура (по умолчанию 0.1)
    """
    return OpenAICompatibleClient(
        api_url=_require("LLM_API_URL"),
        model=_require("LLM_MODEL"),
        token=os.getenv("LLM_API_KEY") or None,
        enable_thinking=os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    )
