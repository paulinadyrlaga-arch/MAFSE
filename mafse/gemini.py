"""Klient Gemini używany wspólnie przez agentów MAFSE."""

import logging
import os
from typing import Protocol

from google.genai.errors import ServerError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


_LOGGER = logging.getLogger(__name__)


def _is_retryable_gemini_error(error: BaseException) -> bool:
    return isinstance(error, ServerError) and error.code == 503


class GeminiResponse(Protocol):
    text: str


class GeminiModel(Protocol):
    def generate_content(self, prompt: str) -> GeminiResponse: ...

    def close(self) -> None: ...


class GeminiTextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiModels(Protocol):
    def generate_content(self, *, model: str, contents: str) -> GeminiResponse: ...


class GeminiAPIClient(Protocol):
    models: GeminiModels

    def close(self) -> None: ...


class _GenAIModel:
    def __init__(self, client: GeminiAPIClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def generate_content(self, prompt: str) -> GeminiResponse:
        return self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )

    def close(self) -> None:
        self._client.close()


class GeminiClient:
    """Prosty klient oparty na pakiecie google-genai."""

    def __init__(self, model: GeminiModel) -> None:
        self._model = model

    @classmethod
    def from_env(cls, timeout_ms: int = 4_000) -> "GeminiClient":
        if not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms musi być dodatnią liczbą całkowitą.")
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Brak klucza API. Ustaw zmienną środowiskową GEMINI_API_KEY."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Brak pakietu google-genai. "
                "Zainstaluj zależności poleceniem: "
                "python -m pip install -r requirements.txt"
            ) from exc

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        return cls(_GenAIModel(client, model_name))

    @retry(
        retry=retry_if_exception(_is_retryable_gemini_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text

    def close(self) -> None:
        self._model.close()
