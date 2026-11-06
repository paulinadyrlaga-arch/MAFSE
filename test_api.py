"""Sprawdza połączenie z API Gemini."""

import os

from google.genai.errors import APIError, ServerError

from mafse import GeminiClient


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print("Brak klucza API. Ustaw zmienną środowiskową GEMINI_API_KEY.")
        return 1

    client: GeminiClient | None = None
    try:
        client = GeminiClient.from_env()
        response = client.generate("Odpowiedz jednym słowem: OK")
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        print(f"Połączenie z API Gemini działa (model: {model_name}).")
        if response:
            print(f"Odpowiedź: {response}")
        return 0
    except ServerError as error:
        if error.code == 503:
            print("API Gemini nadal zwraca HTTP 503 po trzech próbach.")
        else:
            print(f"Błąd serwera API Gemini (HTTP {error.code}): {error}")
        return 1
    except APIError as error:
        print(f"Błąd API Gemini (HTTP {error.code}): {error}")
        return 1
    except RuntimeError as error:
        print(str(error))
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
