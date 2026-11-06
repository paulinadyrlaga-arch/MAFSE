"""PlannerAgent generates a validated research and writing plan with Gemini."""

import json
import re
from typing import Any

from mafse.agent import Agent, AgentContext
from mafse.gemini import GeminiTextGenerator


class PlannerAgent(Agent):
    """Create a JSON plan for downstream research and writing agents."""

    name = "planista"

    _TIMEFRAMES = {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    }
    _INDICATORS = {"RSI", "MACD", "ATR"}
    _RISK_ASSUMPTIONS: dict[str, float | bool] = {
        "risk_per_trade_pct": 1.5,
        "stop_loss_atr_multiplier": 1.5,
        "minimum_reward_risk_ratio": 2.0,
        "leverage_allowed": False,
    }
    _PAIR_PATTERN = re.compile(
        r"\b([A-Z0-9]{2,20})\s*[/\-]\s*([A-Z0-9]{2,20})\b",
        re.IGNORECASE,
    )
    _EXPLICIT_TICKER_PATTERN = re.compile(
        r"\bticker\s*:\s*([A-Z][A-Z0-9.^=_-]{0,19})\b",
        re.IGNORECASE,
    )

    def __init__(self, client: GeminiTextGenerator) -> None:
        self.client = client

    def generate_plan(self, goal: str) -> dict[str, Any]:
        """Return a structured plan, rejecting malformed model output."""
        normalized_goal = goal.strip()
        if not normalized_goal:
            raise ValueError("Cel handlowy nie może być pusty.")
        requested_symbol = self._symbol_from_goal(normalized_goal)

        prompt = (
            "Przygotuj ustrukturyzowany plan analizy rynku dla "
            "ResearchAgenta i RiskManagera. Określ horyzont i ramy czasowe "
            "odpowiednie do celu: dla swing tradingu preferuj 4h i 1d, "
            "ewentualnie 1w jako kontekst trendu; dla krótkoterminowej analizy "
            "dobierz krótsze interwały. Używaj tylko interwałów obsługiwanych "
            "przez Yahoo Finance: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, "
            "12h, 1d, 3d, 1w, 1M. "
            "To ma być plan analizy, nie rekomendacja kupna ani sprzedaży. "
            "Nie wymyślaj danych rynkowych. "
            "Zwróć wyłącznie poprawny JSON, bez markdown i bez dodatkowego "
            "tekstu, z dokładnie następującymi polami: "
            '{"symbol":"BTC-USD","timeframes":["4h","1d"],'
            '"indicators":["RSI","MACD","ATR"],'
            '"research_questions":["..."],'
            '"writer_instructions":"..."}. '
            "symbol ma być tickerem Yahoo Finance. Dla par krypto użyj "
            "formatu BASE-QUOTE, a dla akcji samego symbolu, np. AAPL. "
            "Jeśli użytkownik podał parę, użyj jej zamiast wybierać inną; "
            "jeśli cel zawiera pole Ticker:, użyj dokładnie wskazanego "
            "tickera akcyjnego zamiast wybierać inną parę; "
            "USDT i USDC mapuj na USD, ponieważ Yahoo Finance nie udostępnia "
            "tych par jako tickerów krypto. Jeśli cel nie wskazuje pary, "
            "wybierz BTC-USD. indicators mogą zawierać "
            "wyłącznie RSI, MACD i ATR. Ustal rozsądny zestaw interwałów, "
            "wskaźników oraz pytań badawczych dla celu. Instrukcje dla "
            "WriterAgenta mają jasno zaznaczać niepewność i brak gwarancji "
            "wyników. Nie ustawiaj ani nie modyfikuj parametrów ryzyka; "
            "aplikacja przekaże je osobno do RiskManagera.\n\n"
            f"Cel użytkownika: {normalized_goal}"
        )
        response = self.client.generate(prompt)
        if not isinstance(response, str):
            raise RuntimeError("Gemini zwróciło odpowiedź w nieoczekiwanym formacie.")
        try:
            generated = json.loads(response)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Gemini nie zwróciło poprawnego JSON-u dla planu."
            ) from error

        validated = self._validate_generated_plan(generated)
        symbol = requested_symbol or validated["symbol"]
        return {
            "goal": normalized_goal,
            "symbol": symbol,
            "ticker": symbol,
            "trading_style": self._trading_style(normalized_goal),
            "time_horizon": self._time_horizon(
                self._trading_style(normalized_goal), validated["timeframes"]
            ),
            "research": {
                "timeframes": validated["timeframes"],
                "indicators": validated["indicators"],
                "questions": validated["research_questions"],
            },
            "risk_assumptions": dict(self._RISK_ASSUMPTIONS),
            "risk_management": {
                "risk_fraction": self._RISK_ASSUMPTIONS["risk_per_trade_pct"] / 100,
                "atr_multiplier": self._RISK_ASSUMPTIONS[
                    "stop_loss_atr_multiplier"
                ],
                "reward_risk_ratio": self._RISK_ASSUMPTIONS[
                    "minimum_reward_risk_ratio"
                ],
                "leverage_allowed": self._RISK_ASSUMPTIONS["leverage_allowed"],
            },
            "writer_instructions": validated["writer_instructions"],
        }

    def run(self, context: AgentContext) -> str:
        """Implement the MAFSE Agent interface and return JSON for next agents."""
        return json.dumps(
            self.generate_plan(context.task),
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def _validate_generated_plan(cls, value: Any) -> dict[str, Any]:
        expected_keys = {
            "symbol",
            "timeframes",
            "indicators",
            "research_questions",
            "writer_instructions",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise RuntimeError(
                "Plan Gemini musi zawierać dokładnie: symbol, timeframes, "
                "indicators, research_questions i writer_instructions."
            )

        symbol = value["symbol"]
        if not isinstance(symbol, str):
            raise RuntimeError("Pole symbol w planie Gemini musi być tekstem.")
        symbol = cls._normalize_symbol(symbol)
        timeframes = value["timeframes"]
        if (
            not isinstance(timeframes, list)
            or not 1 <= len(timeframes) <= 5
            or any(
                not isinstance(item, str) or item not in cls._TIMEFRAMES
                for item in timeframes
            )
            or len(set(timeframes)) != len(timeframes)
        ):
            raise RuntimeError(
                "timeframes musi zawierać od 1 do 5 unikalnych, "
                "obsługiwanych interwałów Yahoo Finance."
            )

        indicators = value["indicators"]
        if (
            not isinstance(indicators, list)
            or not 1 <= len(indicators) <= len(cls._INDICATORS)
            or any(
                not isinstance(item, str) or item not in cls._INDICATORS
                for item in indicators
            )
            or len(set(indicators)) != len(indicators)
        ):
            raise RuntimeError(
                "indicators musi zawierać unikalne wskaźniki spośród RSI, MACD i ATR."
            )

        questions = value["research_questions"]
        if (
            not isinstance(questions, list)
            or not 1 <= len(questions) <= 8
            or any(not isinstance(item, str) or not item.strip() for item in questions)
        ):
            raise RuntimeError(
                "research_questions musi zawierać od 1 do 8 niepustych pytań."
            )

        writer_instructions = value["writer_instructions"]
        if (
            not isinstance(writer_instructions, str)
            or not writer_instructions.strip()
        ):
            raise RuntimeError("writer_instructions musi być niepustym tekstem.")

        return {
            "symbol": symbol,
            "timeframes": timeframes,
            "indicators": indicators,
            "research_questions": [item.strip() for item in questions],
            "writer_instructions": writer_instructions.strip(),
        }

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        normalized = symbol.strip().upper()
        match = cls._PAIR_PATTERN.fullmatch(normalized)
        if match is None and re.fullmatch(r"[A-Z][A-Z0-9.^=]{0,19}", normalized):
            return normalized
        if match is None:
            raise RuntimeError(
                "Nieprawidłowy ticker Yahoo Finance."
            )
        base, quote = match.groups()
        if quote in {"USDT", "USDC"}:
            quote = "USD"
        return f"{base}-{quote}"

    @classmethod
    def _symbol_from_goal(cls, goal: str) -> str | None:
        match = cls._PAIR_PATTERN.search(goal)
        if match is not None:
            return cls._normalize_symbol("-".join(match.groups()))
        match = cls._EXPLICIT_TICKER_PATTERN.search(goal)
        if match is not None:
            return cls._normalize_symbol(match.group(1))
        return None

    @staticmethod
    def _trading_style(goal: str) -> str:
        normalized = goal.casefold()
        if "swing" in normalized:
            return "swing trading"
        if any(term in normalized for term in ("scalp", "scalping", "day trading", "intraday")):
            return "day trading"
        return "unspecified"

    @staticmethod
    def _time_horizon(style: str, timeframes: list[str]) -> str:
        if style == "swing trading":
            return "kilka dni do kilku tygodni"
        if style == "day trading":
            return "od kilku minut do końca dnia"
        return f"do oceny na interwałach: {', '.join(timeframes)}"
