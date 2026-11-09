"""Run the full trading-analysis pipeline against deterministic mock services."""

import json
import math
from typing import Any
from unittest.mock import patch

import pandas as pd

from main import _print_report, _run_pipeline
from research_agent import ResearchAgent


class MockGeminiClient:
    """Return fixed, valid planner and writer responses without network access."""

    def __init__(self) -> None:
        self._responses = iter(
            (
                json.dumps(
                    {
                        "symbol": "BTC/USDT",
                        "timeframes": ["1h", "4h"],
                        "indicators": ["RSI", "MACD", "ATR"],
                        "research_questions": [
                            "Czy momentum i trend są zgodne na obu interwałach?"
                        ],
                        "writer_instructions": (
                            "Podsumuj wynik i jasno opisz niepewność."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "signal": "KUP",
                        "rationale": (
                            "Przykładowy sygnał testowy; dane i wynik są symulowane."
                        ),
                    }
                ),
            )
        )

    def generate(self, prompt: str) -> str:
        try:
            return next(self._responses)
        except StopIteration as error:
            raise AssertionError("Pipeline wykonał nieoczekiwane wywołanie Gemini.") from error


class MockTicker:
    def __init__(self, symbol: str) -> None:
        if symbol != "BTC-USD":
            raise AssertionError(f"Nieoczekiwany symbol yfinance: {symbol}")

    def history(self, **_: Any) -> pd.DataFrame:
        periods = 600
        index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("h"),
            periods=periods,
            freq="h",
        )
        close = [
            60_000 + position * 8 + 120 * math.sin(position / 9)
            for position in range(periods)
        ]
        return pd.DataFrame(
            {
                "Open": [price - 5 for price in close],
                "High": [price + 25 for price in close],
                "Low": [price - 25 for price in close],
                "Close": close,
                "Volume": [12.5] * periods,
            },
            index=index,
        )

    def get_info(self) -> dict[str, Any]:
        return {}


def main() -> int:
    print(
        "DRY RUN — nie są wykonywane rzeczywiste żądania do Gemini ani Yahoo Finance."
    )
    offline_news = {
        "status": "no_verified_news",
        "verified_articles": [],
        "rejected_count": 0,
        "sentiment_score": 0.0,
        "verified_source_count": 0,
        "verification_method": "Mock offline.",
    }
    with (
        patch(
            "research_agent.get_testnet_klines",
            return_value=MockTicker("BTC-USD").history(),
        ),
        patch("research_agent.yf.Ticker", MockTicker),
        patch.object(ResearchAgent, "_fetch_news", return_value=offline_news),
        patch.object(
            ResearchAgent,
            "_fetch_order_book",
            return_value={
                "status": "unavailable",
                "reason": "Mock offline.",
            },
        ),
        patch.object(
            ResearchAgent,
            "_fetch_market_regime",
            return_value={
                "status": "unavailable",
                "benchmark": "BTC-USD",
                "reason": "Mock offline.",
            },
        ),
        patch.object(
            ResearchAgent,
            "_fetch_event_risk",
            return_value={
                "status": "incomplete",
                "corporate_calendar_status": "not_applicable",
                "macro_calendar_status": "unavailable",
                "upcoming_events": [],
                "reason": "Mock offline; brak kalendarza makro.",
            },
        ),
    ):
        report = _run_pipeline(
            goal="Sprawdź przykładową parę BTC/USDT",
            capital=10_000.0,
            candle_limit=60,
            gemini_client=MockGeminiClient(),  # type: ignore[arg-type]
        )

    assert report["symbol"] == "BTC-USD"
    assert report["signal"] in {"KUP", "SPRZEDAJ"}
    if report["signal"] == "KUP":
        assert report["entry_price"] > report["stop_loss"]
        assert report["take_profit"] > report["entry_price"]
    else:
        assert report["stop_loss"] > report["entry_price"]
        assert report["take_profit"] < report["entry_price"]
    assert "Zrealizuj cel, gdy cena osiągnie poziom" in report["report_text"]
    assert report["position_size"] > 0
    assert report["rationale"]
    assert 0 <= report["confidence_score_pct"] <= 60
    assert report["news"]["status"] == "no_verified_news"
    json.dumps(report, allow_nan=False)

    _print_report(report)
    print("\n✅ DRY RUN ZAKOŃCZONY — wszystkie etapy pipeline’u przeszły poprawnie.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
