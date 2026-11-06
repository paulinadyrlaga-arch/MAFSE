"""Orchestrate market research, risk sizing, planning, and the final report."""

import argparse
import json
import logging
import math
import sys
from typing import Any

from google.genai.errors import APIError, ServerError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mafse import GeminiClient
from planner_agent import PlannerAgent
from research_agent import ResearchAgent
from risk_manager import RiskManager
from writer_agent import WriterAgent


_LOGGER = logging.getLogger(__name__)


def _is_retryable_research_error(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "HTTP 503" in str(error)


def _analyze_retry_notice(retry_state: Any) -> None:
    error = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number
    delay = retry_state.next_action.sleep if retry_state.next_action else 0
    _LOGGER.warning(
        "Yahoo Finance zwróciło HTTP 503; ponawiam za %g s (próba %d/3): %s",
        delay,
        attempt + 1,
        error,
    )


@retry(
    retry=retry_if_exception(_is_retryable_research_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    before_sleep=_analyze_retry_notice,
    reraise=True,
)
def _analyze_with_retry(
    researcher: ResearchAgent,
    symbol: str,
    timeframe: str,
    limit: int,
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    return researcher.analyze(
        symbol,
        timeframe,
        limit,
        round_trip_cost_bps=round_trip_cost_bps,
    )


def _run_pipeline(
    goal: str,
    capital: float,
    candle_limit: int,
    gemini_client: GeminiClient,
    researcher: ResearchAgent | None = None,
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    if not goal.strip():
        raise ValueError("Cel handlowy nie może być pusty.")
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("Kapitał musi być skończoną wartością większą od zera.")

    client = gemini_client
    planner = PlannerAgent(client)
    writer = WriterAgent(client)
    market_researcher = researcher or ResearchAgent()

    print("1/4 Tworzenie planu analizy...")
    plan = planner.generate_plan(goal)
    symbol = plan["symbol"]
    timeframes = plan["research"]["timeframes"]
    if not timeframes:
        raise RuntimeError("Plan nie zawiera interwałów do zbadania.")

    print(f"2/4 Pobieranie świec dla {symbol}: {', '.join(timeframes)}...")
    analyses = [
        _analyze_with_retry(
            market_researcher,
            symbol,
            timeframe,
            candle_limit,
            round_trip_cost_bps,
        )
        for timeframe in timeframes
    ]
    primary = analyses[0]
    latest = primary["candles"][-1]
    atr = latest.get("atr")
    if not isinstance(atr, (int, float)) or atr <= 0:
        raise RuntimeError(
            f"Brak poprawnego ATR dla głównego interwału {timeframes[0]}."
        )

    research_data = dict(primary)
    research_data["timeframe_summaries"] = [
        {
            "timeframe": result["timeframe"],
            "latest_candle": result["candles"][-1],
        }
        for result in analyses
    ]

    print(f"3/4 Obliczanie poziomów ryzyka na interwale {timeframes[0]}...")
    risk_settings = plan["risk_management"]
    risk_manager = RiskManager(
        risk_fraction=float(risk_settings["risk_fraction"]),
        atr_multiplier=float(risk_settings["atr_multiplier"]),
        reward_risk_ratio=float(risk_settings["reward_risk_ratio"]),
    )
    risk_data = {
        direction: risk_manager.calculate_position(
            entry_price=float(latest["close"]),
            atr=float(atr),
            portfolio_capital=capital,
            direction=direction,
        )
        for direction in ("long", "short")
    }

    print("4/4 Generowanie raportu...")
    report = writer.create_trade_message(research_data, risk_data)
    report["latest_price"] = float(latest["close"])
    report["timeframe"] = primary["timeframe"]
    report["indicators"] = primary["indicators"]
    report["candles"] = primary["candles"]
    report["broker_inputs"] = primary["broker_inputs"]
    report["is_mock"] = primary["is_mock"]
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 52)
    print(report["report_text"])
    print("=" * 52)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uruchom analizę rynku z agentami Gemini i Yahoo Finance."
    )
    parser.add_argument(
        "goal",
        nargs="+",
        help="Cel analizy, np. 'Przeanalizuj parę BTC/USDT'.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        required=True,
        help="Całkowity kapitał portfela w USD.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Liczba świec na interwał (35–1000; domyślnie: 100).",
    )
    args = parser.parse_args()
    goal = " ".join(args.goal)

    if not 35 <= args.limit <= 1000:
        parser.error("--limit musi mieścić się w zakresie 35–1000.")
    if not math.isfinite(args.capital) or args.capital <= 0:
        parser.error("--capital musi być skończoną wartością większą od zera.")

    try:
        gemini_client = GeminiClient.from_env()
        try:
            report = _run_pipeline(
                goal=goal,
                capital=args.capital,
                candle_limit=args.limit,
                gemini_client=gemini_client,
            )
        finally:
            gemini_client.close()
    except ServerError as error:
        if error.code == 503:
            print(
                "Gemini jest chwilowo niedostępne (HTTP 503) po ponowieniu prób.",
                file=sys.stderr,
            )
        else:
            print(f"Błąd serwera Gemini (HTTP {error.code}): {error}", file=sys.stderr)
        return 1
    except APIError as error:
        print(f"Błąd API Gemini (HTTP {error.code}): {error}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as error:
        print(f"Nie udało się przygotować raportu: {error}", file=sys.stderr)
        return 1

    _print_report(report)
    print("\nJSON:")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
