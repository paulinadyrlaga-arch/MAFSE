"""Streamlit market scanner and detailed MAFSE analysis dashboard."""

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from google.genai.errors import APIError, ServerError
from google import genai
from google.genai import types

from mafse import GeminiClient
from main import _run_pipeline
from research_agent import ResearchAgent
from risk_manager import RiskManager
from writer_agent import WriterAgent


LOGGER = logging.getLogger(__name__)
US_STOCK_UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "WMT", "LLY", "V", "XOM", "MA", "ORCL", "COST", "NFLX", "JNJ",
    "PLTR", "AMD", "KO", "PG", "UNH", "HD", "ABBV", "BAC", "CRM", "CVX",
    "CSCO", "CAT", "MRK", "GE", "IBM", "AMAT", "QCOM", "MU", "NOW", "LIN",
    "TMO", "WFC", "ADBE", "DIS", "MCD", "RTX", "C", "GS", "AMGN", "PEP",
    "INTC", "UBER", "PFE", "LOW", "AXP", "SPGI", "TXN", "ISRG", "BKNG", "DE",
    "ACN", "ABT", "ANET", "APP", "ARM", "BX", "COIN", "CRWD", "DDOG", "DELL",
    "F", "GM", "INTU", "KLAC", "LRCX", "MSTR", "PANW", "PLD", "SNOW", "SMCI",
    "SHOP", "TGT", "TMUS", "TSM", "VZ", "PYPL", "SQ", "HOOD", "RDDT", "RIVN",
    "NKE", "BABA", "BMY", "MDT", "CVS", "COP", "SLB", "OXY", "PGR", "CB",
    "BK", "MS", "BLK", "SCHW", "ADP",     "GILD", "VRTX", "REGN",
)
EU_STOCK_UNIVERSE = (
    "ASML.AS", "SAP.DE", "RACE.MI", "MC.PA", "OR.PA", "TTE.PA", "SIE.DE",
    "ALV.DE", "DTE.DE", "BMW.DE", "VOW3.DE", "ADS.DE", "BAS.DE", "AIR.PA",
    "SAN.PA", "BNP.PA", "SU.PA", "ORA.PA", "ENEL.MI", "ISP.MI", "UCG.MI",
    "ABI.BR", "UNA.AS", "ING.AS", "SHEL.L", "AZN.L", "HSBA.L", "BP.L",
    "GSK.L", "ULVR.L", "RIO.L", "BARC.L", "RR.L", "REL.L", "NESN.SW",
    "NOVN.SW", "ROG.SW", "UBSG.SW", "SREN.SW", "NOVO-B.CO", "MAERSK-B.CO",
    "EQNR.OL", "DNB.OL", "VOLV-B.ST", "ERIC-B.ST", "NOKIA.HE", "KNEBV.HE",
    "CRH.L", "PRX.AS", "ADYEN.AS", "STLAM.MI", "PUM.DE", "IFX.DE", "MBG.DE",
    "HEIA.AS", "KER.PA", "RI.PA", "BN.PA", "CAP.PA", "VNA.DE", "ENI.MI",
    "BA.L", "LSEG.L", "III.L", "BATS.L", "IMB.L", "VOD.L",
)
STOCK_UNIVERSE = (*US_STOCK_UNIVERSE, *EU_STOCK_UNIVERSE)
CRYPTO_UNIVERSE = (
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "DOT-USD", "LTC-USD", "BCH-USD", "AVAX-USD", "LINK-USD", "XLM-USD",
    "TRX-USD", "SHIB-USD", "ETC-USD", "UNI-USD", "ATOM-USD", "NEAR-USD",
    "APT-USD", "FIL-USD", "ICP-USD",
)
MARKET_UNIVERSE = (*STOCK_UNIVERSE, *CRYPTO_UNIVERSE)
MARKET_OPTIONS = (
    "Wszystko",
    "Tylko Akcje (Revolut)",
    "Tylko Kryptowaluty (Revolut)",
)
SCAN_TIMEFRAME = "1d"
SCAN_CANDLE_LIMIT = 1000
SHORT_TERM_TIMEFRAME = "15m"
SHORT_TERM_CANDLE_LIMIT = 4000
NETWORK_TIMEOUT_SECONDS = 4.0
SCAN_MAX_WORKERS = 8
STAGE_ONE_MINIMUM_POINTS = 40.0
STAGE_TWO_MAX_ASSETS = 10
QUALIFICATION_THRESHOLD = 110.0
ROUND_TRIP_COST_BPS = 20.0
SHORT_TERM_CATEGORY = "⚡ KATEGORIA 2: Krótkoterminowe / Szybkie"
MULTI_DAY_CATEGORY = "📅 KATEGORIA 1: Kilkudniowe"
SHORT_TERM_NET_TARGET_RETURN = 0.02
SWING_NET_TARGET_RETURN = 0.045
REVOLUT_FEE_PER_SIDE = 0.001


def _symbols_for_market(market: str) -> tuple[str, ...]:
    if market == "Wszystko":
        return MARKET_UNIVERSE
    if market == "Tylko Akcje (Revolut)":
        return STOCK_UNIVERSE
    if market == "Tylko Kryptowaluty (Revolut)":
        return CRYPTO_UNIVERSE
    raise ValueError(f"Nieznany zakres rynku: {market}")


def _historical_snapshot(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock, tzinfo=ZoneInfo("Europe/Warsaw"))


def _market_session_instruction(symbol: str) -> str:
    if symbol in CRYPTO_UNIVERSE or symbol.endswith("-USD"):
        return "Krypto: handel 24/7"
    if symbol in US_STOCK_UNIVERSE:
        return (
            "USA: regularna sesja zwykle od 15:30 czasu PL; "
            "uwzględnij tygodnie zmiany czasu"
        )
    return "Akcje: tylko w godzinach sesji właściwej giełdy"


def _price_currency(symbol: str) -> str:
    if symbol in CRYPTO_UNIVERSE or symbol.endswith("-USD"):
        return "USD"
    suffix = symbol.rpartition(".")[2]
    return {
        "AS": "EUR",
        "DE": "EUR",
        "PA": "EUR",
        "MI": "EUR",
        "BR": "EUR",
        "HE": "EUR",
        "L": "GBp",
        "SW": "CHF",
        "CO": "DKK",
        "OL": "NOK",
        "ST": "SEK",
    }.get(suffix, "USD" if symbol in US_STOCK_UNIVERSE else "waluta Yahoo")


def _buy_trigger_instruction(symbol: str, entry_price: Any) -> str:
    level = _format_market_price(entry_price)
    currency = _price_currency(symbol)
    if symbol in CRYPTO_UNIVERSE or symbol.endswith("-USD"):
        return f"Jeśli kurs osiągnie ≥ {level} {currency}: KUP (rynek 24/7)"
    if symbol in US_STOCK_UNIVERSE:
        return (
            f"Po otwarciu sesji USA, zwykle 15:30 czasu PL*: "
            f"jeśli kurs ≥ {level} {currency}: KUP"
        )
    return (
        f"W godzinach sesji właściwej giełdy: "
        f"jeśli kurs ≥ {level} {currency}: KUP"
    )


def _goal_for_asset(symbol: str) -> str:
    if "-" in symbol:
        return (
            f"Przeanalizuj parę {symbol} pod kątem swing tradingu "
            "i horyzontu od 1 do 3 dni"
        )
    return (
        f"Przeanalizuj ticker: {symbol} pod kątem swing tradingu "
        "i horyzontu od 1 do 3 dni"
    )


def _normalize_user_symbol(value: str) -> str:
    symbol = ResearchAgent._normalize_symbol(value)
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^=-]{0,19}", symbol):
        raise ValueError("Ticker ma nieprawidłowy format.")
    return symbol


def _format_price(value: Any) -> str:
    if value is None:
        return "—"
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${price:,.2f}" if math.isfinite(price) else "—"


def _format_market_price(value: Any) -> str:
    if value is None:
        return "—"
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(price):
        return "—"
    decimals = 8 if abs(price) < 1 else 2
    return f"{price:,.{decimals}f}".rstrip("0").rstrip(".")


def _format_number(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.{decimals}f}" if math.isfinite(number) else "—"


def _chart_data(report: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    candles = report.get("candles", [])
    if not candles:
        return pd.DataFrame(), False

    frame = pd.DataFrame(candles)
    if "timestamp" not in frame or "close" not in frame:
        raise ValueError("Dane świec nie zawierają timestamp albo close.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp")
    frame = frame[["close"]].rename(columns={"close": "Cena zamknięcia"})

    signal = report.get("signal")
    levels: dict[str, float] = {}
    if signal in {"KUP", "SPRZEDAJ"}:
        for key, label in (
            ("entry_price", "Wejście"),
            ("stop_loss", "Stop-Loss"),
            ("trailing_stop", "Trailing Stop"),
            ("take_profit", "Take-Profit"),
        ):
            value = report.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                levels[label] = float(value)
        frame = frame.assign(**{label: value for label, value in levels.items()})
        return frame, False

    indicators = report.get("indicators", {})
    price = report.get("latest_price")
    atr = indicators.get("atr_14") if isinstance(indicators, dict) else None
    if (
        isinstance(price, (int, float))
        and isinstance(atr, (int, float))
        and math.isfinite(price)
        and math.isfinite(atr)
        and price > 0
        and atr > 0
    ):
        try:
            manager = RiskManager()
            for direction, label in (("long", "KUP"), ("short", "SPRZEDAJ")):
                risk = manager.calculate_position(
                    entry_price=float(price),
                    atr=float(atr),
                    portfolio_capital=float(price),
                    direction=direction,
                )
                levels[f"SL {label} (1,5×ATR)"] = float(risk["stop_loss"])
                levels[f"TP {label} (R:R ≥ 1:2)"] = float(risk["take_profit"])
        except ValueError:
            LOGGER.info("Nie można wyznaczyć scenariuszowych poziomów ryzyka dla wykresu.")
    if levels:
        frame = frame.assign(**levels)
    return frame, bool(levels)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_gemini_news_review(
    symbol: str,
    headlines_json: str,
    model_name: str,
    api_key_fingerprint: str,
) -> dict[str, Any]:
    """Cache only successful Gemini source-consistency and sentiment reviews."""
    del api_key_fingerprint
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Brak GEMINI_API_KEY; używam neutralnego sentymentu.")
    prompt = (
        "Oceń wyłącznie dostarczone, wcześniej skorelowane nagłówki i źródła. "
        "Nie dodawaj faktów ani nie traktuj ich jako niezależnie potwierdzonych "
        "poza podanymi źródłami. Jeśli relacje nie wspierają wspólnego twierdzenia, "
        "zwróć supported=false i neutralny sentyment. Zwróć wyłącznie JSON: "
        '{"supported":true|false,"sentiment_score":-1.0..1.0,'
        '"rationale":"krótko po polsku"}.\n'
        f"Aktywo: {symbol}\nNagłówki i źródła:\n{headlines_json}"
    )
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(NETWORK_TIMEOUT_SECONDS * 1000)),
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
    except Exception as error:
        status = getattr(error, "code", None)
        is_rate_limited = status == 429
        is_timeout = isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold()
        LOGGER.warning(
            "Gemini quality review dla %s pominięty (HTTP %s / %s): %s",
            symbol,
            status or ("timeout" if is_timeout else "błąd"),
            type(error).__name__,
            error,
        )
        raise RuntimeError(
            "Gemini quality review niedostępny "
            f"(HTTP {status or ('timeout' if is_timeout else 'błąd')}"
            f"{' rate limit' if is_rate_limited else ''})."
        ) from error
    finally:
        client.close()

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemini zwróciło pustą odpowiedź; używam oceny neutralnej.")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Gemini zwróciło niepoprawny JSON; używam oceny neutralnej."
        ) from error
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("supported"), bool)
        or not isinstance(result.get("sentiment_score"), (int, float))
        or isinstance(result.get("sentiment_score"), bool)
        or not math.isfinite(result["sentiment_score"])
        or not -1 <= result["sentiment_score"] <= 1
        or not isinstance(result.get("rationale"), str)
    ):
        raise RuntimeError(
            "Gemini zwróciło niepoprawny format oceny; używam oceny neutralnej."
        )
    return {
        "status": "reviewed",
        "supported": result["supported"],
        "sentiment_score": float(result["sentiment_score"]),
        "rationale": result["rationale"][:500],
    }


def _review_candidate_news(
    symbol: str,
    news: dict[str, Any],
) -> dict[str, Any]:
    articles = news.get("verified_articles", [])
    if not articles:
        return {
            "status": "no_verified_news",
            "supported": False,
            "sentiment_score": 0.0,
            "rationale": "Brak nagłówków potwierdzonych przez niezależne źródła.",
        }
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    key_fingerprint = hashlib.sha256(key.encode()).hexdigest() if key else "missing"
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    try:
        review = _cached_gemini_news_review(
            symbol,
            json.dumps(articles, ensure_ascii=False, sort_keys=True),
            model_name,
            key_fingerprint,
        )
    except RuntimeError as error:
        LOGGER.warning("Neutralny fallback sentymentu %s: %s", symbol, error)
        return {
            "status": "neutral_fallback",
            "supported": False,
            "sentiment_score": 0.0,
            "rationale": str(error),
        }
    if not review["supported"]:
        review["sentiment_score"] = 0.0
        review["status"] = "unsupported"
    return review


def _scan_item(
    symbol: str,
    research: dict[str, Any],
    score: dict[str, Any],
    technical_and_risk_points: float,
    round_trip_cost_bps: float = 40.0,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    signal = research["signal"]
    if (
        research["news"]["status"] == "deferred"
        or research["fundamentals"]["status"] == "deferred"
    ):
        confidence = {
            "score_pct": None,
            "components": {},
            "note": "Confidence Score zostanie wyliczony po etapie jakościowym.",
        }
    else:
        confidence = WriterAgent.calculate_confidence(research, signal)
    reasons = list(score["disqualification_reasons"])
    if signal != "KUP":
        reasons.append("Skan celu +4,5% jest long-only i wymaga sygnału KUP.")
    price = float(research["price"])
    atr = float(research["indicators"]["atr_14"])
    cost_fraction = round_trip_cost_bps / 20_000
    target_probability = score["filters"]["target_resistance_order_book"][
        "details"
    ]["target_probability_heuristic_pct"]
    stop_loss = price - 1.5 * atr
    return {
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "target_probability_pct": target_probability,
        "timeframe": research["timeframe"],
        "indicators": research["indicators"],
        "confidence_score_pct": confidence["score_pct"],
        "confidence_components": confidence["components"],
        "confidence_note": confidence["note"],
        "broker_score": score,
        "broker_points": score["total_points"],
        "technical_and_risk_points": technical_and_risk_points,
        "qualified": score["qualified"],
        "disqualification_reasons": reasons,
        "news": research["news"],
        "fundamentals": research["fundamentals"],
        "broker_inputs": research["broker_inputs"],
        "is_mock": research["is_mock"],
        "timestamp": research["timestamp"],
        "data_as_of": (
            as_of.astimezone(timezone.utc).isoformat()
            if as_of is not None
            else research.get("as_of")
        ),
        "candles": research["candles"],
        "entry_price": price,
        "take_profit": (
            price * (1 + cost_fraction) * 1.045 / (1 - cost_fraction)
        ),
        "stop_loss": stop_loss if stop_loss > 0 else None,
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
    }


def _finish_candidate(
    researcher: ResearchAgent,
    candidate: tuple[str, dict[str, Any], dict[str, Any], float],
    round_trip_cost_bps: float,
) -> tuple[dict[str, Any], bool]:
    symbol, research, _, stage_one_points = candidate
    enriched = researcher.enrich_qualitative(
        research,
        include_fundamentals=False,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    review = _review_candidate_news(symbol, enriched["news"])
    news = enriched["news"]
    news["gemini_review"] = review
    news["sentiment_score"] = (
        review["sentiment_score"] if review["status"] == "reviewed" else 0.0
    )
    score = RiskManager().calculate_broker_filter_score(
        enriched,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    item = _scan_item(
        symbol,
        enriched,
        score,
        stage_one_points,
        round_trip_cost_bps,
    )
    if score["qualified"] and enriched["signal"] == "KUP":
        manager = RiskManager()
        indicators = enriched["indicators"]
        liquidity = enriched["broker_inputs"]["liquidity"]
        position = manager.calculate_revolut_position(
            purchase_price=float(enriched["price"]),
            current_price=float(enriched["price"]),
            atr=float(indicators["atr_14"]),
            technical_signal=str(enriched["signal"]),
            sentiment_score=(
                float(enriched["news"]["sentiment_score"])
                if enriched["news"]["status"] == "verified"
                else None
            ),
            resistance_price=enriched["broker_inputs"]["target_path"].get(
                "resistance_price"
            ),
            rsi=float(indicators["rsi_14"]),
            macd=float(indicators["macd"]),
            macd_signal=float(indicators["macd_signal"]),
            round_trip_cost_bps=round_trip_cost_bps,
            adx=float(indicators["adx_14"]),
            session_volume_ratio=liquidity.get("session_volume_ratio"),
        )
        risk_reward_ratio = float(
            score["filters"]["risk_reward_costs"]["details"][
                "risk_reward_ratio_net"
            ]
        )
        item.update(
            {
                "entry_price": position["purchase_price"],
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"],
                "gross_target_return_pct": position["gross_target_return_pct"],
                "net_target_return_pct": position["net_target_return_pct"],
                "estimated_net_stop_return_pct": position[
                    "estimated_net_stop_return_pct"
                ],
                "round_trip_cost_bps_assumption": round_trip_cost_bps,
                "risk_reward_ratio": risk_reward_ratio,
            }
        )
    return item, bool(score["qualified"] and enriched["signal"] == "KUP")


def _per_side_fee(round_trip_cost_bps: float) -> float:
    return max(REVOLUT_FEE_PER_SIDE, round_trip_cost_bps / 20_000)


def _net_potential_profit_pct(
    entry: float,
    take_profit: float,
    round_trip_cost_bps: float,
) -> float:
    fee = _per_side_fee(round_trip_cost_bps)
    invested = entry * (1 + fee)
    proceeds = take_profit * (1 - fee)
    return (proceeds / invested - 1) * 100 if invested > 0 else 0.0


def _trade_levels(
    entry: float,
    atr: float,
    net_target: float,
    round_trip_cost_bps: float,
) -> dict[str, float]:
    fee = _per_side_fee(round_trip_cost_bps)
    fee_in = entry * fee
    take_profit = entry * (1 + fee) * (1 + net_target) / (1 - fee)
    stop_loss = entry - 1.5 * atr
    if stop_loss <= 0:
        stop_loss = entry * 0.98
    return {
        "entry_price": entry,
        "entry_fee": fee_in,
        "take_profit": take_profit,
        "potential_profit_pct": _net_potential_profit_pct(
            entry,
            take_profit,
            round_trip_cost_bps,
        ),
        "stop_loss": stop_loss,
        "trailing_stop": stop_loss,
        "trailing_stop_activation_price": entry + atr,
        "trailing_stop_rule": (
            "Początkowy SL = 1,5×ATR; po wzroście o 1×ATR stop podąża "
            "1,5×ATR za maksimum od momentu wejścia i nie obniża się."
        ),
        "net_target_return_pct": net_target * 100,
    }


def _net_risk_reward(
    entry: float,
    take_profit: float,
    stop_loss: float,
    round_trip_cost_bps: float,
) -> float:
    fee = _per_side_fee(round_trip_cost_bps)
    entry_cost = entry * (1 + fee)
    risk = entry_cost - stop_loss * (1 - fee)
    reward = take_profit * (1 - fee) - entry_cost
    return max(0.0, reward) / risk if risk > 0 else 0.0


def _analyst_consensus_points(
    research: dict[str, Any],
    maximum: float,
) -> tuple[float, str]:
    consensus = research.get("analyst_consensus")
    if not isinstance(consensus, dict) or consensus.get("status") != "available":
        return 0.0, "Brak dostępnego konsensusu analityków; nie dodano punktów."
    try:
        rating = float(consensus["consensus_score"])
    except (KeyError, TypeError, ValueError):
        return 0.0, "Brak poprawnego wyniku konsensusu analityków."
    if not math.isfinite(rating):
        return 0.0, "Brak poprawnego wyniku konsensusu analityków."
    points = maximum * max(0.0, min(1.0, (rating - 1.0) / 4.0))
    label = str(consensus.get("recommendation", "brak oceny"))
    count = consensus.get("analyst_count", 0)
    return points, f"Konsensus Yahoo Finance: {label}, ocena {rating:.2f}/5, {count} opinii."


def _macro_clear(research: dict[str, Any]) -> tuple[bool, str]:
    calendar = research.get("macro_calendar")
    if not isinstance(calendar, dict):
        return False, "Brak wiarygodnego kalendarza makro."
    if calendar.get("status") == "high_risk":
        event_titles = ", ".join(
            str(event.get("title", "wydarzenie"))
            for event in calendar.get("upcoming_events", [])
            if isinstance(event, dict)
        )
        return (
            False,
            f"Wykryto wydarzenie makro wysokiego ryzyka: {event_titles} "
            f"(źródło: {calendar.get('source', 'kalendarz makro')}).",
        )
    return (
        calendar.get("status") == "clear",
        f"Brak wysokiego ryzyka w kalendarzu makro "
        f"(źródło: {calendar.get('source', 'niepodane')})."
        if calendar.get("status") == "clear"
        else str(
            calendar.get(
                "reason",
                "Kalendarz makro niekompletny lub niedostępny; punkty bezpieczeństwa = 0.",
            )
        ),
    )


def _verified_news_points(
    research: dict[str, Any],
    maximum: float,
) -> tuple[float, str]:
    news = research.get("news")
    if (
        not isinstance(news, dict)
        or news.get("status") != "verified"
        or not news.get("verified_articles")
    ):
        return 0.0, "Brak wiadomości potwierdzonych przez dwa niezależne źródła."
    try:
        sentiment = float(news["sentiment_score"])
    except (KeyError, TypeError, ValueError):
        return 0.0, "Brak poprawnej oceny sentymentu wiadomości."
    if not math.isfinite(sentiment):
        return 0.0, "Brak poprawnej oceny sentymentu wiadomości."
    points = maximum * max(0.0, min(1.0, (sentiment + 1.0) / 2.0))
    return points, f"Zweryfikowany sentyment wiadomości: {sentiment:+.2f}."


def _score_row(
    label: str,
    points: float,
    maximum: float,
    details: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "points": round(max(0.0, min(maximum, points)), 2),
        "maximum": maximum,
        "details": details,
    }


def _short_term_relative_volume(
    candles: list[dict[str, Any]],
) -> tuple[float | None, int]:
    if len(candles) < 21:
        return None, 0
    latest = candles[-1]
    latest_volume = latest.get("volume")
    try:
        latest_time = pd.Timestamp(latest["timestamp"])
        latest_volume = float(latest_volume)
    except (KeyError, TypeError, ValueError):
        return None, 0
    if not math.isfinite(latest_volume) or latest_volume < 0:
        return None, 0

    reference: list[float] = []
    for candle in reversed(candles[:-1]):
        try:
            candle_time = pd.Timestamp(candle["timestamp"])
            volume = float(candle["volume"])
        except (KeyError, TypeError, ValueError):
            continue
        if candle_time.date() == latest_time.date():
            continue
        same_time_slot = (
            candle_time.hour == latest_time.hour
            and candle_time.minute == latest_time.minute
        )
        if same_time_slot and math.isfinite(volume) and volume >= 0:
            reference.append(volume)
            if len(reference) == 20:
                break
    if len(reference) < 20:
        return None, len(reference)
    average_volume = sum(reference) / len(reference)
    if average_volume <= 0:
        return None, len(reference)
    return latest_volume / average_volume, len(reference)


def _short_term_vwap_and_poc(
    candles: list[dict[str, Any]],
) -> tuple[float | None, float | None, bool, bool]:
    if not candles:
        return None, None, False, False
    try:
        latest_time = pd.Timestamp(candles[-1]["timestamp"])
        session_candles = [
            candle
            for candle in candles
            if pd.Timestamp(candle["timestamp"]).date() == latest_time.date()
        ]
        valid = [
            (
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
                float(candle["open"]),
                float(candle["volume"]),
            )
            for candle in session_candles
            if candle.get("volume") is not None
        ]
    except (KeyError, TypeError, ValueError):
        return None, None, False, False
    if not valid:
        return None, None, False, False
    total_volume = sum(row[4] for row in valid)
    if total_volume <= 0:
        return None, None, False, False
    typical_prices = [(high + low + close) / 3 for high, low, close, _, _ in valid]
    vwap = sum(price * row[4] for price, row in zip(typical_prices, valid)) / total_volume
    closes = [row[2] for row in valid]
    vwap_rising = len(closes) >= 2 and closes[-1] > closes[0]
    price = valid[-1][2]
    vwap_points = (
        price >= vwap
        and (price / vwap - 1) <= 0.005
        and vwap_rising
    )

    bin_width = max(price * 0.001, (max(row[0] - row[1] for row in valid)) / 10)
    volume_by_bin: dict[int, float] = {}
    for typical_price, row in zip(typical_prices, valid):
        bucket = int(typical_price / bin_width)
        volume_by_bin[bucket] = volume_by_bin.get(bucket, 0.0) + row[4]
    poc_bucket = max(volume_by_bin, key=volume_by_bin.get)
    poc = (poc_bucket + 0.5) * bin_width
    latest_low, latest_close, latest_open = valid[-1][1], valid[-1][2], valid[-1][3]
    poc_bounce = (
        len(valid) >= 2
        and latest_low <= poc * 1.0025
        and latest_close > poc
        and latest_close > latest_open
    )
    return vwap, poc, vwap_points, poc_bounce


def _short_term_item(
    symbol: str,
    research: dict[str, Any],
    round_trip_cost_bps: float,
) -> dict[str, Any]:
    indicators = research["indicators"]
    candles = research.get("candles", [])
    price = float(research["price"])
    rsi = float(indicators["rsi_14"])
    atr = float(indicators["atr_14"])
    relative_volume, relative_volume_samples = _short_term_relative_volume(candles)
    relvol_points = (
        30.0
        if relative_volume is not None and relative_volume >= 2.0
        else 20.0
        if relative_volume is not None and relative_volume >= 1.5
        else 10.0
        if relative_volume is not None and relative_volume >= 1.0
        else 0.0
    )
    vwap, poc, above_vwap, poc_bounce = _short_term_vwap_and_poc(candles)
    vwap_points = 15.0 if above_vwap else 0.0
    poc_points = 10.0 if poc_bounce else 0.0
    histogram_value = indicators.get("macd_histogram")
    macd_histogram = (
        float(histogram_value)
        if isinstance(histogram_value, (int, float))
        else 0.0
    )
    prior_histogram = (
        candles[-2].get("macd_histogram")
        if len(candles) > 1
        else None
    )
    histogram_rising = (
        isinstance(prior_histogram, (int, float))
        and macd_histogram > float(prior_histogram)
    )
    rsi_points = 10.0 if 50 <= rsi <= 70 else 0.0
    histogram_points = 10.0 if histogram_rising else 0.0

    prior_candles = candles[-21:-1]
    resistance_values = [
        float(candle["high"])
        for candle in prior_candles
        if isinstance(candle.get("high"), (int, float))
    ]
    resistance = max(resistance_values) if resistance_values else None
    trigger = max(price, resistance) if resistance is not None else price
    levels = _trade_levels(
        trigger,
        atr,
        SHORT_TERM_NET_TARGET_RETURN,
        round_trip_cost_bps,
    )
    risk_reward = _net_risk_reward(
        trigger,
        levels["take_profit"],
        levels["stop_loss"],
        round_trip_cost_bps,
    )
    risk_reward_points = (
        15.0
        if risk_reward >= 2.0
        else 10.0
        if risk_reward >= 1.5
        else 0.0
    )
    session_timing_points = 0.0
    try:
        local_time = pd.Timestamp(research["timestamp"]).tz_convert(
            "Europe/Warsaw"
        ).time()
        if time(14, 0) <= local_time <= time(15, 30):
            session_timing_points = 10.0
    except (KeyError, TypeError, ValueError):
        pass
    macro_is_clear, macro_details = _macro_clear(research)
    macro_points = 10.0 if macro_is_clear else 0.0
    consensus_points, consensus_details = _analyst_consensus_points(
        research,
        10.0,
    )
    news_points, news_details = _verified_news_points(research, 10.0)
    analyst_points = consensus_points + news_points
    trailing_points = 20.0 if math.isfinite(atr) and atr > 0 else 0.0
    trailing_details = (
        "Reguła ATR trailing stop skonfigurowana; time-stop zamyka pozycję "
        "najpóźniej na koniec sesji."
        if trailing_points
        else "Nie można uruchomić trailing stop bez poprawnego ATR."
    )
    time_stop_rule = (
        "Zamknij pozycję najpóźniej przed końcem bieżącej sesji."
    )

    breakdown = [
        _score_row(
            "RelVol",
            relvol_points,
            30,
            (
                f"{relative_volume:.2f}× względem tej samej pory z "
                f"{relative_volume_samples} poprzednich sesji"
                if relative_volume is not None
                else f"Brak 20 porównywalnych sesji (próbki: {relative_volume_samples})."
            ),
        ),
        _score_row(
            "VWAP / POC",
            vwap_points + poc_points,
            25,
            (
                f"VWAP={_format_market_price(vwap)}; POC="
                f"{_format_market_price(poc)}; ponad VWAP={above_vwap}; "
                f"odbicie od POC={poc_bounce}"
            ),
        ),
        _score_row(
            "Szybki impet",
            rsi_points + histogram_points,
            20,
            f"RSI(14)={rsi:.2f}; rosnący histogram MACD={histogram_rising}.",
        ),
        _score_row(
            "Risk/Reward",
            risk_reward_points,
            15,
            f"R/R={risk_reward:.2f}; TP={levels['take_profit']:.8g}; "
            f"SL={levels['stop_loss']:.8g}.",
        ),
        _score_row(
            "Opinie analityków / buzz",
            analyst_points,
            20,
            f"{consensus_details} {news_details}",
        ),
        _score_row(
            "Timing / filtr makro",
            session_timing_points + macro_points,
            20,
            (
                "W oknie 14:00–15:30 czasu PL."
                if session_timing_points
                else "Poza oknem 14:00–15:30 czasu PL lub brak czasu świecy."
            )
            + f" {macro_details}",
        ),
        _score_row(
            "Trailing Stop / Time-Stop",
            trailing_points,
            20,
            f"{trailing_details} {time_stop_rule}",
        ),
    ]
    score = 0.0 if relative_volume is None or relative_volume < 1.0 else sum(
        row["points"] for row in breakdown
    )
    live_quality_reviewed = bool(research.get("quality_reviewed", False))
    has_qualitative_sentiment = (
        consensus_points > 0
        or news_points > 0
        or (
            isinstance(research.get("analyst_consensus"), dict)
            and research["analyst_consensus"].get("status") == "available"
        )
        or (
            isinstance(research.get("news"), dict)
            and research["news"].get("status") == "verified"
            and bool(research["news"].get("verified_articles"))
        )
    )
    qualified = (
        not research.get("is_mock", False)
        and relative_volume is not None
        and relative_volume >= 1.0
        and risk_reward >= 2.0
        and score >= QUALIFICATION_THRESHOLD
        and (research.get("is_historical", False) or (
            live_quality_reviewed and has_qualitative_sentiment
        ))
    )
    reasons = []
    if relative_volume is None or relative_volume < 1.0:
        reasons.append("RelVol poniżej 1,0 lub brak 20 porównywalnych sesji; aktywo odrzucono.")
        breakdown = [
            {
                **row,
                "points": 0.0,
                "details": row["details"] + " Wynik całkowity wyzerowany przez filtr RelVol.",
            }
            for row in breakdown
        ]
    if risk_reward < 2.0:
        reasons.append("Risk/Reward jest niższy niż wymagane minimum 1:2.")
    if score < QUALIFICATION_THRESHOLD:
        reasons.append(
            f"Wynik krótkoterminowy jest niższy niż {QUALIFICATION_THRESHOLD:.0f}/150."
        )
    if not research.get("is_historical", False) and not live_quality_reviewed:
        reasons.append("Brak przeglądu opinii i sentymentu dla kandydata.")
    if (
        not research.get("is_historical", False)
        and live_quality_reviewed
        and not has_qualitative_sentiment
    ):
        reasons.append("Brak wiarygodnych danych opinii analityków lub wiadomości.")
    if research.get("is_mock", False):
        reasons.append("Dane syntetyczne.")
    return {
        "symbol": symbol,
        "signal": "KUP" if qualified else "CZEKAJ",
        "price": price,
        "timestamp": research.get("timestamp"),
        "timeframe": SHORT_TERM_TIMEFRAME,
        "indicators": indicators,
        "candles": candles,
        "is_mock": bool(research.get("is_mock", False)),
        "relative_volume": relative_volume,
        "relative_volume_samples": relative_volume_samples,
        "vwap": vwap,
        "poc": poc,
        "resistance_price": resistance,
        "broker_points": round(score, 2),
        "score_breakdown": breakdown,
        "target_probability_pct": round(max(0.0, min(100.0, score / 1.5)), 1),
        "target_return_pct": SHORT_TERM_NET_TARGET_RETURN * 100,
        **levels,
        "risk_reward_ratio": risk_reward,
        "trailing_stop_rule": levels["trailing_stop_rule"],
        "trailing_stop_activation_price": levels[
            "trailing_stop_activation_price"
        ],
        "time_stop_rule": time_stop_rule,
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "qualified": qualified,
        "disqualification_reasons": reasons,
        "estimated_horizon": "Do końca sesji",
    }


def _swing_item(
    symbol: str,
    research: dict[str, Any],
    round_trip_cost_bps: float,
    market_context_points: float = 0.0,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    candles = research.get("candles", [])
    indicators = research["indicators"]
    price = float(research["price"])
    atr = float(indicators["atr_14"])
    closes = [float(candle["close"]) for candle in candles if candle.get("close") is not None]
    volume_candles = candles
    if as_of is None and candles:
        try:
            latest_candle_time = pd.Timestamp(candles[-1]["timestamp"])
            now = pd.Timestamp.now(tz=latest_candle_time.tz)
            if latest_candle_time.date() == now.date():
                volume_candles = candles[:-1]
        except (KeyError, TypeError, ValueError):
            pass
    volumes = [
        float(candle["volume"])
        for candle in volume_candles
        if isinstance(candle.get("volume"), (int, float))
    ]
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
    sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
    trend_points = (
        (12.5 if price > sma50 else 0.0)
        + (12.5 if sma50 > sma200 else 0.0)
        if sma50 is not None and sma200 is not None
        else 0.0
    )
    rsi = float(indicators["rsi_14"])
    rsi_points = 10.0 if 45 <= rsi <= 65 else 0.0
    current_macd = float(indicators["macd"])
    prior_macd = candles[-2].get("macd") if len(candles) > 1 else None
    macd_rising = (
        isinstance(prior_macd, (int, float))
        and current_macd > float(prior_macd)
    )
    macd_points = 10.0 if macd_rising else 0.0
    mean_5 = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else None
    mean_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    volume_ratio = (
        mean_5 / mean_20
        if mean_5 is not None and mean_20 is not None and mean_20 > 0
        else None
    )
    volume_points = (
        15.0 * min(1.0, max(0.0, volume_ratio - 1.0))
        if volume_ratio is not None
        else 0.0
    )

    low_candles = candles[-21:-1]
    support_levels: list[float] = []
    lows = [
        float(candle["low"])
        for candle in candles
        if isinstance(candle.get("low"), (int, float))
    ]
    for index in range(max(1, len(lows) - 20), len(lows) - 1):
        if lows[index] <= lows[index - 1] and lows[index] <= lows[index + 1] and lows[index] < price:
            support_levels.append(lows[index])
    if not support_levels:
        support_levels = [
            float(candle["low"])
            for candle in low_candles
            if isinstance(candle.get("low"), (int, float))
            and float(candle["low"]) < price
        ]
    support = max(support_levels) if support_levels else None
    support_distance_pct = (
        (price - support) / price * 100 if support is not None and price > 0 else None
    )
    support_points = (
        15.0 * max(0.0, 1 - support_distance_pct / 2)
        if support_distance_pct is not None and support_distance_pct < 2.0
        else 0.0
    )

    seasonality_points, seasonal_samples = _monthly_seasonality_score(candles)
    context_points = max(0.0, min(5.0, market_context_points))
    consensus_points, consensus_details = _analyst_consensus_points(
        research,
        25.0,
    )
    macro_is_clear, macro_details = _macro_clear(research)
    event_risk = (
        research.get("broker_inputs", {}).get("event_risk", {})
        if isinstance(research.get("broker_inputs"), dict)
        else {}
    )
    corporate_status = (
        event_risk.get("corporate_calendar_status")
        if isinstance(event_risk, dict)
        else None
    )
    corporate_events = (
        event_risk.get("upcoming_events", [])
        if isinstance(event_risk, dict)
        else []
    )
    corporate_event_risk = (
        isinstance(event_risk, dict)
        and event_risk.get("status") == "high_risk"
        and bool(corporate_events)
    )
    macro_event_points = (
        15.0
        if macro_is_clear
        and not corporate_event_risk
        and corporate_status in {"available", "not_applicable"}
        else 0.0
    )
    macro_context_points = macro_event_points + seasonality_points + context_points
    breakdown = [
        _score_row(
            "Struktura trendu · SMA",
            trend_points,
            25,
            f"Cena>SMA50: {sma50 is not None and price > sma50}; "
            f"SMA50>SMA200: {sma50 is not None and sma200 is not None and sma50 > sma200}; "
            f"SMA50={_format_market_price(sma50)}; SMA200={_format_market_price(sma200)}.",
        ),
        _score_row(
            "Impet · RSI/MACD",
            rsi_points + macd_points,
            20,
            f"RSI(14)={rsi:.2f}; rosnący MACD={macd_rising}.",
        ),
        _score_row(
            "Wolumen 5/20 sesji",
            volume_points,
            15,
            (
                f"Średnia 5d/20d={volume_ratio:.3f}; "
                f"vol5={mean_5:.0f}, vol20={mean_20:.0f}."
                if volume_ratio is not None
                else "Brak co najmniej 20 dziennych obserwacji wolumenu."
            ),
        ),
        _score_row(
            "Bliskość wsparcia",
            support_points,
            15,
            (
                f"Wsparcie={support:.8g}; odległość={support_distance_pct:.2f}%."
                if support_distance_pct is not None
                else "Nie znaleziono wsparcia poniżej ceny."
            ),
        ),
        _score_row(
            "Opinie analityków",
            consensus_points,
            25,
            consensus_details,
        ),
        _score_row(
            "Makro / kalendarz / kontekst",
            macro_context_points,
            25,
            f"Brak wydarzeń wysokiego ryzyka={macro_event_points > 0}; "
            f"{macro_details} Sezonowość={seasonality_points:.1f}/5 "
            f"({seasonal_samples} lat); reżim rynku={context_points:.1f}/5."
            + (
                " Wykryto raport wyników w horyzoncie."
                if corporate_event_risk
                else ""
            ),
        ),
    ]
    resistance = max(
        (
            float(candle["high"])
            for candle in candles[-21:-1]
            if isinstance(candle.get("high"), (int, float))
        ),
        default=None,
    )
    trigger = max(price, resistance) if resistance is not None else price
    levels = _trade_levels(
        trigger,
        atr,
        SWING_NET_TARGET_RETURN,
        round_trip_cost_bps,
    )
    risk_reward = _net_risk_reward(
        trigger,
        levels["take_profit"],
        levels["stop_loss"],
        round_trip_cost_bps,
    )
    risk_points = (
        25.0
        if risk_reward >= 2.0 and math.isfinite(atr) and atr > 0
        else 15.0
        if risk_reward >= 1.5 and math.isfinite(atr) and atr > 0
        else 0.0
    )
    breakdown.append(
        _score_row(
            "Ryzyko / ATR trailing stop",
            risk_points,
            25,
            f"R/R={risk_reward:.2f}; SL={levels['stop_loss']:.8g}; "
            f"trailing aktywuje się przy "
            f"{levels['trailing_stop_activation_price']:.8g}. "
            f"{levels['trailing_stop_rule']}",
        )
    )
    score = round(sum(row["points"] for row in breakdown), 2)
    live_quality_reviewed = bool(research.get("quality_reviewed", False))
    has_analyst_data = (
        isinstance(research.get("analyst_consensus"), dict)
        and research["analyst_consensus"].get("status") == "available"
    )
    qualified = (
        score >= QUALIFICATION_THRESHOLD
        and sma50 is not None
        and sma200 is not None
        and risk_reward >= 1.5
        and not research.get("is_mock", False)
        and (
            research.get("is_historical", False)
            or (live_quality_reviewed and has_analyst_data)
        )
    )
    reasons = []
    if score < QUALIFICATION_THRESHOLD:
        reasons.append(
            f"Wynik swing jest niższy niż {QUALIFICATION_THRESHOLD:.0f}/150."
        )
    if sma50 is None or sma200 is None:
        reasons.append("Brak wystarczającej historii do SMA50/SMA200.")
    if risk_reward < 1.5:
        reasons.append("Risk/Reward poniżej minimalnego progu 1:1,5.")
    if not research.get("is_historical", False) and not live_quality_reviewed:
        reasons.append("Brak przeglądu opinii i sentymentu dla kandydata.")
    if (
        not research.get("is_historical", False)
        and live_quality_reviewed
        and not has_analyst_data
    ):
        reasons.append("Brak wiarygodnego konsensusu analityków Yahoo Finance.")
    if research.get("is_mock", False):
        reasons.append("Dane syntetyczne.")
    return {
        "symbol": symbol,
        "signal": "KUP" if qualified else "CZEKAJ",
        "price": price,
        "timestamp": research.get("timestamp"),
        "timeframe": SCAN_TIMEFRAME,
        "indicators": indicators,
        "candles": candles,
        "is_mock": bool(research.get("is_mock", False)),
        "sma_50": sma50,
        "sma_200": sma200,
        "volume_ratio_5d_20d": volume_ratio,
        "support_price": support,
        "resistance_price": resistance,
        "broker_points": score,
        "score_breakdown": breakdown,
        "target_probability_pct": round(max(0.0, min(100.0, score / 1.5)), 1),
        "target_return_pct": SWING_NET_TARGET_RETURN * 100,
        **levels,
        "risk_reward_ratio": risk_reward,
        "trailing_stop_rule": levels["trailing_stop_rule"],
        "trailing_stop_activation_price": levels[
            "trailing_stop_activation_price"
        ],
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "qualified": qualified,
        "disqualification_reasons": reasons,
        "estimated_horizon": "3–5 dni",
    }


def _monthly_seasonality_score(
    candles: list[dict[str, Any]],
) -> tuple[float, int]:
    monthly_closes: dict[tuple[int, int], float] = {}
    for candle in candles:
        try:
            stamp = pd.Timestamp(candle["timestamp"])
            close = float(candle["close"])
        except (KeyError, TypeError, ValueError):
            continue
        monthly_closes[(stamp.year, stamp.month)] = close
    if not monthly_closes:
        return 0.0, 0
    latest_stamp = pd.Timestamp(candles[-1]["timestamp"])
    current_period = (latest_stamp.year, latest_stamp.month)
    monthly_returns: list[float] = []
    for year, month in sorted(monthly_closes):
        current = (year, month)
        previous_year = (year - 1, month)
        if month != latest_stamp.month or current >= current_period:
            continue
        if previous_year in monthly_closes and monthly_closes[previous_year] > 0:
            monthly_returns.append(
                monthly_closes[current] / monthly_closes[previous_year] - 1
            )
    samples = len(monthly_returns)
    if samples < 3:
        return 0.0, samples
    positive_fraction = sum(value > 0 for value in monthly_returns) / samples
    return 5.0 * positive_fraction, samples


def _scan_short_term_market(
    symbols: tuple[str, ...],
    round_trip_cost_bps: float,
    as_of: datetime | None,
) -> dict[str, Any]:
    researcher = ResearchAgent(
        timeout=NETWORK_TIMEOUT_SECONDS,
        allow_mock_data=False,
    )
    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def scan_symbol(symbol: str) -> dict[str, Any]:
        research = researcher.analyze(
            symbol=symbol,
            timeframe=SHORT_TERM_TIMEFRAME,
            limit=SHORT_TERM_CANDLE_LIMIT,
            include_qualitative=False,
            allow_mock_data=False,
            as_of=as_of,
        )
        return _short_term_item(symbol, research, round_trip_cost_bps)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(SCAN_MAX_WORKERS, max(1, len(symbols)))
    ) as executor:
        futures = {executor.submit(scan_symbol, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                items.append(future.result(timeout=NETWORK_TIMEOUT_SECONDS + 1))
            except Exception as error:
                LOGGER.warning("Skan krótkoterminowy %s nie powiódł się: %s", symbol, error)
                errors.append(
                    {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}
                )

    failed_assets = [
        {
            "symbol": error["symbol"],
            "signal": "BŁĄD",
            "price": None,
            "broker_points": None,
            "qualified": False,
            "error": error["error"],
            "disqualification_reasons": [
                "Nie pobrano świec rynkowych; nie można ocenić punktacji."
            ],
        }
        for error in errors
    ]
    ranked_assets = sorted(
        [*items, *failed_assets],
        key=lambda item: (
            item.get("broker_points") is not None,
            item.get("broker_points") or -1.0,
        ),
        reverse=True,
    )
    recommendations = [item for item in items if item["qualified"]]
    return {
        "analysis_category": "short_term",
        "recommendations": sorted(
            recommendations,
            key=lambda item: item["broker_points"],
            reverse=True,
        ),
        "watchlist": [item for item in items if not item["qualified"]],
        "ranked_assets": ranked_assets,
        "failed_assets": failed_assets,
        "errors": errors,
        "quality_errors": [],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "mode": "history" if as_of is not None else "live",
        "data_as_of": (
            as_of.astimezone(timezone.utc).isoformat()
            if as_of is not None
            else None
        ),
        "timeframe": SHORT_TERM_TIMEFRAME,
        "qualification_rule": (
            "momentum KUP, wynik > 75/100 i RelVol >= 1,0"
        ),
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "target_return_pct": SHORT_TERM_NET_TARGET_RETURN * 100,
        "scanned_assets": len(symbols),
        "stock_assets": len(set(symbols) & set(STOCK_UNIVERSE)),
        "crypto_assets": len(set(symbols) & set(CRYPTO_UNIVERSE)),
        "stage_two_reviewed_assets": 0,
    }


def _legacy_scan_market(
    symbols: tuple[str, ...],
    round_trip_cost_bps: float = 40.0,
    as_of: datetime | None = None,
    analysis_category: str = MULTI_DAY_CATEGORY,
) -> dict[str, Any]:
    """Scan current or point-in-time market data for all requested assets."""
    if analysis_category == SHORT_TERM_CATEGORY:
        return _scan_short_term_market(symbols, round_trip_cost_bps, as_of)
    if analysis_category != MULTI_DAY_CATEGORY:
        raise ValueError(f"Nieznana kategoria analizy: {analysis_category}")
    researcher = ResearchAgent(
        timeout=NETWORK_TIMEOUT_SECONDS,
        allow_mock_data=False,
    )
    recommendations: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    quality_errors: list[dict[str, str]] = []
    stage_one: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def scan_technical(symbol: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        research = researcher.analyze(
            symbol=symbol,
            timeframe=SCAN_TIMEFRAME,
            limit=SCAN_CANDLE_LIMIT,
            include_qualitative=False,
            allow_mock_data=False,
            as_of=as_of,
        )
        score = RiskManager().calculate_broker_filter_score(
            research,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        return symbol, research, score

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(SCAN_MAX_WORKERS, max(1, len(symbols)))
    ) as executor:
        futures = {executor.submit(scan_technical, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                stage_one.append(future.result(timeout=NETWORK_TIMEOUT_SECONDS + 1))
            except Exception as error:
                LOGGER.warning("Etap techniczny %s nie powiódł się: %s", symbol, error)
                errors.append(
                    {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}
                )

    candidates: list[tuple[str, dict[str, Any], dict[str, Any], float]] = []
    for symbol, research, score in stage_one:
        filters = score["filters"]
        technical_and_risk = (
            filters["technical_momentum"]["points"]
            + filters["risk_reward_costs"]["points"]
        )
        if (
            research["signal"] == "KUP"
            and not research["is_mock"]
            and technical_and_risk >= STAGE_ONE_MINIMUM_POINTS
        ):
            if as_of is None:
                candidates.append((symbol, research, score, technical_and_risk))
            else:
                item = _scan_item(
                    symbol,
                    research,
                    score,
                    technical_and_risk,
                    round_trip_cost_bps,
                    as_of,
                )
                item["disqualification_reasons"].append(
                    "W trybie historycznym wyłączono bieżące wiadomości, "
                    "fundamenty, arkusz zleceń i kalendarze, aby uniknąć "
                    "wykorzystania danych z przyszłości."
                )
                watchlist.append(item)
        else:
            watchlist.append(
                _scan_item(
                    symbol,
                    research,
                    score,
                    technical_and_risk,
                    round_trip_cost_bps,
                    as_of,
                )
            )

    candidates.sort(key=lambda candidate: candidate[3], reverse=True)
    deferred_candidates = candidates[STAGE_TWO_MAX_ASSETS:]
    candidates = candidates[:STAGE_TWO_MAX_ASSETS]
    for symbol, research, score, stage_one_points in deferred_candidates:
        item = _scan_item(
            symbol,
            research,
            score,
            stage_one_points,
            round_trip_cost_bps,
            as_of,
        )
        item["disqualification_reasons"].append(
            "Pominięto etap jakościowy: limit zapytań API dla tego skanu."
        )
        watchlist.append(item)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(STAGE_TWO_MAX_ASSETS, max(1, len(candidates)))
    ) as executor:
        future_map = {
            executor.submit(_finish_candidate, researcher, candidate, round_trip_cost_bps): candidate
            for candidate in candidates
        }
        for future in concurrent.futures.as_completed(future_map):
            symbol, research, stage_one_score, technical_and_risk = future_map[future]
            try:
                item, qualified = future.result(timeout=NETWORK_TIMEOUT_SECONDS + 2)
                if qualified:
                    recommendations.append(item)
                else:
                    watchlist.append(item)
                review = item.get("news", {}).get("gemini_review", {})
                if review.get("status") == "neutral_fallback":
                    quality_errors.append(
                        {
                            "symbol": symbol,
                            "error": str(
                                review.get(
                                    "rationale",
                                    "Gemini review niedostępny; użyto neutralnej oceny.",
                                )
                            ),
                        }
                    )
            except Exception as error:
                LOGGER.warning("Etap jakościowy %s nie powiódł się: %s", symbol, error)
                quality_errors.append(
                    {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}
                )
                fallback_item = _scan_item(
                    symbol,
                    research,
                    stage_one_score,
                    technical_and_risk,
                    round_trip_cost_bps,
                    as_of,
                )
                fallback_item["disqualification_reasons"].append(
                    f"Etap jakościowy pominięty: {type(error).__name__}."
                )
                watchlist.append(fallback_item)

    recommendations.sort(
        key=lambda item: item["broker_points"],
        reverse=True,
    )
    failed_assets = [
        {
            "symbol": error["symbol"],
            "signal": "BŁĄD",
            "price": None,
            "broker_points": None,
            "qualified": False,
            "error": error["error"],
            "disqualification_reasons": [
                "Nie pobrano świec rynkowych; nie można ocenić punktacji."
            ],
        }
        for error in errors
    ]
    ranked_assets = sorted(
        [*recommendations, *watchlist, *failed_assets],
        key=lambda item: (
            item.get("broker_points") is not None,
            item.get("broker_points") or -1.0,
        ),
        reverse=True,
    )
    return {
        "recommendations": recommendations,
        "watchlist": watchlist,
        "ranked_assets": ranked_assets,
        "failed_assets": failed_assets,
        "errors": errors,
        "quality_errors": quality_errors,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "mode": "history" if as_of is not None else "live",
        "data_as_of": (
            as_of.astimezone(timezone.utc).isoformat()
            if as_of is not None
            else None
        ),
        "timeframe": SCAN_TIMEFRAME,
        "analysis_category": "multi_day",
        "qualification_rule": "suma > 75 oraz brak blokady wydarzenia",
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "stage_one_minimum_points": STAGE_ONE_MINIMUM_POINTS,
        "stage_two_max_assets": STAGE_TWO_MAX_ASSETS,
        "stage_two_reviewed_assets": len(candidates),
        "scanned_assets": len(symbols),
        "stock_assets": len(set(symbols) & set(STOCK_UNIVERSE)),
        "crypto_assets": len(set(symbols) & set(CRYPTO_UNIVERSE)),
    }


def _market_context_score(research: dict[str, Any]) -> float:
    candles = research.get("candles", [])
    closes = [
        float(candle["close"])
        for candle in candles
        if isinstance(candle.get("close"), (int, float))
    ]
    if len(closes) < 200:
        return 0.0
    latest = closes[-1]
    sma50 = sum(closes[-50:]) / 50
    sma200 = sum(closes[-200:]) / 200
    if latest > sma50 > sma200:
        return 5.0
    if latest > sma200:
        return 2.5
    return 0.0


def _scan_category_market(
    symbols: tuple[str, ...],
    round_trip_cost_bps: float,
    as_of: datetime | None,
    analysis_category: str,
) -> dict[str, Any]:
    if analysis_category not in {MULTI_DAY_CATEGORY, SHORT_TERM_CATEGORY}:
        raise ValueError(f"Nieznana kategoria analizy: {analysis_category}")
    is_short_term = analysis_category == SHORT_TERM_CATEGORY
    timeframe = SHORT_TERM_TIMEFRAME if is_short_term else SCAN_TIMEFRAME
    candle_limit = SHORT_TERM_CANDLE_LIMIT if is_short_term else SCAN_CANDLE_LIMIT
    researcher = ResearchAgent(
        timeout=NETWORK_TIMEOUT_SECONDS,
        allow_mock_data=False,
    )
    errors: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    research_by_symbol: dict[str, dict[str, Any]] = {}
    quality_errors: list[dict[str, str]] = []
    market_context_points = 0.0
    if not is_short_term and "SPY" not in symbols:
        try:
            market_proxy = researcher.analyze(
                symbol="SPY",
                timeframe=SCAN_TIMEFRAME,
                limit=SCAN_CANDLE_LIMIT,
                include_qualitative=False,
                allow_mock_data=False,
                as_of=as_of,
            )
            market_context_points = _market_context_score(market_proxy)
        except Exception as error:
            LOGGER.warning("Brak kontekstu rynku SPY: %s", error)

    def scan_symbol(symbol: str) -> dict[str, Any]:
        research = researcher.analyze(
            symbol=symbol,
            timeframe=timeframe,
            limit=candle_limit,
            include_qualitative=False,
            allow_mock_data=False,
            as_of=as_of,
        )
        research["is_historical"] = as_of is not None
        item = (
            _short_term_item(symbol, research, round_trip_cost_bps)
            if is_short_term
            else _swing_item(
                symbol,
                research,
                round_trip_cost_bps,
                market_context_points,
                as_of,
            )
        )
        item["data_as_of"] = (
            as_of.astimezone(timezone.utc).isoformat()
            if as_of is not None
            else None
        )
        research_by_symbol[symbol] = research
        return item

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(SCAN_MAX_WORKERS, max(1, len(symbols)))
    ) as executor:
        futures = {executor.submit(scan_symbol, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                items.append(future.result(timeout=NETWORK_TIMEOUT_SECONDS + 1))
            except Exception as error:
                LOGGER.warning("Skan %s nie powiódł się: %s", symbol, error)
                errors.append(
                    {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}
                )

    enrich_qualitative = getattr(researcher, "enrich_qualitative", None)
    candidates = sorted(
        (
            item
            for item in items
            if as_of is None
            and isinstance(item.get("broker_points"), (int, float))
            and item["broker_points"] >= STAGE_ONE_MINIMUM_POINTS
        ),
        key=lambda item: item["broker_points"],
        reverse=True,
    )[:STAGE_TWO_MAX_ASSETS]
    if callable(enrich_qualitative) and candidates:
        def enrich_candidate(symbol: str) -> tuple[str, dict[str, Any] | None, str | None]:
            research = research_by_symbol[symbol]
            try:
                enriched = enrich_qualitative(
                    research,
                    include_fundamentals=False,
                    round_trip_cost_bps=round_trip_cost_bps,
                )
                enriched["quality_reviewed"] = True
                return symbol, enriched, None
            except Exception as error:
                LOGGER.warning("Dane jakościowe %s są niedostępne: %s", symbol, error)
                research["quality_reviewed"] = True
                return symbol, research, f"{type(error).__name__}: {error}"

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(5, len(candidates))
        ) as executor:
            futures = {
                executor.submit(enrich_candidate, item["symbol"]): item["symbol"]
                for item in candidates
            }
            enriched_by_symbol: dict[str, dict[str, Any]] = {}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    enriched_symbol, research, error_text = future.result(
                        timeout=NETWORK_TIMEOUT_SECONDS + 1
                    )
                    if research is not None:
                        enriched_by_symbol[enriched_symbol] = research
                    if error_text is not None:
                        quality_errors.append(
                            {"symbol": symbol, "error": error_text}
                        )
                except Exception as error:
                    quality_errors.append(
                        {"symbol": symbol, "error": f"{type(error).__name__}: {error}"}
                    )
                    research_by_symbol[symbol]["quality_reviewed"] = True

        for index, item in enumerate(items):
            research = enriched_by_symbol.get(item["symbol"])
            if research is None:
                continue
            items[index] = (
                _short_term_item(item["symbol"], research, round_trip_cost_bps)
                if is_short_term
                else _swing_item(
                    item["symbol"],
                    research,
                    round_trip_cost_bps,
                    market_context_points,
                    as_of,
                )
            )
            items[index]["data_as_of"] = item.get("data_as_of")

    failed_assets = [
        {
            "symbol": error["symbol"],
            "signal": "BŁĄD",
            "price": None,
            "broker_points": None,
            "qualified": False,
            "error": error["error"],
            "score_breakdown": [],
            "disqualification_reasons": [
                "Nie pobrano świec rynkowych; nie można ocenić punktacji."
            ],
        }
        for error in errors
    ]
    ranked_assets = sorted(
        [*items, *failed_assets],
        key=lambda item: (
            item.get("broker_points") is not None,
            item.get("broker_points") or -1.0,
        ),
        reverse=True,
    )
    recommendations = [item for item in items if item["qualified"]]
    return {
        "analysis_category": "short_term" if is_short_term else "multi_day",
        "category_label": analysis_category,
        "recommendations": sorted(
            recommendations,
            key=lambda item: item["broker_points"],
            reverse=True,
        ),
        "watchlist": [item for item in items if not item["qualified"]],
        "ranked_assets": ranked_assets,
        "failed_assets": failed_assets,
        "errors": errors,
        "quality_errors": quality_errors,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "mode": "history" if as_of is not None else "live",
        "data_as_of": (
            as_of.astimezone(timezone.utc).isoformat()
            if as_of is not None
            else None
        ),
        "timeframe": timeframe,
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "maximum_score": 150,
        "qualification_threshold": QUALIFICATION_THRESHOLD,
        "qualification_rule": (
            f"wynik >= {QUALIFICATION_THRESHOLD:.0f}/150"
            + (" i RelVol >= 1,0 oraz R/R >= 1:2" if is_short_term else "")
        ),
        "target_return_pct": (
            SHORT_TERM_NET_TARGET_RETURN if is_short_term else SWING_NET_TARGET_RETURN
        )
        * 100,
        "scanned_assets": len(symbols),
        "stock_assets": len(set(symbols) & set(STOCK_UNIVERSE)),
        "crypto_assets": len(set(symbols) & set(CRYPTO_UNIVERSE)),
        "market_context_points": market_context_points,
        "stage_two_reviewed_assets": len(candidates)
        if callable(enrich_qualitative)
        else 0,
    }


def _scan_market(
    symbols: tuple[str, ...],
    round_trip_cost_bps: float = ROUND_TRIP_COST_BPS,
    as_of: datetime | None = None,
    analysis_category: str = MULTI_DAY_CATEGORY,
) -> dict[str, Any]:
    return _scan_category_market(
        symbols,
        round_trip_cost_bps,
        as_of,
        analysis_category,
    )


def _run_detail_analysis(
    symbol: str,
    capital: float,
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    client = GeminiClient.from_env()
    try:
        report = _run_pipeline(
            goal=_goal_for_asset(symbol),
            capital=capital,
            candle_limit=100,
            gemini_client=client,
            researcher=ResearchAgent(
                timeout=NETWORK_TIMEOUT_SECONDS,
                allow_mock_data=False,
            ),
            round_trip_cost_bps=round_trip_cost_bps,
        )
        return report
    finally:
        client.close()


def _calculate_revolut_position(
    symbol: str,
    purchase_price: float,
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    research = ResearchAgent(
        timeout=NETWORK_TIMEOUT_SECONDS,
        allow_mock_data=False,
    ).analyze(
        symbol=symbol,
        timeframe=SCAN_TIMEFRAME,
        limit=SCAN_CANDLE_LIMIT,
        round_trip_cost_bps=round_trip_cost_bps,
        allow_mock_data=False,
    )
    if research["is_mock"]:
        raise RuntimeError(
            "Dane rynkowe są syntetyczne. Odśwież analizę, gdy Yahoo Finance "
            "udostępni rzeczywiste notowania."
        )

    indicators = research["indicators"]
    candles = research["candles"]
    prior_candles = candles[-21:-1] if len(candles) > 1 else candles[-20:]
    resistance_price = (
        max(float(candle["high"]) for candle in prior_candles)
        if prior_candles
        else None
    )
    news = research["news"]
    sentiment_score = (
        float(news["sentiment_score"])
        if news["status"] == "verified" and news["verified_articles"]
        else None
    )
    position = RiskManager().calculate_revolut_position(
        purchase_price=float(purchase_price),
        current_price=float(research["price"]),
        atr=float(indicators["atr_14"]),
        technical_signal=research["signal"],
        sentiment_score=sentiment_score,
        resistance_price=resistance_price,
        rsi=float(indicators["rsi_14"]),
        macd=float(indicators["macd"]),
        macd_signal=float(indicators["macd_signal"]),
        horizon_days=3,
        round_trip_cost_bps=round_trip_cost_bps,
        adx=float(indicators["adx_14"]),
        session_volume_ratio=research["broker_inputs"]["liquidity"][
            "session_volume_ratio"
        ],
    )
    return {
        "symbol": symbol,
        "position": position,
        "news": news,
        "indicators": indicators,
        "timestamp": research["timestamp"],
        "resistance_20": resistance_price,
        "data_source": research["data_source"],
    }


def _render_news(news: Any) -> None:
    st.subheader("Zweryfikowane wiadomości")
    if not isinstance(news, dict):
        st.info("Brak danych o wiadomościach.")
        return

    status = news.get("status", "unavailable")
    if status == "unavailable":
        st.warning(
            "Nie udało się pobrać wiadomości; sentyment nie został użyty. "
            + str(news.get("error", ""))
        )
    elif status == "no_verified_news":
        st.info(
            "Brak najnowszych nagłówków potwierdzonych przez dwa niezależne, "
            "zaufane źródła. Niezweryfikowane treści zostały pominięte."
        )

    st.caption(
        f"Sentyment: {news.get('sentiment_score', 0.0):+.2f} · "
        f"zweryfikowane źródła: {news.get('verified_source_count', 0)} · "
        f"odrzucone nagłówki: {news.get('rejected_count', 0)}"
    )
    articles = news.get("verified_articles", [])
    if articles:
        with st.expander("Nagłówki, daty i źródła"):
            for article in articles:
                st.markdown(f"**{article['title']}**")
                st.caption(
                    f"{article['published_at']} · "
                    f"sentyment nagłówka: {article['sentiment']:+.2f}"
                )
                source_columns = st.columns(min(len(article["sources"]), 3))
                for index, source in enumerate(article["sources"]):
                    source_columns[index % len(source_columns)].link_button(
                        source["publisher"],
                        source["url"],
                    )
    if news.get("verification_method"):
        st.caption(news["verification_method"])


def _render_fundamentals(fundamentals: Any) -> None:
    st.subheader("Analiza fundamentalna")
    if not isinstance(fundamentals, dict):
        st.info("Brak danych fundamentalnych.")
        return

    quality_score = fundamentals.get("quality_score")
    if quality_score is not None:
        st.metric(
            "Ocena jakości fundamentalnej",
            f"{quality_score:.1f}/100",
            help=(
                "Heurystyka oparta na dostępnych danych Yahoo Finance; "
                "nie jest wyceną ani rekomendacją."
            ),
        )
    metrics = fundamentals.get("metrics", {})
    if metrics:
        st.json(metrics)
    elif fundamentals.get("status") == "unavailable":
        st.warning(
            "Dane fundamentalne są niedostępne: "
            + str(fundamentals.get("error", "brak szczegółów"))
        )
    else:
        st.info("Yahoo Finance nie udostępniło danych fundamentalnych dla aktywa.")


def _render_revolut_position(result: dict[str, Any]) -> None:
    position = result["position"]
    st.header(f"Pozycja z Revoluta · {result['symbol']}")
    st.caption(
        f"Dane rynkowe {result['data_source']} · interwał {SCAN_TIMEFRAME} · "
        f"ostatnia świeca: {result['timestamp']}"
    )
    if position["adx_14"] is not None and position["adx_14"] < 20:
        st.warning(
            f"ADX(14) = {position['adx_14']:.1f} (<20): rynek bez silnego trendu, "
            "sygnał kierunkowy jest wstrzymany, a szansa celu obniżona."
        )
    else:
        st.success(
            f"Filtr ADX: {position['adx_status']} "
            f"(ADX(14) = {_format_number(position['adx_14'], 1)}; próg 20)."
        )
    if position["volume_kill_switch_status"] == "blocked":
        st.error(
            "Kill-switch wolumenu: ostatnia ukończona sesja miała mniej niż 50% "
            "średniego wolumenu z poprzednich 20 sesji. Transakcja odrzucona."
        )
    elif position["volume_kill_switch_status"] == "unavailable":
        st.warning(
            "Brak 20 ukończonych sesji referencyjnych; filtr wolumenu nie "
            "zatwierdza transakcji."
        )
    else:
        st.success(
            "Filtr wolumenu zaliczony: "
            f"{position['session_volume_ratio']:.2f}× średniej z 20 sesji."
        )
    if not position["hard_filters_passed"]:
        st.error(
            "Sygnał wstrzymany przez filtr ADX lub wolumenu. Pokazane poziomy "
            "są wyłącznie informacyjne i nie stanowią rekomendacji transakcji."
        )
    elif position["target_reached"]:
        st.success(
            "Bieżąca cena rynkowa jest już równa lub wyższa od celu netto +4,5% "
            "względem podanej ceny zakupu po założonych kosztach."
        )
    else:
        st.info(
            f"{result['symbol']}: {_format_price(position['purchase_price'])} "
            f"→ Sprzedaj na zysku przy {_format_price(position['take_profit'])} "
            f"→ Uciekaj z transakcji przy {_format_price(position['stop_loss'])}."
        )

    cards = st.columns(4)
    cards[0].metric("Cena zakupu", _format_price(position["purchase_price"]))
    cards[1].metric(
        "Sprzedaj na zysku (netto +4,5%)",
        _format_price(position["take_profit"]),
    )
    cards[2].metric(
        "Uciekaj z transakcji (SL)",
        _format_price(position["stop_loss"]),
        help=f"Stop-Loss = cena zakupu − {position['atr_multiplier']} × ATR.",
    )
    cards[3].metric(
        "Szansa osiągnięcia celu*",
        f"{position['success_probability_pct']:.1f}%",
    )
    st.caption(
        f"Cel brutto +{position['gross_target_return_pct']:.2f}% obejmuje "
        f"szacowane koszty round trip {position['estimated_round_trip_cost_pct']:.2f}% "
        f"({position['round_trip_cost_bps_assumption']:.0f} bps); "
        f"cel netto: +{position['net_target_return_pct']:.2f}%. "
        f"SL pozostaje na 1,5×ATR; szacowany wynik netto na SL: "
        f"{position['estimated_net_stop_return_pct']:.2f}%."
    )
    st.caption(position["probability_note"])
    components = position["probability_components"]
    rendered = [
        f"{label}: {_format_number(value, 1)}%"
        for key, label in (
            ("volatility_reachability", "Zmienność ATR"),
            ("technical", "Technika"),
            ("news_sentiment", "Sentyment"),
        )
        if (value := components.get(key)) is not None
    ]
    if rendered:
        st.caption("Składowe estymacji: " + " · ".join(rendered))
    if position["resistance_penalty"] > 0:
        st.warning(
            f"Opór w okolicy {_format_price(position['resistance_price'])} "
            f"obniża estymację o {position['resistance_penalty']:.1f} p.p."
        )
    elif position["resistance_price"] is not None:
        st.caption(
            "Najwyższy szczyt poprzednich świec 4h (okno do 20 świec): "
            f"{_format_price(position['resistance_price'])}."
        )
    news = result["news"]
    if news["status"] == "verified":
        with st.expander("Wiadomości użyte do estymacji"):
            for article in news["verified_articles"]:
                st.markdown(f"**{article['title']}**")
                st.caption(
                    f"{article['published_at']} · "
                    f"sentyment: {article['sentiment']:+.2f}"
                )
                source_columns = st.columns(min(len(article["sources"]), 3))
                for index, source in enumerate(article["sources"]):
                    source_columns[index % len(source_columns)].link_button(
                        source["publisher"],
                        source["url"],
                    )
    else:
        st.caption(
            "Brak potwierdzonych wiadomości: komponent sentymentu nie został użyty."
        )


def _short_term_session_guidance(symbol: str) -> str:
    if symbol in CRYPTO_UNIVERSE or symbol.endswith("-USD"):
        return "Krypto: handel 24/7."
    if symbol in US_STOCK_UNIVERSE:
        return (
            "Skan przygotowawczy od 14:00 czasu PL; dane pre-market zależą od "
            "Yahoo Finance. Regularna sesja USA zwykle 15:30, okresowo 14:30 "
            "przy zmianie czasu."
        )
    return (
        "Skan przygotowawczy od 14:00 czasu PL; handel w sesji właściwej giełdy."
    )


def _render_short_term_scan(scan: dict[str, Any]) -> None:
    mode_label = (
        "Tryb historyczny · backtest"
        if scan.get("mode") == "history"
        else "Tryb na żywo"
    )
    st.header(f"Ranking krótkoterminowy · {mode_label}")
    st.caption(
        f"Model Day Trading na świecach {scan['timeframe']} · "
        f"przeskanowano {scan.get('stock_assets', 0)} akcji US/UE i "
        f"{scan.get('crypto_assets', 0)} kryptowalut · aktualizacja "
        f"{scan['scanned_at']}"
    )
    if scan.get("data_as_of"):
        st.caption(
            f"Moment analizy: {scan['data_as_of']} "
            "(czas Europe/Warsaw, zapisany jako UTC)."
        )
    st.info(
        "Scoring: momentum techniczne (RSI/MACD/cena) = 80%, RelVol i wybicie "
        "oporu = 20%. RelVol jest ostatnim wolumenem względem mediany 20 świec, "
        "a nie wolumenem skorygowanym statystycznie o porę dnia. Estymacja "
        "prawdopodobieństwa jest heurystyczna, nie jest skalibrowaną prognozą."
    )
    if scan.get("mode") == "history":
        st.warning(
            "Używane są wyłącznie świece dostępne do wskazanego czasu. Bieżące "
            "wiadomości i inne dane jakościowe nie są częścią backtestu."
        )
    st.caption(
        "Cel krótkoterminowy: +2,0% netto po założonych kosztach round trip "
        f"{scan.get('round_trip_cost_bps_assumption', 20.0):g} bps. Stop jest "
        "wyliczony tak, aby modelowy R/R netto wynosił co najmniej 1:2. "
        "Scenariusz nie składa zleceń; dane Yahoo mogą być opóźnione."
    )
    ranked_assets = scan.get("ranked_assets", [])
    if not ranked_assets:
        st.error("🚨 BŁĄD: skan nie zwrócił rekordów rankingowych.", icon="🚨")
        return

    ranking_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked_assets, start=1):
        probability = item.get("target_probability_pct")
        ranking_rows.append(
            {
                "#": rank,
                "Aktywo": item["symbol"],
                "Sygnał": item.get("signal", "—"),
                "Ostatnia cena": _format_market_price(item.get("price")),
                "Czas świecy": item.get("timestamp", "—"),
                "Punkty / 100": (
                    f"{item['broker_points']:.1f}"
                    if isinstance(item.get("broker_points"), (int, float))
                    else "—"
                ),
                "Szansa na +2% dziś": (
                    f"{probability:.1f}%"
                    if isinstance(probability, (int, float))
                    else "—"
                ),
                "RelVol": (
                    f"{item['relative_volume']:.2f}×"
                    if isinstance(item.get("relative_volume"), (int, float))
                    else "brak danych"
                ),
                "Trigger kupna": _format_market_price(item.get("entry_price")),
                "Take-Profit": _format_market_price(item.get("take_profit")),
                "Potencjalny zysk": (
                    f"{item['potential_profit_pct']:+.2f}%"
                    if isinstance(item.get("potential_profit_pct"), (int, float))
                    else "—"
                ),
                "Stop-Loss": _format_market_price(item.get("stop_loss")),
                "Status": (
                    "Kwalifikuje się"
                    if item.get("qualified")
                    else "Błąd danych"
                    if item.get("error")
                    else "Obserwowane"
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(ranking_rows),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Pełny ranking: {len(ranked_assets)} aktywów, od najwyższego do "
        "najniższego score. Rozwiń kartę aktywa po szczegóły."
    )
    for rank, item in enumerate(ranked_assets, start=1):
        if item.get("error"):
            with st.expander(f"{rank}. {item['symbol']} · błąd danych"):
                st.error(f"🚨 {item['error']}", icon="🚨")
            continue
        currency = _price_currency(item["symbol"])
        probability = item.get("target_probability_pct")
        probability_label = (
            f"{probability:.1f}% szans heurystycznych na +2,0% podczas sesji"
            if isinstance(probability, (int, float))
            else "Brak estymacji"
        )
        title = (
            f"{rank}. {item['symbol']} · {item['broker_points']:.1f}/100 · "
            f"{probability_label}"
        )
        with st.expander(title):
            st.caption(
                f"Ostatnia cena: {_format_market_price(item.get('price'))} "
                f"{currency} · świeca {item.get('timestamp', '—')} · "
                f"{_short_term_session_guidance(item['symbol'])}"
            )
            metrics = st.columns(4)
            metrics[0].metric("Punktacja", f"{item['broker_points']:.1f}/100")
            metrics[1].metric(
                "Estymacja +2,0% dziś",
                f"{probability:.1f}%"
                if isinstance(probability, (int, float))
                else "—",
            )
            relative_volume = item.get("relative_volume")
            metrics[2].metric(
                "RelVol · mediana 20 świec",
                f"{relative_volume:.2f}×"
                if isinstance(relative_volume, (int, float))
                else "—",
            )
            ratio = item.get("risk_reward_ratio")
            metrics[3].metric(
                "R/R netto",
                f"1:{ratio:.2f}" if isinstance(ratio, (int, float)) else "—",
            )
            st.write(
                f"**WYZWOLENIE KUPNA (Trigger):** jeśli cena osiągnie "
                f"{_format_market_price(item.get('entry_price'))} {currency} "
                "i RelVol wynosi co najmniej 1,2× (wybicie poziomu oporu) → KUP."
            )
            st.write(
                f"**POZIOM WYJŚCIA Z ZYSKIEM (Take-Profit):** sprzedaj przy "
                f"{_format_market_price(item.get('take_profit'))} {currency}. "
                "Cel +2,0% netto uwzględnia modelowe opłaty 0,1% za stronę."
            )
            st.write(
                f"**OCHRONA PRZED STRATĄ (Stop-Loss):** zamknij pozycję przy "
                f"{_format_market_price(item.get('stop_loss'))} {currency}; "
                "poziom dobrano dla minimalnego R/R netto 1:2."
            )
            model = item["score_model"]
            st.caption(
                "Składowe score: "
                f"technika {model['technical_momentum_points']:.1f}/100 × 80% + "
                f"RelVol/wybicie {model['volume_breakout_points']:.1f}/100 × 20%."
            )
            for reason in item.get("disqualification_reasons", []):
                st.caption(f"• {reason}")


def _render_category_scored_scan(scan: dict[str, Any]) -> None:
    is_short_term = scan.get("analysis_category") == "short_term"
    horizon = "do końca sesji" if is_short_term else "3–5 dni"
    target_pct = scan.get("target_return_pct", 2.0 if is_short_term else 4.5)
    mode_label = (
        "Tryb historyczny · backtest"
        if scan.get("mode") == "history"
        else "Tryb na żywo"
    )
    st.header(f"{scan['category_label']} · ranking · {mode_label}")
    st.caption(
        f"Interwał {scan['timeframe']} · aktualizacja {scan['scanned_at']} · "
        f"rynek: {scan.get('stock_assets', 0)} akcji, "
        f"{scan.get('crypto_assets', 0)} kryptowalut."
    )
    if scan.get("data_as_of"):
        st.caption(
            f"Moment analizy: {scan['data_as_of']} "
            "(czas Europe/Warsaw, zapisany jako UTC)."
        )
    if is_short_term:
        st.info(
            "RelVol porównuje wolumen ostatniej świecy 15m z wolumenem tej "
            "samej pory z maksymalnie 20 wcześniejszych sesji. RelVol <1,0 "
            "ustawia cały wynik na 0. Skala 150 pkt: RelVol 30, VWAP/POC 25, "
            "impet 20, R/R 15, analitycy/buzz 20, timing/makro 20 i "
            "Trailing Stop/Time-Stop 20."
        )
    else:
        st.info(
            "Skala swing 150 pkt: trend SMA 25, RSI/MACD 20, wolumen 5/20 "
            "sesji 15, wsparcie 15, konsensus analityków 25, makro/kalendarz/"
            "kontekst 25 i ryzyko/ATR trailing stop 25."
        )
    if scan.get("mode") == "history":
        st.warning(
            "Wszystkie wskaźniki i poziomy bazują na świecach dostępnych do "
            "wskazanego momentu. Aktualne opinie analityków, wiadomości i "
            "kalendarz makro są wyłączone, aby uniknąć look-ahead."
        )
    threshold = scan.get("qualification_threshold", QUALIFICATION_THRESHOLD)
    st.caption(
        f"Sygnał KUP wymaga co najmniej {threshold:.0f}/150 pkt"
        + (" oraz RelVol ≥ 1,0 i R/R ≥ 1:2." if is_short_term else ".")
        + f" Etap jakościowy ocenił {scan.get('stage_two_reviewed_assets', 0)} "
        "kandydatów."
    )
    if scan.get("mode") == "live":
        st.caption(
            "Opinie analityków pochodzą z Yahoo Finance, sentyment newsów z "
            "potwierdzonych nagłówków Google News, a wydarzenia makro z "
            "publicznego kalendarza Forex Factory. Brak lub niepełny feed "
            "oznacza 0 punktów, nie domyślny brak ryzyka."
        )
    st.caption(
        f"Cel netto: +{target_pct:.1f}% po opłatach modelowych "
        f"{scan.get('round_trip_cost_bps_assumption', 20.0):g} bps round trip "
        "(minimum 0,1% na stronę). SL = 1,5×ATR; jeśli poziom wypada poniżej "
        "zera, używany jest fallback 2% poniżej wejścia. "
        "Estymacja jest heurystyczna i nie gwarantuje wyniku."
    )
    ranked_assets = scan.get("ranked_assets", [])
    if not ranked_assets:
        st.error("🚨 BŁĄD: skan nie zwrócił rekordów rankingowych.", icon="🚨")
        return

    rows = []
    for rank, item in enumerate(ranked_assets, start=1):
        rows.append(
            {
                "#": rank,
                "Aktywo": item["symbol"],
                "Sygnał": item.get("signal", "—"),
                "Cena": _format_market_price(item.get("price")),
                "Czas świecy": item.get("timestamp", "—"),
                "Score": (
                    f"{item['broker_points']:.2f}/150"
                    if isinstance(item.get("broker_points"), (int, float))
                    else "—"
                ),
                "RelVol": (
                    f"{item['relative_volume']:.2f}×"
                    if isinstance(item.get("relative_volume"), (int, float))
                    else "—"
                ),
                "Trigger": _format_market_price(item.get("entry_price")),
                "Take-Profit": _format_market_price(item.get("take_profit")),
                "Stop-Loss": _format_market_price(item.get("stop_loss")),
                "Status": (
                    "Kwalifikuje się"
                    if item.get("qualified")
                    else "Błąd danych"
                    if item.get("error")
                    else "Poza kryteriami"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Pełny ranking: {len(ranked_assets)} aktywów, malejąco według "
        "punktacji 0–150. Kliknij kartę aktywa, aby rozwinąć szczegóły."
    )

    for rank, item in enumerate(ranked_assets, start=1):
        points = item.get("broker_points")
        title_score = f"{points:.2f}/150" if isinstance(points, (int, float)) else "brak punktacji"
        with st.expander(f"{rank}. {item['symbol']} · {title_score}"):
            if item.get("error"):
                st.error(f"🚨 {item['error']}", icon="🚨")
                continue
            currency = _price_currency(item["symbol"])
            header_metrics = st.columns(2)
            header_metrics[0].metric(
                f"Cena rynkowa · {currency}",
                _format_market_price(item.get("price")),
                help=f"Ostatnia świeca: {item.get('timestamp', '—')}",
            )
            potential_profit = item.get("potential_profit_pct")
            header_metrics[1].metric(
                "Potencjalny zysk",
                f"{potential_profit:+.2f}%"
                if isinstance(potential_profit, (int, float))
                else "—",
                help=(
                    "Szacowany zysk netto od ceny Trigger do Take-Profit, "
                    "po odjęciu prowizji wejścia i wyjścia."
                ),
            )
            st.caption(
                f"Ostatnia świeca: {item.get('timestamp', '—')} · "
                f"Szacowany horyzont celu: {item.get('estimated_horizon', horizon)}."
            )
            score_rows = [
                {
                    "Składowa": component["label"],
                    "Punkty": f"{component['points']:.2f}/{component['maximum']}",
                    "Szczegóły": component["details"],
                }
                for component in item.get("score_breakdown", [])
            ]
            st.dataframe(
                pd.DataFrame(score_rows),
                hide_index=True,
                use_container_width=True,
            )
            trigger = item.get("entry_price")
            probability = item.get("target_probability_pct")
            if is_short_term:
                st.write(
                    f"**Warunek zakupu (Trigger):** jeśli cena osiągnie "
                    f"{_format_market_price(trigger)} {currency} przy RelVol "
                    f"≥ 1,0 i R/R ≥ 1:2 → KUP."
                )
                st.metric(
                    f"Heurystyczna estymacja score · horyzont {horizon}",
                    f"{probability:.1f}%"
                    if isinstance(probability, (int, float))
                    else "—",
                )
            else:
                st.write(
                    f"**Warunek zakupu (Trigger):** jeśli cena osiągnie "
                    f"{_format_market_price(trigger)} {currency} → KUP."
                )
            st.write(
                f"**Take-Profit:** sprzedaj przy "
                f"{_format_market_price(item.get('take_profit'))} {currency} "
                f"(cel +{target_pct:.1f}% netto po prowizjach)."
            )
            st.write(
                f"**Stop-Loss:** zamknij pozycję przy "
                f"{_format_market_price(item.get('stop_loss'))} {currency}."
            )
            st.write(
                f"**Trailing Stop:** startowo "
                f"{_format_market_price(item.get('trailing_stop'))} {currency}; "
                f"aktywacja po wzroście do "
                f"{_format_market_price(item.get('trailing_stop_activation_price'))} "
                f"{currency}. {item.get('trailing_stop_rule', '')}"
            )
            if item.get("time_stop_rule"):
                st.write(f"**Time-Stop:** {item['time_stop_rule']}")
            if isinstance(item.get("risk_reward_ratio"), (int, float)):
                st.caption(
                    f"Risk/Reward netto po opłatach: "
                    f"1:{item['risk_reward_ratio']:.2f}. "
                    f"Opłata wejścia: {_format_market_price(item.get('entry_fee'))} "
                    f"{currency}."
                )
            if item.get("qualified"):
                st.success("Aktywo spełnia warunki kwalifikacji modelu.")
            else:
                st.warning("Aktywo nie spełnia wszystkich warunków kwalifikacji.")
            for reason in item.get("disqualification_reasons", []):
                st.caption(f"• {reason}")
            candles = item.get("candles", [])
            if candles:
                chart, _ = _chart_data(
                    {
                        "candles": candles[-200:],
                        "signal": item.get("signal"),
                        "entry_price": item.get("entry_price"),
                        "stop_loss": item.get("stop_loss"),
                        "trailing_stop": item.get("trailing_stop"),
                        "take_profit": item.get("take_profit"),
                    }
                )
                if not chart.empty:
                    st.line_chart(chart, height=280)


def _render_scan(scan: dict[str, Any]) -> None:
    if scan.get("analysis_category") in {"short_term", "multi_day"}:
        _render_category_scored_scan(scan)
        return
    mode_label = (
        "Tryb historyczny · backtest"
        if scan.get("mode") == "history"
        else "Tryb na żywo"
    )
    st.header(f"Pełny ranking przeskanowanego rynku · {mode_label}")
    if scan.get("mode") == "history":
        st.warning(
            "Wynik odtworzony z historycznych świec OHLCV. Bieżące newsy, "
            "fundamenty, arkusze zleceń i kalendarze nie są używane jako dane "
            "historyczne; niedostępne filtry mają obniżoną punktację."
        )
    st.caption(
        f"Przeskanowano {scan.get('stock_assets', 0)} akcji US/UE i "
        f"{scan.get('crypto_assets', 0)} aktywów krypto przez Yahoo Finance · "
        f"interwał {scan['timeframe']} · "
        f"aktualizacja: {scan['scanned_at']}"
    )
    if scan.get("data_as_of"):
        st.caption(
            f"Moment analizy: {scan['data_as_of']} "
            "(czas Europe/Warsaw, zapisany jako UTC)."
        )
    st.caption(
        "Dane pobierane z sieci przy każdym skanie; opóźnienie notowań zależy "
        "od Yahoo Finance i giełdy. Czas świecy jest wyświetlany w szczegółach."
    )
    st.caption(
        "Kwalifikacja wymaga sygnału KUP, wyniku >75/100 i zaliczenia filtrów "
        "ADX, wolumenu oraz kalendarza. TP brutto uwzględnia koszty, aby cel "
        "netto wynosił +4,5%; SL = 1,5×ATR. Założony koszt round trip: "
        f"{scan.get('round_trip_cost_bps_assumption', 40.0):g} bps "
        "(estymata, nie wycena taryfy Revolut). Skan nie składa zleceń."
    )

    ranked_assets = scan.get("ranked_assets", [])
    if not ranked_assets:
        st.error("🚨 BŁĄD: skan nie zwrócił rekordów rankingowych.", icon="🚨")
        return

    ranking_rows = []
    for rank, asset in enumerate(ranked_assets, start=1):
        points = asset.get("broker_points")
        price = asset.get("price")
        entry = asset.get("entry_price")
        take_profit = asset.get("take_profit")
        stop_loss = asset.get("stop_loss")
        probability = asset.get("target_probability_pct")
        target_outlook = (
            "Brak estymacji"
            if not isinstance(probability, (int, float))
            else (
                f"Uwaga: tylko {probability:.1f}% w 3 dni"
                if probability < 50
                else f"{probability:.1f}% (heurystyka, 3 dni)"
            )
        )
        ranking_rows.append(
            {
                "#": rank,
                "Aktywo": asset["symbol"],
                "Sygnał": asset.get("signal", "—"),
                f"Ostatnia cena ({_price_currency(asset['symbol'])})": (
                    _format_market_price(price)
                ),
                "Punkty": f"{points:.1f}/100" if points is not None else "—",
                "Prawdopodobieństwo +4,5%": target_outlook,
                "Warunek zakupu": (
                    _buy_trigger_instruction(asset["symbol"], entry)
                    if entry is not None
                    else "Brak danych"
                ),
                "Take-Profit · SPRZEDAJ": (
                    f"Kurs ≥ {_format_market_price(take_profit)} "
                    f"{_price_currency(asset['symbol'])}"
                    if take_profit is not None
                    else "Brak danych"
                ),
                "Stop-Loss · ZAMKNIJ": (
                    f"Kurs ≤ {_format_market_price(stop_loss)} "
                    f"{_price_currency(asset['symbol'])}"
                    if stop_loss is not None
                    else "Brak danych"
                ),
                "Godziny rynku": _market_session_instruction(asset["symbol"]),
                "Status": (
                    "Błąd danych"
                    if asset.get("error")
                    else "Kwalifikuje się"
                    if asset.get("qualified")
                    else "Etap jakościowy pominięty"
                    if asset.get("news", {}).get("status") == "deferred"
                    else "Poza kryteriami"
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(ranking_rows),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Łącznie: {len(ranked_assets)} aktywów · "
        f"{sum(bool(asset.get('qualified')) for asset in ranked_assets)} spełnia "
        "kryteria kwalifikacji."
    )
    selected_symbol = st.selectbox(
        "Wybierz aktywo, aby zobaczyć szczegóły",
        [asset["symbol"] for asset in ranked_assets],
        key="ranked_asset_details",
    )
    item = next(
        asset for asset in ranked_assets if asset["symbol"] == selected_symbol
    )
    st.subheader(f"{item['symbol']} · szczegóły rankingu")

    if item.get("error"):
        st.error(
            f"🚨 Nie udało się pobrać danych dla {item['symbol']}: {item['error']}",
            icon="🚨",
        )
        return

    indicators = item.get("indicators", {})
    metrics = st.columns(4)
    metrics[0].metric(
        f"Ostatnia cena · {_price_currency(item['symbol'])}",
        _format_market_price(item.get("price")),
    )
    metrics[1].metric("Sygnał", item.get("signal", "—"))
    points = item.get("broker_points")
    metrics[2].metric(
        "Punktacja",
        f"{points:.1f}/100" if points is not None else "—",
    )
    metrics[3].metric("ADX (14)", _format_number(indicators.get("adx_14"), 2))
    st.caption(
        f"Źródło: Yahoo Finance · interwał {item.get('timeframe', scan['timeframe'])} · "
        f"ostatnia świeca: {item.get('timestamp', '—')}"
    )
    st.info(
        "Horyzont analizy swing-trading: do 3 dni. Prawdopodobieństwo w rankingowej "
        "kolumnie jest heurystyką, nie skalibrowaną prognozą. Jeśli wynik jest "
        "poniżej 50%, traktuj to jako ostrzeżenie, że cel +4,5% może nie zostać "
        "osiągnięty w tym horyzoncie."
    )
    st.caption(
        "* Godzina otwarcia USA jest zwykle 15:30 czasu polskiego; przez "
        "kilka tygodni różnicy w zmianie czasu może to być 14:30. "
        "Notowania Yahoo mogą być opóźnione."
    )

    liquidity = item.get("broker_inputs", {}).get("liquidity", {})
    volume_columns = st.columns(4)
    volume_columns[0].metric(
        "Wolumen ostatniej sesji",
        _format_number(liquidity.get("latest_completed_session_volume"), 0),
    )
    volume_columns[1].metric(
        "Średni wolumen · 20 sesji",
        _format_number(liquidity.get("average_prior_20_session_volume"), 0),
    )
    volume_ratio = liquidity.get("session_volume_ratio")
    volume_columns[2].metric(
        "Wolumen / średnia",
        f"{volume_ratio:.2f}×"
        if isinstance(volume_ratio, (int, float))
        else "—",
    )
    volume_columns[3].metric(
        "Płynność / filtr wolumenu",
        str(
            liquidity.get(
                "volume_session_status",
                liquidity.get("status", "brak danych"),
            )
        ),
    )

    score = item.get("broker_score")
    filters = score.get("filters", {}) if isinstance(score, dict) else {}
    if filters:
        score_rows = [
            {
                "Filtr": filter_data["label"],
                "Punkty": (
                    f"{filter_data['points']:.1f}/{filter_data['maximum']}"
                ),
                "Status": filter_data["status"],
                "Szczegóły": str(filter_data["details"]),
            }
            for filter_data in filters.values()
        ]
        st.subheader("Rozkład punktacji i statystyki filtrów")
        st.dataframe(
            pd.DataFrame(score_rows),
            hide_index=True,
            use_container_width=True,
        )
        for filter_data in filters.values():
            with st.expander(f"{filter_data['label']} · szczegóły"):
                st.write(filter_data["details"])

    st.subheader("Wytyczne cenowe · Revolut")
    st.caption(
        "Poziomy są scenariuszem obliczeniowym na podstawie ostatniej ceny i ATR; "
        "nie są rekomendacją ani zleceniem. Założenie kosztu obejmuje co najmniej "
        "0,1% przy zakupie i 0,1% przy sprzedaży. "
        f"{_market_session_instruction(item['symbol'])}."
    )
    trade_columns = st.columns(3)
    trade_columns[0].metric(
        f"Cena wejścia · {_price_currency(item['symbol'])}",
        _format_market_price(item.get("entry_price")),
    )
    trade_columns[1].metric(
        f"Take-Profit · {_price_currency(item['symbol'])}",
        _format_market_price(item.get("take_profit")),
    )
    trade_columns[2].metric(
        f"Stop-Loss · {_price_currency(item['symbol'])}",
        _format_market_price(item.get("stop_loss")),
    )
    target_probability = item.get("target_probability_pct")
    if isinstance(target_probability, (int, float)) and target_probability < 50:
        st.warning(
            f"Model szacuje {target_probability:.1f}% heurystycznej szansy "
            "osiągnięcia celu +4,5% w ciągu 3 dni. To poniżej 50% — cel może "
            "nie zostać osiągnięty w zakładanym horyzoncie."
        )
    elif isinstance(target_probability, (int, float)):
        st.caption(
            f"Szansa celu według heurystyki: {target_probability:.1f}% "
            "w ciągu 3 dni; nie jest to gwarancja wyniku."
        )
    if item.get("qualified"):
        st.success(
            "Aktywo spełnia warunki kwalifikacji skanera (>75/100 i brak blokady)."
        )
    else:
        st.warning(
            "Aktywo nie spełnia pełnych warunków kwalifikacji; poziomy cenowe "
            "powyżej są wyłącznie scenariuszem informacyjnym."
        )
    for reason in item.get("disqualification_reasons", []):
        st.caption(f"• {reason}")

    candles = item.get("candles", [])
    if candles:
        detail_chart, _ = _chart_data(
            {
                "candles": candles,
                "signal": item.get("signal"),
                "entry_price": item.get("entry_price"),
                "stop_loss": item.get("stop_loss"),
                "trailing_stop": item.get("trailing_stop"),
                "take_profit": item.get("take_profit"),
            }
        )
        if not detail_chart.empty:
            st.line_chart(detail_chart, height=300)


def _render_detail(
    report: dict[str, Any],
    round_trip_cost_bps: float,
) -> None:
    if report.get("is_mock"):
        st.warning(
            "Yahoo Finance było niedostępne. Dane syntetyczne są tylko testowe; "
            "sygnał transakcyjny jest zablokowany."
        )

    st.subheader(f"{report.get('symbol', 'Aktywo')} — szczegółowa analiza")
    metrics = st.columns(5)
    metrics[0].metric("Aktualna cena", _format_price(report.get("latest_price")))
    metrics[1].metric("Sygnał", report.get("signal", "—"))
    metrics[2].metric("Stop-Loss", _format_price(report.get("stop_loss")))
    metrics[3].metric("Take-Profit", _format_price(report.get("take_profit")))
    confidence = report.get("confidence_score_pct")
    metrics[4].metric(
        "Confidence Score",
        "—" if confidence is None else f"{confidence:.1f}%",
        help="Heurystyka, nie skalibrowane prawdopodobieństwo zysku.",
    )
    if report.get("confidence_note"):
        st.caption(report["confidence_note"])

    broker_inputs = report.get("broker_inputs")
    if isinstance(broker_inputs, dict):
        broker_score = RiskManager().calculate_broker_filter_score(
            {
                "price": report["latest_price"],
                "indicators": report["indicators"],
                "broker_inputs": broker_inputs,
                "is_mock": report["is_mock"],
                "signal": report["signal"],
                "news": report["news"],
            },
            round_trip_cost_bps=round_trip_cost_bps,
        )
        st.subheader("Pięć filtrów maklerskich")
        st.metric(
            "Wynik łączny",
            f"{broker_score['total_points']:.1f}/100",
            help="Kwalifikacja wymaga wyniku ściśle powyżej 75 punktów.",
        )
        if broker_score["qualified"]:
            st.success("Wynik przekracza próg punktowy i nie ma blokad event-risk.")
        else:
            st.warning(
                "Aktywo nie kwalifikuje się do Top 3: "
                + " ".join(broker_score["disqualification_reasons"])
            )
        filter_rows = [
            {
                "Filtr": filter_data["label"],
                "Punkty": (
                    f"{filter_data['points']:.1f}/{filter_data['maximum']}"
                ),
                "Status": filter_data["status"],
            }
            for filter_data in broker_score["filters"].values()
        ]
        st.dataframe(
            pd.DataFrame(filter_rows),
            hide_index=True,
            use_container_width=True,
        )
        for filter_data in broker_score["filters"].values():
            with st.expander(filter_data["label"]):
                st.json(filter_data["details"])

    st.subheader("Wykres cenowy i poziomy ryzyka")
    try:
        chart, scenario_levels = _chart_data(report)
        if chart.empty:
            st.info("Brak świec do wyświetlenia.")
        else:
            st.line_chart(chart, height=400)
            if scenario_levels:
                st.caption(
                    "Sygnał CZEKAJ: linie SL/TP pokazują wyłącznie hipotetyczne "
                    "scenariusze long i short (SL = 1,5×ATR, TP = R:R 1:2); "
                    "nie są rekomendacją."
                )
    except (TypeError, ValueError, KeyError) as error:
        LOGGER.exception("Nie udało się przygotować wykresu")
        st.error(f"Nie udało się wyświetlić wykresu: {error}")

    indicators = report.get("indicators", {})
    if isinstance(indicators, dict):
        st.subheader("Wskaźniki techniczne")
        indicator_cards = st.columns(5)
        indicator_cards[0].metric("RSI (14)", _format_number(indicators.get("rsi_14")))
        indicator_cards[1].metric("MACD", _format_number(indicators.get("macd")))
        indicator_cards[2].metric(
            "Linia sygnału MACD",
            _format_number(indicators.get("macd_signal")),
        )
        indicator_cards[3].metric("ATR (14)", _format_number(indicators.get("atr_14")))
        adx_value = indicators.get("adx_14")
        indicator_cards[4].metric(
            "ADX (14) · filtr",
            _format_number(adx_value, 1),
            delta="Trend" if isinstance(adx_value, (int, float)) and adx_value >= 20 else "Wstrzymany",
            delta_color="normal" if isinstance(adx_value, (int, float)) and adx_value >= 20 else "inverse",
            help="ADX < 20 obniża punkty techniczne i blokuje kwalifikację.",
        )

    components = report.get("confidence_components", {})
    if isinstance(components, dict):
        labels = (
            ("technical", "Technika"),
            ("news_sentiment", "Sentyment wiadomości"),
            ("fundamental", "Fundamenty"),
        )
        rendered = [
            f"{label}: {value:.1f}%"
            for key, label in labels
            if (value := components.get(key)) is not None
        ]
        if rendered:
            st.caption("Składowe Confidence Score: " + " · ".join(rendered))

    _render_news(report.get("news"))
    _render_fundamentals(report.get("fundamentals"))
    st.subheader("Pełny raport WriterAgenta")
    st.text(report.get("report_text", "Brak raportu tekstowego."))


def main() -> None:
    st.set_page_config(
        page_title="MAFSE — Ranking rynku",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 Ranking okazji inwestycyjnych")
    analysis_category = st.radio(
        "Kategoria horyzontu czasowego",
        (MULTI_DAY_CATEGORY, SHORT_TERM_CATEGORY),
        horizontal=True,
        key="analysis_category",
    )
    st.caption(
        "Skan popularnych akcji US/UE i kryptowalut przez Yahoo Finance. "
        "Notowania mogą być opóźnione przez dostawcę lub giełdę; analiza ma "
        "charakter informacyjny i nie składa zleceń."
    )

    analysis_mode = st.radio(
        "Tryb analizy",
        (
            "🔴 Tryb na żywo (Dane w czasie rzeczywistym)",
            "📜 Tryb historyczny (Backtest w wybranym momencie)",
        ),
        horizontal=True,
        key="analysis_mode",
    )
    is_historical = analysis_mode.startswith("📜")
    selected_market = st.selectbox("Skanowany rynek", MARKET_OPTIONS)
    scan_symbols = _symbols_for_market(selected_market)
    round_trip_cost_bps = ROUND_TRIP_COST_BPS
    snapshot_at: datetime | None = None
    if is_historical:
        now_local = datetime.now(ZoneInfo("Europe/Warsaw"))
        date_column, time_column = st.columns(2)
        selected_day = date_column.date_input(
            "Data historyczna",
            value=now_local.date(),
            max_value=now_local.date(),
            key="historical_date",
        )
        selected_clock = time_column.time_input(
            "Godzina i minuta",
            value=now_local.time().replace(second=0, microsecond=0),
            key="historical_time",
        )
        snapshot_at = _historical_snapshot(selected_day, selected_clock)
        st.caption(
            "Godzina podawana w czasie Europe/Warsaw. "
            + (
                "Skan 15m używa ukończonych świec Yahoo Finance; sesje "
                "przed- i posesyjne są uwzględniane, jeśli dostawca je udostępnia. "
                "Historia intraday jest zwykle ograniczona do ok. 60 dni."
                if analysis_category == SHORT_TERM_CATEGORY
                else "Skan dzienny odrzuca świecę z dnia historycznego snapshotu; "
                "źródło Yahoo Finance obejmuje maksymalnie około 5 lat."
            )
        )
    if analysis_category == SHORT_TERM_CATEGORY:
        st.caption(
            "Tryb szybki: świece 15m i przygotowanie do sesji USA od 14:00 "
            "czasu PL (pre-market, gdy dostępny); regularne otwarcie USA "
            "przypada zwykle o 15:30, okresowo o 14:30."
        )
    st.caption(
        f"Zakres: {len(set(scan_symbols) & set(STOCK_UNIVERSE))} akcji US/UE, "
        f"{len(set(scan_symbols) & set(CRYPTO_UNIVERSE))} kryptowalut. "
        "Prowizja modelowa: min. 0,1% przy zakupie + 0,1% przy sprzedaży "
        "(20 bps round trip); spread i rzeczywista taryfa mogą być wyższe. "
        "Dostępność instrumentów Revolut zależy od kraju i konta."
    )

    start_scan = st.button(
        (
            f"📜 Uruchom analizę historyczną · {snapshot_at:%Y-%m-%d %H:%M}"
            if snapshot_at is not None
            else (
                f"⚡ Skanuj {len(scan_symbols)} instrumentów · krótkoterminowo"
                if analysis_category == SHORT_TERM_CATEGORY
                else f"🔴 Skanuj {len(scan_symbols)} instrumentów na żywo"
            )
        ),
        type="primary",
        use_container_width=True,
    )
    if start_scan:
        _cached_gemini_news_review.clear()
        if snapshot_at is not None and snapshot_at > datetime.now(
            ZoneInfo("Europe/Warsaw")
        ):
            st.error(
                "🚨 Data historyczna nie może wskazywać przyszłości.",
                icon="🚨",
            )
        else:
            try:
                with st.spinner(
                    (
                        f"Odtwarzam świece do {snapshot_at:%Y-%m-%d %H:%M} "
                        f"dla {len(scan_symbols)} tickerów..."
                        if snapshot_at is not None
                        else f"Pobieram aktualne dane dla {len(scan_symbols)} "
                        "tickerów i obliczam punktację..."
                    )
                ):
                    st.session_state["market_scan_result"] = _scan_market(
                        scan_symbols,
                        float(round_trip_cost_bps),
                        as_of=snapshot_at,
                        analysis_category=analysis_category,
                    )
                    st.session_state["market_scan_symbols"] = scan_symbols
                    st.session_state["market_scan_category"] = analysis_category
                    st.session_state["market_scan_as_of"] = (
                        snapshot_at.isoformat()
                        if snapshot_at is not None
                        else None
                    )
                st.session_state.pop("market_scan_error", None)
                st.session_state.pop("market_scan_error_symbols", None)
                st.session_state.pop("market_scan_error_category", None)
            except Exception as error:
                LOGGER.exception("Skan całego rynku nie powiódł się")
                st.session_state.pop("market_scan_result", None)
                st.session_state["market_scan_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                st.session_state["market_scan_error_symbols"] = scan_symbols
                st.session_state["market_scan_error_category"] = analysis_category
                st.session_state["market_scan_error_as_of"] = (
                    snapshot_at.isoformat() if snapshot_at is not None else None
                )

    scan_error = st.session_state.get("market_scan_error")
    scan_result = st.session_state.get("market_scan_result")
    requested_snapshot = snapshot_at.isoformat() if snapshot_at is not None else None
    if (
        scan_error
        and st.session_state.get("market_scan_error_symbols") == scan_symbols
        and st.session_state.get("market_scan_error_category") == analysis_category
        and st.session_state.get("market_scan_error_as_of") == requested_snapshot
    ):
        st.error(f"🚨 BŁĄD SKANOWANIA RYNKU\n\n{scan_error}", icon="🚨")
    elif (
        isinstance(scan_result, dict)
        and st.session_state.get("market_scan_symbols") == scan_symbols
        and st.session_state.get("market_scan_category") == analysis_category
        and st.session_state.get("market_scan_as_of") == requested_snapshot
    ):
        _render_scan(scan_result)
        errors = scan_result.get("errors", [])
        if errors:
            st.error(
                f"🚨 BŁĄD DANYCH: pobieranie nie powiodło się dla "
                f"{len(errors)} z {scan_result.get('scanned_assets', len(scan_symbols))} "
                "instrumentów. Wynik skanu jest niepełny; nie traktuj go jako "
                "pełnego obrazu rynku.",
                icon="🚨",
            )
            with st.expander("Szczegóły błędów pobierania danych"):
                st.dataframe(
                    pd.DataFrame(errors),
                    hide_index=True,
                    use_container_width=True,
                )
        quality_errors = scan_result.get("quality_errors", [])
        if quality_errors:
            st.error(
                f"🚨 BŁĄD ANALIZY JAKOŚCIOWEJ: nie udało się pobrać części "
                f"konsensusu, wiadomości lub kalendarza dla "
                f"{len(quality_errors)} kandydatów. Niedostępne źródła otrzymują "
                "zero punktów.",
                icon="🚨",
            )
            with st.expander("Szczegóły błędów źródeł jakościowych"):
                st.dataframe(
                    pd.DataFrame(quality_errors),
                    hide_index=True,
                    use_container_width=True,
                )
        if (
            scan_result.get("scanned_assets", 0)
            and not scan_result.get("ranked_assets")
        ):
            st.error(
                "🚨 BŁĄD: nie udało się uzyskać żadnych użytecznych danych "
                "rynkowych. Sprawdź połączenie z internetem i dostępność "
                "Yahoo Finance.",
                icon="🚨",
            )
    else:
        st.info(
            "Wybierz zakres rynku i uruchom skan przyciskiem, aby pobrać świeże "
            "dane sieciowe oraz zobaczyć pełny ranking wszystkich aktywów."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("Nieobsłużony błąd dashboardu")
        st.error(
            f"🚨 WYSTĄPIŁ BŁĄD APLIKACJI: {type(error).__name__}: {error}",
            icon="🚨",
        )
