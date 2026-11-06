"""Streamlit market scanner and detailed MAFSE analysis dashboard."""

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import Any

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
MARKET_UNIVERSE = ("BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "AAPL")
ASSET_OPTIONS = (*MARKET_UNIVERSE, "Inny ticker")
SCAN_TIMEFRAME = "4h"
SCAN_CANDLE_LIMIT = 60
NETWORK_TIMEOUT_SECONDS = 4.0
STAGE_ONE_MINIMUM_POINTS = 40.0
STAGE_TWO_MAX_ASSETS = 3


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
    return {
        "symbol": symbol,
        "signal": signal,
        "price": research["price"],
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
        "is_mock": research["is_mock"],
        "timestamp": research["timestamp"],
        "candles": research["candles"],
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
    item = _scan_item(symbol, enriched, score, stage_one_points)
    if score["qualified"] and enriched["signal"] == "KUP":
        risk = RiskManager().calculate_position(
            entry_price=float(enriched["price"]),
            atr=float(enriched["indicators"]["atr_14"]),
            portfolio_capital=10_000.0,
            direction="long",
        )
        item.update(
            {
                "entry_price": risk["entry_price"],
                "stop_loss": risk["stop_loss"],
                "take_profit": risk["take_profit"],
                "position_size": risk["position_size"],
                "risk_reward_ratio": risk["risk_reward_ratio"],
            }
        )
    return item, bool(score["qualified"] and enriched["signal"] == "KUP")


@st.cache_data(ttl=900, show_spinner=False)
def _scan_market(
    symbols: tuple[str, ...],
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    """Cache a two-stage scan; only strong technical candidates reach Gemini."""
    researcher = ResearchAgent(timeout=NETWORK_TIMEOUT_SECONDS)
    recommendations: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    stage_one: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def scan_technical(symbol: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        research = researcher.analyze(
            symbol=symbol,
            timeframe=SCAN_TIMEFRAME,
            limit=SCAN_CANDLE_LIMIT,
            include_qualitative=False,
        )
        score = RiskManager().calculate_broker_filter_score(
            research,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        return symbol, research, score

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(5, max(1, len(symbols)))
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
            candidates.append((symbol, research, score, technical_and_risk))
        else:
            watchlist.append(
                _scan_item(symbol, research, score, technical_and_risk)
            )

    candidates.sort(key=lambda candidate: candidate[3], reverse=True)
    deferred_candidates = candidates[STAGE_TWO_MAX_ASSETS:]
    candidates = candidates[:STAGE_TWO_MAX_ASSETS]
    for symbol, research, score, stage_one_points in deferred_candidates:
        item = _scan_item(symbol, research, score, stage_one_points)
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
            except Exception as error:
                LOGGER.warning("Etap jakościowy %s nie powiódł się: %s", symbol, error)
                fallback_item = _scan_item(
                    symbol,
                    research,
                    stage_one_score,
                    technical_and_risk,
                )
                fallback_item["disqualification_reasons"].append(
                    f"Etap jakościowy pominięty: {type(error).__name__}."
                )
                watchlist.append(fallback_item)

    recommendations.sort(
        key=lambda item: item["broker_points"],
        reverse=True,
    )
    return {
        "recommendations": recommendations[:3],
        "watchlist": watchlist,
        "errors": errors,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": SCAN_TIMEFRAME,
        "qualification_rule": "suma > 75 oraz brak blokady wydarzenia",
        "round_trip_cost_bps_assumption": round_trip_cost_bps,
        "stage_one_minimum_points": STAGE_ONE_MINIMUM_POINTS,
        "stage_two_max_assets": STAGE_TWO_MAX_ASSETS,
        "stage_two_reviewed_assets": len(candidates),
    }


@st.cache_data(ttl=900, show_spinner=False)
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
            round_trip_cost_bps=round_trip_cost_bps,
        )
        return report
    finally:
        client.close()


@st.cache_data(ttl=900, show_spinner=False)
def _calculate_revolut_position(
    symbol: str,
    purchase_price: float,
    round_trip_cost_bps: float = 40.0,
) -> dict[str, Any]:
    research = ResearchAgent(timeout=NETWORK_TIMEOUT_SECONDS).analyze(
        symbol=symbol,
        timeframe=SCAN_TIMEFRAME,
        limit=SCAN_CANDLE_LIMIT,
        round_trip_cost_bps=round_trip_cost_bps,
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


def _select_detail_asset(symbol: str) -> None:
    st.session_state["detail_asset_option"] = symbol


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


def _render_scan(scan: dict[str, Any]) -> None:
    st.header("Top 3 Okazje na 1–3 dni")
    st.caption(
        "Skan techniczny i sentyment zweryfikowanych wiadomości dla "
        f"{', '.join(MARKET_UNIVERSE)} · interwał {scan['timeframe']} · "
        f"czas skanu: {scan['scanned_at']}"
    )
    st.caption(
        "Selekcja dwuetapowa: etap jakościowy uruchomiono dla "
        f"{scan.get('stage_two_reviewed_assets', 0)} najlepszych kandydatów "
        f"(próg wstępny {scan.get('stage_one_minimum_points', STAGE_ONE_MINIMUM_POINTS):g}/55)."
    )
    st.warning(
        "Do Top 3 trafiają wyłącznie pozycje KUP z punktacją ściśle >75/100 "
        "i bez blokady event-risk. Punktacja jest modelem regułowym, nie "
        "skalibrowanym prawdopodobieństwem zysku. Skan nie składa zleceń."
    )
    st.info(
        "Kompletny kalendarz makro nie jest obecnie dostępny. Zgodnie z "
        "konserwatywną regułą fail-closed blokuje to kwalifikację do Top 3; "
        "brak danych nie jest traktowany jako brak wydarzeń."
    )
    st.caption(
        "Koszt round trip użyty w R/R: "
        f"{scan.get('round_trip_cost_bps_assumption', 40.0):g} bps "
        "(założenie użytkownika, nie aktualna taryfa brokera)."
    )

    recommendations = scan.get("recommendations", [])
    if recommendations:
        columns = st.columns(len(recommendations))
        for index, item in enumerate(recommendations):
            with columns[index]:
                with st.container(border=True):
                    st.subheader(f"#{index + 1} · {item['symbol']}")
                    st.metric("Sygnał", item["signal"])
                    score = item.get("confidence_score_pct")
                    st.metric(
                        "Filtry maklerskie",
                        f"{item['broker_points']:.1f}/100",
                        help="Kwalifikacja wymaga wyniku ściśle powyżej 75 punktów.",
                    )
                    st.caption(
                        f"Estymata heurystyczna celu +4,5%: "
                        f"{'—' if score is None else f'{score:.1f}%'}"
                    )
                    review = item["news"].get("gemini_review")
                    if isinstance(review, dict):
                        st.caption(
                            "Gemini — zgodność źródeł: "
                            f"{review['status']}. {review['rationale']}"
                        )
                    st.metric("Cena", _format_price(item["price"]))
                    st.caption(
                        f"RSI {item['indicators']['rsi_14']:.1f} · "
                        f"MACD {item['indicators']['macd']:.4f} · "
                        f"ATR {item['indicators']['atr_14']:.4f} · "
                        f"ADX {item['indicators']['adx_14']:.1f} "
                        f"({'trend' if item['indicators']['adx_14'] >= 20 else 'wstrzymany'})"
                    )
                    st.caption(
                        f"SL {_format_price(item['stop_loss'])} · "
                        f"TP {_format_price(item['take_profit'])} · "
                        f"R:R {item['risk_reward_ratio']:.1f}:1"
                    )
                    detail_chart, _ = _chart_data(
                        {
                            "candles": item["candles"],
                            "signal": item["signal"],
                            "entry_price": item["entry_price"],
                            "stop_loss": item["stop_loss"],
                            "take_profit": item["take_profit"],
                        }
                    )
                    if not detail_chart.empty:
                        st.line_chart(detail_chart, height=180)
                    filters = item["broker_score"]["filters"]
                    score_rows = [
                        {
                            "Filtr": filter_data["label"],
                            "Punkty": f"{filter_data['points']:.1f}/{filter_data['maximum']}",
                            "Status": filter_data["status"],
                        }
                        for filter_data in filters.values()
                    ]
                    with st.expander("Breakdown 5 filtrów i uzasadnienie"):
                        st.dataframe(
                            pd.DataFrame(score_rows),
                            hide_index=True,
                            use_container_width=True,
                        )
                        for filter_data in filters.values():
                            st.caption(
                                f"{filter_data['label']}: "
                                f"{filter_data['details']}"
                            )
                    st.button(
                        "Szczegółowa analiza",
                        key=f"scan_detail_{item['symbol']}",
                        on_click=_select_detail_asset,
                        args=(item["symbol"],),
                        use_container_width=True,
                    )
    else:
        st.info(
            "Brak aktywów spełniających jednocześnie sygnał KUP, wynik >75/100 "
            "i wszystkie blokady kwalifikacyjne. Aktywa niespełniające kryteriów "
            "nie są prezentowane jako rekomendacje."
        )

    if scan.get("watchlist"):
        with st.expander("Pozostałe aktywa — brak kwalifikacji do Top 3"):
            for item in scan["watchlist"]:
                score = item.get("broker_points", 0.0)
                st.markdown(
                    f"**{item['symbol']}** · {item['signal']} · "
                    f"{_format_price(item['price'])} · {score:.1f}/100"
                    + (" · dane syntetyczne" if item["is_mock"] else "")
                )
                reasons = item.get("disqualification_reasons", [])
                if reasons:
                    st.caption(" · ".join(reasons))
                filters = item.get("broker_score", {}).get("filters", {})
                if filters:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Filtr": filter_data["label"],
                                    "Punkty": (
                                        f"{filter_data['points']:.1f}/"
                                        f"{filter_data['maximum']}"
                                    ),
                                    "Status": filter_data["status"],
                                }
                                for filter_data in filters.values()
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

    if scan.get("errors"):
        with st.expander("Aktywa, których nie udało się przeskanować"):
            for item in scan["errors"]:
                st.error(f"{item['symbol']}: {item['error']}")


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
        page_title="MAFSE — Skaner rynku",
        page_icon="📈",
        layout="wide",
    )
    st.title("📈 MAFSE — Skaner rynku i dashboard")
    st.caption(
        "PlannerAgent · ResearchAgent · RiskManager · WriterAgent. "
        "Aplikacja analizuje dane, nie składa zleceń."
    )

    with st.sidebar:
        st.header("Ustawienia")
        selected_option = st.selectbox(
            "Aktywo do szczegółowej analizy",
            ASSET_OPTIONS,
            index=0,
            key="detail_asset_option",
        )
        custom_symbol = ""
        if selected_option == "Inny ticker":
            custom_symbol = st.text_input(
                "Ticker Yahoo Finance",
                placeholder="np. MSFT lub BTC-USD",
            )
        capital = st.number_input(
            "Kapitał portfela (USD)",
            min_value=100.0,
            value=10_000.0,
            step=500.0,
        )
        round_trip_cost_bps = st.number_input(
            "Założony koszt transakcji (round trip, bps)",
            min_value=0.0,
            max_value=1_000.0,
            value=40.0,
            step=5.0,
            help=(
                "Założenie dla spreadu i prowizji w obie strony, nie rzeczywista "
                "wycena brokera. 100 bps = 1%."
            ),
        )
        run_detail = st.button(
            "Uruchom szczegółową analizę",
            type="primary",
            use_container_width=True,
        )
        refresh_scan = st.button("Odśwież skan rynku", use_container_width=True)
        st.caption("Skan automatyczny jest cache’owany przez 15 minut.")
        with st.expander("Pozycja z Revoluta", expanded=True):
            revolut_asset_option = st.selectbox(
                "Walor",
                ASSET_OPTIONS,
                key="revolut_asset_option",
            )
            revolut_custom_symbol = ""
            if revolut_asset_option == "Inny ticker":
                revolut_custom_symbol = st.text_input(
                    "Ticker pozycji",
                    placeholder="np. MSFT lub BTC-USD",
                    key="revolut_custom_symbol",
                )
            revolut_purchase_price = st.number_input(
                "Cena zakupu za jednostkę",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.8f",
                key="revolut_purchase_price",
                help="Wpisz cenę zakupu z Revoluta w walucie notowań danego tickera.",
            )
            calculate_revolut = st.button(
                "Wylicz cele pozycji",
                key="calculate_revolut_position",
                use_container_width=True,
            )

    if refresh_scan:
        _scan_market.clear()
        _cached_gemini_news_review.clear()
        st.session_state.pop("market_scan_result", None)
    if "market_scan_result" not in st.session_state:
        try:
            with st.spinner(
                "Etap 1: technika i ryzyko. Etap 2: weryfikacja jakościowa "
                "wyłącznie dla najlepszych kandydatów..."
            ):
                st.session_state["market_scan_result"] = _scan_market(
                    MARKET_UNIVERSE,
                    float(round_trip_cost_bps),
                )
        except Exception as error:
            LOGGER.exception("Skan rynku nie powiódł się")
            st.session_state["market_scan_result"] = {
                "recommendations": [],
                "watchlist": [],
                "errors": [
                    {"symbol": "Skan rynku", "error": f"{type(error).__name__}: {error}"}
                ],
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": SCAN_TIMEFRAME,
            }

    _render_scan(st.session_state["market_scan_result"])
    st.divider()

    if calculate_revolut:
        try:
            if revolut_asset_option == "Inny ticker":
                revolut_symbol = _normalize_user_symbol(revolut_custom_symbol)
            else:
                revolut_symbol = revolut_asset_option
            if (
                not math.isfinite(revolut_purchase_price)
                or revolut_purchase_price <= 0
            ):
                raise ValueError("Cena zakupu musi być większa od zera.")
            with st.spinner(
                f"Pobieram ATR i dane rynkowe dla {revolut_symbol}..."
            ):
                st.session_state["revolut_position_result"] = (
                    _calculate_revolut_position(
                        revolut_symbol,
                        float(revolut_purchase_price),
                        float(round_trip_cost_bps),
                    )
                )
            st.session_state.pop("revolut_position_error", None)
        except (RuntimeError, ValueError) as error:
            LOGGER.exception("Nie udało się obliczyć pozycji z Revoluta")
            st.session_state.pop("revolut_position_result", None)
            st.session_state["revolut_position_error"] = str(error)
        except Exception as error:
            LOGGER.exception("Nieoczekiwany błąd wyliczania pozycji z Revoluta")
            st.session_state.pop("revolut_position_result", None)
            st.session_state["revolut_position_error"] = (
                f"{type(error).__name__}: {error}"
            )

    revolut_error = st.session_state.get("revolut_position_error")
    if revolut_error:
        st.error(f"Nie udało się obliczyć celów pozycji: {revolut_error}")
    try:
        current_revolut_symbol = (
            _normalize_user_symbol(revolut_custom_symbol)
            if revolut_asset_option == "Inny ticker"
            and revolut_custom_symbol.strip()
            else (
                None
                if revolut_asset_option == "Inny ticker"
                else revolut_asset_option
            )
        )
    except ValueError:
        current_revolut_symbol = None
    revolut_result = st.session_state.get("revolut_position_result")
    if (
        isinstance(revolut_result, dict)
        and revolut_result["position"]["purchase_price"] == revolut_purchase_price
        and revolut_result["symbol"] == current_revolut_symbol
    ):
        _render_revolut_position(revolut_result)
        st.divider()

    detail_symbol: str | None = None
    try:
        if selected_option == "Inny ticker":
            if custom_symbol.strip():
                detail_symbol = _normalize_user_symbol(custom_symbol)
        else:
            detail_symbol = selected_option
    except ValueError as error:
        st.sidebar.error(str(error))

    if run_detail:
        if detail_symbol is None:
            st.error("Wybierz aktywo lub podaj poprawny ticker.")
        elif not math.isfinite(capital) or capital <= 0:
            st.error("Kapitał musi być skończony i większy od zera.")
        else:
            try:
                with st.spinner(
                    f"Planner, Research, RiskManager i Writer analizują {detail_symbol}..."
                ):
                    st.session_state["detail_report"] = _run_detail_analysis(
                        detail_symbol,
                        float(capital),
                        float(round_trip_cost_bps),
                    )
                    st.session_state["detail_report_symbol"] = detail_symbol
                st.success("Szczegółowa analiza zakończona.")
            except ServerError as error:
                LOGGER.exception("Błąd serwera Gemini")
                if error.code == 503:
                    st.error(
                        "Gemini jest chwilowo niedostępne (HTTP 503); "
                        "automatyczne ponowienia zostały wyczerpane."
                    )
                else:
                    st.error(f"Błąd serwera Gemini (HTTP {error.code}): {error}")
            except APIError as error:
                LOGGER.exception("Błąd API Gemini")
                st.error(f"Błąd API Gemini (HTTP {error.code}): {error}")
            except (RuntimeError, ValueError) as error:
                LOGGER.exception("Nie udało się wykonać szczegółowej analizy")
                st.error(f"Nie udało się wykonać analizy: {error}")
            except Exception as error:
                LOGGER.exception("Nieoczekiwany błąd szczegółowej analizy")
                st.error(
                    "Nieoczekiwany błąd podczas analizy: "
                    f"{type(error).__name__}: {error}"
                )

    report = st.session_state.get("detail_report")
    report_symbol = st.session_state.get("detail_report_symbol")
    if isinstance(report, dict) and report_symbol == detail_symbol:
        _render_detail(report, float(round_trip_cost_bps))
    elif not run_detail:
        st.info(
            "Wybierz aktywo w panelu bocznym i uruchom szczegółową analizę. "
            "Możesz też wybrać aktywo z kart Top 3."
        )


if __name__ == "__main__":
    main()
