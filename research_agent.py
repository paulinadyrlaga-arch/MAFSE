"""Fetch market OHLCV data and calculate pandas-ta indicators."""

import argparse
import concurrent.futures
import email.utils
import json
import logging
import math
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from requests.exceptions import RequestException
import yfinance as yf
from binance.exceptions import BinanceAPIException, BinanceRequestException

from binance_testnet import (
    get_futures_testnet_klines,
    get_testnet_klines,
    get_testnet_order_book,
    to_binance_symbol,
)


_LOGGER = logging.getLogger(__name__)
_YFINANCE_DATA_LOCK = threading.RLock()


class ResearchAgent:
    """Fetch OHLCV candles, technical indicators, and a preliminary signal."""

    MINIMUM_ROUND_TRIP_COST_BPS = 20.0
    _TRUSTED_NEWS_DOMAINS = {
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "bloomberg.com",
        "cnbc.com",
        "ft.com",
        "reuters.com",
        "wsj.com",
        "yahoo.com",
    }
    _POSITIVE_WORDS = {
        "approved", "beat", "growth", "gain", "launch", "partnership", "profit",
        "record", "rally", "surge", "upgrade", "wzrost", "zysk", "zatwierdz",
        "partnerstwo", "rekord", "zwyżk", "podwyższ",
    }
    _NEGATIVE_WORDS = {
        "ban", "decline", "downgrade", "drop", "fraud", "loss", "lawsuit",
        "probe", "recall", "reject", "risk", "selloff", "slump", "spad",
        "strat", "pozew", "śledztw", "oszustw", "zakaz", "obniż", "ryzyk",
    }
    _NEWS_STOP_WORDS = {
        "about", "after", "amid", "from", "into", "over", "that", "their",
        "this", "with", "and", "for", "the", "of", "a", "an", "to", "in",
        "is", "on", "as", "by", "at", "or", "new", "says", "said", "dla",
        "oraz", "jest", "się", "przez", "pod", "nad", "po", "na", "w", "z",
        "do", "od", "i", "o",
    }
    _RUMOR_MARKERS = {
        "rumor",
        "rumour",
        "unconfirmed",
        "speculation",
        "speculative",
        "alleged",
        "unverified",
        "plotka",
        "niepotwierdz",
        "spekulac",
        "rzekom",
    }
    _ASSET_SEARCH_TERMS = {
        "BTC-USD": "Bitcoin BTC",
        "ETH-USD": "Ethereum Ether ETH",
        "AAPL": "Apple Inc AAPL",
    }

    INTERVALS = {
        "1m": ("1m", None),
        "3m": ("1m", "3min"),
        "5m": ("5m", None),
        "15m": ("15m", None),
        "30m": ("30m", None),
        "1h": ("1h", None),
        "2h": ("1h", "2h"),
        "4h": ("1h", "4h"),
        "6h": ("1h", "6h"),
        "8h": ("1h", "8h"),
        "12h": ("1h", "12h"),
        "1d": ("1d", None),
        "3d": ("1d", "3D"),
        "1w": ("1wk", None),
        "1M": ("1mo", None),
    }
    NET_TARGET_RETURN = 0.045

    def __init__(
        self,
        timeout: float = 4.0,
        *,
        allow_mock_data: bool = False,
        binance_futures: bool = False,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout musi być skończoną wartością większą od zera.")
        self.timeout = min(timeout, 4.0)
        self.allow_mock_data = allow_mock_data
        self.binance_futures = binance_futures
        self._fundamentals_cache: dict[str, dict[str, Any]] = {}
        self._news_cache: dict[str, dict[str, Any]] = {}
        self._analyst_cache: dict[str, dict[str, Any]] = {}
        self._macro_calendar_cache: dict[str, dict[str, Any]] = {}
        self._frame_cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    def analyze(
        self,
        symbol: str = "BTC-USD",
        timeframe: str = "1d",
        limit: int = 100,
        *,
        include_qualitative: bool = True,
        round_trip_cost_bps: float = 40.0,
        allow_mock_data: bool | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        """Return OHLCV, indicators, and optionally slower qualitative inputs."""
        ticker_symbol = self._normalize_symbol(symbol)
        if timeframe not in self.INTERVALS:
            raise ValueError(f"Nieobsługiwany interwał: {timeframe}")
        if not 35 <= limit <= 10_000:
            raise ValueError("Liczba świec musi mieścić się w zakresie 35–10000.")
        if (
            not math.isfinite(round_trip_cost_bps)
            or not self.MINIMUM_ROUND_TRIP_COST_BPS
            <= round_trip_cost_bps
            < 10_000
        ):
            raise ValueError(
                "Koszt round trip musi wynosić co najmniej 20 bps "
                "(10 bps przy zakupie i 10 bps przy sprzedaży) "
                "i być niższy niż 10 000 bps."
            )

        source_interval, resample_rule = self.INTERVALS[timeframe]
        include_extended_hours = (
            timeframe in {"5m", "15m"} and not ticker_symbol.endswith("-USD")
        )
        snapshot = None
        if as_of is not None:
            snapshot = pd.Timestamp(as_of)
            if snapshot.tzinfo is None:
                raise ValueError("as_of musi zawierać informację o strefie czasowej.")
            snapshot = snapshot.tz_convert("UTC")
        fallback_reason: str | None = None
        is_binance_crypto = ticker_symbol.endswith("-USD")
        source_name = (
            "Binance USD-M Futures Testnet"
            if is_binance_crypto and self.binance_futures
            else "Binance Spot Testnet"
            if is_binance_crypto
            else "Yahoo Finance"
        )
        try:
            if is_binance_crypto:
                binance_interval = {
                    "1wk": "1w",
                    "1mo": "1M",
                }.get(source_interval, source_interval)
                resample_multiplier = {
                    "3min": 3,
                    "2h": 2,
                    "4h": 4,
                    "6h": 6,
                    "8h": 8,
                    "12h": 12,
                    "3D": 3,
                }.get(resample_rule, 1)
                source_limit = (
                    min(10_000, limit * resample_multiplier + resample_multiplier)
                    if resample_rule is not None
                    else limit
                )
                get_klines = (
                    get_futures_testnet_klines
                    if self.binance_futures
                    else get_testnet_klines
                )
                history = get_klines(
                    ticker_symbol,
                    binance_interval,
                    source_limit,
                    end_time=snapshot.to_pydatetime() if snapshot is not None else None,
                    timeout=self.timeout,
                )
            else:
                ticker = yf.Ticker(ticker_symbol)
                if snapshot is None:
                    period = (
                        "60d"
                        if source_interval.endswith(("m", "h"))
                        else "5y"
                        if source_interval == "1d"
                        else "max"
                    )
                    history = ticker.history(
                        period=period,
                        interval=source_interval,
                        auto_adjust=False,
                        actions=False,
                        prepost=include_extended_hours,
                        timeout=self.timeout,
                    )
                else:
                    lookback = (
                        timedelta(days=59)
                        if source_interval.endswith(("m", "h"))
                        else timedelta(days=365 * 5 + 2)
                        if source_interval == "1d"
                        else timedelta(days=365 * 10)
                    )
                    history = ticker.history(
                        start=(snapshot - lookback).to_pydatetime(),
                        end=(snapshot + timedelta(seconds=1)).to_pydatetime(),
                        interval=source_interval,
                        auto_adjust=False,
                        actions=False,
                        prepost=include_extended_hours,
                        timeout=self.timeout,
                    )
            if not isinstance(history, pd.DataFrame) or history.empty:
                raise RuntimeError(
                    f"{source_name} nie zwrócił danych dla {ticker_symbol} "
                    f"(interwał {source_interval})."
                )
            frame = self._prepare_history(history)
            if snapshot is not None:
                frame = self._filter_historical_frame(
                    frame,
                    snapshot,
                    source_interval,
                    timeframe,
                    resample_rule,
                )
            elif source_interval.endswith(("m", "h")):
                frame = self._filter_incomplete_live_frame(
                    frame,
                    source_interval,
                )
            if resample_rule is not None:
                frame = self._resample(frame, resample_rule)
            if len(frame) < 35:
                raise RuntimeError(
                    f"{source_name} zwrócił tylko {len(frame)} świec dla {ticker_symbol} "
                    f"({timeframe}); potrzeba co najmniej 35."
                )
        except Exception as error:
            use_mock_data = (
                self.allow_mock_data
                if allow_mock_data is None
                else allow_mock_data
            )
            if not use_mock_data:
                raise RuntimeError(
                    f"Nie udało się pobrać rzeczywistych danych {source_name} "
                    f"dla {ticker_symbol} ({timeframe}): "
                    f"{type(error).__name__}: {error}"
                ) from error
            frame = self._fallback_frame(ticker_symbol, timeframe, limit, error)
            fallback_reason = f"{type(error).__name__}: {error}"

        if fallback_reason is not None:
            _LOGGER.warning(
                "Nie udało się pobrać danych %s (%s); "
                "używam syntetycznych danych testowych dla %s. "
                "Nie są to aktualne notowania.",
                source_name,
                fallback_reason,
                ticker_symbol,
            )

        frame["rsi"] = ta.rsi(frame["Close"], length=14)
        macd = ta.macd(frame["Close"], fast=12, slow=26, signal=9)
        atr = ta.atr(frame["High"], frame["Low"], frame["Close"], length=14)
        adx = ta.adx(frame["High"], frame["Low"], frame["Close"], length=14)
        if macd is None or macd.empty or atr is None or adx is None or adx.empty:
            raise RuntimeError("Nie udało się obliczyć wskaźników pandas-ta.")

        macd_columns = {
            "macd": self._find_indicator_column(macd, "MACD_"),
            "macd_histogram": self._find_indicator_column(macd, "MACDh_"),
            "macd_signal": self._find_indicator_column(macd, "MACDs_"),
        }
        frame["macd"] = macd[macd_columns["macd"]]
        frame["macd_histogram"] = macd[macd_columns["macd_histogram"]]
        frame["macd_signal"] = macd[macd_columns["macd_signal"]]
        frame["atr"] = atr
        adx_column = self._find_indicator_column(adx, "ADX_")
        frame["adx"] = adx[adx_column]
        frame["ema_20"] = frame["Close"].ewm(span=20, adjust=False).mean()
        frame["ema_50"] = frame["Close"].ewm(span=50, adjust=False).mean()
        frame["ema_200"] = frame["Close"].ewm(span=200, adjust=False).mean()

        valid = frame.dropna(
            subset=(
                "rsi",
                "macd",
                "macd_signal",
                "macd_histogram",
                "atr",
                "adx",
            )
        ).tail(limit)
        if valid.empty:
            raise RuntimeError(
                "Za mało poprawnych świec, aby obliczyć RSI, MACD i ATR."
            )
        latest = valid.iloc[-1]
        latest_timestamp = valid.index[-1]
        quality = (
            self._fetch_qualitative_bundle(
                ticker_symbol,
                float(latest["Close"]),
                round_trip_cost_bps=round_trip_cost_bps,
            )
            if include_qualitative and fallback_reason is None
            else None
        )
        if quality is None:
            deferred_status = (
                "deferred" if not include_qualitative else "unavailable"
            )
            fundamentals = {
                "status": deferred_status,
                "metrics": {},
                "quality_score": None,
            }
            news = {
                "status": deferred_status,
                "verified_articles": [],
                "rejected_count": 0,
                "sentiment_score": 0.0,
                "verified_source_count": 0,
                "verification_method": (
                    "Pominięto w etapie technicznym."
                    if not include_qualitative
                    else "Pominięto dla syntetycznych danych rynkowych."
                ),
            }
        else:
            fundamentals = quality["fundamentals"]
            news = quality["news"]
        analyst_consensus = (
            quality["analyst_consensus"]
            if quality is not None
            else {
                "status": "deferred" if not include_qualitative else "unavailable",
                "counts": {},
                "consensus_score": None,
                "recommendation": "unavailable",
            }
        )
        macro_calendar = (
            quality["macro_calendar"]
            if quality is not None
            else {
                "status": "deferred" if not include_qualitative else "unavailable",
                "upcoming_events": [],
                "source": "Forex Factory calendar feed",
            }
        )

        candles = [
            self._serialize_candle(timestamp, row)
            for timestamp, row in valid.iterrows()
        ]
        self._frame_cache[(ticker_symbol, timeframe, len(candles))] = frame
        indicators = {
            "rsi_14": float(latest["rsi"]),
            "macd": float(latest["macd"]),
            "macd_signal": float(latest["macd_signal"]),
            "macd_histogram": float(latest["macd_histogram"]),
            "atr_14": float(latest["atr"]),
            "adx_14": float(latest["adx"]),
        }
        momentum_signal = self._preliminary_signal(
            indicators["rsi_14"],
            indicators["macd"],
            indicators["macd_signal"],
        )
        signal = momentum_signal
        broker_inputs = self._build_broker_inputs(
            frame,
            latest,
            indicators,
            signal,
        )
        if quality is not None:
            broker_inputs["market_regime"] = quality["market_regime"]
            broker_inputs["event_risk"] = self._combined_event_risk(
                quality["event_risk"],
                quality["macro_calendar"],
            )
            broker_inputs["order_book"] = quality["order_book"]
            broker_inputs["analyst_consensus"] = quality["analyst_consensus"]
            broker_inputs["macro_calendar"] = quality["macro_calendar"]

        return {
            "exchange": (
                "Synthetic mock data"
                if fallback_reason is not None
                else source_name
                if is_binance_crypto
                else "Yahoo Finance"
            ),
            "symbol": ticker_symbol,
            "timeframe": timeframe,
            "source_interval": source_interval,
            "data_source": (
                "mock"
                if fallback_reason is not None
                else (
                    "binance_futures_testnet"
                    if self.binance_futures
                    else "binance_testnet"
                )
                if is_binance_crypto
                else "yfinance"
            ),
            "is_mock": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "timestamp": self._serialize_timestamp(latest_timestamp),
            "as_of": (
                self._serialize_timestamp(snapshot)
                if snapshot is not None
                else None
            ),
            "price": float(latest["Close"]),
            "indicators": indicators,
            "signal": signal,
            "signal_basis": self._signal_basis(signal)
            + (
                " ADX(14) < 20 oznacza słaby trend i obniża kwalifikację, "
                "ale nie usuwa kierunkowej etykiety."
                if indicators["adx_14"] < 20
                else ""
            ),
            "fundamentals": fundamentals,
            "news": news,
            "analyst_consensus": analyst_consensus,
            "macro_calendar": macro_calendar,
            "broker_inputs": broker_inputs,
            "candles": candles,
        }

    @staticmethod
    def _filter_historical_frame(
        frame: pd.DataFrame,
        snapshot: pd.Timestamp,
        source_interval: str,
        timeframe: str,
        resample_rule: str | None,
    ) -> pd.DataFrame:
        """Keep only source bars and resampled candles completed by the snapshot."""
        index = frame.index
        localized_snapshot = (
            snapshot.tz_convert(index.tz)
            if index.tz is not None
            else snapshot.tz_localize(None)
        )
        if index.tz is None:
            index = index.tz_localize("UTC")
            localized_snapshot = snapshot

        if source_interval.endswith(("m", "h")):
            unit = source_interval[-1]
            length = int(source_interval[:-1])
            duration = (
                timedelta(minutes=length)
                if unit == "m"
                else timedelta(hours=length)
            )
            frame = frame.loc[index + duration <= localized_snapshot]
        else:
            # Daily-and-longer Yahoo bars can include a partial snapshot-day bar.
            frame = frame.loc[index.normalize() < localized_snapshot.normalize()]

        if resample_rule is not None and not frame.empty:
            duration_by_timeframe = {
                "2h": timedelta(hours=2),
                "4h": timedelta(hours=4),
                "6h": timedelta(hours=6),
                "8h": timedelta(hours=8),
                "12h": timedelta(hours=12),
                "3d": timedelta(days=3),
            }
            bucket_duration = duration_by_timeframe.get(timeframe)
            if bucket_duration is not None:
                frame = frame.loc[
                    frame.index + bucket_duration <= localized_snapshot
                ]
        return frame

    @staticmethod
    def _filter_incomplete_live_frame(
        frame: pd.DataFrame,
        source_interval: str,
    ) -> pd.DataFrame:
        unit = source_interval[-1]
        length = int(source_interval[:-1])
        duration = (
            timedelta(minutes=length)
            if unit == "m"
            else timedelta(hours=length)
        )
        now = pd.Timestamp.now(tz="UTC")
        index = frame.index
        comparison_now = now.tz_convert(index.tz) if index.tz is not None else now.tz_localize(None)
        return frame.loc[index + duration <= comparison_now]

    def _with_yfinance_timeout(
        self,
        symbol: str,
        operation: Callable[[yf.Ticker], Any],
    ) -> Any:
        ticker = yf.Ticker(symbol)
        data_client = getattr(ticker, "_data", None)
        if data_client is None:
            return operation(ticker)
        with _YFINANCE_DATA_LOCK:
            original_get_raw_json = data_client.get_raw_json

            def bounded_get_raw_json(
                url: str,
                params: dict[str, Any] | None = None,
                timeout: float = 4.0,
            ) -> Any:
                return original_get_raw_json(
                    url,
                    params=params,
                    timeout=min(timeout, self.timeout, 4.0),
                )

            data_client.get_raw_json = bounded_get_raw_json
            try:
                return operation(ticker)
            finally:
                data_client.get_raw_json = original_get_raw_json

    def _cached_fundamentals(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._fundamentals_cache:
            self._fundamentals_cache[symbol] = self._fetch_fundamentals(symbol)
        return self._fundamentals_cache[symbol]

    def _cached_news(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._news_cache:
            self._news_cache[symbol] = self._fetch_news(symbol)
        return self._news_cache[symbol]

    def enrich_qualitative(
        self,
        research: dict[str, Any],
        *,
        include_fundamentals: bool = True,
        round_trip_cost_bps: float = 40.0,
    ) -> dict[str, Any]:
        """Add news, fundamentals, and other slow inputs after technical screening."""
        symbol = str(research["symbol"])
        cache_key = (symbol, str(research["timeframe"]), len(research["candles"]))
        frame = self._frame_cache.get(cache_key)
        if frame is None:
            raise RuntimeError(
                "Brak lokalnej historii do wzbogacenia analizy jakościowej; "
                "uruchom ponownie etap techniczny."
            )

        outputs = self._fetch_qualitative_bundle(
            symbol,
            float(research["price"]),
            include_fundamentals=include_fundamentals,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        research["fundamentals"] = outputs["fundamentals"]
        research["news"] = outputs["news"]
        research["analyst_consensus"] = outputs["analyst_consensus"]
        research["macro_calendar"] = outputs["macro_calendar"]
        latest = frame.iloc[-1]
        research["broker_inputs"] = self._build_broker_inputs(
            frame,
            latest,
            research["indicators"],
            research["signal"],
        )
        research["broker_inputs"]["market_regime"] = outputs["market_regime"]
        research["broker_inputs"]["event_risk"] = self._combined_event_risk(
            outputs["event_risk"],
            outputs["macro_calendar"],
        )
        research["broker_inputs"]["order_book"] = outputs["order_book"]
        research["broker_inputs"]["analyst_consensus"] = outputs[
            "analyst_consensus"
        ]
        research["broker_inputs"]["macro_calendar"] = outputs["macro_calendar"]
        return research

    def _fetch_qualitative_bundle(
        self,
        symbol: str,
        current_price: float,
        *,
        include_fundamentals: bool = True,
        round_trip_cost_bps: float = 40.0,
    ) -> dict[str, dict[str, Any]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            jobs = {
                "news": executor.submit(self._cached_news, symbol),
                "market_regime": executor.submit(self._fetch_market_regime, symbol),
                "event_risk": executor.submit(self._fetch_event_risk, symbol),
                "macro_calendar": executor.submit(
                    self._cached_macro_calendar,
                    symbol,
                ),
                "analyst_consensus": executor.submit(
                    self._cached_analyst_consensus,
                    symbol,
                ),
                "order_book": executor.submit(
                    self._fetch_order_book,
                    symbol,
                    current_price,
                    round_trip_cost_bps,
                ),
            }
            if include_fundamentals:
                jobs["fundamentals"] = executor.submit(
                    self._cached_fundamentals,
                    symbol,
                )
            outputs: dict[str, dict[str, Any]] = {}
            for name, future in jobs.items():
                try:
                    outputs[name] = future.result(timeout=self.timeout + 0.5)
                except Exception as error:
                    _LOGGER.warning(
                        "Pominięto dane jakościowe %s dla %s: %s",
                        name,
                        symbol,
                        error,
                    )
                    outputs[name] = self._neutral_quality_value(name, error)
        if not include_fundamentals:
            outputs["fundamentals"] = {
                "status": "unavailable",
                "metrics": {},
                "quality_score": None,
                "error": "Pominięto w skanerze dwuetapowym.",
            }
        return outputs

    @staticmethod
    def _neutral_quality_value(name: str, error: Exception) -> dict[str, Any]:
        reason = f"{type(error).__name__}: {error}"
        if name == "fundamentals":
            return {
                "status": "unavailable",
                "error": reason,
                "metrics": {},
                "quality_score": None,
            }
        if name == "news":
            return {
                "status": "unavailable",
                "error": reason,
                "verified_articles": [],
                "rejected_count": 0,
                "sentiment_score": 0.0,
                "verified_source_count": 0,
                "verification_method": "Neutralny fallback po błędzie lub timeout.",
            }
        if name == "market_regime":
            return {"status": "unavailable", "reason": reason}
        if name == "event_risk":
            return {
                "status": "incomplete",
                "macro_calendar_status": "unavailable",
                "corporate_calendar_status": "unavailable",
                "upcoming_events": [],
                "reason": reason,
            }
        if name == "analyst_consensus":
            return {
                "status": "unavailable",
                "error": reason,
                "counts": {},
                "consensus_score": None,
                "recommendation": "unavailable",
            }
        if name == "macro_calendar":
            return {
                "status": "unavailable",
                "error": reason,
                "upcoming_events": [],
                "source": "Forex Factory calendar feed",
            }
        return {"status": "unavailable", "reason": reason}

    @staticmethod
    def _combined_event_risk(
        corporate: dict[str, Any],
        macro: dict[str, Any],
    ) -> dict[str, Any]:
        corporate_events = corporate.get("upcoming_events", [])
        macro_events = macro.get("upcoming_events", [])
        events = [
            event
            for event in (*corporate_events, *macro_events)
            if isinstance(event, dict)
        ]
        macro_status = (
            "available"
            if macro.get("status") in {"clear", "high_risk"}
            else "unavailable"
        )
        corporate_status = corporate.get("corporate_calendar_status")
        status = (
            "high_risk"
            if events
            else "clear"
            if macro_status == "available"
            and corporate_status in {"available", "not_applicable"}
            else "incomplete"
        )
        return {
            **corporate,
            "status": status,
            "macro_calendar_status": macro_status,
            "upcoming_events": events,
            "macro_calendar": macro,
        }

    def _cached_analyst_consensus(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._analyst_cache:
            self._analyst_cache[symbol] = self._fetch_analyst_consensus(symbol)
        return self._analyst_cache[symbol]

    def _cached_macro_calendar(self, symbol: str) -> dict[str, Any]:
        if symbol not in self._macro_calendar_cache:
            self._macro_calendar_cache[symbol] = self._fetch_macro_calendar(symbol)
        return self._macro_calendar_cache[symbol]

    def _build_broker_inputs(
        self,
        frame: pd.DataFrame,
        latest: pd.Series,
        indicators: dict[str, float],
        signal: str,
    ) -> dict[str, Any]:
        close = float(latest["Close"])
        ema20 = float(latest["ema_20"])
        ema50 = float(latest["ema_50"])
        ema200 = float(latest["ema_200"])
        previous = frame.iloc[-2]
        technical_parts = {
            "moving_averages": 8.0
            if close > ema20 > ema50 > ema200
            else 4.0
            if close > ema20 and ema20 > ema50
            else 0.0,
            "rsi": 6.0
            if 50 <= indicators["rsi_14"] <= 65
            else 3.0
            if 45 <= indicators["rsi_14"] < 50
            else 0.0,
            "macd": 8.0
            if indicators["macd"] > indicators["macd_signal"]
            and indicators["macd_histogram"] > 0
            else 4.0
            if indicators["macd"] > indicators["macd_signal"]
            else 0.0,
            "price_action": 8.0
            if close > float(previous["High"]) and close > float(latest["Open"])
            else 4.0
            if close > float(previous["Close"])
            else 0.0,
        }
        recent_highs = frame["High"].iloc[-21:-1]
        resistance_price = (
            float(recent_highs.max()) if not recent_highs.empty else None
        )
        volume_median = float(frame["Volume"].tail(20).median())
        latest_volume = float(latest["Volume"])
        volume_ratio = (
            latest_volume / volume_median if volume_median > 0 else None
        )
        session_volume = self._completed_session_volume_ratio(frame)
        return {
            "technical": {
                "points": round(sum(technical_parts.values()), 1),
                "maximum": 30.0,
                "components": technical_parts,
                "signal": signal,
                "ema_20": ema20,
                "ema_50": ema50,
                "ema_200": ema200,
            },
            "target_path": {
                "resistance_price": resistance_price,
                "distance_to_resistance_pct": (
                    (resistance_price / close - 1) * 100
                    if resistance_price is not None
                    else None
                ),
                "resistance_status": (
                    "available" if resistance_price is not None else "unavailable"
                ),
            },
            "order_book": {"status": "deferred"},
            "liquidity": {
                "status": (
                    "available"
                    if volume_ratio is not None
                    and session_volume["session_volume_ratio"] is not None
                    else "unavailable"
                ),
                "latest_volume": latest_volume,
                "median_volume_20": volume_median,
                "volume_ratio": volume_ratio,
                **session_volume,
            },
            "market_regime": {"status": "deferred"},
            "event_risk": {
                "status": "deferred",
                "macro_calendar_status": "unavailable",
                "corporate_calendar_status": "unavailable",
                "upcoming_events": [],
                "reason": "Pominięto w etapie technicznym.",
            },
        }

    @staticmethod
    def _completed_session_volume_ratio(
        frame: pd.DataFrame,
    ) -> dict[str, float | str | None]:
        volume = pd.to_numeric(frame["Volume"], errors="coerce").dropna()
        if volume.empty or not isinstance(volume.index, pd.DatetimeIndex):
            return {
                "session_volume_ratio": None,
                "latest_completed_session_volume": None,
                "average_prior_20_session_volume": None,
                "volume_session_status": "unavailable",
            }

        session_volumes = volume.groupby(volume.index.normalize()).sum()
        if isinstance(volume.index, pd.DatetimeIndex):
            today = (
                pd.Timestamp.now(tz=volume.index.tz).normalize()
                if volume.index.tz is not None
                else pd.Timestamp.now().normalize()
            )
            session_volumes = session_volumes[session_volumes.index < today]
        if len(session_volumes) < 21:
            return {
                "session_volume_ratio": None,
                "latest_completed_session_volume": None,
                "average_prior_20_session_volume": None,
                "volume_session_status": "insufficient_history",
            }

        latest_session = float(session_volumes.iloc[-1])
        prior_average = float(session_volumes.iloc[-21:-1].mean())
        ratio = latest_session / prior_average if prior_average > 0 else None
        if ratio is None or not math.isfinite(ratio):
            return {
                "session_volume_ratio": None,
                "latest_completed_session_volume": latest_session,
                "average_prior_20_session_volume": prior_average,
                "volume_session_status": "unavailable",
            }
        return {
            "session_volume_ratio": ratio,
            "latest_completed_session_volume": latest_session,
            "average_prior_20_session_volume": prior_average,
            "volume_session_status": "available",
        }

    def _fetch_order_book(
        self,
        symbol: str,
        current_price: float,
        round_trip_cost_bps: float = 40.0,
    ) -> dict[str, Any]:
        _, separator, quote = symbol.partition("-")
        if not separator or quote != "USD":
            return {
                "status": "unavailable",
                "reason": "Brak skonfigurowanego publicznego źródła arkusza dla tickera.",
            }
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError("Cena do oceny arkusza musi być dodatnia i skończona.")
        if (
            not math.isfinite(round_trip_cost_bps)
            or not self.MINIMUM_ROUND_TRIP_COST_BPS
            <= round_trip_cost_bps
            < 10_000
        ):
            raise ValueError(
                "Koszt round trip musi wynosić co najmniej 20 bps "
                "(10 bps przy zakupie i 10 bps przy sprzedaży) "
                "i być niższy niż 10 000 bps."
            )
        try:
            depth = get_testnet_order_book(
                symbol,
                limit=1000,
                timeout=self.timeout,
                futures=self.binance_futures,
            )
        except (
            BinanceAPIException,
            BinanceRequestException,
            RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            _LOGGER.warning(
                "Nie udało się pobrać arkusza Binance Testnet dla %s: %s",
                symbol,
                error,
            )
            return {
                "status": "unavailable",
                "reason": f"{type(error).__name__}: {error}",
            }
        if not isinstance(depth, dict):
            return {"status": "unavailable", "reason": "Nieprawidłowy format danych order book."}
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        cost_fraction = round_trip_cost_bps / 20_000
        target_price = (
            current_price
            * (1 + cost_fraction)
            * 1.045
            / (1 - cost_fraction)
        )
        downside_boundary = current_price * 0.955
        try:
            parsed_asks = [
                (float(price), float(quantity)) for price, quantity in asks
            ]
            parsed_bids = [
                (float(price), float(quantity)) for price, quantity in bids
            ]
            if any(
                not math.isfinite(price)
                or not math.isfinite(quantity)
                or price <= 0
                or quantity < 0
                for price, quantity in parsed_asks + parsed_bids
            ):
                return {
                    "status": "unavailable",
                    "reason": "Nieprawidłowe wartości poziomów order book.",
                }
            asks_in_target_path = [
                (price, quantity)
                for price, quantity in parsed_asks
                if price <= target_price
            ]
            bids_in_downside_path = [
                (price, quantity)
                for price, quantity in parsed_bids
                if price >= downside_boundary
            ]
            ask_notional = sum(
                price * quantity for price, quantity in asks_in_target_path
            )
            bid_notional = sum(
                price * quantity for price, quantity in bids_in_downside_path
            )
            total_ask_notional = sum(
                price * quantity for price, quantity in parsed_asks
            )
            total_bid_notional = sum(
                price * quantity for price, quantity in parsed_bids
            )
            largest_ask_wall = max(
                (price * quantity for price, quantity in asks_in_target_path),
                default=0.0,
            )
        except (TypeError, ValueError, OverflowError):
            return {"status": "unavailable", "reason": "Nieprawidłowy format danych order book."}
        if not all(
            math.isfinite(value)
            for value in (
                ask_notional,
                bid_notional,
                total_ask_notional,
                total_bid_notional,
                largest_ask_wall,
            )
        ):
            return {"status": "unavailable", "reason": "Nieprawidłowe wartości order book."}
        return {
            "status": "available",
            "venue": (
                "Binance USD-M Futures Testnet"
                if self.binance_futures
                else "Binance Spot Testnet"
            ),
            "symbol": to_binance_symbol(symbol),
            "ask_notional": ask_notional,
            "bid_notional": bid_notional,
            "target_price": target_price,
            "downside_boundary": downside_boundary,
            "total_ask_notional": total_ask_notional,
            "total_bid_notional": total_bid_notional,
            "largest_ask_wall_notional": largest_ask_wall,
            "largest_ask_wall_share": (
                largest_ask_wall / ask_notional if ask_notional > 0 else None
            ),
            "ask_bid_depth_ratio": (
                ask_notional / bid_notional if bid_notional > 0 else None
            ),
            "note": (
                "Notional ofert sprzedaży przed celem brutto pokrywającym "
                "cel netto +4,5% po uwzględnieniu szacowanego kosztu "
                "round trip, zestawiony z "
                "notionalem bidów w pasie ±4,5%; dane pochodzą z migawki "
                "Binance Spot Testnet w USDT."
            ),
        }

    def _fetch_market_regime(self, symbol: str) -> dict[str, Any]:
        benchmark = "BTC-USD" if symbol.endswith("-USD") else "SPY"
        if benchmark == symbol:
            benchmark = "SPY"
        try:
            history = yf.Ticker(benchmark).history(
                period="3mo",
                interval="1d",
                auto_adjust=True,
                actions=False,
                timeout=self.timeout,
            )
            if not isinstance(history, pd.DataFrame) or len(history) < 20:
                raise RuntimeError("Za mało danych benchmarku do oceny reżimu rynku.")
            closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
            average = float(closes.tail(20).mean())
            current = float(closes.iloc[-1])
            if not math.isfinite(average) or average <= 0 or not math.isfinite(current):
                raise RuntimeError("Nieprawidłowe dane benchmarku.")
            return {
                "status": "available",
                "benchmark": benchmark,
                "current": current,
                "sma_20": average,
                "regime": "risk_on" if current >= average else "risk_off",
            }
        except Exception as error:
            _LOGGER.warning("Nie udało się określić reżimu rynku: %s", error)
            return {
                "status": "unavailable",
                "benchmark": benchmark,
                "reason": f"{type(error).__name__}: {error}",
            }

    def _fetch_event_risk(self, symbol: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        horizon_end = now.date() + timedelta(days=3)
        upcoming_events: list[dict[str, str]] = []
        corporate_calendar_status = (
            "not_applicable" if symbol.endswith("-USD") else "unavailable"
        )
        if not symbol.endswith("-USD"):
            try:
                calendar = self._with_yfinance_timeout(
                    symbol,
                    lambda ticker: ticker.calendar,
                )
                if not isinstance(calendar, dict):
                    raise RuntimeError(
                        "Yahoo Finance nie zwróciło poprawnego kalendarza spółki."
                    )
                earnings_dates = calendar.get("Earnings Date", [])
                if not isinstance(earnings_dates, (list, tuple)):
                    earnings_dates = [earnings_dates]
                corporate_calendar_status = "available"
                for event_date in earnings_dates:
                    if isinstance(event_date, datetime):
                        event_day = event_date.date()
                    elif isinstance(event_date, date):
                        event_day = event_date
                    else:
                        continue
                    if now.date() <= event_day <= horizon_end:
                        upcoming_events.append(
                            {
                                "type": "earnings",
                                "date": event_day.isoformat(),
                                "source": "Yahoo Finance calendar",
                            }
                        )
            except Exception as error:
                _LOGGER.warning("Nie udało się pobrać kalendarza %s: %s", symbol, error)
                corporate_calendar_status = "unavailable"
        return {
            "status": (
                "high_risk"
                if upcoming_events
                else "clear"
                if corporate_calendar_status in {"available", "not_applicable"}
                else "incomplete"
            ),
            "corporate_calendar_status": corporate_calendar_status,
            "macro_calendar_status": "unavailable",
            "upcoming_events": upcoming_events,
            "reason": (
                "Wykryto raport wyników w oknie 1–3 dni."
                if upcoming_events
                else "Brak kompletnego, wiarygodnego kalendarza makro dla horyzontu 1–3 dni."
            ),
        }

    def _fetch_macro_calendar(self, symbol: str) -> dict[str, Any]:
        """Read high-impact macro events from the public Forex Factory feed."""
        now = datetime.now(timezone.utc)
        horizon_end = now + timedelta(days=3)
        sunday = now.date() - timedelta(days=(now.weekday() + 1) % 7)
        saturday = sunday + timedelta(days=6)
        coverage_complete = horizon_end.date() <= saturday

        request = urllib.request.Request(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers={"User-Agent": "MAFSE market research/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, json.JSONDecodeError, urllib.error.URLError) as error:
            return {
                "status": "unavailable",
                "upcoming_events": [],
                "source": "Forex Factory calendar feed",
                "reason": f"{type(error).__name__}: {error}",
            }

        if not isinstance(payload, list):
            return {
                "status": "unavailable",
                "upcoming_events": [],
                "source": "Forex Factory calendar feed",
                "reason": "Nieprawidłowy format kalendarza.",
            }
        relevant_currencies = self._relevant_macro_currencies(symbol)
        high_impact_terms = (
            "cpi",
            "consumer price",
            "fomc",
            "federal funds",
            "fed interest rate",
            "non-farm",
            "nonfarm",
            "payroll",
            "pce",
            "gross domestic product",
            "gdp",
        )
        events: list[dict[str, str]] = []
        valid_current_week_entries = 0
        for item in payload:
            if not isinstance(item, dict):
                continue
            event_date = item.get("date")
            try:
                stamp = datetime.fromisoformat(
                    str(event_date).replace("Z", "+00:00")
                )
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                stamp = stamp.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            country = str(item.get("country", "")).upper()
            title = str(item.get("title", "")).strip()
            impact = str(item.get("impact", "")).casefold()
            if sunday <= stamp.date() <= saturday and country and title:
                valid_current_week_entries += 1
            is_high_impact = impact == "high" or any(
                term in title.casefold() for term in high_impact_terms
            )
            if (
                country in relevant_currencies
                and is_high_impact
                and now <= stamp <= horizon_end
            ):
                events.append(
                    {
                        "title": title,
                        "country": country,
                        "date": stamp.isoformat(),
                        "impact": impact or "high",
                        "source": "Forex Factory calendar feed",
                    }
                )
        return {
            "status": (
                "high_risk"
                if events
                else "unavailable"
                if valid_current_week_entries == 0
                else "clear"
                if coverage_complete
                else "incomplete"
            ),
            "upcoming_events": events,
            "source": "Forex Factory calendar feed",
            "coverage_through": saturday.isoformat(),
            "valid_week_entries": valid_current_week_entries,
            "reason": (
                None
                if coverage_complete and valid_current_week_entries > 0
                else "Feed nie zawiera poprawnych wpisów dla bieżącego tygodnia."
                if valid_current_week_entries == 0
                else "Feed obejmuje bieżący tydzień, nie cały horyzont 1–3 dni."
            ),
        }

    @staticmethod
    def _relevant_macro_currencies(symbol: str) -> set[str]:
        if symbol.endswith("-USD"):
            return {"USD"}
        suffix = symbol.rpartition(".")[2]
        return {
            "L": {"GBP"},
            "AS": {"EUR"},
            "DE": {"EUR"},
            "PA": {"EUR"},
            "MI": {"EUR"},
            "BR": {"EUR"},
            "HE": {"EUR"},
            "SW": {"CHF"},
            "CO": {"DKK"},
            "OL": {"NOK"},
            "ST": {"SEK"},
        }.get(suffix, {"USD"})

    def _fetch_news(self, symbol: str) -> dict[str, Any]:
        """Return recent headlines corroborated by two trusted, distinct outlets."""
        query = self._ASSET_SEARCH_TERMS.get(symbol, symbol.replace("-", " "))
        url = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode(
                {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
            )
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MAFSE market research/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                root = ET.fromstring(response.read())
        except (OSError, ET.ParseError, urllib.error.URLError) as error:
            _LOGGER.warning(
                "Nie udało się pobrać Google News RSS dla %s: %s", symbol, error
            )
            return {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "verified_articles": [],
                "rejected_count": 0,
                "sentiment_score": 0.0,
                "verified_source_count": 0,
                "verification_method": self._news_verification_method(),
            }

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        candidates: list[dict[str, Any]] = []
        considered_count = 0
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            source_element = item.find("source")
            publisher = (
                source_element.text.strip()
                if source_element is not None and source_element.text
                else ""
            )
            source_url = source_element.get("url", "") if source_element is not None else ""
            domain = urllib.parse.urlparse(source_url).hostname or ""
            domain = domain.lower().removeprefix("www.")
            link = (item.findtext("link") or "").strip()
            published_text = (item.findtext("pubDate") or "").strip()
            if not title or not link or not publisher:
                continue
            try:
                published = email.utils.parsedate_to_datetime(published_text)
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                published = published.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
            if published < cutoff:
                continue
            considered_count += 1
            title_tokens = self._headline_tokens(title)
            trusted_domain = self._trusted_root_domain(domain)
            if (
                trusted_domain is None
                or any(
                    token.startswith(marker)
                    for token in title_tokens
                    for marker in self._RUMOR_MARKERS
                )
            ):
                continue
            candidates.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "domain": trusted_domain,
                    "url": link,
                    "published_at": published.isoformat(),
                    "tokens": title_tokens,
                }
            )

        groups: list[list[dict[str, Any]]] = []
        for candidate in candidates:
            for group in groups:
                if (
                    self._headline_similarity(candidate["tokens"], group[0]["tokens"])
                    >= 0.35
                ):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])

        verified_articles: list[dict[str, Any]] = []
        corroborated_candidate_count = 0
        for group in groups:
            publishers: dict[str, dict[str, str]] = {}
            for article in group:
                publishers.setdefault(
                    article["domain"],
                    {
                        "publisher": article["publisher"],
                        "domain": article["domain"],
                        "url": article["url"],
                    },
                )
            if len(publishers) < 2:
                continue
            corroborated_candidate_count += len(group)
            latest_article = max(group, key=lambda item: item["published_at"])
            title = latest_article["title"]
            verified_articles.append(
                {
                    "title": title,
                    "published_at": latest_article["published_at"],
                    "sources": list(publishers.values()),
                    "sentiment": self._headline_sentiment(title),
                }
            )

        verified_articles.sort(
            key=lambda article: article["published_at"],
            reverse=True,
        )
        sentiments = [item["sentiment"] for item in verified_articles]
        sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return {
            "status": "verified" if verified_articles else "no_verified_news",
            "verified_articles": verified_articles[:10],
            "rejected_count": max(0, considered_count - corroborated_candidate_count),
            "sentiment_score": round(sentiment_score, 3),
            "verified_source_count": len(
                {
                    source["domain"]
                    for article in verified_articles
                    for source in article["sources"]
                }
            ),
            "verification_method": self._news_verification_method(),
        }

    def _fetch_analyst_consensus(self, symbol: str) -> dict[str, Any]:
        """Return Yahoo Finance's current analyst recommendation distribution."""
        def fetch_summary(ticker: yf.Ticker) -> Any:
            getter = getattr(ticker, "get_recommendations_summary", None)
            return getter() if callable(getter) else None

        try:
            recommendations = self._with_yfinance_timeout(
                symbol,
                fetch_summary,
            )
        except Exception as error:
            _LOGGER.warning(
                "Nie udało się pobrać konsensusu analityków dla %s: %s",
                symbol,
                error,
            )
            return {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "counts": {},
                "consensus_score": None,
                "recommendation": "unavailable",
            }

        if (
            not isinstance(recommendations, pd.DataFrame)
            or recommendations.empty
        ):
            return {
                "status": "not_available",
                "counts": {},
                "consensus_score": None,
                "recommendation": "not_available",
            }

        table = recommendations
        if "period" in table.columns:
            current_period = table.loc[
                table["period"].astype(str).str.casefold().isin(
                    {"0m", "current", "now"}
                )
            ]
            table = current_period if not current_period.empty else table.iloc[:1]
        elif table.index.name and table.index.name.casefold() == "period":
            current_period = table.loc[
                table.index.astype(str).str.casefold().isin(
                    {"0m", "current", "now"}
                )
            ]
            table = current_period if not current_period.empty else table.iloc[:1]
        else:
            table = table.iloc[:1]

        latest = table.iloc[0]
        counts: dict[str, int] = {}
        for column in ("strongBuy", "buy", "hold", "sell", "strongSell"):
            value = latest.get(column)
            try:
                count = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if count >= 0 and not pd.isna(value):
                counts[column] = count
        total = sum(counts.values())
        if total <= 0:
            return {
                "status": "not_available",
                "counts": counts,
                "consensus_score": None,
                "recommendation": "not_available",
            }

        rating_score = (
            5 * counts.get("strongBuy", 0)
            + 4 * counts.get("buy", 0)
            + 3 * counts.get("hold", 0)
            + 2 * counts.get("sell", 0)
            + counts.get("strongSell", 0)
        ) / total
        recommendation = (
            "strong_buy"
            if rating_score >= 4.5
            else "buy"
            if rating_score >= 3.5
            else "hold"
            if rating_score >= 2.5
            else "sell"
            if rating_score >= 1.5
            else "strong_sell"
        )
        return {
            "status": "available",
            "counts": counts,
            "analyst_count": total,
            "consensus_score": round(rating_score, 2),
            "recommendation": recommendation,
            "source": "Yahoo Finance analyst recommendations",
        }

    def _fetch_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Read Yahoo Finance company fundamentals without inventing missing values."""
        try:
            info = self._with_yfinance_timeout(
                symbol,
                lambda ticker: ticker.get_info(),
            )
        except Exception as error:
            _LOGGER.warning(
                "Nie udało się pobrać fundamentów Yahoo Finance dla %s: %s",
                symbol,
                error,
            )
            return {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "metrics": {},
                "quality_score": None,
            }
        if not isinstance(info, dict) or not info:
            return {
                "status": "not_available",
                "metrics": {},
                "quality_score": None,
            }

        metrics: dict[str, Any] = {}
        for key in (
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "profitMargins",
            "revenueGrowth",
            "debtToEquity",
            "beta",
        ):
            value = info.get(key)
            if isinstance(value, str) or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                metrics[key] = value
        quality_score = self._fundamental_quality_score(metrics)
        return {
            "status": "available" if quality_score is not None else "not_available",
            "metrics": metrics,
            "quality_score": quality_score,
        }

    @staticmethod
    def _fundamental_quality_score(metrics: dict[str, Any]) -> float | None:
        score = 50.0
        observations = 0
        margins = metrics.get("profitMargins")
        if isinstance(margins, (int, float)):
            score += 15 if margins > 0 else -15 if margins < 0 else 0
            observations += 1
        growth = metrics.get("revenueGrowth")
        if isinstance(growth, (int, float)):
            score += (
                15 if growth > 0.05
                else 5 if growth > 0
                else -15 if growth < 0
                else 0
            )
            observations += 1
        debt_to_equity = metrics.get("debtToEquity")
        if isinstance(debt_to_equity, (int, float)):
            score += 5 if debt_to_equity < 200 else -10 if debt_to_equity > 300 else 0
            observations += 1
        return round(max(0.0, min(100.0, score)), 1) if observations else None

    @classmethod
    def _is_trusted_source(cls, domain: str) -> bool:
        return cls._trusted_root_domain(domain) is not None

    @classmethod
    def _trusted_root_domain(cls, domain: str) -> str | None:
        for trusted in cls._TRUSTED_NEWS_DOMAINS:
            if domain == trusted or domain.endswith(f".{trusted}"):
                return trusted
        return None

    @classmethod
    def _headline_tokens(cls, title: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[^\W_]{2,}", title.casefold())
            if token not in cls._NEWS_STOP_WORDS
        }

    @staticmethod
    def _headline_similarity(first: set[str], second: set[str]) -> float:
        union = first | second
        return len(first & second) / len(union) if union else 0.0

    @classmethod
    def _headline_sentiment(cls, title: str) -> float:
        tokens = cls._headline_tokens(title)
        positive = sum(
            1
            for token in tokens
            if any(token.startswith(word) for word in cls._POSITIVE_WORDS)
        )
        negative = sum(
            1
            for token in tokens
            if any(token.startswith(word) for word in cls._NEGATIVE_WORDS)
        )
        total = positive + negative
        return 0.0 if total == 0 else (positive - negative) / total

    @staticmethod
    def _news_verification_method() -> str:
        return (
            "Nagłówek musi pochodzić z dozwolonego źródła i być podobnie "
            "relacjonowany przez co najmniej dwa różne, zaufane wydawnictwa. "
            "To corroboracja nagłówków, nie niezależny audyt prawdziwości artykułów."
        )

    @classmethod
    def _fallback_frame(
        cls,
        symbol: str,
        timeframe: str,
        limit: int,
        cause: Exception,
    ) -> pd.DataFrame:
        if symbol != "BTC-USD":
            raise RuntimeError(
                f"Nie udało się pobrać danych yfinance dla {symbol}; "
                "syntetyczny fallback jest dostępny wyłącznie dla BTC-USD."
            ) from cause

        frequencies = {
            "1m": "1min",
            "3m": "3min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1D",
            "3d": "3D",
            "1w": "7D",
            "1M": "30D",
        }
        count = limit + 100
        rng = np.random.default_rng(20261106)
        frequency = frequencies[timeframe]
        if timeframe.endswith("m") or timeframe.endswith("h"):
            end = pd.Timestamp.now(tz="UTC").floor("min")
        else:
            end = pd.Timestamp.now(tz="UTC").floor("D")
        index = pd.date_range(end=end, periods=count, freq=frequency)

        if timeframe in {"1m", "3m", "5m", "15m", "30m"}:
            volatility = 0.0015
        elif timeframe in {"1h", "2h", "4h", "6h", "8h", "12h"}:
            volatility = 0.005
        elif timeframe == "1d":
            volatility = 0.02
        elif timeframe == "1w":
            volatility = 0.045
        else:
            volatility = 0.08

        returns = np.clip(
            rng.normal(loc=0.0001, scale=volatility, size=count),
            -0.15,
            0.15,
        )
        log_prices = np.log(60_000.0) + np.cumsum(returns)
        close = 60_000.0 * np.exp(np.clip(log_prices - np.log(60_000.0), -1.5, 1.5))
        open_prices = np.empty(count)
        open_prices[0] = close[0] / (1 + returns[0])
        open_prices[1:] = close[:-1]
        intrabar_range = np.maximum(
            close * rng.uniform(0.001, max(volatility * 1.5, 0.003), size=count),
            np.abs(close - open_prices),
        )
        high = np.maximum(open_prices, close) + intrabar_range
        low = np.maximum(
            np.minimum(open_prices, close) - intrabar_range,
            np.finfo(float).tiny,
        )
        volume = rng.lognormal(mean=1.2, sigma=0.7, size=count)

        return pd.DataFrame(
            {
                "Open": open_prices,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if "/" in normalized:
            base, quote = normalized.split("/", maxsplit=1)
            if not base or not quote:
                raise ValueError("Nieprawidłowy symbol; użyj np. BTC-USD.")
            if quote in {"USDT", "USDC"}:
                quote = "USD"
            normalized = f"{base}-{quote}"
        if not normalized or any(char.isspace() for char in normalized):
            raise ValueError("Nieprawidłowy symbol.")
        return normalized

    @staticmethod
    def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
        if isinstance(history.columns, pd.MultiIndex):
            history = history.copy()
            history.columns = history.columns.get_level_values(0)

        required = ("Open", "High", "Low", "Close", "Volume")
        missing = [column for column in required if column not in history.columns]
        if missing:
            raise RuntimeError(
                "Dane rynkowe nie zawierają wymaganych kolumn OHLCV: "
                + ", ".join(missing)
            )

        frame = history.loc[:, required].copy()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=("Open", "High", "Low", "Close"))
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if frame.empty:
            raise RuntimeError("Brak poprawnych świec OHLCV w danych rynkowych.")
        return frame

    @staticmethod
    def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            frame.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(subset=("Open", "High", "Low", "Close"))
        )

    @staticmethod
    def _find_indicator_column(frame: pd.DataFrame, prefix: str) -> str:
        matches = [column for column in frame.columns if column.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Nieoczekiwany wynik pandas-ta dla wskaźnika {prefix}: {matches}"
            )
        return matches[0]

    @staticmethod
    def _preliminary_signal(rsi: float, macd: float, macd_signal: float) -> str:
        if macd > macd_signal:
            return "KUP"
        if macd < macd_signal:
            return "SPRZEDAJ"
        return "KUP" if rsi >= 50 else "SPRZEDAJ"

    @staticmethod
    def _signal_basis(signal: str) -> str:
        if signal == "KUP":
            return "MACD powyżej linii sygnału, RSI w przedziale 50–70."
        if signal == "SPRZEDAJ":
            return "Kierunek spadkowy: MACD poniżej linii sygnału."
        return "Kierunek wzrostowy: MACD powyżej linii sygnału."

    @staticmethod
    def _serialize_timestamp(timestamp: Any) -> str:
        return timestamp.isoformat()

    @classmethod
    def _serialize_candle(cls, timestamp: Any, row: pd.Series) -> dict[str, Any]:
        return {
            "timestamp": cls._serialize_timestamp(timestamp),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": None if pd.isna(row["Volume"]) else float(row["Volume"]),
            "rsi": None if pd.isna(row["rsi"]) else float(row["rsi"]),
            "macd": None if pd.isna(row["macd"]) else float(row["macd"]),
            "macd_signal": (
                None if pd.isna(row["macd_signal"]) else float(row["macd_signal"])
            ),
            "macd_histogram": (
                None
                if pd.isna(row["macd_histogram"])
                else float(row["macd_histogram"])
            ),
            "atr": None if pd.isna(row["atr"]) else float(row["atr"]),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pobierz świece OHLCV i oblicz RSI, MACD oraz ATR."
    )
    parser.add_argument("--symbol", default="BTC-USD", help="Symbol, np. BTC-USD.")
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Interwał, np. 1h, 4h, 1d albo 1w.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Liczba świec (35–10000).")
    args = parser.parse_args()

    try:
        result = ResearchAgent().analyze(args.symbol, args.timeframe, args.limit)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Błąd: {error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
