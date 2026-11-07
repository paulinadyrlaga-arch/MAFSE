"""Tests for the scanner's two-stage API gate and neutral quality fallback."""

import unittest
from datetime import timedelta
from unittest.mock import patch
from typing import Any

import app
from test_broker_filters import _research


class ScannerScreeningTests(unittest.TestCase):
    def test_scanner_covers_a_broad_unique_stock_universe(self) -> None:
        self.assertGreaterEqual(len(app.US_STOCK_UNIVERSE), 50)
        self.assertGreaterEqual(len(app.EU_STOCK_UNIVERSE), 30)
        self.assertEqual(len(app.STOCK_UNIVERSE), len(set(app.STOCK_UNIVERSE)))
        self.assertFalse(hasattr(app._scan_market, "clear"))
        self.assertIn("NVDA", app.STOCK_UNIVERSE)
        self.assertIn("AAPL", app.STOCK_UNIVERSE)
        self.assertIn("TSLA", app.STOCK_UNIVERSE)
        self.assertIn("ASML.AS", app.STOCK_UNIVERSE)

    def test_scanner_market_selection_returns_the_selected_universe(self) -> None:
        self.assertEqual(app._symbols_for_market("Wszystko"), app.MARKET_UNIVERSE)
        self.assertEqual(
            app._symbols_for_market("Tylko Akcje (Revolut)"),
            app.STOCK_UNIVERSE,
        )
        self.assertEqual(
            app._symbols_for_market("Tylko Kryptowaluty (Revolut)"),
            app.CRYPTO_UNIVERSE,
        )
        self.assertIn("XRP-USD", app.CRYPTO_UNIVERSE)
        self.assertIn("ADA-USD", app.CRYPTO_UNIVERSE)
        self.assertIn("DOGE-USD", app.CRYPTO_UNIVERSE)

    def test_fee_assumption_is_at_least_ten_bps_per_side(self) -> None:
        self.assertGreaterEqual(app.ROUND_TRIP_COST_BPS / 2, 10.0)

    def test_trade_levels_include_entry_and_exit_fee_and_safe_stop_fallback(self) -> None:
        levels = app._trade_levels(100.0, 100.0, 0.045, 20.0)
        self.assertEqual(levels["entry_fee"], 0.1)
        self.assertAlmostEqual(
            levels["take_profit"],
            100.0 * 1.001 * 1.045 / 0.999,
        )
        self.assertAlmostEqual(levels["potential_profit_pct"], 4.5)
        self.assertEqual(levels["stop_loss"], 98.0)
        self.assertEqual(levels["trailing_stop"], levels["stop_loss"])
        self.assertEqual(levels["trailing_stop_activation_price"], 200.0)

    def test_short_term_scoring_uses_150_point_model_and_net_levels(self) -> None:
        research = _research()
        candles = []
        latest_date = app.date(2026, 11, 6)
        for offset in range(20, -1, -1):
            session_day = latest_date - timedelta(days=offset)
            candles.append(
                {
                    "timestamp": f"{session_day.isoformat()}T15:15:00+00:00",
                    "high": 100.5 if offset == 0 else 101.0,
                    "low": 99.5,
                    "close": 100.0,
                    "open": 99.9,
                    "volume": 200.0 if offset == 0 else 100.0,
                    "macd_histogram": 0.8 if offset == 0 else 0.5,
                }
            )
        research.update(
            {
                "price": 100.0,
                "timestamp": "2026-11-06T15:15:00+00:00",
                "candles": candles,
            }
        )
        research["indicators"]["macd_histogram"] = 1.0

        item = app._short_term_item("AAPL", research, 20.0)

        self.assertEqual(item["timeframe"], "15m")
        self.assertEqual(item["relative_volume"], 2.0)
        self.assertEqual(item["score_breakdown"][0]["points"], 30.0)
        self.assertEqual(
            sum(row["maximum"] for row in item["score_breakdown"]),
            150,
        )
        self.assertEqual(item["entry_price"], 101.0)
        self.assertAlmostEqual(
            item["take_profit"],
            101.0 * 1.001 * 1.02 / 0.999,
        )
        self.assertAlmostEqual(item["stop_loss"], 101.0 - 1.5 * 1.5)
        self.assertLess(item["risk_reward_ratio"], 1.5)
        self.assertAlmostEqual(
            item["take_profit"] * 0.999 - item["entry_price"] * 1.001,
            app.SHORT_TERM_NET_TARGET_RETURN
            * item["entry_price"]
            * 1.001,
        )
        self.assertAlmostEqual(
            item["potential_profit_pct"],
            app.SHORT_TERM_NET_TARGET_RETURN * 100,
        )

    def test_short_term_buy_requires_110_points_and_rr_at_least_two(self) -> None:
        research = _research()
        research["indicators"]["atr_14"] = 0.1
        candles = []
        latest_date = app.date(2026, 11, 6)
        for offset in range(20, -1, -1):
            session_day = latest_date - timedelta(days=offset)
            candles.append(
                {
                    "timestamp": f"{session_day.isoformat()}T14:15:00+00:00",
                    "high": 100.5 if offset == 0 else 101.0,
                    "low": 99.5,
                    "close": 100.0,
                    "open": 99.9,
                    "volume": 200.0 if offset == 0 else 100.0,
                    "macd_histogram": 0.8 if offset == 0 else 0.5,
                }
            )
        research.update(
            {
                "price": 100.0,
                "timestamp": candles[-1]["timestamp"],
                "candles": candles,
                "quality_reviewed": True,
                "analyst_consensus": {
                    "status": "available",
                    "consensus_score": 5.0,
                    "recommendation": "strong_buy",
                    "analyst_count": 8,
                },
                "news": {
                    "status": "verified",
                    "verified_articles": [{"title": "Positive verified news"}],
                    "sentiment_score": 1.0,
                },
                "macro_calendar": {"status": "clear"},
            }
        )
        research["indicators"]["macd_histogram"] = 1.0

        item = app._short_term_item("AAPL", research, 20.0)

        self.assertGreaterEqual(item["broker_points"], 110.0)
        self.assertGreaterEqual(item["risk_reward_ratio"], 2.0)
        self.assertTrue(item["qualified"])
        self.assertEqual(item["signal"], "KUP")
        self.assertEqual(
            [row["maximum"] for row in item["score_breakdown"]],
            [30, 25, 20, 15, 20, 20, 20],
        )

    def test_exactly_110_points_qualifies_and_109_does_not(self) -> None:
        research = _research()
        research["indicators"]["atr_14"] = 0.1
        candles = [
            {
                "timestamp": (
                    f"{(app.date(2026, 11, 6) - timedelta(days=offset)).isoformat()}"
                    "T14:15:00+00:00"
                ),
                "high": 100.5 if offset == 0 else 101.0,
                "low": 99.5,
                "close": 100.0,
                "open": 99.9,
                "volume": 200.0 if offset == 0 else 100.0,
                "macd_histogram": 1.0 if offset == 0 else 0.5,
            }
            for offset in range(20, -1, -1)
        ]
        research.update(
            {
                "price": 100.0,
                "timestamp": candles[-1]["timestamp"],
                "candles": candles,
                "quality_reviewed": True,
                "macro_calendar": {"status": "clear"},
            }
        )
        research["indicators"]["macd_histogram"] = 1.0
        with patch(
            "app._short_term_vwap_and_poc",
            return_value=(None, None, False, False),
        ):
            research["analyst_consensus"] = {
                "status": "available",
                "consensus_score": 3.0,
                "analyst_count": 5,
            }
            exactly_threshold = app._short_term_item("AAPL", research, 20.0)
            research["analyst_consensus"]["consensus_score"] = 2.6
            below_threshold = app._short_term_item("AAPL", research, 20.0)

        self.assertEqual(exactly_threshold["broker_points"], 110.0)
        self.assertEqual(exactly_threshold["signal"], "KUP")
        self.assertEqual(below_threshold["broker_points"], 109.0)
        self.assertEqual(below_threshold["signal"], "CZEKAJ")

    def test_swing_scoring_adds_analyst_macro_and_trailing_risk_points(self) -> None:
        research = self._daily_research(elevated_recent_volume=True)
        research.update(
            {
                "quality_reviewed": True,
                "analyst_consensus": {
                    "status": "available",
                    "consensus_score": 5.0,
                    "recommendation": "strong_buy",
                    "analyst_count": 8,
                },
                "macro_calendar": {"status": "clear"},
            }
        )
        research["broker_inputs"] = {
            "event_risk": {
                "status": "clear",
                "corporate_calendar_status": "available",
                "upcoming_events": [],
            }
        }

        item = app._swing_item("AAPL", research, 20.0, 5.0)

        self.assertEqual(
            [row["maximum"] for row in item["score_breakdown"]],
            [25, 20, 15, 15, 25, 25, 25],
        )
        self.assertEqual(item["score_breakdown"][4]["points"], 25.0)
        self.assertEqual(item["score_breakdown"][5]["points"], 20.0)
        self.assertEqual(item["score_breakdown"][6]["points"], 25.0)
        self.assertGreaterEqual(item["broker_points"], 110.0)
        self.assertTrue(item["qualified"])
        self.assertAlmostEqual(
            item["potential_profit_pct"],
            app.SWING_NET_TARGET_RETURN * 100,
        )
        self.assertIn("nie obniża się", item["trailing_stop_rule"])

    def test_short_term_relative_volume_uses_matching_time_slot_from_20_sessions(self) -> None:
        session_dates = [
            app.date(2026, 11, 6) - timedelta(days=offset)
            for offset in range(40, -1, -1)
            if (app.date(2026, 11, 6) - timedelta(days=offset)).weekday() < 5
        ][-21:]
        candles = [
            {
                "timestamp": f"{session_day.isoformat()}T14:00:00-05:00",
                "volume": 100.0,
            }
            for session_day in session_dates
        ]
        candles[-1]["volume"] = 150.0

        relative_volume, samples = app._short_term_relative_volume(candles)

        self.assertEqual(relative_volume, 1.5)
        self.assertEqual(samples, 20)
        for expected, threshold_points in ((1.0, 10.0), (1.5, 20.0), (2.0, 30.0)):
            candles[-1]["volume"] = 100.0 * expected
            result, _ = app._short_term_relative_volume(candles)
            self.assertEqual(result, expected)
            self.assertEqual(
                app._short_term_item(
                    "AAPL",
                    {
                        **_research(),
                        "price": 100.0,
                        "timestamp": candles[-1]["timestamp"],
                        "candles": [
                            {
                                **candle,
                                "high": 100.5,
                                "low": 99.5,
                                "close": 100.0,
                                "open": 99.9,
                            }
                            for candle in candles
                        ],
                    },
                    20.0,
                )["score_breakdown"][0]["points"],
                threshold_points,
            )

    def test_relvol_below_one_sets_entire_short_term_score_to_zero(self) -> None:
        good = _research()
        candles = []
        latest_date = app.date(2026, 11, 6)
        for offset in range(20, -1, -1):
            session_day = latest_date - timedelta(days=offset)
            candles.append(
                {
                    "timestamp": f"{session_day.isoformat()}T15:15:00+00:00",
                    "high": 100.5,
                    "low": 99.5,
                    "close": 100.0,
                    "open": 99.9,
                    "volume": 50.0 if offset == 0 else 100.0,
                }
            )
        good.update({"price": 100.0, "candles": candles})

        item = app._short_term_item("AAPL", good, 20.0)

        self.assertEqual(item["broker_points"], 0.0)
        self.assertFalse(item["qualified"])
        self.assertEqual(item["relative_volume"], 0.5)
        self.assertEqual(
            sum(row["points"] for row in item["score_breakdown"]),
            0.0,
        )

    def _daily_research(self, *, elevated_recent_volume: bool = False) -> dict[str, Any]:
        research = _research()
        candles = []
        start = app.date(2023, 10, 1)
        for index in range(1000):
            close = 50.0 + index * 0.1
            day = start + timedelta(days=index)
            volume = (
                200.0
                if elevated_recent_volume and index >= 995
                else 100.0
            )
            candles.append(
                {
                    "timestamp": f"{day.isoformat()}T00:00:00+00:00",
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.1,
                    "close": close,
                    "volume": volume,
                    "macd": 2.0 if index == 999 else 0.5,
                    "macd_signal": 1.0 if index == 999 else 1.0,
                }
            )
        research.update(
            {
                "price": candles[-1]["close"],
                "timestamp": candles[-1]["timestamp"],
                "candles": candles,
                "is_mock": False,
            }
        )
        research["indicators"].update(
            {
                "rsi_14": 55.0,
                "macd": 2.0,
                "macd_signal": 1.0,
                "atr_14": 1.0,
            }
        )
        candles[-2]["macd"] = 0.5
        candles[-2]["macd_signal"] = 1.0
        return research

    def test_swing_scoring_uses_defined_pillar_maxima_and_rising_macd(self) -> None:
        research = self._daily_research(elevated_recent_volume=True)
        item = app._swing_item("AAPL", research, 20.0, 5.0)

        self.assertEqual(
            [row["maximum"] for row in item["score_breakdown"]],
            [25, 20, 15, 15, 25, 25, 25],
        )
        self.assertEqual(sum(row["maximum"] for row in item["score_breakdown"]), 150)
        self.assertEqual(item["score_breakdown"][0]["points"], 25.0)
        self.assertEqual(item["score_breakdown"][1]["points"], 20.0)
        self.assertGreater(item["score_breakdown"][2]["points"], 0.0)
        self.assertEqual(item["sma_50"], sum(
            candle["close"] for candle in research["candles"][-50:]
        ) / 50)

    def test_short_term_scanner_uses_intraday_and_preserves_full_ranking(self) -> None:
        def intraday_research(volume: float, rsi: float) -> dict[str, Any]:
            research = _research()
            candles = []
            latest_date = app.date(2026, 11, 6)
            for offset in range(20, -1, -1):
                session_day = latest_date - timedelta(days=offset)
                candles.append(
                    {
                        "timestamp": f"{session_day.isoformat()}T15:15:00+00:00",
                        "high": 100.0,
                        "low": 99.0,
                        "close": 99.9,
                        "open": 99.5,
                        "volume": volume if offset == 0 else 100.0,
                        "macd_histogram": 1.0 if offset == 0 else 0.5,
                    }
                )
            research.update(
                {
                    "price": 99.9,
                    "timestamp": candles[-1]["timestamp"],
                    "candles": candles,
                }
            )
            research["indicators"]["rsi_14"] = rsi
            research["indicators"]["macd_histogram"] = 1.0
            return research

        analyses: list[dict[str, Any]] = []
        research_by_symbol = {
            "HIGH": intraday_research(250.0, 65.0),
            "LOW": intraday_research(100.0, 50.0),
        }

        class FakeResearcher:
            def analyze(self, **kwargs: Any) -> dict[str, Any]:
                analyses.append(kwargs)
                return research_by_symbol[kwargs["symbol"]]

        snapshot = app._historical_snapshot(
            app.date(2026, 11, 6),
            app.time(15, 30),
        )
        with patch("app.ResearchAgent", return_value=FakeResearcher()):
            result = app._scan_market(
                ("LOW", "HIGH"),
                app.ROUND_TRIP_COST_BPS,
                as_of=snapshot,
                analysis_category=app.SHORT_TERM_CATEGORY,
            )

        self.assertEqual(result["analysis_category"], "short_term")
        self.assertEqual(result["timeframe"], "15m")
        self.assertEqual(
            [item["symbol"] for item in result["ranked_assets"]],
            ["HIGH", "LOW"],
        )
        self.assertTrue(all(call["timeframe"] == "15m" for call in analyses))
        self.assertTrue(all(call["as_of"] == snapshot for call in analyses))
        self.assertTrue(all(not call["include_qualitative"] for call in analyses))
        self.assertTrue(all(call["limit"] == app.SHORT_TERM_CANDLE_LIMIT for call in analyses))

    def test_live_scan_enriches_only_top_technical_candidates(self) -> None:
        candles = [
            {
                "timestamp": f"2026-11-{index:02d}T14:15:00+00:00",
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "open": 99.9,
                "volume": 200.0 if index == 21 else 100.0,
                "macd_histogram": 1.0 if index == 21 else 0.5,
            }
            for index in range(1, 22)
        ]
        research = _research()
        research.update(
            {
                "price": 100.0,
                "timestamp": candles[-1]["timestamp"],
                "candles": candles,
            }
        )
        research["indicators"]["macd_histogram"] = 1.0
        enriched_symbols: list[str] = []

        class FakeResearcher:
            def analyze(self, **_: Any) -> dict[str, Any]:
                return research.copy()

            def enrich_qualitative(
                self,
                current: dict[str, Any],
                **_: Any,
            ) -> dict[str, Any]:
                enriched_symbols.append("AAPL")
                current["analyst_consensus"] = {
                    "status": "available",
                    "consensus_score": 4.0,
                    "recommendation": "buy",
                    "analyst_count": 6,
                }
                current["news"] = {
                    "status": "verified",
                    "verified_articles": [{"title": "Confirmed news"}],
                    "sentiment_score": 0.5,
                }
                current["macro_calendar"] = {"status": "clear"}
                return current

        with patch("app.ResearchAgent", return_value=FakeResearcher()):
            result = app._scan_market(
                ("AAPL",),
                app.ROUND_TRIP_COST_BPS,
                analysis_category=app.SHORT_TERM_CATEGORY,
            )

        self.assertEqual(enriched_symbols, ["AAPL"])
        self.assertEqual(result["stage_two_reviewed_assets"], 1)
        item = result["ranked_assets"][0]
        analyst_row = next(
            row for row in item["score_breakdown"]
            if row["label"] == "Opinie analityków / buzz"
        )
        self.assertGreater(analyst_row["points"], 0.0)

    def test_market_instructions_distinguish_crypto_from_us_stocks(self) -> None:
        self.assertIn("24/7", app._market_session_instruction("BTC-USD"))
        self.assertIn("15:30", app._market_session_instruction("AAPL"))
        self.assertIn("≥ 100", app._buy_trigger_instruction("BTC-USD", 100.0))

    def test_historical_snapshot_uses_warsaw_timezone(self) -> None:
        snapshot = app._historical_snapshot(
            app.date(2026, 11, 6),
            app.time(15, 30),
        )

        self.assertEqual(snapshot.isoformat(), "2026-11-06T15:30:00+01:00")

    def test_unknown_market_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            app._symbols_for_market("nieznany")

    def test_qualified_opportunity_uses_net_target_and_atr_stop(self) -> None:
        research = _research()
        research.update(
            {
                "symbol": "TEST",
                "timeframe": "4h",
                "timestamp": "2026-11-06T00:00:00+00:00",
                "candles": [
                    {
                        "timestamp": "2026-11-06T00:00:00+00:00",
                        "close": 100.0,
                        "rsi": 60.0,
                        "macd": 2.0,
                        "macd_signal": 1.0,
                        "atr": 1.5,
                    }
                ],
                "fundamentals": {
                    "status": "unavailable",
                    "metrics": {},
                    "quality_score": None,
                },
            }
        )
        research["news"] = {
            "status": "verified",
            "verified_articles": [],
            "sentiment_score": 0.0,
            "verified_source_count": 2,
            "rejected_count": 0,
        }

        class FakeResearcher:
            def enrich_qualitative(
                self,
                current: dict[str, Any],
                **_: object,
            ) -> dict[str, Any]:
                return current

        item, qualified = app._finish_candidate(
            FakeResearcher(),
            ("TEST", research, {}, 45.0),
            round_trip_cost_bps=40.0,
        )

        self.assertTrue(qualified)
        self.assertEqual(item["entry_price"], 100.0)
        self.assertAlmostEqual(item["take_profit"], 100 * 1.002 * 1.045 / 0.998)
        self.assertEqual(item["stop_loss"], 97.75)
        self.assertAlmostEqual(item["net_target_return_pct"], 4.5)

    def test_ranking_contains_all_scanned_assets_in_score_order(self) -> None:
        research_by_symbol = {
            "LOW": self._daily_research(),
            "HIGH": self._daily_research(elevated_recent_volume=True),
        }

        class FakeResearcher:
            def analyze(self, *, symbol: str, **_: object) -> dict[str, object]:
                return research_by_symbol.get(symbol, research_by_symbol["HIGH"])

        with patch("app.ResearchAgent", return_value=FakeResearcher()):
            result = app._scan_market(("LOW", "HIGH"), app.ROUND_TRIP_COST_BPS)

        ranked_assets = result["ranked_assets"]
        self.assertEqual([item["symbol"] for item in ranked_assets], ["HIGH", "LOW"])
        self.assertGreaterEqual(
            ranked_assets[0]["broker_points"],
            ranked_assets[1]["broker_points"],
        )

    def test_ranking_keeps_assets_with_market_data_errors(self) -> None:
        research = _research()
        research.update(
            {
                "symbol": "GOOD",
                "timeframe": "4h",
                "timestamp": "2026-11-06T00:00:00+00:00",
                "candles": [],
                "signal": "CZEKAJ",
                "news": {"status": "deferred", "verified_articles": []},
                "fundamentals": {"status": "deferred", "metrics": {}},
            }
        )

        class FakeResearcher:
            def analyze(self, *, symbol: str, **_: object) -> dict[str, object]:
                if symbol == "BAD":
                    raise RuntimeError("Yahoo Finance unavailable")
                return research

        with patch("app.ResearchAgent", return_value=FakeResearcher()):
            result = app._scan_market(("BAD", "GOOD"), app.ROUND_TRIP_COST_BPS)

        by_symbol = {item["symbol"]: item for item in result["ranked_assets"]}
        self.assertEqual(set(by_symbol), {"BAD", "GOOD"})
        self.assertEqual(by_symbol["BAD"]["signal"], "BŁĄD")
        self.assertIn("Yahoo Finance unavailable", by_symbol["BAD"]["error"])

    def test_historical_scan_passes_snapshot_and_skips_current_quality_stage(self) -> None:
        snapshot = app._historical_snapshot(
            app.date(2026, 11, 6),
            app.time(15, 30),
        )
        research = _research(technical_points=30.0)
        research["indicators"]["atr_14"] = 1.0
        research.update(
            {
                "symbol": "TEST",
                "timeframe": "4h",
                "timestamp": "2026-11-06T13:00:00+00:00",
                "candles": [],
                "signal": "KUP",
                "news": {"status": "deferred", "verified_articles": []},
                "fundamentals": {"status": "deferred", "metrics": {}},
            }
        )
        analyzed_with: list[dict[str, Any]] = []

        class FakeResearcher:
            def analyze(self, **kwargs: object) -> dict[str, object]:
                analyzed_with.append(kwargs)
                return research

        with patch("app.ResearchAgent", return_value=FakeResearcher()):
            result = app._scan_market(
                ("TEST",),
                app.ROUND_TRIP_COST_BPS,
                as_of=snapshot,
            )

        self.assertEqual(analyzed_with[0]["as_of"], snapshot)
        self.assertEqual(result["mode"], "history")
        expected_utc = snapshot.astimezone(app.timezone.utc).isoformat()
        self.assertEqual(result["data_as_of"], expected_utc)
        self.assertEqual(result["stage_two_reviewed_assets"], 0)
        self.assertEqual(result["ranked_assets"][0]["data_as_of"], expected_utc)

    def test_main_displays_live_and_historical_modes(self) -> None:
        from streamlit.testing.v1 import AppTest

        test_app = AppTest.from_file("app.py", default_timeout=20).run()
        self.assertFalse(test_app.exception)
        self.assertEqual(
            list(test_app.radio[1].options),
            [
                "🔴 Tryb na żywo (Dane w czasie rzeczywistym)",
                "📜 Tryb historyczny (Backtest w wybranym momencie)",
            ],
        )
        self.assertEqual(
            list(test_app.radio[0].options),
            [app.MULTI_DAY_CATEGORY, app.SHORT_TERM_CATEGORY],
        )
        self.assertFalse(test_app.date_input)

        test_app.radio[0].set_value(app.SHORT_TERM_CATEGORY).run()
        self.assertFalse(test_app.exception)
        self.assertTrue(
            any("przygotowanie do sesji USA od 14:00" in item.value for item in test_app.caption)
        )

        test_app.radio[1].set_value(
            "📜 Tryb historyczny (Backtest w wybranym momencie)"
        ).run()
        self.assertFalse(test_app.exception)
        self.assertTrue(test_app.date_input)
        self.assertTrue(
            any("Uruchom analizę historyczną" in button.label for button in test_app.button)
        )

    def test_news_timeout_falls_back_to_neutral_sentiment(self) -> None:
        with patch(
            "app._cached_gemini_news_review",
            side_effect=RuntimeError("HTTP 429"),
        ):
            review = app._review_candidate_news(
                "AAPL",
                {
                    "verified_articles": [
                        {
                            "title": "Headline",
                            "sources": [{"publisher": "Reuters"}],
                        }
                    ]
                },
            )

        self.assertEqual(review["status"], "neutral_fallback")
        self.assertEqual(review["sentiment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
