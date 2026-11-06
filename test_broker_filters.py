"""Focused tests for the five-pillar market scanner score."""

import json
import math
import unittest
from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pandas as pd
from research_agent import ResearchAgent
from risk_manager import RiskManager


def _research(
    *,
    technical_points: float = 30.0,
    macro_status: str = "available",
    upcoming_events: list[dict[str, str]] | None = None,
    is_mock: bool = False,
) -> dict[str, object]:
    return {
        "price": 100.0,
        "signal": "KUP",
        "is_mock": is_mock,
        "indicators": {
            "atr_14": 1.5,
            "rsi_14": 60.0,
            "macd": 2.0,
            "macd_signal": 1.0,
            "adx_14": 25.0,
        },
        "news": {"status": "unavailable"},
        "broker_inputs": {
            "technical": {
                "points": technical_points,
                "components": {"momentum": technical_points},
            },
            "target_path": {
                "resistance_status": "available",
                "resistance_price": 110.0,
            },
            "order_book": {
                "status": "available",
                "ask_bid_depth_ratio": 1.0,
                "largest_ask_wall_share": 0.1,
            },
            "liquidity": {
                "status": "available",
                "volume_ratio": 1.5,
                "session_volume_ratio": 1.5,
                "latest_completed_session_volume": 150.0,
                "average_prior_20_session_volume": 100.0,
            },
            "market_regime": {
                "status": "available",
                "regime": "risk_on",
                "benchmark": "SPY",
            },
            "event_risk": {
                "status": "high_risk" if upcoming_events else "incomplete",
                "macro_calendar_status": macro_status,
                "corporate_calendar_status": "available",
                "upcoming_events": upcoming_events or [],
            },
        },
    }


class BrokerFilterScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = RiskManager()

    def _score(self, research: dict[str, object]) -> dict[str, object]:
        with patch.object(
            self.manager,
            "calculate_revolut_position",
            return_value={"success_probability_pct": 100.0},
        ):
            return self.manager.calculate_broker_filter_score(
                research,
                round_trip_cost_bps=0.0,
            )

    def test_pillar_weights_sum_to_100_and_total_is_bounded(self) -> None:
        score = self._score(_research())
        filters = score["filters"]
        self.assertEqual(
            sum(item["maximum"] for item in filters.values()),
            100,
        )
        self.assertLessEqual(score["total_points"], 100)
        self.assertGreaterEqual(score["total_points"], 0)
        self.assertTrue(score["qualified"])

    def test_qualification_threshold_is_strictly_greater_than_75(self) -> None:
        exact_threshold = self._score(_research(technical_points=5.0))
        above_threshold = self._score(_research(technical_points=5.1))

        self.assertEqual(exact_threshold["total_points"], 75.0)
        self.assertFalse(exact_threshold["qualified"])
        self.assertEqual(above_threshold["total_points"], 75.1)
        self.assertTrue(above_threshold["qualified"])

    def test_missing_macro_calendar_blocks_qualification(self) -> None:
        research = _research(macro_status="unavailable")
        score = self._score(research)

        self.assertFalse(score["qualified"])
        self.assertEqual(score["filters"]["macro_event_risk"]["points"], 0.0)
        self.assertTrue(
            any("kalendarza makro" in reason for reason in score["disqualification_reasons"])
        )

    def test_concentrated_ask_wall_reduces_target_path_points(self) -> None:
        research = _research()
        research["broker_inputs"]["order_book"]["largest_ask_wall_share"] = 0.5

        score = self._score(research)
        target_details = score["filters"]["target_resistance_order_book"]["details"]

        self.assertEqual(target_details["order_book_points"], 0.0)

    def test_high_event_and_synthetic_market_data_are_disqualified(self) -> None:
        event_score = self._score(
            _research(
                upcoming_events=[
                    {"type": "earnings", "date": "2026-01-01"},
                ]
            )
        )
        mock_score = self._score(_research(is_mock=True))

        self.assertFalse(event_score["qualified"])
        self.assertEqual(event_score["filters"]["macro_event_risk"]["points"], 0.0)
        self.assertFalse(mock_score["qualified"])
        self.assertTrue(
            any("syntetyczne" in reason for reason in mock_score["disqualification_reasons"])
        )

    def test_invalid_cost_assumption_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.calculate_broker_filter_score(
                _research(),
                round_trip_cost_bps=-1.0,
            )

    def test_adx_below_20_blocks_signal_and_reduces_technical_points(self) -> None:
        research = _research()
        research["indicators"]["adx_14"] = 19.9
        score = self._score(research)

        self.assertFalse(score["qualified"])
        self.assertLess(score["filters"]["technical_momentum"]["points"], 30.0)
        self.assertTrue(
            any("ADX(14) poniżej 20" in reason for reason in score["disqualification_reasons"])
        )

    def test_adx_at_20_does_not_trigger_veto(self) -> None:
        research = _research()
        research["indicators"]["adx_14"] = 20.0
        score = self._score(research)

        self.assertTrue(score["qualified"])
        self.assertEqual(score["filters"]["technical_momentum"]["points"], 30.0)

    def test_latest_session_volume_below_half_of_prior_20_triggers_kill_switch(self) -> None:
        research = _research()
        research["broker_inputs"]["liquidity"]["session_volume_ratio"] = 0.499
        score = self._score(research)

        self.assertFalse(score["qualified"])
        self.assertTrue(
            any("Kill-switch" in reason for reason in score["disqualification_reasons"])
        )

    def test_session_volume_ratio_at_half_passes_kill_switch(self) -> None:
        research = _research()
        research["broker_inputs"]["liquidity"]["session_volume_ratio"] = 0.5
        score = self._score(research)

        self.assertTrue(score["qualified"])
        self.assertFalse(
            any("Kill-switch" in reason for reason in score["disqualification_reasons"])
        )

    def test_revolut_target_gross_up_preserves_net_4_5_percent(self) -> None:
        position = self.manager.calculate_revolut_position(
            purchase_price=100.0,
            current_price=100.0,
            atr=1.0,
            technical_signal="KUP",
            round_trip_cost_bps=40.0,
            adx=25.0,
            session_volume_ratio=1.0,
        )

        self.assertAlmostEqual(position["take_profit"], 104.9)
        self.assertAlmostEqual(position["net_target_return_pct"], 4.5)
        self.assertAlmostEqual(position["estimated_round_trip_cost_pct"], 0.4)
        self.assertTrue(position["hard_filters_passed"])

    def test_revolut_kill_switch_rejects_thin_session_volume(self) -> None:
        position = self.manager.calculate_revolut_position(
            purchase_price=100.0,
            current_price=100.0,
            atr=1.0,
            technical_signal="KUP",
            adx=25.0,
            session_volume_ratio=0.49,
        )

        self.assertFalse(position["hard_filters_passed"])
        self.assertEqual(position["volume_kill_switch_status"], "blocked")

    def test_completed_session_volume_uses_prior_20_sessions_and_ignores_today(self) -> None:
        today = pd.Timestamp.now(tz="UTC").normalize()
        index = pd.date_range(end=today - pd.Timedelta(days=1), periods=21, freq="D")
        index = index.append(pd.DatetimeIndex([today + pd.Timedelta(hours=1)]))
        frame = pd.DataFrame(
            {"Volume": [100.0] * 20 + [49.0, 10_000.0]},
            index=index,
        )

        result = ResearchAgent._completed_session_volume_ratio(frame)

        self.assertAlmostEqual(result["session_volume_ratio"], 0.49)
        self.assertEqual(result["latest_completed_session_volume"], 49.0)
        self.assertEqual(result["average_prior_20_session_volume"], 100.0)

    def test_same_day_earnings_are_included_in_event_window(self) -> None:
        class CalendarTicker:
            @property
            def calendar(self) -> dict[str, list[object]]:
                return {
                    "Earnings Date": [
                        datetime.now(timezone.utc).date(),
                    ]
                }

        def with_mock_calendar(
            symbol: str,
            operation: Callable[[Any], Any],
        ) -> Any:
            self.assertEqual(symbol, "AAPL")
            return operation(CalendarTicker())

        with patch.object(
            ResearchAgent,
            "_with_yfinance_timeout",
            side_effect=with_mock_calendar,
        ):
            event_risk = ResearchAgent()._fetch_event_risk("AAPL")

        self.assertEqual(event_risk["status"], "high_risk")
        self.assertEqual(
            event_risk["upcoming_events"][0]["date"],
            datetime.now(timezone.utc).date().isoformat(),
        )

    def test_order_book_only_counts_levels_inside_target_corridor(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "asks": [["104", "1"], ["104.4", "0.1"], ["106", "10"]],
                "bids": [["100", "2"], ["95", "50"]],
            }
        ).encode()

        with patch("research_agent.urllib.request.urlopen", return_value=response):
            order_book = ResearchAgent()._fetch_order_book("BTC-USD", 100.0)

        self.assertEqual(order_book["status"], "available")
        self.assertTrue(math.isclose(order_book["ask_notional"], 114.44))
        self.assertEqual(order_book["bid_notional"], 200.0)
        self.assertTrue(
            math.isclose(order_book["largest_ask_wall_share"], 104 / 114.44)
        )

    def test_technical_only_analysis_skips_all_quality_network_fetches(self) -> None:
        class HistoryTicker:
            def history(self, **_: object) -> pd.DataFrame:
                index = pd.date_range(
                    end=pd.Timestamp.now(tz="UTC"),
                    periods=600,
                    freq="h",
                )
                close = [100 + index * 0.1 + math.sin(index / 8) for index in range(600)]
                return pd.DataFrame(
                    {
                        "Open": [price - 0.05 for price in close],
                        "High": [price + 0.3 for price in close],
                        "Low": [price - 0.3 for price in close],
                        "Close": close,
                        "Volume": [1_000.0] * len(close),
                    },
                    index=index,
                )

        researcher = ResearchAgent(timeout=0.1)
        with (
            patch("research_agent.yf.Ticker", return_value=HistoryTicker()),
            patch.object(
                researcher,
                "_fetch_qualitative_bundle",
                side_effect=AssertionError("Stage 1 must not fetch quality inputs."),
            ),
            patch(
                "research_agent.ta.adx",
                side_effect=lambda high, low, close, length: pd.DataFrame(
                    {"ADX_14": 10.0},
                    index=high.index,
                ),
            ),
        ):
            result = researcher.analyze(
                symbol="AAPL",
                timeframe="4h",
                limit=60,
                include_qualitative=False,
            )

        self.assertEqual(result["news"]["status"], "deferred")
        self.assertEqual(result["fundamentals"]["status"], "deferred")
        self.assertEqual(result["broker_inputs"]["order_book"]["status"], "deferred")
        self.assertEqual(result["broker_inputs"]["event_risk"]["status"], "deferred")
        self.assertIn("adx_14", result["indicators"])
        self.assertEqual(result["indicators"]["adx_14"], 10.0)
        self.assertEqual(result["signal"], "CZEKAJ")
        self.assertEqual(
            result["broker_inputs"]["liquidity"]["volume_session_status"],
            "available",
        )

    def test_yfinance_quote_requests_receive_bounded_timeout(self) -> None:
        class SharedDataClient:
            def get_raw_json(
                self,
                url: str,
                params: object = None,
                timeout: float = 30.0,
            ) -> float:
                del url, params
                return timeout

        class QuoteTicker:
            _data = SharedDataClient()

        data_client = QuoteTicker._data
        original_method = data_client.get_raw_json
        researcher = ResearchAgent(timeout=0.1)
        with patch("research_agent.yf.Ticker", return_value=QuoteTicker()):
            actual_timeout = researcher._with_yfinance_timeout(
                "AAPL",
                lambda ticker: ticker._data.get_raw_json(
                    "https://example.test",
                    timeout=30.0,
                ),
            )

        self.assertEqual(actual_timeout, 0.1)
        self.assertEqual(data_client.get_raw_json, original_method)

    def test_research_agent_timeout_cannot_exceed_four_seconds(self) -> None:
        self.assertEqual(ResearchAgent(timeout=30.0).timeout, 4.0)


if __name__ == "__main__":
    unittest.main()
