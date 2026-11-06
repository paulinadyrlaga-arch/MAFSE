"""Tests for the scanner's two-stage API gate and neutral quality fallback."""

import unittest
from unittest.mock import patch
from typing import Any

import app
from test_broker_filters import _research


class ScannerScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        app._scan_market.clear()

    def tearDown(self) -> None:
        app._scan_market.clear()

    def _run_scan(
        self,
        technical_points: float,
        signal: str,
    ) -> tuple[dict[str, Any], Any]:
        research = _research(technical_points=technical_points)
        research.update(
            {
                "symbol": "TEST",
                "timeframe": "4h",
                "timestamp": "2026-11-06T00:00:00+00:00",
                "candles": [{"timestamp": "2026-11-06T00:00:00+00:00", "close": 100.0}],
            }
        )
        research["signal"] = signal
        research["news"] = {
            "status": "deferred",
            "verified_articles": [],
            "sentiment_score": 0.0,
            "verified_source_count": 0,
            "rejected_count": 0,
        }
        research["fundamentals"] = {
            "status": "deferred",
            "metrics": {},
            "quality_score": None,
        }

        class FakeResearcher:
            def analyze(self, **_: object) -> dict[str, object]:
                return research

        with (
            patch("app.ResearchAgent", return_value=FakeResearcher()),
            patch(
                "app._finish_candidate",
                return_value=({"broker_points": 95.0}, True),
            ) as finish_candidate,
        ):
            result = app._scan_market((f"test-{technical_points}-{signal}",), 0.0)
        return result, finish_candidate

    def test_weak_technical_candidate_does_not_enter_quality_stage(self) -> None:
        result, finish_candidate = self._run_scan(0.0, "KUP")

        finish_candidate.assert_not_called()
        self.assertEqual(result["stage_two_reviewed_assets"], 0)
        self.assertEqual(len(result["watchlist"]), 1)

    def test_only_high_scoring_buy_candidates_enter_quality_stage(self) -> None:
        result, finish_candidate = self._run_scan(30.0, "KUP")

        finish_candidate.assert_called_once()
        self.assertEqual(result["stage_two_reviewed_assets"], 1)
        self.assertEqual(len(result["recommendations"]), 1)

    def test_non_buy_signal_never_enters_quality_stage(self) -> None:
        result, finish_candidate = self._run_scan(30.0, "CZEKAJ")

        finish_candidate.assert_not_called()
        self.assertEqual(result["stage_two_reviewed_assets"], 0)

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
