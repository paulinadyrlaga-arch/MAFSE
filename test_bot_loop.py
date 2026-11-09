"""Tests for guarded autonomous Futures Testnet execution."""

import logging
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import app
from bot_loop import run_cycle, scan_crypto_market


class TradingBotLoopTests(unittest.TestCase):
    def test_scanner_uses_futures_testnet_for_crypto_universe(self) -> None:
        fake_researcher = Mock()
        expected = {"mode": "live", "ranked_assets": []}
        with (
            patch("bot_loop.ResearchAgent", return_value=fake_researcher) as constructor,
            patch("bot_loop.app._scan_market", return_value=expected) as scanner,
        ):
            result = scan_crypto_market()

        self.assertIs(result, expected)
        self.assertTrue(constructor.call_args.kwargs["binance_futures"])
        self.assertEqual(scanner.call_args.args[0], app.CRYPTO_UNIVERSE)
        self.assertIs(scanner.call_args.kwargs["researcher"], fake_researcher)

    def test_cycle_trades_only_qualified_crypto_from_ranked_top_three(self) -> None:
        scan = {
            "mode": "live",
            "ranked_assets": [
                {
                    "symbol": "BTC-USD",
                    "signal": "KUP",
                    "qualified": True,
                    "broker_points": 130.0,
                },
                {
                    "symbol": "AAPL",
                    "signal": "KUP",
                    "qualified": True,
                    "broker_points": 129.0,
                },
                {
                    "symbol": "ETH-USD",
                    "signal": "SPRZEDAJ",
                    "qualified": False,
                    "broker_points": 125.0,
                },
                {
                    "symbol": "SOL-USD",
                    "signal": "KUP",
                    "qualified": True,
                    "broker_points": 120.0,
                },
            ],
        }
        leverage_setter = Mock(return_value={})
        position_checker = Mock(return_value=False)
        position_counter = Mock(return_value=0)
        order_sender = Mock(return_value={"orderId": 99, "status": "NEW"})
        logger = Mock(spec=logging.Logger)

        results = run_cycle(
            scanner=lambda: scan,
            leverage_setter=leverage_setter,
            position_checker=position_checker,
            position_counter=position_counter,
            order_sender=order_sender,
            logger=logger,
        )

        self.assertEqual(
            results,
            [
                {"symbol": "BTC-USD", "status": "submitted", "order_id": 99},
                {"symbol": "ETH-USD", "status": "not_qualified"},
            ],
        )
        self.assertEqual(leverage_setter.call_count, len(app.CRYPTO_UNIVERSE))
        leverage_setter.assert_any_call(
            "BTC-USD",
            5,
            timeout=app.NETWORK_TIMEOUT_SECONDS,
        )
        self.assertEqual(position_checker.call_count, 1)
        position_counter.assert_called_once_with(
            timeout=app.NETWORK_TIMEOUT_SECONDS,
        )
        order_sender.assert_called_once_with(
            "BTC-USD",
            10.0,
            leverage=5,
            timeout=app.NETWORK_TIMEOUT_SECONDS,
        )

    def test_cycle_skips_symbol_with_an_open_position(self) -> None:
        scan = {
            "mode": "live",
            "ranked_assets": [
                {
                    "symbol": "BTC-USD",
                    "signal": "KUP",
                    "qualified": True,
                    "broker_points": 130.0,
                }
            ],
        }
        order_sender = Mock()

        results = run_cycle(
            scanner=lambda: scan,
            leverage_setter=Mock(return_value={}),
            position_checker=Mock(return_value=True),
            order_sender=order_sender,
            logger=Mock(spec=logging.Logger),
        )

        self.assertEqual(
            results,
            [{"symbol": "BTC-USD", "status": "already_open"}],
        )
        order_sender.assert_not_called()

    def test_cycle_refuses_historical_scans(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "na żywo"):
            run_cycle(
                scanner=lambda: {"mode": "history", "ranked_assets": []},
                leverage_setter=Mock(return_value={}),
            )

    def test_cycle_does_not_exceed_three_open_positions(self) -> None:
        scan = {
            "mode": "live",
            "ranked_assets": [
                {
                    "symbol": "BTC-USD",
                    "signal": "KUP",
                    "qualified": True,
                    "broker_points": 130.0,
                }
            ],
        }
        order_sender = Mock()

        results = run_cycle(
            scanner=lambda: scan,
            leverage_setter=Mock(return_value={}),
            position_checker=Mock(return_value=False),
            position_counter=Mock(return_value=3),
            order_sender=order_sender,
            logger=Mock(spec=logging.Logger),
        )

        self.assertEqual(
            results,
            [{"symbol": "BTC-USD", "status": "position_limit"}],
        )
        order_sender.assert_not_called()


if __name__ == "__main__":
    unittest.main()
