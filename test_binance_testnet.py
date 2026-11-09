"""Tests for Binance Spot Testnet configuration and request behavior."""

import os
import unittest
from unittest.mock import Mock, patch

from binance.client import Client

from binance_testnet import (
    _create_testnet_client,
    get_futures_testnet_account_metrics,
    get_testnet_klines,
    get_open_futures_testnet_position_count,
    has_open_futures_testnet_position,
    set_futures_testnet_leverage,
    submit_testnet_market_buy,
    submit_testnet_futures_market_buy,
    to_binance_symbol,
)


def _kline(open_time: int) -> list[object]:
    return [
        open_time,
        "10",
        "12",
        "9",
        "11",
        "5",
        open_time + 59_999,
        "55",
        3,
        "2",
        "22",
        "0",
    ]


class BinanceTestnetTests(unittest.TestCase):
    def test_python_binance_routes_requests_to_spot_testnet_url(self) -> None:
        client = Client(
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
            ping=False,
        )
        try:
            self.assertEqual(
                client._create_api_uri("klines", signed=False),
                "https://testnet.binance.vision/api/v3/klines",
            )
        finally:
            client.close_connection()

    def test_python_binance_routes_futures_calls_to_futures_testnet_url(self) -> None:
        client = Client(
            api_key="test-key",
            api_secret="test-secret",
            testnet=True,
            ping=False,
        )
        try:
            self.assertEqual(
                client._create_futures_api_uri("klines"),
                "https://testnet.binancefuture.com/fapi/v1/klines",
            )
        finally:
            client.close_connection()

    def test_testnet_client_uses_dedicated_environment_keys(self) -> None:
        fake_client = Mock()
        with (
            patch.dict(
                os.environ,
                {
                    "BINANCE_TESTNET_API_KEY": "test-key",
                    "BINANCE_TESTNET_SECRET_KEY": "test-secret",
                },
            ),
            patch("binance_testnet.Client", return_value=fake_client) as client_type,
        ):
            result = _create_testnet_client(3.0)

        self.assertIs(result, fake_client)
        kwargs = client_type.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["api_secret"], "test-secret")
        self.assertTrue(kwargs["testnet"])
        self.assertFalse(kwargs["ping"])
        self.assertEqual(kwargs["requests_params"], {"timeout": 3.0})

    def test_missing_testnet_credentials_are_an_explicit_error(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("binance_testnet.load_dotenv"),
        ):
            with self.assertRaisesRegex(RuntimeError, "BINANCE_TESTNET_API_KEY"):
                _create_testnet_client(3.0)

    def test_crypto_symbol_maps_to_usdt_and_equities_are_rejected(self) -> None:
        self.assertEqual(to_binance_symbol("BTC-USD"), "BTCUSDT")
        with self.assertRaises(ValueError):
            to_binance_symbol("AAPL")

    def test_klines_are_paginated_and_return_chronological_ohlcv(self) -> None:
        client = Mock()
        first_page = [
            _kline(open_time)
            for open_time in range(60_000_000, 120_000_000, 60_000)
        ]
        client.get_klines.side_effect = [
            first_page,
            [_kline(0)],
        ]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            frame = get_testnet_klines("BTC-USD", "1m", 1001, timeout=1.0)

        self.assertEqual(len(frame), 1001)
        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(frame.index.is_monotonic_increasing)
        self.assertEqual(frame.iloc[-1]["Close"], 11.0)
        self.assertEqual(client.get_klines.call_count, 2)
        first_call, second_call = client.get_klines.call_args_list
        self.assertEqual(first_call.kwargs["symbol"], "BTCUSDT")
        self.assertEqual(first_call.kwargs["limit"], 1000)
        self.assertEqual(second_call.kwargs["limit"], 1)
        self.assertEqual(
            second_call.kwargs["endTime"],
            int(first_page[0][0]) - 1,
        )
        client.close_connection.assert_called_once()

    def test_market_buy_uses_quote_quantity_on_testnet(self) -> None:
        client = Mock()
        client.order_market_buy.return_value = {"orderId": 123, "status": "FILLED"}
        with patch(
            "binance_testnet._create_testnet_client",
            return_value=client,
        ):
            result = submit_testnet_market_buy("ETH-USD", 50.0)

        self.assertEqual(result["orderId"], 123)
        client.order_market_buy.assert_called_once_with(
            symbol="ETHUSDT",
            quoteOrderQty=50.0,
        )
        client.close_connection.assert_called_once()

    def test_futures_klines_use_futures_endpoint(self) -> None:
        client = Mock()
        client.futures_klines.return_value = [_kline(1)]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            frame = get_testnet_klines(
                "BTC-USD",
                "1d",
                35,
                futures=True,
            )

        self.assertEqual(len(frame), 1)
        client.futures_klines.assert_called_once_with(
            symbol="BTCUSDT",
            interval="1d",
            limit=35,
            endTime=client.futures_klines.call_args.kwargs["endTime"],
        )
        client.get_klines.assert_not_called()
        client.close_connection.assert_called_once()

    def test_futures_account_metrics_include_balance_and_realized_pnl(self) -> None:
        client = Mock()
        client.futures_account.return_value = {
            "assets": [{"asset": "USDT", "walletBalance": "100.25"}],
            "totalUnrealizedProfit": "-2.50",
        }
        client.futures_income_history.return_value = [
            {"income": "3.25", "time": 1000},
            {"income": "-0.75", "time": 2000},
        ]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            metrics = get_futures_testnet_account_metrics()

        self.assertEqual(metrics, {
            "balance_usdt": 100.25,
            "unrealized_pnl_usdt": -2.5,
            "realized_pnl_usdt": 2.5,
        })
        client.futures_income_history.assert_called_once()
        self.assertEqual(
            client.futures_income_history.call_args.kwargs["incomeType"],
            "REALIZED_PNL",
        )
        client.close_connection.assert_called_once()

    def test_futures_leverage_is_set_for_the_symbol(self) -> None:
        client = Mock()
        client.futures_change_leverage.return_value = {
            "symbol": "BTCUSDT",
            "leverage": 5,
        }
        with patch("binance_testnet._create_testnet_client", return_value=client):
            result = set_futures_testnet_leverage("BTC-USD")

        self.assertEqual(result["leverage"], 5)
        client.futures_change_leverage.assert_called_once_with(
            symbol="BTCUSDT",
            leverage=5,
        )
        client.close_connection.assert_called_once()

    def test_futures_buy_sizes_market_order_by_five_times_margin(self) -> None:
        client = Mock()
        client.futures_position_information.return_value = [
            {"positionAmt": "0"},
        ]
        client.futures_account.return_value = {"availableBalance": "100"}
        client.futures_mark_price.return_value = {"markPrice": "100"}
        client.futures_exchange_info.return_value = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "filters": [
                        {
                            "filterType": "LOT_SIZE",
                            "stepSize": "0.001",
                            "minQty": "0.001",
                        },
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
            ]
        }
        client.futures_create_order.return_value = {
            "orderId": 456,
            "status": "NEW",
        }
        with patch("binance_testnet._create_testnet_client", return_value=client):
            order = submit_testnet_futures_market_buy("BTC-USD", 10.0, leverage=5)

        self.assertEqual(order["orderId"], 456)
        client.futures_change_leverage.assert_called_once_with(
            symbol="BTCUSDT",
            leverage=5,
        )
        client.futures_create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            quantity="0.5",
        )
        client.close_connection.assert_called_once()

    def test_futures_buy_refuses_to_add_to_an_existing_position(self) -> None:
        client = Mock()
        client.futures_position_information.return_value = [
            {"positionAmt": "0.25"},
        ]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "już otwarta pozycja"):
                submit_testnet_futures_market_buy("BTC-USD", 10.0)

        client.futures_create_order.assert_not_called()
        client.close_connection.assert_called_once()

    def test_open_position_check_detects_any_nonzero_position(self) -> None:
        client = Mock()
        client.futures_position_information.return_value = [
            {"positionAmt": "0"},
            {"positionAmt": "-0.01"},
        ]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            self.assertTrue(has_open_futures_testnet_position("ETH-USD"))

    def test_open_position_counter_counts_distinct_nonzero_contracts(self) -> None:
        client = Mock()
        client.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0"},
            {"symbol": "ETHUSDT", "positionAmt": "0.5"},
            {"symbol": "SOLUSDT", "positionAmt": "-2"},
        ]
        with patch("binance_testnet._create_testnet_client", return_value=client):
            self.assertEqual(get_open_futures_testnet_position_count(), 2)
        client.close_connection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
