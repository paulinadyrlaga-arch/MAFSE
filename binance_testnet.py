"""Binance Spot and USD-M Futures Testnet access."""

import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from time import time
from typing import Any

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv


_ENV_FILE = Path(__file__).with_name(".env")
_MAX_KLINES_PER_REQUEST = 1000
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def to_binance_symbol(symbol: str) -> str:
    """Map app crypto tickers such as BTC-USD to Binance USDT pairs."""
    base, separator, quote = symbol.strip().upper().partition("-")
    if not separator or not base or quote != "USD":
        raise ValueError(f"Nieobsługiwana para Binance Testnet: {symbol}.")
    return f"{base}USDT"


def _create_testnet_client(timeout: float) -> Client:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout Binance musi być dodatni i skończony.")
    load_dotenv(_ENV_FILE, override=False)
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "").strip()
    
    try:
        with open("private.pem", "r") as f:
            private_key_content = f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Nie udało się wczytać pliku private.pem: {e}")

    if not api_key or not private_key_content:
        raise RuntimeError(
            "Brak BINANCE_TESTNET_API_KEY w pliku .env lub pliku private.pem."
        )
        
    return Client(
        api_key=api_key,
        private_key=private_key_content,  # KLUCZOWA ZMIANA: używamy private_key zamiast api_secret
        requests_params={"timeout": timeout},
        testnet=True,
        ping=False,
    )


def get_testnet_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    end_time: datetime | None = None,
    timeout: float = 4.0,
    futures: bool = False,
) -> pd.DataFrame:
    if not 35 <= limit <= 10_000:
        raise ValueError("Liczba świec Binance musi mieścić się w zakresie 35–10000.")
    if end_time is not None:
        if end_time.tzinfo is None:
            raise ValueError("end_time Binance musi zawierać strefę czasową.")
        cursor = int(end_time.astimezone(timezone.utc).timestamp() * 1000)
    else:
        cursor = int(datetime.now(timezone.utc).timestamp() * 1000)

    client = _create_testnet_client(timeout)
    pages: list[list[list[Any]]] = []
    received = 0
    try:
        binance_pair = to_binance_symbol(symbol)
        while received < limit:
            requested = min(_MAX_KLINES_PER_REQUEST, limit - received)
            get_klines = client.futures_klines if futures else client.get_klines
            response = get_klines(
                symbol=binance_pair,
                interval=interval,
                limit=requested,
                endTime=cursor,
            )
            if not isinstance(response, list):
                raise RuntimeError("Binance Testnet zwrócił nieprawidłowe świece.")
            if not response:
                break
            pages.append(response)
            received += len(response)
            if len(response) < requested:
                break
            earliest_open_time = int(response[0][0])
            if earliest_open_time >= cursor:
                raise RuntimeError(
                    "Binance Testnet zwrócił powtarzające się strony świec."
                )
            cursor = earliest_open_time - 1
    finally:
        client.close_connection()

    rows = [row for page in reversed(pages) for row in page]
    if not rows:
        raise RuntimeError(
            f"Binance Testnet nie zwrócił świec dla {to_binance_symbol(symbol)}."
        )
    frame = pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [row[5] for row in rows],
        },
        index=pd.to_datetime([int(row[0]) for row in rows], unit="ms", utc=True),
    )
    frame = frame.loc[:, _OHLCV_COLUMNS]
    for column in _OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[~frame.index.duplicated(keep="last")].sort_index().tail(limit)


def get_futures_testnet_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    end_time: datetime | None = None,
    timeout: float = 4.0,
) -> pd.DataFrame:
    return get_testnet_klines(
        symbol,
        interval,
        limit,
        end_time=end_time,
        timeout=timeout,
        futures=True,
    )


def get_testnet_order_book(
    symbol: str,
    *,
    limit: int = 1000,
    timeout: float = 4.0,
    futures: bool = False,
) -> dict[str, Any]:
    client = _create_testnet_client(timeout)
    try:
        get_order_book = client.futures_order_book if futures else client.get_order_book
        return get_order_book(
            symbol=to_binance_symbol(symbol),
            limit=limit,
        )
    finally:
        client.close_connection()


def submit_testnet_market_buy(
    symbol: str,
    quote_order_qty: float,
    *,
    timeout: float = 4.0,
) -> dict[str, Any]:
    if not math.isfinite(quote_order_qty) or quote_order_qty <= 0:
        raise ValueError("Kwota zlecenia musi być dodatnia i skończona.")
    client = _create_testnet_client(timeout)
    try:
        return client.order_market_buy(
            symbol=to_binance_symbol(symbol),
            quoteOrderQty=quote_order_qty,
        )
    finally:
        client.close_connection()


def get_futures_testnet_account_metrics(
    *,
    timeout: float = 4.0,
    realized_lookback_days: int = 30,
) -> dict[str, float]:
    if realized_lookback_days <= 0:
        raise ValueError("Zakres zrealizowanego PnL musi być dodatni.")
    client = _create_testnet_client(timeout)
    try:
        account = client.futures_account()
        usdt_asset = next(
            (
                asset
                for asset in account.get("assets", [])
                if asset.get("asset") == "USDT"
            ),
            None,
        )
        if usdt_asset is None:
            raise RuntimeError("Konto Futures Testnet nie zawiera salda USDT.")
        start_time = int((time() - realized_lookback_days * 86400) * 1000)
        end_time = int(time() * 1000)
        realized_income = 0.0
        while True:
            entries = client.futures_income_history(
                incomeType="REALIZED_PNL",
                startTime=start_time,
                endTime=end_time,
                limit=1000,
            )
            if not isinstance(entries, list):
                raise RuntimeError(
                    "Binance Futures Testnet zwrócił nieprawidłową historię PnL."
                )
            for entry in entries:
                try:
                    realized_income += float(entry["income"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "Nieprawidłowy wpis REALIZED_PNL z Binance Testnet."
                    ) from error
            if len(entries) < 1000:
                break
            latest_time = max(int(entry["time"]) for entry in entries)
            if latest_time < start_time:
                raise RuntimeError(
                    "Binance Futures Testnet zwrócił powtarzającą się historię PnL."
                )
            start_time = latest_time + 1
        return {
            "balance_usdt": float(usdt_asset["walletBalance"]),
            "unrealized_pnl_usdt": float(account["totalUnrealizedProfit"]),
            "realized_pnl_usdt": realized_income,
        }
    finally:
        client.close_connection()


def get_spot_testnet_account_balance(
    *,
    timeout: float = 4.0,
) -> float:
    client = _create_testnet_client(timeout)
    try:
        account = client.get_account()
        usdt_asset = next(
            (
                asset
                for asset in account.get("balances", [])
                if asset.get("asset") == "USDT"
            ),
            None,
        )
        if usdt_asset is None:
            raise RuntimeError("Konto Spot Testnet nie zawiera salda USDT.")
        return float(usdt_asset["free"]) + float(usdt_asset["locked"])
    finally:
        client.close_connection()


def set_futures_testnet_leverage(
    symbol: str,
    leverage: int = 5,
    *,
    timeout: float = 4.0,
) -> dict[str, Any]:
    if not 1 <= leverage <= 125:
        raise ValueError("Dźwignia Futures musi mieścić się w zakresie 1–125.")
    client = _create_testnet_client(timeout)
    try:
        return client.futures_change_leverage(
            symbol=to_binance_symbol(symbol),
            leverage=leverage,
        )
    finally:
        client.close_connection()


def has_open_futures_testnet_position(
    symbol: str,
    *,
    timeout: float = 4.0,
) -> bool:
    client = _create_testnet_client(timeout)
    try:
        positions = client.futures_position_information(
            symbol=to_binance_symbol(symbol),
        )
        return any(
            Decimal(str(position.get("positionAmt", "0"))) != 0
            for position in positions
        )
    finally:
        client.close_connection()


def get_open_futures_testnet_position_count(
    *,
    timeout: float = 4.0,
) -> int:
    client = _create_testnet_client(timeout)
    try:
        positions = client.futures_position_information()
        return sum(
            Decimal(str(position.get("positionAmt", "0"))) != 0
            for position in positions
        )
    finally:
        client.close_connection()


def submit_testnet_futures_market_buy(
    symbol: str,
    margin_usdt: float,
    *,
    leverage: int = 5,
    timeout: float = 4.0,
) -> dict[str, Any]:
    if not math.isfinite(margin_usdt) or margin_usdt <= 0:
        raise ValueError("Margin zlecenia musi być dodatni i skończony.")
    if not 1 <= leverage <= 125:
        raise ValueError("Dźwignia Futures musi mieścić się w zakresie 1–125.")

    pair = to_binance_symbol(symbol)
    client = _create_testnet_client(timeout)
    try:
        positions = client.futures_position_information(symbol=pair)
        if any(float(position.get("positionAmt", 0)) != 0 for position in positions):
            raise RuntimeError(
                f"Istnieje już otwarta pozycja Futures dla {pair}; pomijam nowe zlecenie."
            )

        account = client.futures_account()
        available_balance = float(account.get("availableBalance", 0))
        if available_balance < margin_usdt:
            raise RuntimeError(
                f"Niewystarczający dostępny margin USDT dla zlecenia {pair}."
            )
        mark = client.futures_mark_price(symbol=pair)
        mark_price = Decimal(str(mark["markPrice"]))
        if not mark_price.is_finite() or mark_price <= 0:
            raise RuntimeError(f"Binance zwrócił nieprawidłową cenę mark dla {pair}.")

        exchange_info = client.futures_exchange_info()
        symbol_info = next(
            (
                item
                for item in exchange_info.get("symbols", [])
                if item.get("symbol") == pair and item.get("status") == "TRADING"
            ),
            None,
        )
        if symbol_info is None:
            raise RuntimeError(f"Para {pair} nie jest aktywna na Futures Testnet.")
        lot_filter = next(
            (
                item
                for item in symbol_info.get("filters", [])
                if item.get("filterType") == "LOT_SIZE"
            ),
            None,
        )
        if lot_filter is None:
            raise RuntimeError(f"Brak reguł LOT_SIZE dla pary Futures {pair}.")
        step_size = Decimal(str(lot_filter["stepSize"]))
        minimum_quantity = Decimal(str(lot_filter["minQty"]))
        desired_notional = Decimal(str(margin_usdt)) * leverage
        raw_quantity = desired_notional / mark_price
        quantity = (
            (raw_quantity / step_size).to_integral_value(rounding=ROUND_DOWN)
            * step_size
        )
        minimum_notional = Decimal("0")
        for filter_info in symbol_info.get("filters", []):
            if filter_info.get("filterType") in {"MIN_NOTIONAL", "NOTIONAL"}:
                minimum_notional = Decimal(
                    str(
                        filter_info.get(
                            "notional",
                            filter_info.get("minNotional", "0"),
                        )
                    )
                )
                break
        if quantity < minimum_quantity or quantity <= 0:
            raise RuntimeError(
                f"Wyliczona wielkość zlecenia {pair} jest poniżej minimum giełdy."
            )
        if quantity * mark_price < minimum_notional:
            raise RuntimeError(
                f"Zlecenie {pair} jest poniżej minimalnej wartości nominalnej giełdy."
            )

        client.futures_change_leverage(symbol=pair, leverage=leverage)
        return client.futures_create_order(
            symbol=pair,
            side="BUY",
            type="MARKET",
            quantity=format(quantity, "f"),
        )
    finally:
        client.close_connection()
