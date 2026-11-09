"""Autonomous Binance USD-M Futures Testnet scanner and LONG trader."""

import argparse
import logging
import math
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from binance.exceptions import BinanceAPIException, BinanceRequestException
from requests.exceptions import RequestException

import app
from binance_testnet import (
    get_open_futures_testnet_position_count,
    has_open_futures_testnet_position,
    set_futures_testnet_leverage,
    submit_testnet_futures_market_buy,
)
from research_agent import ResearchAgent


DEFAULT_INTERVAL_SECONDS = 60.0
FUTURES_TESTNET_LEVERAGE = 5
FUTURES_TESTNET_MARGIN_USDT = 10.0
MAX_OPEN_FUTURES_POSITIONS = 3
LOG_PATH = Path(__file__).with_name("trading.log")
LOGGER = logging.getLogger("mafse.trading")

_EXCHANGE_ERRORS = (
    BinanceAPIException,
    BinanceRequestException,
    RequestException,
    OSError,
    RuntimeError,
    ValueError,
)


def configure_logging(log_path: Path = LOG_PATH) -> logging.Logger:
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in LOGGER.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        LOGGER.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in LOGGER.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        LOGGER.addHandler(console_handler)
    return LOGGER


def scan_crypto_market() -> dict[str, Any]:
    researcher = ResearchAgent(
        timeout=app.NETWORK_TIMEOUT_SECONDS,
        allow_mock_data=False,
        binance_futures=True,
    )
    return app._scan_market(
        app.CRYPTO_UNIVERSE,
        app.ROUND_TRIP_COST_BPS,
        analysis_category=app.MULTI_DAY_CATEGORY,
        researcher=researcher,
    )


def run_cycle(
    *,
    scanner: Callable[[], dict[str, Any]] = scan_crypto_market,
    leverage_setter: Callable[..., dict[str, Any]] = set_futures_testnet_leverage,
    position_checker: Callable[..., bool] = has_open_futures_testnet_position,
    position_counter: Callable[..., int] = get_open_futures_testnet_position_count,
    order_sender: Callable[..., dict[str, Any]] = submit_testnet_futures_market_buy,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, Any]]:
    scan = scanner()
    ranked_assets = scan.get("ranked_assets")
    if not isinstance(ranked_assets, list):
        raise RuntimeError("Skan nie zwrócił prawidłowego rankingu aktywów.")
    if scan.get("mode") != "live":
        raise RuntimeError("Handel automatyczny wymaga skanu na żywo.")

    leverage_configured: set[str] = set()
    for symbol in app.CRYPTO_UNIVERSE:
        try:
            leverage_setter(
                symbol,
                FUTURES_TESTNET_LEVERAGE,
                timeout=app.NETWORK_TIMEOUT_SECONDS,
            )
        except _EXCHANGE_ERRORS as error:
            logger.error(
                "LEWAR: nie ustawiono %dx dla %s (%s)",
                FUTURES_TESTNET_LEVERAGE,
                symbol,
                type(error).__name__,
            )
        else:
            leverage_configured.add(symbol)

    cycle_results: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked_assets[: app.TOP_RANKED_DETAIL_ASSETS], start=1):
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        if symbol not in app.CRYPTO_UNIVERSE:
            continue
        if item.get("error"):
            logger.warning(
                "SYGNAŁ: %s ranga=%d zawiera błąd danych; zlecenie pominięte",
                symbol,
                rank,
            )
            cycle_results.append({"symbol": symbol, "status": "data_error"})
            continue
        if symbol not in leverage_configured:
            logger.error(
                "CYKL: %s ranga=%d pominięto, bo nie ustawiono dźwigni %dx",
                symbol,
                rank,
                FUTURES_TESTNET_LEVERAGE,
            )
            cycle_results.append(
                {"symbol": symbol, "status": "leverage_error"}
            )
            continue

        signal = item.get("signal")
        points = item.get("broker_points")
        if signal != "KUP" or not item.get("qualified") or item.get("is_mock"):
            logger.info(
                "SYGNAŁ: %s ranga=%d sygnał=%s punkty=%s kwalifikacja=%s; "
                "zlecenie pominięte",
                symbol,
                rank,
                signal,
                points,
                bool(item.get("qualified")),
            )
            cycle_results.append({"symbol": symbol, "status": "not_qualified"})
            continue
        try:
            if position_checker(
                symbol,
                timeout=app.NETWORK_TIMEOUT_SECONDS,
            ):
                logger.info(
                    "POZYCJA: %s ranga=%d już ma otwartą pozycję; "
                    "nie dokładam zlecenia",
                    symbol,
                    rank,
                )
                cycle_results.append({"symbol": symbol, "status": "already_open"})
                continue
            open_positions = position_counter(timeout=app.NETWORK_TIMEOUT_SECONDS)
            if open_positions >= MAX_OPEN_FUTURES_POSITIONS:
                logger.warning(
                    "LIMIT: osiągnięto limit %d otwartych pozycji; "
                    "%s pominięto",
                    MAX_OPEN_FUTURES_POSITIONS,
                    symbol,
                )
                cycle_results.append({"symbol": symbol, "status": "position_limit"})
                continue
            order = order_sender(
                symbol,
                FUTURES_TESTNET_MARGIN_USDT,
                leverage=FUTURES_TESTNET_LEVERAGE,
                timeout=app.NETWORK_TIMEOUT_SECONDS,
            )
        except _EXCHANGE_ERRORS as error:
            logger.error(
                "ZLECENIE: BUY %s ranga=%d nie powiodło się (%s)",
                symbol,
                rank,
                type(error).__name__,
            )
            cycle_results.append({"symbol": symbol, "status": "order_error"})
            continue

        logger.info(
            "TRANSAKCJA: Futures Testnet BUY %s ranga=%d punkty=%s "
            "margin=%.2f USDT dźwignia=%dx orderId=%s status=%s qty=%s; "
            "automatyczne zamknięcie nie jest skonfigurowane",
            symbol,
            rank,
            points,
            FUTURES_TESTNET_MARGIN_USDT,
            FUTURES_TESTNET_LEVERAGE,
            order.get("orderId", "brak"),
            order.get("status", "nieznany"),
            order.get("executedQty", order.get("origQty", "brak")),
        )
        cycle_results.append(
            {
                "symbol": symbol,
                "status": "submitted",
                "order_id": order.get("orderId"),
            }
        )
    return cycle_results


def run_forever(interval_seconds: float = DEFAULT_INTERVAL_SECONDS) -> None:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Odstęp pętli musi być dodatni i skończony.")
    logger = configure_logging()
    logger.warning(
        "BOT URUCHOMIONY: wyłącznie Futures Testnet; LONG KUP; Top 3 krypto; "
        "margin=%.2f USDT, dźwignia=%dx, odstęp=%.1f s. "
        "Brak automatycznych zleceń wyjścia.",
        FUTURES_TESTNET_MARGIN_USDT,
        FUTURES_TESTNET_LEVERAGE,
        interval_seconds,
    )
    try:
        while True:
            logger.info("CYKL: rozpoczęcie skanu rynku.")
            try:
                results = run_cycle(logger=logger)
                logger.info("CYKL: zakończono; wyników=%d.", len(results))
            except _EXCHANGE_ERRORS as error:
                logger.error("CYKL: błąd skanowania (%s).", type(error).__name__)
            except Exception as error:
                logger.error(
                    "CYKL: nieoczekiwany błąd (%s).",
                    type(error).__name__,
                )
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.warning("BOT ZATRZYMANY przez użytkownika.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchom autonomiczny skaner i LONG trader na Binance Futures Testnet."
        )
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Minimalny odstęp między skanami w sekundach (domyślnie: 60).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Wykonaj jeden skan i zakończ (może złożyć zlecenie Testnet).",
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval musi być dodatni i skończony.")
    logger = configure_logging()
    if args.once:
        try:
            run_cycle(logger=logger)
        except _EXCHANGE_ERRORS as error:
            logger.error("Skan jednorazowy nie powiódł się (%s).", type(error).__name__)
            return 1
        return 0
    run_forever(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
