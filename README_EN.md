# MAFSE

MAFSE is a Python multi-agent market-analysis project with a Streamlit market
scanner. Its agents use Gemini for planning and report generation, Yahoo
Finance for equities, and Binance Spot Testnet for cryptocurrency market data
and demo orders. The scanner can rank equities and cryptocurrencies using
separate multi-day (swing) and short-term (day-trading) models.

> **Important:** This project provides heuristic analysis, not financial
> advice. It does not monitor positions. Explicit buttons can send manual
> market BUY orders for qualifying Top 3 crypto assets to Binance Spot Testnet
> only; no production orders are sent and SELL signals are not executed.

## Requirements and setup

Python 3 and the dependencies listed in `requirements.txt` are required:

```bash
python3 -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in your environment to use Gemini-backed planning,
qualitative reviews, and report generation. Do not put API keys in source
files. Put Binance Testnet credentials in `.env` as
`BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_SECRET_KEY`. The Futures loop
requires these credentials to have USD-M Futures Testnet access; Spot and
Futures are separate account environments. Never use production keys.
`GEMINI_MODEL` is optional; the default is `gemini-3.8-flash`.

Run the Streamlit dashboard:

```bash
python3 -m streamlit run app.py
```

The scanner does not start automatically. Choose a market scope and analysis
category, then explicitly start a live scan or historical analysis. The
dashboard supports:

- Live and point-in-time historical scans.
- Multi-day swing and short-term 15-minute models.
- A broad watchlist of popular US/EU stocks and major cryptocurrencies.
  Equities use Yahoo Finance; crypto candles and order books use Binance Spot
  Testnet. The watchlist is not an authoritative list of instruments available
  in Revolut; availability varies by country and account.
- A complete descending ranking, score breakdowns, source information, and
  conditional entry/exit levels.
- Individual manual market BUY buttons for qualifying crypto assets in the
  Top 3, using a fixed 50 USDT quote amount per order. SELL signals are shown
  but not executed.

  The scanner computes the lightweight ranking for the full selected universe,
  then performs qualitative enrichment and retains detailed reports, indicators,
  and charts only for the **Top 3** highest-ranked assets.

Historical intraday availability is limited by Yahoo Finance (typically around
60 days). Historical scans exclude current news, analyst opinions, and
calendars to avoid look-ahead bias. Missing market data is shown as an error;
the application does not replace it with fabricated prices.

## Scanner models and risk

Both scanner categories use a score from 0 to 150. **KUP (BUY)** or
**SPRZEDAJ (SELL)** is the trend direction; the 110-point threshold and other
filters separately determine whether an asset qualifies for a BUY
recommendation. A low score does not replace the directional signal with
"WAIT":

- **Multi-day (swing):** trend/SMA structure, RSI and MACD, 5/20-session
  volume, support proximity, analyst consensus, macro and market context, and
  risk/reward.
- **Short-term (day trading):** relative volume, session VWAP and volume
  profile, RSI/MACD momentum, risk/reward, analyst/news sentiment, session
  timing and macro filters, plus a trailing-stop/time-stop model. Relative
  volume below 1.0 or net risk/reward below 1:2 disqualifies the setup.

The initial Stop-Loss buffer is **1.8×ATR**. If the calculated stop would be
zero or negative, the scanner uses its documented 2% fallback. The trailing
stop activates after a 1×ATR move and follows the maximum price by 1.8×ATR; the
short-term model also calls for closing by the end of the session. These are
calculated scenarios only: the dashboard does not monitor a position or
automatically execute exits.

Take-Profit targets are set to +4.5% net for swing and +2.0% net for the
short-term model. The fee assumption is at least **0.1% per side (0.2%
round-trip)**. The Take-Profit calculation grosses up the target to account
for modeled entry and exit fees:

```text
entry × (1 + entry_fee) × (1 + net_target) / (1 - exit_fee)
```

The **Potencjalny zysk (Potential profit)** metric is shown in both categories.
It reports the estimated net return from the entry/trigger price to Take-Profit
after modeled entry and exit fees. The fee assumption is not a live Revolut
quote and does not include any additional spread or slippage.

For US equities, the regular session usually opens at **15:30 Polish local
time** (often referred to as 15:30 CET); during the weeks when US and European
daylight-saving changes do not align, it can be 14:30 in Poland. The short-term
scanner may start evaluating pre-market data around 14:00 Polish time when
Yahoo Finance provides it. Crypto markets trade 24/7.

News sentiment is included only for similar headlines appearing in at least
two approved publisher sources. This is a conservative source/corroboration
check, not proof that an article's claims are true. Macro-calendar or
qualitative-data failures are surfaced and do not silently become evidence of
low risk.

## Autonomous Futures Testnet loop

The dashboard refreshes Futures Testnet USDT balance, unrealized PnL, and
realized PnL over the last 30 days without caching when the page renders.

Start a separate process to scan the 21-crypto watchlist using USD-M Futures
Testnet candles at least once per minute and send LONG market BUY orders only
for qualified KUP signals ranked in the crypto Top 3:

```bash
python3 bot_loop.py
```

Each order uses 10 USDT of margin at 5x leverage (approximately 50 USDT
notional, subject to exchange quantity filters). The loop sets 5x leverage for
all analyzed crypto symbols and checks for existing positions before trading; it
will not add to a position for the same symbol, including after a restart, and
limits the account to at most three open positions.
SELL signals are logged but not executed. The bot does **not** create stop-loss
or take-profit orders and does **not** automatically close positions. Monitor
the demo account and close positions manually.

Activity and order results, excluding API credentials, are written to the
rotating `trading.log`. `python3 bot_loop.py --once` runs one cycle and may
place a Testnet order. Use `--interval 60` to set the pause after each cycle;
stop the loop with Ctrl+C.

## Command-line analysis

The multi-agent pipeline plans an analysis with Gemini, fetches OHLCV candles
from Binance Spot Testnet for crypto and Yahoo Finance for equities, calculates
indicators and risk levels, then generates a Polish report:

```bash
python3 main.py "Analyze BTC/USDT" --capital 10000
```

Capital is denominated in USD. Binance crypto pairs are quoted in USDT; symbol
normalization does not perform currency conversion.

Test the Gemini API connection (requires `GEMINI_API_KEY`):

```bash
python3 test_api.py
```

The Gemini client retries HTTP 503 responses up to three times. For an offline
deterministic pipeline smoke test, run:

```bash
python3 dry_run.py
```

## Market data and dependencies

`research_agent.py` fetches OHLCV candles from Binance Spot Testnet for crypto
and Yahoo Finance for equities. It calculates RSI(14), MACD(12, 26, 9),
ATR(14), and ADX(14) using `pandas-ta-classic`. Crypto analysis requires the
Testnet credentials above but no Gemini key:

```bash
python3 research_agent.py --symbol BTC-USD --timeframe 1d --limit 100
```

`requirements.txt` includes the direct runtime dependencies used by the
application, including `python-binance` and `python-dotenv`. Tests use
Python's built-in `unittest` framework.

Run all unit tests:

```bash
python3 -m unittest discover -v
```
