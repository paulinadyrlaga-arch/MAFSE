# MAFSE

MAFSE is a Python multi-agent market-analysis project with a Streamlit market
scanner. Its agents use Gemini for planning and report generation and Yahoo
Finance for market data. The scanner can rank equities and cryptocurrencies
using separate multi-day (swing) and short-term (day-trading) models.

> **Important:** This project provides heuristic analysis, not financial
> advice. It does not place or monitor orders. Prices and indicators may be
> delayed, and no score or estimated probability guarantees a result.

## Requirements and setup

Python 3 and the dependencies listed in `requirements.txt` are required:

```bash
python3 -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in your environment to use Gemini-backed planning,
qualitative reviews, and report generation. Do not put API keys in source
files. `GEMINI_MODEL` is optional; the default is `gemini-3.8-flash`.

Run the Streamlit dashboard:

```bash
python3 -m streamlit run app.py
```

The scanner does not start automatically. Choose a market scope and analysis
category, then explicitly start a live scan or historical analysis. The
dashboard supports:

- Live and point-in-time historical scans.
- Multi-day swing and short-term 15-minute models.
- A broad Yahoo Finance watchlist of popular US/EU stocks and major
  cryptocurrencies. It is not an authoritative list of instruments available
  in Revolut; availability varies by country and account.
- A complete descending ranking, score breakdowns, source information, and
  conditional entry/exit levels.

Historical intraday availability is limited by Yahoo Finance (typically around
60 days). Historical scans exclude current news, analyst opinions, and
calendars to avoid look-ahead bias. Missing market data is shown as an error;
the application does not replace it with fabricated prices.

## Scanner models and risk

Both scanner categories use a score from 0 to 150. A **KUP (BUY)** signal
requires at least 110 points and passing additional category-specific checks:

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
calculated scenarios only: the dashboard does not monitor a position or send
orders.

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

## Command-line analysis

The multi-agent pipeline plans an analysis with Gemini, fetches Yahoo Finance
OHLCV candles, calculates indicators and risk levels, then generates a Polish
report:

```bash
python3 main.py "Analyze BTC/USDT" --capital 10000
```

Capital is denominated in USD. Yahoo Finance data for `BTC-USD` is also in USD;
normalizing a `*/USDT` pair to `*-USD` does not convert currencies.

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

`research_agent.py` fetches OHLCV candles from Yahoo Finance and calculates
RSI(14), MACD(12, 26, 9), ATR(14), and ADX(14) using `pandas-ta-classic`.
The ResearchAgent's standalone market-data analysis does not require a Gemini
key:

```bash
python3 research_agent.py --symbol BTC-USD --timeframe 1d --limit 100
```

`requirements.txt` includes the direct runtime dependencies used by the
application: `google-genai`, `numpy`, `pandas`, `yfinance`,
`pandas-ta-classic`, `tenacity`, and `streamlit`. Tests use Python's built-in
`unittest` framework.

Run all unit tests:

```bash
python3 -m unittest discover -v
```
