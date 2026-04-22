"""
backtest_engine.py
------------------
Alexander Elder's Triple Screen Trading Strategy backtester.

All indicator logic and entry/exit rules are unchanged — they are wrapped
in a single run_backtest() function so the Streamlit UI can call it with
user-selected parameters.

Indicators used:
    - EMA(win_short) and EMA(win_long)  → trend direction (Screen 1)
    - MACD + Signal line                → momentum confirmation (Screen 2)
    - RSI(14)                           → entry timing / exit trigger (Screen 3)

Entry / Exit logic :
    LONG  entry : EMA_short > EMA_long  AND  MACD > Signal  AND  RSI < RSI_lower
    SHORT entry : EMA_short < EMA_long  AND  MACD < Signal  AND  RSI > RSI_upper
    EXIT  long  : RSI > RSI_upper
    EXIT  short : RSI < RSI_lower
"""

import yfinance as yf
import pandas as pd
import numpy as np


# ── Indicator functions ──────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast   = ema(series, fast)
    ema_slow   = ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = pd.Series(np.where(delta > 0, delta, 0), index=series.index)
    loss     = pd.Series(np.where(delta < 0, -delta, 0), index=series.index)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── Performance metric helpers ─────────────────────────────────────────────

def _compute_metrics(
    equity_curve: list,
    close_prices: pd.Series,
    trade_log:    list,
    initial_capital: float,
) -> dict:
    """
    Compute performance metrics from an equity curve, the underlying price
    series, and a list of closed trades.

    trade_log entries are dicts:
        {"side": "long"/"short", "entry_price": float, "exit_price": float, "pnl": float}
    """
    equity = pd.Series(equity_curve, index=close_prices.index)

    # ── Total return & CAGR ────────────────────────────────────────────────
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital * 100

    n_days = (equity.index[-1] - equity.index[0]).days
    years  = n_days / 365.25 if n_days > 0 else 1.0
    if final_equity > 0 and initial_capital > 0:
        cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # ── Sharpe ratio (annualized, risk-free = 0) ───────────────────────────
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # ── Max drawdown ───────────────────────────────────────────────────────
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = float(drawdown.min()) * 100   # negative number, e.g. -23.4

    # ── Trade statistics ───────────────────────────────────────────────────
    n_trades   = len(trade_log)
    wins       = [t for t in trade_log if t["pnl"] > 0]
    losses     = [t for t in trade_log if t["pnl"] <= 0]
    n_wins     = len(wins)
    win_rate   = (n_wins / n_trades * 100) if n_trades > 0 else 0.0

    avg_win    = float(np.mean([t["pnl"] for t in wins]))   if wins   else 0.0
    avg_loss   = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0

    # Profit factor = gross wins / gross losses  (abs)
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    # ── Buy & Hold benchmark ───────────────────────────────────────────────
    # Same capital, same period, just hold the underlying from day 1.
    first_price = float(close_prices.iloc[0])
    last_price  = float(close_prices.iloc[-1])
    bh_return   = (last_price - first_price) / first_price * 100
    bh_final    = initial_capital * (1 + bh_return / 100)
    alpha       = total_return - bh_return   # strategy - benchmark, in %

    return {
        "total_return":   total_return,
        "cagr":           cagr,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "n_trades":       n_trades,
        "n_wins":         n_wins,
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "bh_return":      bh_return,
        "bh_final":       bh_final,
        "alpha":          alpha,
    }


# ── Main backtest function ─────────────────────────────────────────────────

def run_backtest(
    symbol:          str   = "SPY",
    start:           str   = "2020-01-01",
    end:             str   = "2025-12-31",
    win_short:       int   = 15,
    win_long:        int   = 200,
    rsi_lower:       float = 50.0,
    rsi_upper:       float = 53.0,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Run the Elder Triple Screen backtest and return all results needed
    for the Streamlit dashboard charts and metrics.

    Parameters
    ----------
    symbol          : Yahoo Finance ticker (e.g. 'AAPL', 'SPY')
    start           : backtest start date  YYYY-MM-DD
    end             : backtest end date    YYYY-MM-DD
    win_short       : short EMA window
    win_long        : long  EMA window
    rsi_lower       : RSI threshold for long entry / short exit
    rsi_upper       : RSI threshold for short entry / long exit
    initial_capital : starting portfolio value in USD

    Returns
    -------
    dict with keys:
        data          : pd.DataFrame  full OHLCV + indicators + equity
        buy_dates     : list[datetime]           — long entries
        sell_dates    : list[datetime]           — short entries
        buy_prices    : list[float]
        sell_prices   : list[float]
        final_equity  : float
        total_return  : float  (%)
        n_long_trades : int    — long entries taken
        n_short_trades: int    — short entries taken
        metrics       : dict   — performance metrics (see _compute_metrics)
        trade_log     : list[dict]  — closed round-trip trades
        error         : str | None  — non-None if download/calc failed
    """

    # 1. Download data
    try:
        raw = yf.download(symbol, start=start, end=end, progress=False)
        if raw.empty:
            return {"error": f"No data returned for '{symbol}'. Check the ticker."}

        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw.xs(symbol, axis=1, level="Ticker")
        else:
            data = raw.copy()

        if len(data) < win_long + 30:
            return {"error": (
                f"Not enough data for {symbol} with win_long={win_long}. "
                "Try a wider date range or a smaller win_long value."
            )}
    except Exception as e:
        return {"error": f"Download failed: {e}"}

    # 2. Compute indicators
    data[f"EMA{win_short}"] = ema(data["Close"], win_short)
    data[f"EMA{win_long}"]  = ema(data["Close"], win_long)
    data["MACD"], data["MACD_signal"] = macd(data["Close"])
    data["RSI"] = rsi(data["Close"])

    # 3. Simulate trades
    capital  = initial_capital
    cash     = capital
    position = 0        #  0=flat, 1=long, -1=short
    entry_price = 0.0

    equity_curve  = []
    buy_indices   = []   # long entries
    sell_indices  = []   # short entries
    trade_log     = []   # closed round-trip trades

    for i in range(len(data)):
        price      = float(data["Close"].iloc[i])
        trend_up   = data[f"EMA{win_short}"].iloc[i] > data[f"EMA{win_long}"].iloc[i]
        trend_down = not trend_up
        macd_up    = data["MACD"].iloc[i] > data["MACD_signal"].iloc[i]
        macd_down  = not macd_up
        rsi_val    = data["RSI"].iloc[i]
        rsi_low    = rsi_val < rsi_lower
        rsi_high   = rsi_val > rsi_upper

        # LONG ENTRY
        if position == 0 and trend_up and macd_up and rsi_low:
            position    = 1
            entry_price = price
            buy_indices.append(i)

        # SHORT ENTRY
        elif position == 0 and trend_down and macd_down and rsi_high:
            position    = -1
            entry_price = price
            sell_indices.append(i)

        # EXIT LONG
        elif position == 1 and rsi_high:
            pnl      = price - entry_price
            cash    += pnl
            trade_log.append({
                "side":        "long",
                "entry_price": entry_price,
                "exit_price":  price,
                "pnl":         pnl,
            })
            position = 0

        # EXIT SHORT
        elif position == -1 and rsi_low:
            pnl      = entry_price - price
            cash    += pnl
            trade_log.append({
                "side":        "short",
                "entry_price": entry_price,
                "exit_price":  price,
                "pnl":         pnl,
            })
            position = 0

        # EQUITY SNAPSHOT
        if position == 1:
            equity = cash + (price - entry_price)
        elif position == -1:
            equity = cash + (entry_price - price)
        else:
            equity = cash
        equity_curve.append(equity)

    data["Equity"] = equity_curve

    # 4. Collect results
    final_equity   = equity_curve[-1] if equity_curve else initial_capital
    total_return   = (final_equity - initial_capital) / initial_capital * 100

    buy_dates  = [data.index[i] for i in buy_indices]
    sell_dates = [data.index[i] for i in sell_indices]
    buy_prices = [float(data["Close"].iloc[i]) for i in buy_indices]
    sell_prices= [float(data["Close"].iloc[i]) for i in sell_indices]

    # Performance metrics
    metrics = _compute_metrics(
        equity_curve   = equity_curve,
        close_prices   = data["Close"],
        trade_log      = trade_log,
        initial_capital= initial_capital,
    )

    return {
        "data":            data,
        "buy_dates":       buy_dates,
        "sell_dates":      sell_dates,
        "buy_prices":      buy_prices,
        "sell_prices":     sell_prices,
        "final_equity":    final_equity,
        "total_return":    total_return,
        "n_long_trades":   len(buy_indices),
        "n_short_trades":  len(sell_indices),
        "initial_capital": initial_capital,
        "metrics":         metrics,
        "trade_log":       trade_log,
        "error":           None,
    }