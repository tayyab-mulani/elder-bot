# Backtester — Strategy and Metrics

## Strategy: Elder Triple Screen

A trade is only taken when **all three screens align**. This is the core Elder principle that filters out most false signals.

| Screen | Indicator | Condition | Role |
|---|---|---|---|
| 1 — Trend | EMA | `EMA(short) > EMA(long)` | Uptrend filter |
| 2 — Momentum | MACD | `MACD > MACD signal` | Trend confirmation |
| 3 — Timing | RSI(14) | `RSI < RSI_lower` | Pullback entry |

**Long entry:** uptrend + MACD up + RSI oversold
**Short entry:** downtrend + MACD down + RSI overbought
**Exit long:** RSI crosses above `RSI_upper`
**Exit short:** RSI crosses below `RSI_lower`

**Default parameters:** `win_short=15`, `win_long=200`, `RSI_lower=50`, `RSI_upper=53`, `capital=$10,000`. All adjustable in the UI.

## Performance metrics

| Metric | Formula / Meaning |
|---|---|
| **Total Return** | `(final_equity − initial_capital) / initial_capital × 100` |
| **CAGR** | `(final/initial)^(1/years) − 1` — compound annual growth rate |
| **Sharpe Ratio** | Annualized, risk-free = 0: `mean(daily_returns) / std(daily_returns) × √252` |
| **Max Drawdown** | Largest peak-to-trough decline in the equity curve, as a percentage |
| **Win Rate** | Percentage of closed trades with positive P&L |
| **Profit Factor** | Gross winning P&L ÷ gross losing P&L (absolute); `>1` means profitable overall |
| **Total Trades** | Count of closed round-trip trades |
| **Buy & Hold Return** | Return of simply holding the underlying over the same period with the same capital |
| **Alpha vs. B&H** | Strategy return minus buy-and-hold return |

A collapsible **Trade Log** below the charts lists every closed trade with side, entry price, exit price, P&L, and a win/loss marker — useful for verifying the engine by hand.

---

## Design note: position sizing

> The metrics above should be read with one implementation choice in mind.

The backtester uses **unit-position sizing**: each trade enters and exits exactly **one share** of the underlying. The engine does not deploy a percentage of capital per trade, and does not compound dollar returns across trades as a real account would.

### Why this matters for the numbers

With a $10,000 starting balance trading SPY at ~$400, one share is only ~4% of capital. Even a 75% win rate with a double-digit profit factor produces a small absolute return, because the strategy is almost never fully invested. In buy-and-hold comparisons during a bull market, this shows up as a large negative alpha that reflects **capital under-deployment**, not a flaw in the signal itself.

### What the metrics *are* measuring

Unit-position P&L effectively turns the backtest into a **signal-quality test**, independent of sizing:

- **Win Rate, Profit Factor, Sharpe** → statements about the *signals*
- **Total Return, CAGR, Alpha** → outcomes of the specific (conservative) sizing choice

### Why we chose unit-position sizing

1. **Simplicity and transparency.** P&L per trade is exactly `exit − entry`, trivial to verify by hand against the trade log. No hidden assumptions.
2. **Separation of concerns.** Position sizing and signal generation are two independent problems in trading-system design. Isolating them lets the metrics speak to one thing at a time.
3. **Honest reporting.** We considered implementing Elder's 2% Rule, but it introduces a second degree of freedom (stop-loss placement) whose effect on results is large. Rather than ship an under-tested sizing layer, we left sizing explicit and trivial, and documented the trade-off here.

### What a future iteration would change

A production-ready version would apply Elder's 2% rule: `shares = floor(equity × 0.02 / |entry − stop|)`, add a hard stop-loss exit alongside the RSI exit, and compound returns across trades so equity grows or shrinks realistically. With those changes the buy-and-hold benchmark becomes a fairer comparison.