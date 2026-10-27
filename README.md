# Trend Following with EMA Crossovers and an ADX Filter

**CFA Club, IIT Indore: Induction Task 2026 | PS2 Finance & Quant | Track 2 (Trend Following)**
Author: Ritesh Agarwal (250003065)

A long-only EMA-crossover strategy, with and without an ADX trend-strength filter, backtested on three Indian large-caps (Bajaj Finance, Titan, ITC). Parameters are chosen on 2015–2021 data and evaluated once on a held-out 2022–2025 period, with realistic costs, next-open execution and a Nifty 50 benchmark.

> **Headline result (honest summary):** the strategy did **not** generalise. The selected setting (EMA 10/50, ADX > 15) had a mean development Sharpe of 0.31, which fell to **-0.33** in the test period. It beat Nifty 50 only for Bajaj Finance in development, lost money on two of three stocks in the test period, and lagged buy-and-hold in every development stock and in two of three test stocks. Its main benefit was shallower drawdowns than buy-and-hold in most cases, not higher returns.

---

## 1. Repository contents

| Item | Description |
|---|---|
| `finance.ipynb` | Full pipeline: data loading, cleaning, EDA, indicators, backtest engine, grid search, final run, robustness analysis |
| `Report.pdf` | 8-page technical report (strategy, rules, assumptions, results, limitations) |
| `CFA_InductionTask2026.pdf` | Problem statement (PS2 Track 2 is the one attempted) |
| `data/` (created on first run) | Cached raw OHLCV CSVs, `performance_metrics.csv`, `trade_log_final.csv`, `analysis_*.csv` |
| `plots/` (created on first run) | All figures (see section 6) |

## 2. How to run

**Requirements:** Python 3.x, `numpy`, `pandas`, `matplotlib`, `yfinance` (plus Jupyter).

```bash
pip install numpy pandas matplotlib yfinance jupyter
jupyter notebook finance.ipynb      # Kernel > Restart & Run All
```

- Run cells **top to bottom**. Cell 25 asserts that earlier cells have been run.
- The first run downloads data from Yahoo Finance and caches it in `data/*_raw.csv`. Later runs read the cache, so results stay stable even if Yahoo revises data. Delete the cache to re-download.
- Internet access is needed on the first run only.
- The pipeline is deterministic (no random numbers).

## 3. Data

| Series | Ticker | Source |
|---|---|---|
| Bajaj Finance | `BAJFINANCE.NS` | Yahoo Finance via `yfinance` |
| Titan | `TITAN.NS` | Yahoo Finance via `yfinance` |
| ITC | `ITC.NS` | Yahoo Finance via `yfinance` |
| Nifty 50 (benchmark) | `^NSEI` | Yahoo Finance via `yfinance` |

- Daily OHLCV, requested 2015-01-01 to 2026-01-01 (end exclusive), so the series cover 2015–2025.
- `auto_adjust=True`: prices are adjusted for splits, bonuses and dividends, so corporate actions are handled consistently.
- **Quality checks:** missing values, duplicate dates, zero-volume days, non-positive prices, OHLC consistency, gaps > 5 calendar days, daily moves > 10%.
- **Cleaning:** drop duplicate dates, sort by date, drop rows missing OHLC, drop non-positive prices, fill missing volume with 0.
- **Why these stocks:** liquid, well-known names with different price behaviour (Bajaj: strong persistent uptrend; Titan: strong uptrend; ITC: long range-bound phase, then a rally). This was done **with hindsight** and is a real selection bias (see section 9).

## 4. Strategy

### Indicators

| Indicator | Definition |
|---|---|
| EMA(n) | `EMA_t = α·C_t + (1-α)·EMA_{t-1}`, `α = 2/(n+1)`, `adjust=False`, full-period warm-up |
| True Range | `max(H-L, abs(H-C_prev), abs(L-C_prev))` |
| ATR(14) | Wilder smoothing of True Range (`α = 1/14`) |
| ADX(14) | Wilder-smoothed DX, where `DX = 100·abs(+DI − −DI)/(+DI + −DI)` |

ADX measures trend **strength**, not direction. Direction comes from the EMA ordering.

### Rules (final frozen setting: EMA 10/50, ADX > 15, ATR 14, stop 3×ATR)

1. **Basic strategy:** long while `EMA_fast > EMA_slow`.
2. **ADX-filtered strategy:** long while `EMA_fast > EMA_slow` **and** `ADX > 15`.
3. **Timing (no look-ahead):** the signal is computed on day *t*'s close and **shifted one session**. The trade executes at day *t+1*'s **open**.
4. **Exit:** when the long condition turns false (executed at next open), or when the trailing stop is hit.
5. **Trailing stop:** `highest close since entry − 3 × ATR(prior day)`, checked against the current day's **low**. If gapped through, the exit fills at the open. After a stop-out, re-entry is blocked until the signal turns off and resets.
6. **Position sizing:** long-only, all-in; buy the maximum whole shares affordable (including entry cost). Each stock is traded independently.

## 5. Backtest assumptions

| Assumption | Value |
|---|---|
| Capital | INR 1,000,000 per stock, per period |
| Fees | 0.10% per side |
| Slippage | 0.05% per side (entry price up, exit price down) |
| Shares | Whole shares only (leftover cash is held) |
| Direction | Long-only (no shorting) |
| Development period | 2015-01-02 to 2021-12-31 |
| Test period | 2022-01-03 to 2025-12-31 (starts flat; open positions are force-closed at period end) |
| Buy & hold | Buy at first open, sell at final close, same costs |
| Nifty 50 | Rebased to the same capital and dates, **no costs** (an index cannot be bought directly) |
| Risk-free rate | 6% p.a. (approx. Indian 91-day T-bill) |
| Return frequency | Daily equity returns |
| Sharpe / Sortino | `mean(excess) / std(excess) × √252` (Sortino uses downside RMS) |
| Annualised return | Geometric, over elapsed calendar years |
| Max drawdown | Largest fall from a prior equity peak |
| Undefined metrics | Shown as **N/A** (e.g. profit factor with no losing trade) |

**Parameter selection.** Grid: EMA fast ∈ {10, 20, 30} × slow ∈ {50, 100, 150} × ADX ∈ {15, 20, 25, 30, 35}, plus a "Basic" (no-ADX) column. The score is the mean development Sharpe across the three stocks, **smoothed over neighbouring grid cells** (±1 step in each dimension) to prefer a plateau over an isolated spike. Result: **EMA 10/50, ADX > 15** (raw development Sharpe 0.31, the highest cell in the heatmap).

**Correctness checks built into the notebook:**
- Toy-example unit tests of entry, exit, stop and cost logic (cell 15)
- Causality tests: indicators computed on truncated vs full history must match at several cutoffs
- Signal-shift assertion, trade order and period assertions, open-price execution checks
- Equity reconciles to initial capital + sum of trade P&L
- Equity cannot change while flat
- Zero-cost run must be at least as good as the costed run

## 6. Results

### Development (2015-01-02 to 2021-12-31)

| Stock | Method | Cum. % | Ann. % | Sharpe | Sortino | Max DD % | Trades | Win % | PF |
|---|---|---|---|---|---|---|---|---|---|
| Bajaj Finance | Basic | 249.75 | 19.59 | 0.79 | 1.43 | -15.88 | 12 | 66.67 | 12.96 |
| | ADX-filtered | 402.68 | 25.95 | 0.95 | 1.64 | -19.15 | 26 | 57.69 | 7.33 |
| | Buy & hold | 1927.39 | 53.73 | 1.17 | 1.73 | -62.44 | N/A | N/A | N/A |
| Titan | Basic | 85.31 | 9.22 | 0.26 | 0.46 | -23.95 | 17 | 58.82 | 2.40 |
| | ADX-filtered | 67.22 | 7.62 | 0.18 | 0.31 | -25.56 | 18 | 55.56 | 2.08 |
| | Buy & hold | 584.17 | 31.63 | 0.84 | 1.31 | -41.73 | N/A | N/A | N/A |
| ITC | Basic | 9.29 | 1.28 | -0.30 | -0.44 | -20.14 | 20 | 30.00 | 1.23 |
| | ADX-filtered | 14.23 | 1.92 | -0.21 | -0.30 | -24.84 | 27 | 40.74 | 1.36 |
| | Buy & hold | 9.04 | 1.24 | -0.03 | -0.04 | -55.28 | N/A | N/A | N/A |
| Nifty 50 | (no costs) | 109.37 | 11.14 | 0.37 | 0.50 | -38.44 | N/A | N/A | N/A |

### Test (2022-01-03 to 2025-12-31)

| Stock | Method | Cum. % | Ann. % | Sharpe | Sortino | Max DD % | Trades | Win % | PF |
|---|---|---|---|---|---|---|---|---|---|
| Bajaj Finance | Basic | -1.23 | -0.31 | -0.34 | -0.53 | -31.05 | 15 | 33.33 | 0.97 |
| | ADX-filtered | -1.44 | -0.36 | -0.36 | -0.57 | -27.36 | 20 | 30.00 | 0.97 |
| | Buy & hold | 43.76 | 9.51 | 0.26 | 0.38 | -32.71 | N/A | N/A | N/A |
| Titan | Basic | -1.94 | -0.49 | -0.42 | -0.59 | -25.81 | 18 | 33.33 | 0.95 |
| | ADX-filtered | -1.68 | -0.42 | -0.47 | -0.67 | -23.10 | 18 | 38.89 | 0.96 |
| | Buy & hold | 62.99 | 13.01 | 0.40 | 0.59 | -28.60 | N/A | N/A | N/A |
| ITC | Basic | -10.49 | -2.74 | -0.73 | -1.00 | -22.71 | 12 | 16.67 | 0.68 |
| | ADX-filtered | 13.44 | 3.21 | -0.16 | -0.23 | -14.45 | 15 | 46.67 | 1.45 |
| | Buy & hold | 123.12 | 22.25 | 0.89 | 1.37 | -20.32 | N/A | N/A | N/A |
| Nifty 50 | (no costs) | 50.28 | 10.74 | 0.41 | 0.57 | -16.47 | N/A | N/A | N/A |

Per-trade statistics (average profit/loss, max profit/loss) are in `Report.pdf` Tables 1–2 and `data/performance_metrics.csv`.

### Key findings

1. **No out-of-sample edge.** Mean selected-setting Sharpe went from +0.31 (dev) to -0.33 (test). Test Sharpes for the ADX strategy were -0.36 (Bajaj), -0.47 (Titan), -0.16 (ITC). Only ITC's filtered strategy made money in test, and it still trailed both ITC buy-and-hold (+123%) and Nifty (+50%).
2. **Weak parameter persistence.** Spearman rank correlation of development vs test Sharpe across the 45 ADX-filtered settings is only about 0.21 (the notebook's scatter title shows 0.22; the report text says 0.21, so re-check the printed value). Neighbouring settings gave mixed test results.
3. **The ADX filter did not help consistently.** It lifted Bajaj and ITC in development but hurt Titan; in test it was roughly neutral except for ITC.
4. **Drawdown is the one consistent benefit.** In development the strategies had much smaller drawdowns than buy-and-hold (e.g. Bajaj -19% vs -62%). In test the advantage shrank, and Nifty's drawdown (-16.5%) was smaller than the strategies' for Bajaj and Titan.
5. **Profits are concentrated.** Top 3 trades gave 55.5% (Bajaj) and 54.9% (ITC) of gross profit in development. Without the top 3, development returns fall from 402.68% to 146.12% (Bajaj), from 14.23% to -11.80% (ITC), and from 67.22% to -20.58% (Titan). Test returns without the top 3 are negative for Bajaj, Titan and ITC-ADX.
6. **Whipsaws and lag.** 44–72% of trades ended on the ATR stop. Median test entries came 32.5–48 calendar days after the prior low, with 62–85% of the move already gone at entry. Median captured move was negative in all six test groups.
7. **Regimes.** Sideways markets were difficult for all three stocks in test (-10.1%, -16.9%, -3.4%). Regime tables are descriptive only (200-day SMA + ADX > 20, known at the previous close), not a validated signal.
8. **Costs matter.** Raising cost + slippage from 0.15% to 0.30% per side pair cut returns by 78.5 pp (Bajaj dev), 18.1 pp (Titan dev), 18.5 pp (ITC dev), and 11.8 / 10.6 / 10.2 pp for the three test stocks.

### Figures (`plots/`)

| File | Shows |
|---|---|
| `price_bajaj_finance.png`, `price_titan.png`, `price_itc.png` | Adjusted close, 200-day SMA, volume, dev/test split |
| `rebased_comparison.png` | Growth of 100 invested, stocks vs Nifty (log scale) |
| `return_distribution.png` | Daily return histograms (mean, std in titles) |
| `parameter_heatmap.png` | Development Sharpe grid, selected cell boxed |
| `equity_curves_final.png` | Equity curves vs buy-and-hold and Nifty, both periods |
| `drawdown_strategy_development.png`, `drawdown_strategy_test.png` | Drawdown from prior peak |
| `trades_bajaj_finance.png`, `trades_titan.png`, `trades_itc.png` | Price, EMAs, entry/exit markers, win/loss shading |
| `regime_daily_returns.png` | Average daily return by prior-close regime |
| `stability_dev_vs_test.png` | Development vs test Sharpe across the grid |

## 7. Notebook map

| Cells | Purpose |
|---|---|
| 0–1 | Config (tickers, dates, split) and cached data loader |
| 2–6 | Inspection, quality report, cleaning, common dates, largest daily moves |
| 7–9 | EDA plots (price + 200-day SMA, rebased comparison, return histograms) |
| 10–12 | Indicators (EMA, ATR, ADX, Wilder) and signals, plus causality tests |
| 13–15 | Cost parameters, backtest engine, buy-and-hold, Nifty benchmark, toy unit tests |
| 16 | Metric functions (equity metrics, trade metrics, N/A display) |
| 17–18 | Development grid search, neighbourhood-smoothed selection, heatmap, robustness table |
| 19–21 | Final run on both periods, sanity assertions, metrics tables, trade log CSV |
| 22–24 | Equity curves, drawdown charts, trade-marker charts |
| 25–32 | Extra analysis: regimes, whipsaws, profit concentration, signal lag, stability, cost sensitivity |

## 8. Output data files (`data/`)

`performance_metrics.csv`, `trade_log_final.csv`, `analysis_regimes.csv`, `analysis_whipsaws.csv`, `analysis_profit_concentration.csv`, `analysis_signal_lag.csv`, `analysis_stability_selected.csv`, `analysis_stability_neighbours.csv`, `analysis_cost_sensitivity.csv`.

## 9. Limitations

- **Selection / survivorship bias.** The three stocks were picked knowing they did well over 2015–2025. The weak test results are therefore, if anything, optimistic for the method.
- **Tiny sample.** Three stocks, long-only, 12–27 trades per stock-period. Win rates and profit factors are statistically noisy.
- **Overfitting risk.** 45 filtered settings were searched on 7 years of data. The weak dev-to-test rank correlation shows the "best" setting was largely luck.
- **Cost assumptions.** Fees and slippage are flat assumptions, and results are sensitive to them. Real impact costs, taxes, STT and stamp duty are not modelled in detail.
- **Whole-share sizing** leaves residual cash and may not scale to other capital levels.
- **Data quality.** Yahoo Finance data can be revised. Corporate-action handling relies on Yahoo's adjustments and is not independently verified.
- **Notebook defaults vs frozen setting.** Cell 10 holds exploratory defaults (EMA 20/50, ADX 25). The reported results use the frozen EMA 10/50, ADX > 15 (`EMA_FAST_F`, `EMA_SLOW_F`, `ADX_MIN_F`). Do not confuse them.
- **Test-period grid.** Cell 31 runs the grid on the test period for diagnostics (stability plot and rank correlation). It is **not** used for selection, but the test period is therefore examined across many settings after the choice was frozen.
- **Nifty comparison** ignores costs, which flatters the benchmark slightly.

**Possible improvements (untested proposals):** a broader, pre-defined stock universe; walk-forward evaluation; alternative noise filters; portfolio-level position sizing; more realistic execution modelling.

## 10. Open items to finish before submission

These are flagged as `[TODO]` in `Report.pdf` and need your action:

1. Report the before/after cleaning counts (run cells 3–5 and copy the output).
2. Add sources for corporate-action statements (Bajaj Finance 2016 split/bonus and 2025 bonus; ITC 2016 1:2 bonus and January 2025 ITC Hotels demerger; no split/bonus for Titan).
3. Confirm from a fresh top-to-bottom run that all look-ahead and sanity assertions pass, and keep the saved notebook output.
4. List any tutorials/articles used, or state that none were used.
5. Reconcile the rank-correlation figure (0.21 in the report vs 0.22 in the plot title).
6. Check the report's cost-sensitivity wording ("0.15% to 0.30%") against the notebook, which compares 0%, 0.15% and 0.30% combined cost + slippage per side.
7. Optionally add a one-line note that short-trade analysis is N/A because the system is long-only.

## 11. Task-requirement checklist (PS2 Track 2)

| Requirement | Where |
|---|---|
| Select data, state source | Section 3; `Report.pdf` §2 |
| Explore and clean data, plots | Notebook cells 2–9; price plots |
| Clear entry/exit, sizing, risk control | Section 4; notebook cells 10–14 |
| Realistic backtest, costs, no look-ahead | Section 5; cells 12, 14, 19–20 |
| Dev/test split, benchmark, limitations | Sections 5, 6, 9 |
| EMA crossover plus ADX confirmation, compared with basic | Both strategies in all tables |
| Trailing / volatility stop | 3×ATR trailing stop |
| Whipsaws, signal lag, profit concentration, parameter stability | Cells 28–31; Report §7 |
| Equity curve, drawdown chart, trade-marker chart | `equity_curves_final.png`, `drawdown_strategy_*.png`, `trades_*.png` |
| Metrics table (cum/ann return, Sharpe, Sortino, MDD, win rate, trades, PF, avg/max win/loss, benchmark) | Section 6; `performance_metrics.csv` |
| Return frequency, annualisation, risk-free rate stated | Section 5 |
| Cite sources, acknowledge AI help | Section 12 |

## 12. Resources and acknowledgements

- **Data:** Yahoo Finance, accessed via the `yfinance` Python package.
- **Libraries:** pandas, numpy, matplotlib, yfinance.
- **Indicators:** ATR, ADX and Wilder smoothing from J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* (1978).
- **AI assistance:** Claude and ChatGPT were used for coding and drafting. The author is prepared to explain the code and results.
- **Tutorials/articles:** [TODO: add any used, or state none].
