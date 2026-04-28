# run_backtest_all.py — Backtest ETFTRADE universe using the installed skill engine
import sys
import os
import csv
from pathlib import Path
from datetime import datetime, timedelta

# Add skill scripts to path
SKILL_DIR = Path(os.path.expanduser("~")) / ".agents" / "skills" / "backtesting-trading-strategies"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from backtest import run_backtest, load_data
from metrics import format_results

ETF_UNIVERSE = ["SPY", "VGT", "SCHD", "IFRA", "SCHF"]
STRATEGIES   = ["macd", "sma_crossover", "rsi_reversal", "bollinger_bands"]

CAPITAL      = 100_000
STOP_LOSS    = 0.05
TAKE_PROFIT  = 0.12
MAX_POS_SIZE = 0.10
PERIOD_DAYS  = 730   # ~2 years

risk = {"stop_loss": STOP_LOSS, "take_profit": TAKE_PROFIT, "max_position_size": MAX_POS_SIZE}
data_dir = SKILL_DIR / "data"
end   = datetime.now()
start = end - timedelta(days=PERIOD_DAYS + 60)

rows = []
print(f"{'Symbol':<6} {'Strategy':<18} {'Return':>8} {'CAGR':>7} {'Sharpe':>7} {'WinRate':>8} {'PF':>5} {'MaxDD':>8} {'Trades':>7}")
print("-" * 90)

for etf in ETF_UNIVERSE:
    try:
        df = load_data(etf, start, end, data_dir)
        df.attrs["symbol"] = etf
        df_period = df[df.index >= (end - timedelta(days=PERIOD_DAYS)).strftime("%Y-%m-%d")]
    except Exception as e:
        print(f"[{etf}] Data error: {e}")
        continue

    for strat in STRATEGIES:
        try:
            r = run_backtest(strat, df_period, initial_capital=CAPITAL, risk_settings=risk)
            tr    = r.total_return * 100  if r.total_return  else 0
            cagr  = r.cagr * 100          if r.cagr          else 0
            sharpe= r.sharpe_ratio        if r.sharpe_ratio   else 0
            wr    = r.win_rate * 100      if r.win_rate       else 0
            pf    = r.profit_factor       if r.profit_factor  else 0
            mdd   = r.max_drawdown * 100  if r.max_drawdown   else 0
            nt    = len(r.trades)

            print(f"{etf:<6} {strat:<18} {tr:>+7.1f}% {cagr:>+6.1f}% {sharpe:>7.2f} {wr:>7.1f}% {pf:>5.2f} {mdd:>+7.1f}% {nt:>7}")
            rows.append({
                "symbol": etf, "strategy": strat,
                "total_return_pct": round(tr, 2),
                "cagr_pct": round(cagr, 2),
                "sharpe": round(sharpe, 3),
                "win_rate_pct": round(wr, 1),
                "profit_factor": round(pf, 3),
                "max_drawdown_pct": round(mdd, 2),
                "total_trades": nt,
                "final_capital": round(r.final_capital, 2),
            })
        except Exception as e:
            print(f"{etf:<6} {strat:<18} ERROR: {e}")

# Save CSV
out_path = Path(__file__).parent / "backtest_results.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"\nResults saved to: {out_path}")
print(f"\nConfig: capital=${CAPITAL:,}  stop_loss={STOP_LOSS*100:.0f}%  take_profit={TAKE_PROFIT*100:.0f}%  position_size={MAX_POS_SIZE*100:.0f}%  period=2y")
