import pandas as pd
import numpy as np

df = pd.read_csv("backtest_results.csv")
df["entry_date"] = pd.to_datetime(df["entry_date"])
df["exit_date"]  = pd.to_datetime(df["exit_date"])

wins   = df[df["pnl"] > 0]
losses = df[df["pnl"] < 0]

gross_profit = wins["pnl"].sum()
gross_loss   = abs(losses["pnl"].sum())
net_pnl      = df["pnl"].sum()
win_rate     = len(wins) / len(df) * 100
profit_factor = gross_profit / gross_loss
avg_win      = wins["pnl"].mean()
avg_loss     = abs(losses["pnl"].mean())
rr_ratio     = avg_win / avg_loss

# Max drawdown via equity curve (sorted by exit_date)
df_sorted = df.sort_values("exit_date").copy()
df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum()
equity = 100_000 + df_sorted["cum_pnl"]
peak   = equity.cummax()
dd     = (equity - peak) / peak * 100
max_dd = dd.min()

# By hold days buckets
df["category"] = pd.cut(df["hold_days"], bins=[0,14,45,200],
                        labels=["Short (<15d)", "Swing (15-45d)", "Position (>45d)"])

print("=" * 60)
print("  ANÁLISIS COMPLETO DE TRADES — BACKTEST (Ene 2025 – Abr 2026)")
print("=" * 60)

print(f"\n📊 RESUMEN GENERAL")
print(f"  Total trades      : {len(df)}")
print(f"  Wins / Losses     : {len(wins)} / {len(losses)}")
print(f"  Win Rate          : {win_rate:.1f}%")
print(f"  Profit Factor     : {profit_factor:.2f}x")
print(f"  Net PnL           : ${net_pnl:,.2f}")
print(f"  Retorno s/ $100k  : {net_pnl/100_000*100:.1f}%")
print(f"  Max Drawdown      : {max_dd:.2f}%")

print(f"\n💰 WINS vs LOSSES")
print(f"  Avg Win           : ${avg_win:,.2f}")
print(f"  Avg Loss          : ${avg_loss:,.2f}")
print(f"  R:R Ratio         : {rr_ratio:.2f}")
print(f"  Gross Profit      : ${gross_profit:,.2f}")
print(f"  Gross Loss        : ${gross_loss:,.2f}")

print(f"\n📈 RESULTADO POR ETF")
by_ticker = df.groupby("ticker").agg(
    trades=("pnl","count"),
    wins=("pnl", lambda x: (x>0).sum()),
    net_pnl=("pnl","sum"),
    avg_pnl=("pnl","mean"),
    avg_hold=("hold_days","mean")
).sort_values("net_pnl", ascending=False)
by_ticker["win_rate"] = (by_ticker["wins"] / by_ticker["trades"] * 100).round(1)
for t, r in by_ticker.iterrows():
    bar = "🟢" if r.net_pnl > 0 else "🔴"
    print(f"  {bar} {t:<5} | ${r.net_pnl:>8,.0f} | WR {r.win_rate:.0f}% ({r.wins:.0f}/{r.trades:.0f}) | avg hold {r.avg_hold:.0f}d")

print(f"\n⏱  RESULTADO POR TIMEFRAME")
by_cat = df.groupby("category", observed=True).agg(
    trades=("pnl","count"),
    wins=("pnl", lambda x: (x>0).sum()),
    net_pnl=("pnl","sum")
)
by_cat["win_rate"] = (by_cat["wins"] / by_cat["trades"] * 100).round(1)
for c, r in by_cat.iterrows():
    print(f"  {c:<20} | {r.trades:.0f} trades | WR {r.win_rate:.0f}% | Net ${r.net_pnl:,.0f}")

print(f"\n🚪 RESULTADO POR TIPO DE SALIDA")
by_exit = df.groupby("exit_reason").agg(
    trades=("pnl","count"),
    wins=("pnl", lambda x: (x>0).sum()),
    net_pnl=("pnl","sum"),
    avg_pnl=("pnl","mean")
).sort_values("net_pnl", ascending=False)
by_exit["win_rate"] = (by_exit["wins"]/by_exit["trades"]*100).round(1)
for e, r in by_exit.iterrows():
    print(f"  {e:<20} | {r.trades:.0f} trades | WR {r.win_rate:.0f}% | avg ${r.avg_pnl:,.0f} | total ${r.net_pnl:,.0f}")

print(f"\n📅 RESULTADO POR AÑO/TRIMESTRE")
df["quarter"] = df["exit_date"].dt.to_period("Q")
by_q = df.groupby("quarter")["pnl"].agg(["sum","count"])
for q, r in by_q.iterrows():
    bar = "+" if r["sum"] >= 0 else "-"
    print(f"  {q}  {bar}${abs(r['sum']):>8,.0f}  ({r['count']:.0f} trades)")

print(f"\n⚠️  PEORES TRADES")
worst = df.nsmallest(5,"pnl")[["ticker","entry_date","exit_date","pnl","pnl_pct","exit_reason","hold_days"]]
for _, r in worst.iterrows():
    print(f"  {r.ticker} | ${r.pnl:,.2f} ({r.pnl_pct:.1f}%) | {r.entry_date.date()} → {r.exit_date.date()} | {r.exit_reason}")

print(f"\n🏆 MEJORES TRADES")
best = df.nlargest(5,"pnl")[["ticker","entry_date","exit_date","pnl","pnl_pct","exit_reason","hold_days"]]
for _, r in best.iterrows():
    print(f"  {r.ticker} | ${r.pnl:,.2f} ({r.pnl_pct:.1f}%) | {r.entry_date.date()} → {r.exit_date.date()} | {r.exit_reason}")
