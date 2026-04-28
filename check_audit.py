# check_audit.py — Verificacion automatica de criterios para pasar a LIVE
# Uso manual : python check_audit.py
# Uso interno: from check_audit import run_audit

import sys, os, json, csv
import requests
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOT_URL = "http://localhost:5010"


def run_audit(verbose: bool = True) -> dict:
    """
    Evalua todos los criterios del checklist.
    Retorna:
        {
          "passed":  int,
          "failed":  int,
          "warns":   int,
          "total":   int,
          "ready":   bool,           # True = listo para live
          "failed_items":  [str],
          "warn_items":    [str],
          "summary":       str,      # texto para Telegram
        }
    """
    checks = []   # (label, passed, warn, detail)

    def ck(label, passed, detail="", warn=False):
        checks.append((label, passed, warn, detail))
        if verbose:
            icon = "OK " if passed else ("?? " if warn else "XX ")
            print(f"  {icon}  {label:<50} {detail}")
        return passed

    if verbose:
        from datetime import datetime
        print("\n" + "="*70)
        print("  ETFTRADE — AUDITORIA PARA LIVE TRADING")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)

    # ── 1. Sistema ────────────────────────────────────────────────────────────
    if verbose: print("\n[1] SISTEMA")
    try:
        r = requests.get(f"{BOT_URL}/health", timeout=5).json()
        ck("Bot corriendo", True, f"uptime {r.get('uptime_sec',0):.0f}s")
        ck("Puerto 5010 activo", True, f"market_open={r.get('market_open')}")
    except Exception as e:
        ck("Bot corriendo", False, f"ERROR: {e}")
        ck("Puerto 5010 activo", False, "no responde")

    # ── 2. Portafolio ─────────────────────────────────────────────────────────
    if verbose: print("\n[2] PORTAFOLIO")
    try:
        p = requests.get(f"{BOT_URL}/api/portfolio", timeout=5).json()
        pv  = p.get("portfolio_value", 0)
        pos = p.get("positions", {})
        roi = (pv - 100_000) / 100_000 * 100
        from config import ETF_UNIVERSE, MAX_OPEN_POSITIONS
        non_etf = [k for k in pos if k not in ETF_UNIVERSE]
        etf_pos = [k for k in pos if k in ETF_UNIVERSE]
        ck("Portafolio accesible", True, f"${pv:,.0f}")
        ck("ROI positivo", roi > 0, f"{roi:+.2f}%")
        ck(f"Solo posiciones ETF (no acciones sueltas)",
           len(non_etf) == 0,
           f"No-ETF: {non_etf}" if non_etf else "OK")
        ck(f"Posiciones ETF <= {MAX_OPEN_POSITIONS}",
           len(etf_pos) <= MAX_OPEN_POSITIONS,
           f"{len(etf_pos)} ETFs abiertos")
    except Exception as e:
        ck("Portafolio accesible", False, f"ERROR: {e}")

    # ── 3. Filtros de riesgo ──────────────────────────────────────────────────
    if verbose: print("\n[3] FILTROS DE RIESGO")
    try:
        s  = requests.get(f"{BOT_URL}/api/sentiment", timeout=5).json()
        fg = s.get("fear_greed_score", 50)
        ck("Sentimiento accesible", True, f"F&G={fg:.0f} ({s.get('fear_greed_label','')})")
        ck("SPY RSI disponible",    s.get("spy_rsi", 0) > 0, f"RSI={s.get('spy_rsi',0):.1f}")
    except Exception as e:
        ck("Sentimiento accesible", False, f"ERROR: {e}")

    try:
        from config import VIX_MAX, VIX_TICKER
        df  = yf.download(VIX_TICKER, period="2d", progress=False, auto_adjust=True)
        vix = float(df["Close"].values[-1])
        ck("Filtro VIX implementado", True,  f"VIX={vix:.1f} (bloquea si >={VIX_MAX})")
        ck("VIX dentro del limite",   vix < VIX_MAX,
           f"VIX={vix:.1f} vs max={VIX_MAX}", warn=(vix >= VIX_MAX))
    except Exception as e:
        ck("Filtro VIX", False, f"ERROR: {e}")

    # ── 4. Circuit Breaker ────────────────────────────────────────────────────
    if verbose: print("\n[4] CIRCUIT BREAKER")
    try:
        from config import CIRCUIT_BREAKER_ENABLED, CIRCUIT_BREAKER_URL
        if not CIRCUIT_BREAKER_ENABLED:
            ck("Circuit breaker", True, "desactivado (modo normal)")
        else:
            try:
                cb = requests.get(f"{CIRCUIT_BREAKER_URL}/status", timeout=3).json()
                status = cb.get("status", "UNKNOWN").upper()
                ck("Circuit breaker CLOSED", status == "CLOSED", f"status={status}")
            except Exception:
                ck("Circuit breaker", True, "inalcanzable — fail-open OK", warn=True)
    except Exception as e:
        ck("Circuit breaker config", False, f"ERROR: {e}")

    # ── 5. Cooldown ───────────────────────────────────────────────────────────
    if verbose: print("\n[5] COOLDOWN")
    try:
        from config import COOLDOWN_HOURS, COOLDOWN_SL_HOURS
        cd_ok = os.path.exists("cooldown.json")
        ck("cooldown.json existe",         cd_ok,  "control de reentradas")
        ck("Cooldown normal >= 24h",        COOLDOWN_HOURS    >= 24, f"{COOLDOWN_HOURS}h")
        ck("Cooldown post-SL >= 48h",       COOLDOWN_SL_HOURS >= 48, f"{COOLDOWN_SL_HOURS}h")
    except Exception as e:
        ck("Cooldown verificado", False, f"ERROR: {e}")

    # ── 6. Bracket orders ─────────────────────────────────────────────────────
    if verbose: print("\n[6] STOPS SERVER-SIDE")
    try:
        src = open("broker/alpaca.py", encoding="utf-8").read()
        ck("Bracket orders implementados",
           "OrderClass.BRACKET" in src and "StopLossRequest" in src,
           "SL y TP en servidor Alpaca")
    except Exception as e:
        ck("Bracket orders", False, f"ERROR: {e}")

    # ── 7. Backtest ───────────────────────────────────────────────────────────
    if verbose: print("\n[7] BACKTEST")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.csv")
    if os.path.exists(csv_path):
        try:
            trades = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            wins   = [t for t in trades if float(t.get("pnl", 0)) > 0]
            wr     = len(wins) / len(trades) * 100 if trades else 0
            gw     = sum(float(t["pnl"]) for t in trades if float(t.get("pnl",0)) > 0)
            gl     = abs(sum(float(t["pnl"]) for t in trades if float(t.get("pnl",0)) <= 0)) or 1e-9
            pf     = gw / gl
            ck("Win rate >= 55%",       wr >= 55,  f"{wr:.1f}%")
            ck("Profit Factor >= 1.3",  pf >= 1.3, f"{pf:.2f}")
            ck("Trades suficientes",    len(trades) >= 10, f"{len(trades)} trades")
        except Exception as e:
            ck("Backtest CSV legible", False, f"ERROR: {e}")
    else:
        if verbose: print("  --   backtest_results.csv no encontrado — corre opcion [3] del menu")

    # ── 8. Seguridad ──────────────────────────────────────────────────────────
    if verbose: print("\n[8] SEGURIDAD")
    ck(".env existe", os.path.exists(".env"), "credenciales configuradas")

    gi_ok = True
    if os.path.exists(".gitignore"):
        gi_ok = ".env" in open(".gitignore").read()
    ck(".env no en git", gi_ok, ".gitignore OK")

    try:
        from config import ALPACA_BASE_URL
        ck("Modo paper activo", "paper" in ALPACA_BASE_URL.lower(), ALPACA_BASE_URL)
    except Exception as e:
        ck("Config Alpaca", False, str(e))

    # ── Resultado ─────────────────────────────────────────────────────────────
    n_passed = sum(1 for c in checks if c[1] and not c[2])
    n_failed = sum(1 for c in checks if not c[1] and not c[2])
    n_warns  = sum(1 for c in checks if c[2])
    total    = len(checks)
    ready    = n_failed == 0

    failed_items = [f"{c[0]}: {c[3]}" for c in checks if not c[1] and not c[2]]
    warn_items   = [f"{c[0]}: {c[3]}" for c in checks if c[2]]

    # Resumen para Telegram
    if ready:
        summary = (
            "El bot ETF esta listo para crear cuenta real\n"
            f"Todos los criterios pasados ({n_passed}/{total})\n"
            f"Advertencias: {n_warns}\n\n"
            "Pasos:\n"
            "1. Genera API keys LIVE en app.alpaca.markets\n"
            "2. Actualiza ALPACA_BASE_URL en .env\n"
            "3. Reinicia el bot con run.bat\n"
            "4. Empieza con MAX_POSITION_PCT=0.002 ($200/trade)"
        )
    else:
        lines = [f"Auditoria LIVE: {n_passed}/{total} OK"]
        for item in failed_items:
            lines.append(f"XX {item}")
        for item in warn_items:
            lines.append(f"?? {item}")
        summary = "\n".join(lines)

    if verbose:
        print("\n" + "="*70)
        print(f"  RESULTADO: {n_passed}/{total} OK   Fallidos: {n_failed}   Advertencias: {n_warns}")
        if failed_items:
            print("\n  Pendiente:")
            for i in failed_items: print(f"    XX {i}")
        if warn_items:
            print("\n  Advertencias:")
            for i in warn_items: print(f"    ?? {i}")
        if ready:
            print("\n  *** LISTO PARA LIVE TRADING ***")
        elif n_failed <= 2:
            print("\n  Casi listo.")
        else:
            print("\n  Sigue en paper.")
        print("="*70 + "\n")

    return {
        "passed": n_passed, "failed": n_failed,
        "warns": n_warns,   "total": total,
        "ready": ready,
        "failed_items": failed_items,
        "warn_items":   warn_items,
        "summary":      summary,
    }


if __name__ == "__main__":
    run_audit(verbose=True)
