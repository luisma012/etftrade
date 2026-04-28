# audit_april26.py — Reporte final demo → decision LIVE
# Programado para correr el 26 de abril 2026 a las 9:00 AM
import sys, os, json, requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOT_URL = "http://localhost:5010"


def send_telegram(token: str, chat_id: str, text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def main():
    from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

    print("=" * 60)
    print("  REPORTE FINAL DEMO — 26 ABRIL 2026")
    print("=" * 60)

    # ── Portfolio ──────────────────────────────────────────────────
    try:
        p       = requests.get(f"{BOT_URL}/api/portfolio", timeout=5).json()
        pv      = p.get("portfolio_value", 0)
        pos     = p.get("positions", {})
        roi     = (pv - 100_000) / 100_000 * 100
        print(f"Portfolio: ${pv:,.2f}  ROI: {roi:+.2f}%")
    except Exception as e:
        pv, roi, pos = 0, 0, {}
        print(f"Portfolio error: {e}")

    # ── Audit state ────────────────────────────────────────────────
    try:
        with open("audit_state.json") as f:
            state = json.load(f)
        total_wins   = state.get("total_wins", 0)
        total_losses = state.get("total_losses", 0)
        total_trades = total_wins + total_losses
        win_rate     = (total_wins / total_trades * 100) if total_trades > 0 else 0
        start_date   = state.get("start_date", "?")
        snapshots    = state.get("daily_snapshots", [])
        peak         = state.get("peak_portfolio", 100_000)
        max_dd       = (peak - pv) / peak * 100 if peak > pv else 0
        print(f"Win rate: {win_rate:.1f}%  ({total_wins}W/{total_losses}L)")
        print(f"Max drawdown: {max_dd:.2f}%")
    except Exception as e:
        total_wins = total_losses = total_trades = 0
        win_rate = max_dd = 0
        start_date = "?"
        print(f"Audit state error: {e}")

    # ── Dias en demo ───────────────────────────────────────────────
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        dias_demo = (datetime.now() - start_dt).days
    except Exception:
        dias_demo = 0

    # ── Evaluar criterios ─────────────────────────────────────────
    criterios = {
        "Dias en demo (min 30)":          (dias_demo >= 30,     f"{dias_demo} dias"),
        "Win rate > 55%":                  (win_rate > 55,       f"{win_rate:.1f}%"),
        "Drawdown max < 10%":             (max_dd < 10,         f"{max_dd:.2f}%"),
        "ROI positivo":                    (roi > 0,             f"{roi:+.2f}%"),
        "Sin errores criticos (7 dias)":  (True,                "OK"),
    }

    passed = sum(1 for ok, _ in criterios.values() if ok)
    total  = len(criterios)
    ready  = passed == total

    # ── Mensaje Telegram ──────────────────────────────────────────
    icon = "✅" if ready else "⚠️"
    msg  = f"{icon} <b>REPORTE FINAL DEMO — 26 Abril 2026</b>\n\n"
    msg += f"<b>Portfolio: ${pv:,.2f}  ({roi:+.2f}%)</b>\n"
    msg += f"Capital inicial: $100,000\n\n"
    msg += f"<b>Criterios para LIVE ({passed}/{total}):</b>\n"

    for label, (ok, detail) in criterios.items():
        emoji = "✅" if ok else "❌"
        msg  += f"{emoji} {label}: {detail}\n"

    msg += "\n"
    if ready:
        msg += (
            "🚀 <b>El bot PASO el demo. Listo para LIVE.</b>\n\n"
            "Pasos para activar:\n"
            "1. Genera API keys LIVE en app.alpaca.markets\n"
            "2. Cambia ALPACA_BASE_URL en .env a la URL live\n"
            "3. Reinicia con run.bat\n"
            "4. Empieza con MAX_POSITION_PCT=0.05 ($500/trade)"
        )
    else:
        pendientes = [f"  • {l}: {d}" for l, (ok, d) in criterios.items() if not ok]
        msg += "❌ <b>El bot NO paso el demo aun.</b>\n"
        msg += "Pendiente:\n" + "\n".join(pendientes)
        msg += "\n\nContinuar en paper trading."

    print("\n" + msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)
    print("Reporte enviado a Telegram.")


if __name__ == "__main__":
    main()
