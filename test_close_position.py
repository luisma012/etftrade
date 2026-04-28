"""Unit tests for close_position() and the AUTO-SELL path that uses it.

Run from the project root:
    python test_close_position.py

No live broker calls — broker.sell, broker.get_order_status,
broker.get_positions, telegram, and cooldown/trailing-state file IO are all
mocked or redirected to a temp directory.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Make sure we import the project's scanner module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scanner  # noqa: E402


class CloseFnTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.trail_file    = os.path.join(self.tmp.name, "trailing_state.json")
        self.cooldown_file = os.path.join(self.tmp.name, "cooldown.json")
        self._orig_trail    = scanner._TRAILING_STATE_FILE
        self._orig_cooldown = scanner._COOLDOWN_FILE if hasattr(scanner, "_COOLDOWN_FILE") else None
        scanner._TRAILING_STATE_FILE = self.trail_file
        # Also patch the cooldown filename if it exists as a module constant
        if self._orig_cooldown is not None:
            scanner._COOLDOWN_FILE = self.cooldown_file

        # Seed empty state files
        with open(self.trail_file, "w") as f:
            f.write("{}")

    def tearDown(self):
        scanner._TRAILING_STATE_FILE = self._orig_trail
        if self._orig_cooldown is not None:
            scanner._COOLDOWN_FILE = self._orig_cooldown
        self.tmp.cleanup()

    def write_trail(self, state):
        with open(self.trail_file, "w") as f:
            json.dump(state, f)

    def read_trail(self):
        if not os.path.exists(self.trail_file):
            return {}
        with open(self.trail_file) as f:
            return json.load(f)


class CloseSuccessTests(CloseFnTestBase):
    """Success path: cleans state, writes cooldown, sends Telegram, returns success."""

    def test_success_path_cleans_state(self):
        self.write_trail({
            "SPY": {"high_price": 720.0, "trailing_active": True},
            "VTI": {"high_price": 360.0, "trailing_active": False},
        })

        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "ord-123"}
        broker_mock.get_order_status.return_value = {
            "status":           "filled",
            "filled_avg_price": 715.50,
            "filled_qty":       1.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown") as cd:
            result = scanner.close_position("SPY", reason="manual_sell")

        # Result contract
        self.assertTrue(result["success"])
        self.assertEqual(result["fill_price"], 715.50)
        self.assertEqual(result["fill_qty"], 1.0)
        self.assertIsNone(result["error_msg"])

        # Broker was called correctly
        broker_mock.sell.assert_called_once_with("SPY", None)
        broker_mock.get_order_status.assert_called_with("ord-123")

        # State cleanup happened on disk
        trail = self.read_trail()
        self.assertNotIn("SPY", trail, "SPY should have been purged from trailing_state")
        self.assertIn("VTI", trail, "Other tickers must be untouched")

        # Cooldown was written
        cd.assert_called_once_with("SPY", stop_loss=False)

        # Telegram called once with success message containing the reason
        tg.assert_called_once()
        sent = tg.call_args[0][0]
        self.assertIn("SPY", sent)
        self.assertIn("manual_sell", sent)
        self.assertIn("$715.50", sent)

    def test_success_with_qty_and_extra_msg(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "x"}
        broker_mock.get_order_status.return_value = {
            "filled_avg_price": 100.0, "filled_qty": 5.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position(
                "QQQ", reason="auto_sell_signal",
                qty=5.0, extra_msg="Score: 32.1/100",
            )

        self.assertTrue(result["success"])
        broker_mock.sell.assert_called_once_with("QQQ", 5.0)
        sent = tg.call_args[0][0]
        self.assertIn("auto_sell_signal", sent)
        self.assertIn("Score: 32.1/100", sent)

    def test_stop_loss_passes_through_to_cooldown(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "x"}
        broker_mock.get_order_status.return_value = {
            "filled_avg_price": 50.0, "filled_qty": 10.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send"), \
             patch.object(scanner, "write_cooldown") as cd:
            scanner.close_position("XLE", reason="trailing_stop", stop_loss=True)

        cd.assert_called_once_with("XLE", stop_loss=True)

    def test_fill_polling_times_out_returns_success_with_none_fill(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "slow"}
        # Always return no fill info (simulate polling timeout)
        broker_mock.get_order_status.return_value = {"filled_avg_price": None, "filled_qty": 0}

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send"), \
             patch.object(scanner, "write_cooldown"), \
             patch("scanner.time.sleep"):  # skip the 5s wait
            result = scanner.close_position("SPY", reason="manual_sell")

        self.assertTrue(result["success"], "broker.sell ok ⇒ success even if fill not confirmed")
        self.assertIsNone(result["fill_price"])
        self.assertIsNone(result["fill_qty"])


class CloseFailureTests(CloseFnTestBase):
    """Failure path: leaves state intact, sends error Telegram."""

    def test_broker_error_keeps_state(self):
        seeded = {"SPY": {"high_price": 720.0, "trailing_active": True}}
        self.write_trail(seeded)

        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "error", "message": "insufficient funds"}

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown") as cd:
            result = scanner.close_position("SPY", reason="manual_sell")

        # Result contract
        self.assertFalse(result["success"])
        self.assertEqual(result["error_msg"], "insufficient funds")
        self.assertIsNone(result["fill_price"])
        self.assertIsNone(result["fill_qty"])

        # State must be untouched
        self.assertEqual(self.read_trail(), seeded,
                         "trailing_state must NOT be cleaned when broker.sell fails")

        # Cooldown must NOT be written
        cd.assert_not_called()

        # Telegram alert was still sent (specs: obligatorio en éxito y error)
        tg.assert_called_once()
        sent = tg.call_args[0][0]
        self.assertIn("FALL", sent.upper())  # "FALLÓ"
        self.assertIn("insufficient funds", sent)

    def test_broker_error_without_message_uses_fallback(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "error"}

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send"), \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position("SPY", reason="manual_sell")

        self.assertFalse(result["success"])
        self.assertEqual(result["error_msg"], "unknown broker error")


class CloseValidationTests(CloseFnTestBase):
    def test_invalid_reason_raises(self):
        with self.assertRaises(ValueError) as ctx:
            scanner.close_position("SPY", reason="random_string")
        self.assertIn("invalid reason", str(ctx.exception).lower())

    def test_all_valid_reasons_accepted(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "x"}
        broker_mock.get_order_status.return_value = {"filled_avg_price": 1.0, "filled_qty": 1.0}

        for reason in scanner.VALID_CLOSE_REASONS:
            with patch.dict("sys.modules", {"broker": broker_mock}), \
                 patch.object(scanner, "_tg_send"), \
                 patch.object(scanner, "write_cooldown"):
                # Should not raise
                result = scanner.close_position("SPY", reason=reason)
                self.assertTrue(result["success"], f"reason={reason}")


class AutoSellPathTest(CloseFnTestBase):
    """The AUTO-SELL branch in run_scan() must route through close_position()."""

    def test_auto_sell_calls_close_position_with_correct_reason(self):
        # Build minimal scoring result that triggers AUTO-SELL
        score_results = [{
            "ticker":        "SPY",
            "score":         28.5,
            "signal":        "SELL",
            "current_price": 712.34,
            "technical": {}, "ml": {}, "sentiment": {}, "timestamp": "2026-04-28T00:00:00Z",
        }]

        broker_mock = MagicMock()
        broker_mock.get_positions.return_value = {
            "SPY": {"avg_buy_price": 700.0, "current_price": 712.34, "quantity": 1.0, "unrealized_pnl": 12.34}
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_circuit_breaker_ok", return_value=True), \
             patch.object(scanner, "_vix_ok", return_value=True), \
             patch.object(scanner, "_monitor_positions"), \
             patch.object(scanner, "_expire_pending"), \
             patch.object(scanner, "score_all_etfs", return_value=score_results), \
             patch.object(scanner, "AUTO_TRADE", True), \
             patch.object(scanner, "close_position") as cp:
            cp.return_value = {"success": True, "fill_price": 712.0, "fill_qty": 1.0, "error_msg": None}
            scanner.run_scan()

        cp.assert_called_once()
        args, kwargs = cp.call_args
        self.assertEqual(args[0], "SPY")
        self.assertEqual(kwargs.get("reason"), "auto_sell_signal")
        self.assertIn("28.5", kwargs.get("extra_msg", ""))
        self.assertIn("712.34", kwargs.get("extra_msg", ""))


class RealizedPnlTests(CloseFnTestBase):
    """realized_pnl = (fill_price - avg_entry_price) * fill_qty."""

    def test_realized_pnl_in_result_and_telegram(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "ord-pnl"}
        broker_mock.get_order_status.return_value = {
            "filled_avg_price": 115.0,
            "filled_qty":       5.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position(
                "QQQ",
                reason="manual_sell",
                avg_entry_price=100.0,
            )

        # (115.0 - 100.0) * 5.0 = 75.0
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["realized_pnl"], 75.0)

        # Telegram must mention the realized amount
        sent = tg.call_args[0][0]
        self.assertIn("$+75.00", sent)

    def test_realized_pnl_negative(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "ord-loss"}
        broker_mock.get_order_status.return_value = {
            "filled_avg_price": 90.0,
            "filled_qty":       10.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position(
                "XLE",
                reason="trailing_stop",
                stop_loss=True,
                avg_entry_price=100.0,
            )

        # (90.0 - 100.0) * 10.0 = -100.0
        self.assertAlmostEqual(result["realized_pnl"], -100.0)
        sent = tg.call_args[0][0]
        self.assertIn("$-100.00", sent)

    def test_no_avg_entry_price_gives_none(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "ok", "order_id": "ord-nopnl"}
        broker_mock.get_order_status.return_value = {
            "filled_avg_price": 200.0,
            "filled_qty":       3.0,
        }

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send") as tg, \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position("GLD", reason="manual_sell")

        self.assertIsNone(result["realized_pnl"])
        sent = tg.call_args[0][0]
        self.assertNotIn("Realizado", sent)

    def test_failure_result_has_realized_pnl_none(self):
        broker_mock = MagicMock()
        broker_mock.sell.return_value = {"status": "error", "message": "no position"}

        with patch.dict("sys.modules", {"broker": broker_mock}), \
             patch.object(scanner, "_tg_send"), \
             patch.object(scanner, "write_cooldown"):
            result = scanner.close_position("SPY", reason="manual_sell", avg_entry_price=500.0)

        self.assertFalse(result["success"])
        self.assertIsNone(result["realized_pnl"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
