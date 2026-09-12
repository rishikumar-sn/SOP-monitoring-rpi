from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import integrated_ui_app as app_module


class IntegratedUiClockTests(unittest.TestCase):
    def tearDown(self):
        with app_module.APPLICATION_CLOCK_LOCK:
            app_module.APPLICATION_CLOCK_SYNC = None

    def test_wifi_connected_requires_up_wireless_carrier(self):
        with tempfile.TemporaryDirectory() as directory:
            interface = Path(directory) / "wlan0"
            (interface / "wireless").mkdir(parents=True)
            (interface / "operstate").write_text("up\n", encoding="utf-8")
            (interface / "carrier").write_text("1\n", encoding="utf-8")
            self.assertTrue(app_module.wifi_connected(Path(directory)))

            (interface / "carrier").write_text("0\n", encoding="utf-8")
            self.assertFalse(app_module.wifi_connected(Path(directory)))

    @patch("integrated_ui_app.time.monotonic", return_value=1000.0)
    @patch("integrated_ui_app.wifi_connected", return_value=False)
    def test_browser_time_is_used_when_wifi_is_disconnected(self, _wifi, _monotonic):
        epoch_ms = int(
            datetime(2026, 9, 9, 6, 30, tzinfo=timezone.utc).timestamp() * 1000
        )
        response = app_module.app.test_client().post(
            "/api/time/sync",
            json={"epoch_ms": epoch_ms, "utc_offset_minutes": 330},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["source"], "ethernet_browser_time")
        self.assertEqual(payload["current_time"], "2026-09-09 12:00:00")

    @patch("integrated_ui_app.wifi_connected", return_value=True)
    def test_wifi_time_clears_browser_fallback(self, _wifi):
        app_module.APPLICATION_CLOCK_SYNC = {
            "epoch_seconds": 1.0,
            "utc_offset_minutes": 0,
            "monotonic_seconds": 1.0,
            "client_ip": "127.0.0.1",
        }
        response = app_module.app.test_client().post(
            "/api/time/sync",
            json={"epoch_ms": 1788935400000, "utc_offset_minutes": 330},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["source"], "raspberry_pi_wifi_time")
        self.assertIsNone(app_module.APPLICATION_CLOCK_SYNC)

    def test_purity_manager_uses_application_time_provider(self):
        expected = datetime(2026, 9, 9, 12, 30)
        manager = app_module.PurityTestManager(
            base_dir=Path("."),
            frame_getter=lambda: None,
            now_fn=lambda: expected,
        )
        self.assertEqual(manager._stamp(), "2026-09-09 12:30:00")


if __name__ == "__main__":
    unittest.main()
