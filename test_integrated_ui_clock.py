from __future__ import annotations

import tempfile
import subprocess
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

    def test_pdf_keeps_previous_acid_details_and_prints_temporal_result(self):
        for acid_result in ("22K gold", "Non-gold"):
            with self.subTest(acid_result=acid_result), tempfile.TemporaryDirectory() as directory:
                state = app_module.build_empty_state()
                state["session_id"] = "acid-report-test"
                state["pledge_id"] = "acid-report-test"
                state["jewel_index"] = 1
                state["classification"].update(
                    {"confirmed": True, "confirmed_label": "Ring"}
                )
                state["purity_test"] = {
                    "started_at": "2026-09-15 10:00:00",
                    "stopped_at": "2026-09-15 10:00:10",
                    "completed_at": "2026-09-15 10:00:09",
                    "stage": "COMPLETED",
                    "status": f"Acid test completed. {acid_result}.",
                    "result": acid_result,
                    "acid_result": acid_result,
                    "rubbing_ok": True,
                    "acid_ok": True,
                    "running": False,
                }
                evidence = app_module.np.full((80, 120, 3), 180, dtype=app_module.np.uint8)
                for artifact_key in (
                    "rubbing_image",
                    "rubbing_zoom_image",
                    "acid_success_image",
                    "acid_zoom_image",
                ):
                    image_path = Path(directory) / f"{artifact_key}.jpg"
                    self.assertTrue(app_module.cv2.imwrite(str(image_path), evidence))
                    state["purity_test"][artifact_key] = {
                        "name": image_path.name,
                        "path": str(image_path),
                    }
                pdf_path = Path(directory) / "report.pdf"
                pdf_path.write_bytes(
                    app_module.generate_pdf_report(
                        [state],
                        {"pledge_id": "acid-report-test", "jewel_count": 1},
                    ).getvalue()
                )

                extracted = subprocess.run(
                    ["pdftotext", str(pdf_path), "-"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout

                self.assertIn(f"Acid Result: {acid_result}", extracted)
                self.assertIn("Stage: COMPLETED", extracted)
                self.assertIn("Rubbing OK: Yes", extracted)
                self.assertIn("Acid OK: Yes", extracted)
                embedded_images = subprocess.run(
                    ["pdfimages", "-list", str(pdf_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                image_rows = [
                    line for line in embedded_images.splitlines()
                    if line.split() and line.split()[0].isdigit()
                ]
                self.assertGreaterEqual(len(image_rows), 4)


if __name__ == "__main__":
    unittest.main()
