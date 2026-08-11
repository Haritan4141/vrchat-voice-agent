from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from vrchat_ai_tool.chatgpt_ui_state import UiActivityState
from vrchat_ai_tool.control_server import (
    CONTROL_HTML,
    VoiceControlService,
    load_or_create_token,
    make_handler,
    should_show_thinking,
)
from vrchat_ai_tool.voice_config import (
    ChatGPTVoiceConfig,
    VoiceAudioConfig,
    VoiceControlConfig,
    VoiceLoopGuardConfig,
    VoiceOscConfig,
    VoiceParsecConfig,
    VoiceProcessConfig,
)


class FakeService:
    def __init__(self) -> None:
        self.muted = False
        self.loop_enabled = True
        self.motion_enabled = True
        self.ui_monitor_enabled = True
        self.thinking_test_enabled = False
        self.diagnostic_calls: list[tuple[str, int | None]] = []

    def snapshot(self) -> dict[str, object]:
        return {
            "status": 0,
            "avatar": {
                "status": 0,
                "status_target": 0,
                "motion_enabled": True,
                "motion_enabled_target": True,
                "activity": 0,
                "activity_target": 0,
                "energy": 0.0,
                "energy_target": 0.0,
                "gesture": 0,
                "gesture_target": 0,
                "expression": 0,
                "expression_target": 0,
                "thinking": False,
                "thinking_target": False,
            },
            "muted": self.muted,
            "loop": {"enabled": self.loop_enabled, "running": self.loop_enabled, "triggered": False},
            "motion": {
                "enabled": self.motion_enabled,
                "activity_name": "IDLE",
                "input_rms": 0.0,
                "energy": 0.0,
            },
            "ui_monitor": {
                "enabled": self.ui_monitor_enabled,
                "running": self.ui_monitor_enabled,
                "available": True,
                "state": "idle",
                "thinking": False,
                "searching": False,
                "element_count": 240,
                "last_error": "",
                "test_override": self.thinking_test_enabled,
            },
            "last_error": "",
        }

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        self.muted = False

    def set_status(self, _value: int) -> None:
        return

    def reset_loop(self) -> None:
        return

    def set_loop_guard_enabled(self, enabled: bool) -> None:
        self.loop_enabled = enabled

    def set_motion_enabled(self, enabled: bool) -> None:
        self.motion_enabled = enabled

    def set_ui_monitor_enabled(self, enabled: bool) -> None:
        self.ui_monitor_enabled = enabled

    def set_thinking_test(self, enabled: bool) -> None:
        self.thinking_test_enabled = enabled

    def start_motion_diagnostic_test(self) -> None:
        self.diagnostic_calls.append(("test", None))

    def stop_motion_diagnostic(self) -> None:
        self.diagnostic_calls.append(("stop", None))

    def set_motion_diagnostic_activity(self, value: int) -> None:
        self.diagnostic_calls.append(("activity", value))

    def play_motion_diagnostic_gesture(self, value: int) -> None:
        self.diagnostic_calls.append(("gesture", value))

    def set_motion_diagnostic_expression(self, value: int) -> None:
        self.diagnostic_calls.append(("expression", value))


class ControlServerTests(unittest.TestCase):
    def test_thinking_display_test_pulses_off_before_on(self) -> None:
        class RecordingOsc:
            def __init__(self) -> None:
                self.values: list[bool] = []

            def send_thinking(self, value: bool) -> None:
                self.values.append(value)

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._thinking_test_override = False
        service._thinking_output = False
        service.osc = RecordingOsc()

        with patch("vrchat_ai_tool.control_server.time.sleep") as sleep:
            service.set_thinking_test(True)

        self.assertEqual(service.osc.values, [False, True])
        self.assertTrue(service._thinking_test_override)
        self.assertTrue(service._thinking_output)
        sleep.assert_called_once_with(0.15)

    def test_thinking_is_hidden_only_while_voice_is_speaking(self) -> None:
        self.assertFalse(should_show_thinking(UiActivityState.IDLE, 0))
        self.assertTrue(should_show_thinking(UiActivityState.WORKING, 0))
        self.assertFalse(should_show_thinking(UiActivityState.SEARCHING, 1))
        self.assertTrue(should_show_thinking(UiActivityState.SEARCHING, 2))

    def test_control_page_persists_token_in_browser(self) -> None:
        self.assertIn("localStorage.setItem('voiceAgentToken'", CONTROL_HTML)
        self.assertIn("sessionStorage.getItem('voiceAgentToken'", CONTROL_HTML)
        self.assertIn("forgetToken()", CONTROL_HTML)
        self.assertIn("/api/loop/enabled", CONTROL_HTML)
        self.assertIn("/api/motion/enabled", CONTROL_HTML)
        self.assertIn("/api/ui-monitor/enabled", CONTROL_HTML)
        self.assertIn("/api/thinking/test", CONTROL_HTML)
        self.assertIn("/api/motion/test", CONTROL_HTML)
        self.assertIn("/api/motion/diagnostic/activity", CONTROL_HTML)
        self.assertIn("/api/motion/diagnostic/gesture", CONTROL_HTML)
        self.assertIn("/api/motion/diagnostic/expression", CONTROL_HTML)
        self.assertIn("status_target", CONTROL_HTML)
        self.assertIn("未反映→", CONTROL_HTML)
        self.assertIn("VRChat ${actualEnergy}", CONTROL_HTML)
        self.assertIn("全動作テスト（約49秒）", CONTROL_HTML)
        self.assertIn("アバター自動モーション", CONTROL_HTML)
        self.assertIn("監視を無効化", CONTROL_HTML)
        self.assertIn("ChatGPT画面状態監視", CONTROL_HTML)
        self.assertIn("thinking_target", CONTROL_HTML)
        self.assertIn("考え中表示テスト", CONTROL_HTML)

    def test_token_is_generated_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ChatGPTVoiceConfig(
                VoiceAudioConfig(), VoiceProcessConfig(), VoiceOscConfig(),
                VoiceControlConfig(), VoiceLoopGuardConfig(), VoiceParsecConfig(),
                Path(directory) / "voice.toml",
            )
            first, path, created = load_or_create_token(config)
            second, same_path, created_again = load_or_create_token(config)
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first, second)
            self.assertEqual(path, same_path)
            self.assertGreaterEqual(len(first), 32)

    def test_api_requires_token_and_can_mute(self) -> None:
        service = FakeService()
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(service, "x" * 32, ("127.0.0.1",))
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(base + "/api/status", timeout=2)
            self.assertEqual(context.exception.code, 401)

            request = urllib.request.Request(
                base + "/api/mic/mute",
                data=b"{}",
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertTrue(service.muted)

            request = urllib.request.Request(
                base + "/api/loop/enabled",
                data=json.dumps({"enabled": False}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertFalse(service.loop_enabled)

            request = urllib.request.Request(
                base + "/api/motion/enabled",
                data=json.dumps({"enabled": False}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertFalse(service.motion_enabled)

            request = urllib.request.Request(
                base + "/api/ui-monitor/enabled",
                data=json.dumps({"enabled": False}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertFalse(service.ui_monitor_enabled)

            request = urllib.request.Request(
                base + "/api/thinking/test",
                data=json.dumps({"enabled": True}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertTrue(service.thinking_test_enabled)

            request = urllib.request.Request(
                base + "/api/motion/diagnostic/gesture",
                data=json.dumps({"value": 6}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertIn(("gesture", 6), service.diagnostic_calls)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
