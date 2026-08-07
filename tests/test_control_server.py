from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from vrchat_ai_tool.control_server import CONTROL_HTML, load_or_create_token, make_handler
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

    def snapshot(self) -> dict[str, object]:
        return {
            "status": 0,
            "muted": self.muted,
            "loop": {"enabled": self.loop_enabled, "running": self.loop_enabled, "triggered": False},
            "motion": {
                "enabled": self.motion_enabled,
                "activity_name": "IDLE",
                "input_rms": 0.0,
                "energy": 0.0,
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


class ControlServerTests(unittest.TestCase):
    def test_control_page_persists_token_in_browser(self) -> None:
        self.assertIn("localStorage.setItem('voiceAgentToken'", CONTROL_HTML)
        self.assertIn("sessionStorage.getItem('voiceAgentToken'", CONTROL_HTML)
        self.assertIn("forgetToken()", CONTROL_HTML)
        self.assertIn("/api/loop/enabled", CONTROL_HTML)
        self.assertIn("/api/motion/enabled", CONTROL_HTML)
        self.assertIn("アバター自動モーション", CONTROL_HTML)
        self.assertIn("監視を無効化", CONTROL_HTML)

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
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
