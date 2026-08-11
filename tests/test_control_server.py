from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
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
        self.caption_mode = "off"
        self.caption_quality = "standard"
        self.caption_tests = 0
        self.preflight_started = False
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
                "probe": False,
                "probe_target": False,
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
            "captions": {
                "mode": self.caption_mode,
                "running": True,
                "speaking": False,
                "typing": False,
                "last_text": "",
                "last_source": "",
                "last_error": "",
                "send_count": self.caption_tests,
                "stt_state": "not_loaded",
                "stt_quality": self.caption_quality,
                "stt_model": "small" if self.caption_quality == "standard" else "medium",
                "stt_beam_size": 1 if self.caption_quality == "standard" else 5,
            },
            "osc": {"target": "127.0.0.1:9000", "listen": "127.0.0.1:9001"},
            "preflight": {
                "state": "ready" if self.preflight_started else "not_run",
                "message": "OSC同期済み・ONLINE" if self.preflight_started else "未実行",
                "osc_ok": self.preflight_started,
                "probe_ok": self.preflight_started,
                "baseline_ok": self.preflight_started,
                "probe_rtt_ms": 1.2 if self.preflight_started else None,
                "avatar_id": None,
                "avatar_generation": 0,
            },
            "last_error": "",
        }

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        self.muted = False

    def set_status(self, _value: int) -> None:
        return

    def preflight_and_start(self) -> dict[str, object]:
        self.preflight_started = True
        return {"state": "ready"}

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

    def set_caption_mode(self, mode: str) -> None:
        self.caption_mode = mode

    def set_caption_stt_quality(self, quality: str) -> None:
        self.caption_quality = quality

    def send_caption_test(self) -> None:
        self.caption_tests += 1

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
    def test_preflight_probes_resets_and_enters_online(self) -> None:
        class PreflightOsc:
            def __init__(self) -> None:
                self.statuses: list[int] = []
                self.thinking: list[bool] = []
                self.events: list[str] = []

            def reload_current_avatar(self) -> float:
                self.events.append("reload")
                return 825.0

            def set_status_confirmed(self, value: object) -> None:
                self.statuses.append(int(value))
                self.events.append(f"status:{int(value)}")

            def confirm_probe_roundtrip(self) -> float:
                self.events.append("probe")
                return 7.4

            def send_thinking(self, value: bool) -> None:
                self.thinking.append(value)

            @staticmethod
            def feedback_snapshot() -> dict[str, object]:
                return {
                    "osc_listener_running": True,
                    "probe": False,
                    "avatar_id": "avtr_test",
                    "avatar_generation": 2,
                }

        class PreflightMotion:
            def __init__(self) -> None:
                self.stopped = False
                self.enabled: bool | None = None

            def stop_diagnostic_test(self) -> None:
                self.stopped = True

            def set_enabled(self, enabled: bool) -> None:
                self.enabled = enabled

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._preflight_lock = threading.Lock()
        service._thinking_output_lock = threading.RLock()
        service._thinking_test_override = True
        service._thinking_output = True
        service._thinking_last_sent_at = 0.0
        service._preflight = {}
        service.last_error = "old error"
        service.osc = PreflightOsc()
        service.motion = PreflightMotion()
        service.config = SimpleNamespace(motion=SimpleNamespace(enabled=True))
        service.loop_guard = SimpleNamespace(
            detector=SimpleNamespace(last=SimpleNamespace(triggered=False))
        )

        result = service.preflight_and_start()

        self.assertEqual(service.osc.statuses, [3, 1])
        self.assertEqual(service.osc.events, ["reload", "status:3", "probe", "status:1"])
        self.assertEqual(service.osc.thinking, [False])
        self.assertTrue(service.motion.stopped)
        self.assertTrue(service.motion.enabled)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["probe_rtt_ms"], 7.4)
        self.assertTrue(result["avatar_reload_ok"])
        self.assertEqual(result["avatar_reload_ms"], 825.0)
        self.assertEqual(result["avatar_generation"], 2)
        self.assertFalse(service._thinking_test_override)

    def test_ready_preflight_becomes_stale_after_avatar_change(self) -> None:
        service = object.__new__(VoiceControlService)
        service._preflight = {
            "state": "ready",
            "message": "OSC同期済み・ONLINE",
            "probe_ok": True,
            "baseline_ok": True,
            "avatar_generation": 2,
        }

        result = service._preflight_snapshot({"avatar_generation": 3})

        self.assertEqual(result["state"], "stale")
        self.assertFalse(result["probe_ok"])
        self.assertFalse(result["baseline_ok"])

    def test_preflight_stops_before_online_when_avatar_reload_fails(self) -> None:
        class FailingOsc:
            def __init__(self) -> None:
                self.sent_statuses: list[int] = []

            @staticmethod
            def reload_current_avatar() -> float:
                raise RuntimeError("reload unavailable")

            def send_status(self, value: object) -> None:
                self.sent_statuses.append(int(value))

            @staticmethod
            def feedback_snapshot() -> dict[str, object]:
                return {
                    "osc_listener_running": True,
                    "avatar_id": None,
                    "avatar_generation": 0,
                }

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._preflight_lock = threading.Lock()
        service._preflight = {}
        service.last_error = ""
        service.osc = FailingOsc()
        service.loop_guard = SimpleNamespace(
            detector=SimpleNamespace(last=SimpleNamespace(triggered=False))
        )

        with self.assertRaisesRegex(RuntimeError, "reload unavailable"):
            service.preflight_and_start()

        self.assertEqual(service._preflight["state"], "error")
        self.assertFalse(service._preflight["avatar_reload_ok"])
        self.assertEqual(service.osc.sent_statuses, [2])

    def test_thinking_display_test_pulses_off_before_on(self) -> None:
        class RecordingOsc:
            def __init__(self) -> None:
                self.values: list[bool] = []

            def send_thinking(self, value: bool) -> None:
                self.values.append(value)

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._thinking_output_lock = threading.RLock()
        service._thinking_test_override = False
        service._thinking_output = False
        service._thinking_last_sent_at = 0.0
        service.osc = RecordingOsc()

        with patch("vrchat_ai_tool.control_server.time.sleep") as sleep:
            service.set_thinking_test(True)

        self.assertEqual(service.osc.values, [False, True])
        self.assertTrue(service._thinking_test_override)
        self.assertTrue(service._thinking_output)
        sleep.assert_called_once_with(0.15)

    def test_thinking_packets_cannot_be_reordered_across_threads(self) -> None:
        class BlockingOsc:
            def __init__(self) -> None:
                self.values: list[bool] = []
                self.true_send_started = threading.Event()
                self.release_true_send = threading.Event()

            def send_thinking(self, value: bool) -> None:
                if value and not self.true_send_started.is_set():
                    self.true_send_started.set()
                    self.release_true_send.wait(timeout=2.0)
                self.values.append(value)

        class MutableMotion:
            def __init__(self) -> None:
                self.activity = 0

            def snapshot(self) -> dict[str, int]:
                return {"activity": self.activity}

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._thinking_output_lock = threading.RLock()
        service._thinking_test_override = False
        service._thinking_output = False
        service._thinking_last_sent_at = 0.0
        service._ui_state = UiActivityState.WORKING
        service.osc = BlockingOsc()
        service.motion = MutableMotion()

        ui_thread = threading.Thread(target=service._update_thinking_output)
        ui_thread.start()
        self.assertTrue(service.osc.true_send_started.wait(timeout=1.0))

        service.motion.activity = 1
        speech_thread = threading.Thread(target=service._update_thinking_output)
        speech_thread.start()
        service.osc.release_true_send.set()
        ui_thread.join(timeout=1.0)
        speech_thread.join(timeout=1.0)

        self.assertFalse(ui_thread.is_alive())
        self.assertFalse(speech_thread.is_alive())
        self.assertEqual(service.osc.values, [True, False])
        self.assertFalse(service._thinking_output)

    def test_thinking_output_is_reasserted_after_one_second(self) -> None:
        class RecordingOsc:
            def __init__(self) -> None:
                self.values: list[bool] = []

            def send_thinking(self, value: bool) -> None:
                self.values.append(value)

        class IdleMotion:
            @staticmethod
            def snapshot() -> dict[str, int]:
                return {"activity": 0}

        service = object.__new__(VoiceControlService)
        service._lock = threading.RLock()
        service._thinking_output_lock = threading.RLock()
        service._thinking_test_override = False
        service._thinking_output = False
        service._thinking_last_sent_at = 0.0
        service._ui_state = UiActivityState.WORKING
        service.osc = RecordingOsc()
        service.motion = IdleMotion()

        with patch(
            "vrchat_ai_tool.control_server.time.monotonic",
            side_effect=(10.0, 10.5, 11.1),
        ):
            service._update_thinking_output()
            service._update_thinking_output()
            service._update_thinking_output()

        self.assertEqual(service.osc.values, [True, True])

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
        self.assertIn("/api/captions/mode", CONTROL_HTML)
        self.assertIn("/api/captions/quality", CONTROL_HTML)
        self.assertIn("/api/captions/test", CONTROL_HTML)
        self.assertIn("高精度（medium・低速）", CONTROL_HTML)
        self.assertIn("AI発話字幕（VRChatチャットボックス）", CONTROL_HTML)
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
        self.assertIn("/api/preflight/start", CONTROL_HTML)
        self.assertIn("同期確認して開始", CONTROL_HTML)
        self.assertLess(CONTROL_HTML.index("緊急ミュート"), CONTROL_HTML.index("アバター状態表示"))
        self.assertLess(CONTROL_HTML.index("アバター状態表示"), CONTROL_HTML.index("自己ループ対策"))

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
                base + "/api/captions/mode",
                data=json.dumps({"mode": "uia"}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(service.caption_mode, "uia")

            request = urllib.request.Request(
                base + "/api/captions/quality",
                data=json.dumps({"quality": "accuracy"}).encode("utf-8"),
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(service.caption_quality, "accuracy")

            request = urllib.request.Request(
                base + "/api/captions/test",
                data=b"{}",
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(service.caption_tests, 1)

            request = urllib.request.Request(
                base + "/api/preflight/start",
                data=b"{}",
                method="POST",
                headers={"Authorization": "Bearer " + "x" * 32, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertTrue(service.preflight_started)

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
