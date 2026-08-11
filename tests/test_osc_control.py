from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vrchat_ai_tool.osc_control import AgentStatus, VRChatOscController
from vrchat_ai_tool.voice_config import VoiceOscConfig


class FakeClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []
        self.controller: VRChatOscController | None = None
        self.actual_muted = False

    def send_message(self, address: str, value: object) -> None:
        self.messages.append((address, value))
        if address == "/input/Voice" and value is True:
            self.actual_muted = not self.actual_muted
            assert self.controller is not None
            self.controller._on_mute_self("/avatar/parameters/MuteSelf", self.actual_muted)
        elif address == "/avatar/parameters/VoiceAgentStatus":
            assert self.controller is not None
            self.controller._on_status(address, value)
        elif address == "/avatar/parameters/VoiceAgentThinking":
            assert self.controller is not None
            self.controller._on_thinking(address, value)
        elif address == "/avatar/parameters/VoiceAgentOscProbe":
            assert self.controller is not None
            self.controller._on_probe(address, value)
        elif address == "/avatar/change":
            assert self.controller is not None
            self.controller._on_avatar_change(address, value)


class OscControlTests(unittest.TestCase):
    def test_current_avatar_reload_is_sent_once_and_confirmed(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(
            VoiceOscConfig(avatar_reload_settle_sec=0.0),
            client_factory=lambda _h, _p: fake,
        )
        fake.controller = controller
        controller._server = object()
        controller._on_avatar_change(
            "/avatar/change", "avtr_12345678-1234-1234-1234-123456789abc"
        )
        fake.messages.clear()

        elapsed_ms = controller.reload_current_avatar()

        self.assertGreaterEqual(elapsed_ms, 0.0)
        self.assertEqual(
            fake.messages.count(
                ("/avatar/change", "avtr_12345678-1234-1234-1234-123456789abc")
            ),
            1,
        )
        self.assertEqual(controller.feedback_snapshot()["avatar_generation"], 2)

    def test_avatar_reload_fails_without_a_safe_current_id(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller
        controller._server = object()

        with (
            patch.object(controller, "_discover_current_avatar_id_from_log", return_value=None),
            self.assertRaisesRegex(RuntimeError, "avatar ID"),
        ):
            controller.reload_current_avatar()

    def test_current_avatar_id_is_discovered_from_vrchat_log_and_probe_schema(self) -> None:
        avatar_id = "avtr_12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            vrchat = profile / "AppData" / "LocalLow" / "VRChat" / "VRChat"
            vrchat.mkdir(parents=True)
            (vrchat / "output_log_2026-08-12_00-00-00.txt").write_text(
                f"Loading Avatar Data:{avatar_id}\n",
                encoding="utf-8",
            )
            schema_dir = vrchat / "OSC" / "usr_test" / "Avatars"
            schema_dir.mkdir(parents=True)
            (schema_dir / f"{avatar_id}.json").write_text(
                '{"parameters":[{"name":"VoiceAgentOscProbe"}]}',
                encoding="utf-8",
            )
            controller = VRChatOscController(
                VoiceOscConfig(), client_factory=lambda _h, _p: FakeClient()
            )

            with patch.dict(os.environ, {"USERPROFILE": directory}):
                discovered = controller._discover_current_avatar_id_from_log()

        self.assertEqual(discovered, avatar_id)

    def test_confirmed_mute_works_from_unknown_initial_state(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller
        controller._server = object()  # listener availability; feedback is supplied by FakeClient

        self.assertTrue(controller.set_muted(True))
        self.assertTrue(controller.mute_state)
        self.assertFalse(controller.set_muted(False))
        self.assertFalse(controller.mute_state)

    def test_status_uses_generic_parameter(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller
        controller.send_status(AgentStatus.MAINTENANCE)
        self.assertEqual(
            fake.messages[-3:],
            [("/avatar/parameters/VoiceAgentStatus", 3)] * 3,
        )

    def test_chatbox_caption_is_sent_once_with_typing_state(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)

        controller.send_chatbox_typing(True)
        controller.send_chatbox("AI: こんにちは")
        controller.send_chatbox_typing(False)

        self.assertEqual(
            fake.messages,
            [
                ("/chatbox/typing", True),
                ("/chatbox/input", ["AI: こんにちは", True, False]),
                ("/chatbox/typing", False),
            ],
        )

    def test_chatbox_rejects_text_beyond_vrchat_limit(self) -> None:
        controller = VRChatOscController(
            VoiceOscConfig(), client_factory=lambda _h, _p: FakeClient()
        )

        with self.assertRaisesRegex(ValueError, "144"):
            controller.send_chatbox("x" * 145)

    def test_status_change_is_confirmed_by_vrchat_feedback(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller
        controller._server = object()

        self.assertEqual(controller.set_status_confirmed(AgentStatus.ONLINE), AgentStatus.ONLINE)
        feedback = controller.feedback_snapshot()
        self.assertEqual(feedback["status"], 1)
        self.assertTrue(feedback["status_confirmed"])

    def test_motion_parameters_use_generic_avatar_addresses(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller.send_motion_enabled(True)
        controller.send_motion_activity(1)
        controller.send_motion_energy(0.62)
        controller.send_motion_gesture(4)
        controller.send_motion_expression(3)
        controller.send_thinking(True)

        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentMotionEnabled", True)), 3)
        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentActivity", 1)), 3)
        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentEnergy", 0.62)), 1)
        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentGesture", 4)), 3)
        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentExpression", 3)), 3)
        self.assertEqual(fake.messages.count(("/avatar/parameters/VoiceAgentThinking", True)), 3)

    def test_avatar_menu_can_report_motion_toggle(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller._on_motion_enabled("/avatar/parameters/VoiceAgentMotionEnabled", False)

        self.assertFalse(controller.motion_enabled)

    def test_motion_gesture_supports_nine_motions(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller.send_motion_gesture(9)
        controller.send_motion_gesture(99)

        self.assertEqual(
            fake.messages[-6:],
            [("/avatar/parameters/VoiceAgentGesture", 9)] * 6,
        )

    def test_motion_expression_is_clamped_to_seven_faces(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller.send_motion_expression(6)
        controller.send_motion_expression(99)

        self.assertEqual(
            fake.messages[-6:],
            [("/avatar/parameters/VoiceAgentExpression", 6)] * 6,
        )

    def test_feedback_snapshot_reports_observed_motion_values(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller._on_motion_activity("/avatar/parameters/VoiceAgentActivity", 1)
        controller._on_motion_energy("/avatar/parameters/VoiceAgentEnergy", 0.73)
        controller._on_motion_gesture("/avatar/parameters/VoiceAgentGesture", 6)
        controller._on_motion_expression("/avatar/parameters/VoiceAgentExpression", 4)
        controller._on_thinking("/avatar/parameters/VoiceAgentThinking", True)

        feedback = controller.feedback_snapshot()
        self.assertEqual(feedback["activity"], 1)
        self.assertEqual(feedback["energy"], 0.73)
        self.assertEqual(feedback["gesture"], 6)
        self.assertEqual(feedback["expression"], 4)
        self.assertTrue(feedback["thinking"])

    def test_thinking_parameter_reports_target_and_feedback(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller.send_thinking(True)

        feedback = controller.feedback_snapshot()
        self.assertTrue(feedback["thinking"])
        self.assertTrue(feedback["thinking_target"])

    def test_dedicated_probe_confirms_both_edges(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller
        controller._server = object()

        rtt_ms = controller.confirm_probe_roundtrip()

        feedback = controller.feedback_snapshot()
        self.assertGreaterEqual(rtt_ms, 0.0)
        self.assertFalse(feedback["probe"])
        self.assertFalse(feedback["probe_target"])
        self.assertIn(("/avatar/parameters/VoiceAgentOscProbe", True), fake.messages)
        self.assertIn(("/avatar/parameters/VoiceAgentOscProbe", False), fake.messages)

    def test_probe_error_explains_that_avatar_upload_is_required(self) -> None:
        class SilentClient:
            def __init__(self) -> None:
                self.messages: list[tuple[str, object]] = []

            def send_message(self, address: str, value: object) -> None:
                self.messages.append((address, value))

        fake = SilentClient()
        controller = VRChatOscController(
            VoiceOscConfig(mute_confirm_timeout_sec=0.01, mute_retry_count=0),
            client_factory=lambda _h, _p: fake,
        )
        controller._server = object()

        with self.assertRaisesRegex(RuntimeError, "VoiceAgentOscProbe"):
            controller.confirm_probe_roundtrip()


if __name__ == "__main__":
    unittest.main()
