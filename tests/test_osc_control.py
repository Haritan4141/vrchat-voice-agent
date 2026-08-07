from __future__ import annotations

import unittest

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


class OscControlTests(unittest.TestCase):
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
            fake.messages[-1],
            ("/avatar/parameters/VoiceAgentStatus", 3),
        )

    def test_motion_parameters_use_generic_avatar_addresses(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller.send_motion_enabled(True)
        controller.send_motion_activity(1)
        controller.send_motion_energy(0.62)
        controller.send_motion_gesture(4)

        self.assertEqual(
            fake.messages[-4:],
            [
                ("/avatar/parameters/VoiceAgentMotionEnabled", True),
                ("/avatar/parameters/VoiceAgentActivity", 1),
                ("/avatar/parameters/VoiceAgentEnergy", 0.62),
                ("/avatar/parameters/VoiceAgentGesture", 4),
            ],
        )

    def test_avatar_menu_can_report_motion_toggle(self) -> None:
        fake = FakeClient()
        controller = VRChatOscController(VoiceOscConfig(), client_factory=lambda _h, _p: fake)
        fake.controller = controller

        controller._on_motion_enabled("/avatar/parameters/VoiceAgentMotionEnabled", False)

        self.assertFalse(controller.motion_enabled)


if __name__ == "__main__":
    unittest.main()
