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


if __name__ == "__main__":
    unittest.main()
