from __future__ import annotations

import unittest

from vrchat_ai_tool.motion_control import MotionActivity, MotionService
from vrchat_ai_tool.osc_control import AgentStatus
from vrchat_ai_tool.voice_config import VoiceMotionConfig


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRandom:
    def uniform(self, low: float, _high: float) -> float:
        return low

    def choice(self, values: list[int]) -> int:
        return values[0]


class FakeOsc:
    def __init__(self) -> None:
        self.mute_state: bool | None = False
        self.status = AgentStatus.ONLINE
        self.motion_enabled = True
        self.messages: list[tuple[str, object]] = []

    def send_motion_enabled(self, enabled: bool) -> None:
        self.motion_enabled = enabled
        self.messages.append(("enabled", enabled))

    def send_motion_activity(self, activity: int) -> None:
        self.messages.append(("activity", activity))

    def send_motion_energy(self, energy: float) -> None:
        self.messages.append(("energy", energy))

    def send_motion_gesture(self, gesture: int) -> None:
        self.messages.append(("gesture", gesture))

    def send_motion_expression(self, expression: int) -> None:
        self.messages.append(("expression", expression))


class MotionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.osc = FakeOsc()
        self.config = VoiceMotionConfig(
            attack_ms=200,
            release_ms=300,
            settling_ms=200,
            energy_smoothing=1.0,
            idle_gesture_min_sec=1.0,
            idle_gesture_max_sec=1.0,
            speaking_gesture_min_sec=0.5,
            speaking_gesture_max_sec=0.5,
            speaking_expression_min_sec=0.5,
            speaking_expression_max_sec=0.5,
        )
        self.service = MotionService(
            self.config,
            self.osc,
            clock=self.clock,
            rng=FakeRandom(),
        )
        self.service.start()

    def test_audio_hysteresis_moves_through_speaking_and_settling(self) -> None:
        self.service.on_audio_level(1800.0)
        self.clock.advance(0.1)
        self.service.on_audio_level(1800.0)
        self.clock.advance(0.1)
        self.service.on_audio_level(1800.0)

        speaking = self.service.snapshot()
        self.assertEqual(speaking["activity"], int(MotionActivity.SPEAKING))
        self.assertGreater(speaking["energy"], 0.0)

        for _ in range(4):
            self.clock.advance(0.1)
            self.service.on_audio_level(0.0)
        settling = self.service.snapshot()
        self.assertEqual(settling["activity"], int(MotionActivity.SETTLING))

        self.clock.advance(0.2)
        self.service.on_audio_level(0.0)
        self.assertEqual(self.service.snapshot()["activity"], int(MotionActivity.IDLE))
        self.assertEqual(self.service.snapshot()["last_expression"], 0)

    def test_speaking_expression_changes_without_repeating_and_resets(self) -> None:
        self.service.on_audio_level(1800.0)
        self.clock.advance(0.2)
        self.service.on_audio_level(1800.0)

        first = self.service.snapshot()["last_expression"]
        self.assertEqual(first, 1)

        self.clock.advance(0.5)
        self.service.on_audio_level(1800.0)
        second = self.service.snapshot()["last_expression"]
        self.assertEqual(second, 0)
        self.assertNotEqual(first, second)

        for _ in range(4):
            self.clock.advance(0.1)
            self.service.on_audio_level(0.0)
        self.assertEqual(self.service.snapshot()["last_expression"], 0)
        self.assertIn(("expression", 0), self.osc.messages)

    def test_diagnostic_test_bypasses_audio_and_restores_neutral(self) -> None:
        self.service.start_diagnostic_test()

        snapshot = self.service.snapshot()
        self.assertTrue(snapshot["diagnostic_running"])
        self.assertEqual(snapshot["activity"], int(MotionActivity.SPEAKING))
        self.assertEqual(snapshot["last_expression"], 1)
        self.assertIn(("energy", 0.72), self.osc.messages)

        self.service.on_audio_level(0.0)
        self.assertEqual(
            self.service.snapshot()["activity"],
            int(MotionActivity.SPEAKING),
        )

        self.service.stop_diagnostic_test()
        stopped = self.service.snapshot()
        self.assertFalse(stopped["diagnostic_running"])
        self.assertEqual(stopped["activity"], int(MotionActivity.IDLE))
        self.assertEqual(stopped["last_expression"], 0)

    def test_idle_gesture_is_randomised_without_repeating_last_value(self) -> None:
        self.clock.advance(1.0)
        self.service.on_audio_level(0.0)
        first = self.service.snapshot()["last_gesture"]
        self.assertEqual(first, 1)

        self.clock.advance(1.0)
        self.service.on_audio_level(0.0)
        second = self.service.snapshot()["last_gesture"]
        self.assertEqual(second, 5)

    def test_mute_suppresses_speaking_motion(self) -> None:
        self.osc.mute_state = True
        for _ in range(4):
            self.clock.advance(0.1)
            self.service.on_audio_level(3000.0)

        snapshot = self.service.snapshot()
        self.assertEqual(snapshot["activity"], int(MotionActivity.IDLE))
        self.assertEqual(snapshot["energy"], 0.0)


if __name__ == "__main__":
    unittest.main()
