from __future__ import annotations

import unittest

import numpy as np

from vrchat_ai_tool.loop_guard import LoopDetector
from vrchat_ai_tool.voice_config import VoiceLoopGuardConfig


class LoopDetectorTests(unittest.TestCase):
    def make_config(self) -> VoiceLoopGuardConfig:
        return VoiceLoopGuardConfig(
            rms_threshold=100,
            correlation_threshold=0.9,
            min_consecutive_matches=2,
            feature_ms=20,
            comparison_window_ms=400,
            min_delay_ms=100,
            max_delay_ms=500,
        )

    def test_delayed_cable_b_envelope_triggers_and_latches(self) -> None:
        rng = np.random.default_rng(4141)
        cable_b = rng.uniform(200, 1800, 180).astype(np.float32)
        lag = 10
        cable_a = np.concatenate((np.zeros(lag, dtype=np.float32), cable_b[:-lag]))
        detector = LoopDetector(self.make_config(), sample_rate=48000)

        result = None
        for start in range(0, cable_a.size, 5):
            result = detector.add_features(cable_a[start : start + 5], cable_b[start : start + 5])

        self.assertIsNotNone(result)
        self.assertTrue(result.triggered)
        self.assertEqual(result.delay_ms, 200)
        self.assertGreaterEqual(result.score, 0.99)

        unrelated = np.arange(5, dtype=np.float32) + 300
        result = detector.add_features(unrelated, unrelated[::-1].copy())
        self.assertTrue(result.triggered)
        self.assertEqual(result.delay_ms, 200)
        self.assertGreaterEqual(result.score, 0.99)

        detector.reset()
        self.assertFalse(detector.last.triggered)

    def test_unrelated_audio_does_not_trigger(self) -> None:
        rng = np.random.default_rng(7)
        detector = LoopDetector(self.make_config(), sample_rate=48000)
        for _ in range(40):
            a = rng.uniform(200, 1800, 5).astype(np.float32)
            b = rng.uniform(200, 1800, 5).astype(np.float32)
            result = detector.add_features(a, b)
        self.assertFalse(result.triggered)

    def test_high_correlation_short_transient_does_not_trigger(self) -> None:
        rng = np.random.default_rng(19)
        cable_b = rng.uniform(200, 1800, 60).astype(np.float32)
        lag = 10
        cable_a = np.concatenate((np.zeros(lag, dtype=np.float32), cable_b[:-lag]))
        detector = LoopDetector(self.make_config(), sample_rate=48000)

        for start in range(0, cable_a.size, 5):
            result = detector.add_features(cable_a[start : start + 5], cable_b[start : start + 5])

        self.assertGreaterEqual(result.score, 0.99)
        self.assertFalse(result.triggered)
        self.assertLess(result.candidate_duration_ms, 1500)

    def test_delay_beyond_reliable_limit_is_ignored(self) -> None:
        rng = np.random.default_rng(23)
        cable_b = rng.uniform(200, 1800, 300).astype(np.float32)
        lag = 106  # 2120 ms, matching the observed false positive.
        cable_a = np.concatenate((np.zeros(lag, dtype=np.float32), cable_b[:-lag]))
        config = self.make_config()
        config.max_delay_ms = 5000
        config.reliable_max_delay_ms = 1800
        config.correlation_threshold = 0.99
        detector = LoopDetector(config, sample_rate=48000)

        for start in range(0, cable_a.size, 5):
            result = detector.add_features(cable_a[start : start + 5], cable_b[start : start + 5])

        self.assertFalse(result.triggered)


if __name__ == "__main__":
    unittest.main()
