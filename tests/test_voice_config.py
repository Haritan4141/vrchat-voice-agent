from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vrchat_ai_tool.voice_config import (
    load_voice_config,
    resolve_config_relative,
    save_loop_guard_enabled,
    save_motion_enabled,
)


class VoiceConfigTests(unittest.TestCase):
    def test_defaults_and_relative_token_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.toml"
            path.write_text("[loop_guard]\nauto_mute = true\n", encoding="utf-8")
            config = load_voice_config(path)

            self.assertEqual(config.audio.vrchat_output, "CABLE-A Input")
            self.assertTrue(config.loop_guard.auto_mute)
            self.assertEqual(
                resolve_config_relative(config, config.control.token_file),
                (Path(directory) / "control-token.txt").resolve(),
            )
            self.assertEqual(config.loop_guard.reliable_max_delay_ms, 1800)
            self.assertEqual(config.loop_guard.min_match_duration_ms, 1500)
            self.assertTrue(config.motion.enabled)
            self.assertEqual(config.osc.motion_activity_parameter, "VoiceAgentActivity")

    def test_loop_guard_switch_is_persisted_without_losing_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.toml"
            path.write_text(
                "# keep this comment\n[loop_guard]\nenabled = true\nauto_mute = true\n\n"
                "[osc]\ntarget_host = \"127.0.0.1\"\n",
                encoding="utf-8",
            )
            config = load_voice_config(path)

            save_loop_guard_enabled(config, False)

            saved = path.read_text(encoding="utf-8")
            self.assertIn("# keep this comment", saved)
            self.assertIn("enabled = false", saved)
            self.assertIn("auto_mute = true", saved)
            self.assertIn('[osc]\ntarget_host = "127.0.0.1"', saved)
            self.assertFalse(config.loop_guard.enabled)
            self.assertFalse(load_voice_config(path).loop_guard.enabled)

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.toml"
            path.write_text("[osc]\nwrong = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong"):
                load_voice_config(path)

    def test_motion_switch_is_persisted_in_its_own_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.toml"
            path.write_text("[osc]\ntarget_host = \"127.0.0.1\"\n", encoding="utf-8")
            config = load_voice_config(path)

            save_motion_enabled(config, False)

            saved = path.read_text(encoding="utf-8")
            self.assertIn("[motion]", saved)
            self.assertIn("enabled = false", saved)
            self.assertFalse(load_voice_config(path).motion.enabled)


if __name__ == "__main__":
    unittest.main()
