from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vrchat_ai_tool.voice_config import load_voice_config, resolve_config_relative


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

    def test_unknown_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.toml"
            path.write_text("[osc]\nwrong = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong"):
                load_voice_config(path)


if __name__ == "__main__":
    unittest.main()
