from __future__ import annotations

from pathlib import Path
import os
import textwrap
import unittest

from vrchat_ai_tool.config import load_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        config_text = textwrap.dedent(
            """
            [audio.capture]
            mode = "virtual_device"
            input_device = "Capture A"
            sample_rate = 16000
            channels = 1
            chunk_ms = 200
            silence_timeout_ms = 900

            [audio.output]
            tts_output_device = "Playback B"
            monitor_output_device = ""

            [stt]
            backend = "faster_whisper"
            model = "small"
            device = "auto"
            compute_type = "int8"
            language = "ja"

            [llm]
            backend = "ollama"
            base_url = "http://127.0.0.1:11434"
            model = "qwen2.5:7b-instruct"
            temperature = 0.7
            max_tokens = 120
            system_prompt = "test prompt"

            [tts]
            backend = "voicevox"
            base_url = "http://127.0.0.1:50021"
            speaker = 3
            speed_scale = 1.0

            [conversation]
            max_response_chars = 80
            min_reply_interval_sec = 8
            allow_topic_suggestions = true
            pause_listening_while_speaking = true
            """
        ).strip()

        path = Path.cwd() / "tests" / "_tmp_settings.toml"
        try:
            path.write_text(config_text, encoding="utf-8")
            config = load_config(path)
        finally:
            if path.exists():
                os.remove(path)

        self.assertEqual(config.audio_capture.input_device, "Capture A")
        self.assertEqual(config.audio_output.tts_output_device, "Playback B")
        self.assertEqual(config.llm.model, "qwen2.5:7b-instruct")
        self.assertTrue(config.conversation.allow_topic_suggestions)
        self.assertEqual(config.audio_capture.rms_threshold, 500.0)
        self.assertEqual(config.stt.timeout_sec, 8)


if __name__ == "__main__":
    unittest.main()
