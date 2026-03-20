from __future__ import annotations

from pathlib import Path
import os
import textwrap
import unittest

from vrchat_ai_tool.config import dump_config, ensure_config_file, load_config, save_config  # noqa: E402


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
            device = "cuda"
            compute_type = "int8"
            language = "ja"
            beam_size = 3
            vad_filter = true
            vad_min_silence_ms = 400

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
        self.assertEqual(config.stt.beam_size, 3)
        self.assertTrue(config.stt.vad_filter)

    def test_save_config_round_trip(self) -> None:
        config_text = textwrap.dedent(
            """
            [audio.capture]
            mode = "virtual_device"
            input_device = "CABLE-A Output"
            sample_rate = 16000
            channels = 1
            chunk_ms = 200
            silence_timeout_ms = 900
            rms_threshold = 450
            min_speech_ms = 300
            max_utterance_ms = 8000

            [audio.output]
            tts_output_device = "CABLE-B Input"
            monitor_output_device = ""

            [stt]
            backend = "faster_whisper"
            model = "large-v3"
            device = "cuda"
            compute_type = "float16"
            language = "ja"
            timeout_sec = 20
            beam_size = 5
            vad_filter = true
            vad_min_silence_ms = 500

            [llm]
            backend = "ollama"
            base_url = "http://127.0.0.1:11434"
            model = "gemma3:12b"
            temperature = 0.7
            max_tokens = 120
            timeout_sec = 60
            system_prompt = '''
            line one
            line two
            '''

            [tts]
            backend = "voicevox"
            base_url = "http://127.0.0.1:50021"
            speaker = 3
            speed_scale = 1.0
            timeout_sec = 30

            [conversation]
            max_response_chars = 80
            min_reply_interval_sec = 4
            allow_topic_suggestions = true
            pause_listening_while_speaking = true
            """
        ).strip()

        source_path = Path.cwd() / "tests" / "_tmp_source_settings.toml"
        output_path = Path.cwd() / "tests" / "_tmp_saved_settings.toml"
        try:
            source_path.write_text(config_text, encoding="utf-8")
            config = load_config(source_path)
            save_config(config, output_path)
            loaded = load_config(output_path)
        finally:
            if source_path.exists():
                os.remove(source_path)
            if output_path.exists():
                os.remove(output_path)

        self.assertEqual(loaded.stt.model, "large-v3")
        self.assertEqual(loaded.llm.model, "gemma3:12b")
        self.assertEqual(loaded.llm.system_prompt, "line one\nline two")
        self.assertEqual(loaded.conversation.min_reply_interval_sec, 4)
        self.assertIn("system_prompt =", dump_config(loaded))

    def test_ensure_config_file_copies_example(self) -> None:
        example_path = Path.cwd() / "tests" / "_tmp_example_settings.toml"
        target_path = Path.cwd() / "tests" / "_tmp_gui_settings.toml"
        try:
            example_path.write_text("[conversation]\nmax_response_chars = 80\n", encoding="utf-8")
            ensure_config_file(target_path, example_path)
            copied = target_path.read_text(encoding="utf-8")
        finally:
            if example_path.exists():
                os.remove(example_path)
            if target_path.exists():
                os.remove(target_path)

        self.assertEqual(copied, "[conversation]\nmax_response_chars = 80\n")


if __name__ == "__main__":
    unittest.main()
