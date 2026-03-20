from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vrchat_ai_tool.config import SttConfig
from vrchat_ai_tool.stt import (
    FasterWhisperTranscriber,
    create_transcriber,
    normalize_whisper_language,
)


class FakeWhisperModel:
    last_init: dict | None = None
    last_transcribe: dict | None = None

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        FakeWhisperModel.last_init = {
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
        }

    def transcribe(self, path: str, **kwargs):
        FakeWhisperModel.last_transcribe = {
            "path": path,
            "kwargs": kwargs,
        }
        segments = [SimpleNamespace(text="こんにちは"), SimpleNamespace(text=" world")]
        info = SimpleNamespace(language="ja")
        return iter(segments), info


class SttTests(unittest.TestCase):
    def test_normalize_whisper_language(self) -> None:
        self.assertEqual(normalize_whisper_language("ja-JP"), "ja")
        self.assertEqual(normalize_whisper_language("EN"), "en")
        self.assertIsNone(normalize_whisper_language("auto"))

    def test_create_transcriber_for_faster_whisper(self) -> None:
        config = SttConfig(
            backend="faster_whisper",
            model="small",
            device="cuda",
            compute_type="float16",
            language="ja",
            timeout_sec=20,
            beam_size=5,
            vad_filter=True,
            vad_min_silence_ms=500,
        )

        transcriber = create_transcriber(config)

        self.assertIsInstance(transcriber, FasterWhisperTranscriber)

    def test_faster_whisper_transcribe_uses_expected_kwargs(self) -> None:
        transcriber = FasterWhisperTranscriber(
            model_name="medium",
            device="cuda",
            compute_type="float16",
            language="ja-JP",
            beam_size=3,
            vad_filter=True,
            vad_min_silence_ms=700,
        )

        with patch("vrchat_ai_tool.stt.import_module", return_value=SimpleNamespace(WhisperModel=FakeWhisperModel)):
            text = transcriber.transcribe_wav(Path("dummy.wav"))

        self.assertEqual(text, "こんにちは world")
        self.assertEqual(
            FakeWhisperModel.last_init,
            {
                "model_name": "medium",
                "device": "cuda",
                "compute_type": "float16",
            },
        )
        self.assertEqual(FakeWhisperModel.last_transcribe["path"], "dummy.wav")
        self.assertEqual(FakeWhisperModel.last_transcribe["kwargs"]["beam_size"], 3)
        self.assertEqual(FakeWhisperModel.last_transcribe["kwargs"]["language"], "ja")
        self.assertTrue(FakeWhisperModel.last_transcribe["kwargs"]["vad_filter"])
        self.assertEqual(
            FakeWhisperModel.last_transcribe["kwargs"]["vad_parameters"]["min_silence_duration_ms"],
            700,
        )


if __name__ == "__main__":
    unittest.main()
