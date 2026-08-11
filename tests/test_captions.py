from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from vrchat_ai_tool.captions import CaptionService, UiaCaptionExtractor, format_chatbox_text
from vrchat_ai_tool.chatgpt_ui_diagnostic import UiElementRecord, UiScanResult
from vrchat_ai_tool.voice_config import VoiceCaptionConfig


def text_record(locator: str, text: str, top: int) -> UiElementRecord:
    return UiElementRecord(
        locator=locator,
        process_id=1,
        window_handle=2,
        window_title="ChatGPT",
        control_type="Text",
        name=text,
        automation_id="",
        class_name="",
        is_enabled=True,
        is_offscreen=False,
        rectangle=f"0,{top},600,{top + 20}",
    )


def scan(*records: UiElementRecord) -> UiScanResult:
    return UiScanResult((1,), 1, {record.locator: record for record in records})


class FakeOsc:
    def __init__(self) -> None:
        self.chatbox: list[str] = []
        self.typing: list[bool] = []
        self.sent = threading.Event()

    def send_chatbox(self, text: str, *, notify: bool = False) -> None:
        self.chatbox.append(text)
        self.sent.set()

    def send_chatbox_typing(self, typing: bool) -> None:
        self.typing.append(typing)


class FakeTranscriber:
    def __init__(self, text: str = "こんにちは、音声字幕です") -> None:
        self.text = text
        self.warmed = False
        self.paths: list[Path] = []

    def warm_up(self) -> None:
        self.warmed = True

    def transcribe_wav(self, wave_path: Path) -> str:
        self.paths.append(wave_path)
        self.assert_wave_exists = wave_path.exists()
        return self.text


class CaptionTests(unittest.TestCase):
    def test_chatbox_formatter_uses_rolling_tail_and_144_character_limit(self) -> None:
        result = format_chatbox_text("あ" * 200, "AI: ", 144)

        self.assertEqual(len(result), 144)
        self.assertTrue(result.startswith("AI: …"))
        self.assertTrue(result.endswith("あ" * 10))

    def test_uia_extractor_ignores_history_and_prefers_newest_response(self) -> None:
        extractor = UiaCaptionExtractor()
        old = text_record("old", "以前の回答です", 100)
        extractor.begin(scan(old))

        newest = text_record("new", "こんにちは、今回の回答です", 500)
        result = extractor.update(scan(old, newest))

        self.assertEqual(result, "こんにちは、今回の回答です")

    def test_uia_extractor_follows_streaming_text_updates(self) -> None:
        extractor = UiaCaptionExtractor()
        extractor.begin(scan(text_record("old", "以前の回答", 100)))

        self.assertEqual(
            extractor.update(scan(text_record("answer", "こん", 500))),
            "こん",
        )
        self.assertEqual(
            extractor.update(scan(text_record("answer", "こんにちは", 500))),
            "こんにちは",
        )

    def test_stt_segments_cable_b_and_sends_local_transcript(self) -> None:
        osc = FakeOsc()
        transcriber = FakeTranscriber()
        config = VoiceCaptionConfig(
            mode="stt",
            prefix="AI: ",
            min_send_interval_sec=0.0,
            stt_speech_on_rms=10.0,
            stt_speech_off_rms=5.0,
            stt_end_silence_ms=100,
            stt_min_audio_ms=100,
            stt_partial_interval_sec=60.0,
        )
        service = CaptionService(
            config,
            osc,
            transcriber_factory=lambda: transcriber,
            sample_rate=1000,
            channels=1,
        )
        service.start()
        try:
            one_hundred_ms = b"\x01\x00" * 100
            service.on_audio_chunk(one_hundred_ms, 20.0)
            service.on_audio_chunk(b"\x00\x00" * 100, 0.0)

            self.assertTrue(osc.sent.wait(2.0))
            self.assertEqual(osc.chatbox[-1], "AI: こんにちは、音声字幕です")
            self.assertTrue(transcriber.warmed)
            self.assertIn(True, osc.typing)
            self.assertEqual(osc.typing[-1], False)
            self.assertEqual(service.snapshot()["stt_state"], "ready")
        finally:
            service.stop()

    def test_switching_off_drops_pending_caption(self) -> None:
        osc = FakeOsc()
        service = CaptionService(
            VoiceCaptionConfig(mode="uia", min_send_interval_sec=5.0),
            osc,
        )
        service.start()
        try:
            service._publish("送られない字幕", "uia")
            service.set_mode("off")
            time.sleep(0.05)
            self.assertEqual(osc.chatbox, [])
        finally:
            service.stop()


if __name__ == "__main__":
    unittest.main()
