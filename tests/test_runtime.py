from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import threading
import unittest

from vrchat_ai_tool.runtime import BotRuntime, HeardUtterance, clean_reply_text


class StubRuntime(BotRuntime):
    def __init__(self, stop_event: threading.Event) -> None:
        self.config = SimpleNamespace(
            conversation=SimpleNamespace(
                min_reply_interval_sec=0,
                max_response_chars=80,
                allow_topic_suggestions=True,
                pause_listening_while_speaking=True,
            )
        )
        self.history = deque(maxlen=8)
        self.last_reply_at = 0.0
        self._stop_event = stop_event

    def capture_and_transcribe_once(
        self,
        max_wait_sec: float | None = None,
        save_audio: bool = False,
    ) -> HeardUtterance:
        return HeardUtterance(text="hello", wav_path=None)

    def generate_reply(self, transcript: str) -> str:
        return f"reply to {transcript}"

    def speak_text(self, text: str, save_audio: bool = False):
        self._stop_event.set()
        return None


class WarmupComponent:
    def __init__(self, label: str, calls: list[str]) -> None:
        self.label = label
        self.calls = calls

    def warm_up(self) -> None:
        self.calls.append(self.label)


class RuntimeTests(unittest.TestCase):
    def test_clean_reply_text(self) -> None:
        self.assertEqual(clean_reply_text("  hello   world  ", 20), "hello world")
        self.assertEqual(clean_reply_text("abcdef", 4), "abcd")
        self.assertEqual(clean_reply_text("   ", 10), "")

    def test_run_forever_honors_stop_event_and_logger(self) -> None:
        stop_event = threading.Event()
        runtime = StubRuntime(stop_event)
        logs: list[str] = []

        runtime.run_forever(stop_event=stop_event, logger=logs.append, listen_timeout_sec=0.01)

        self.assertEqual(logs, ["[heard] hello", "[reply] reply to hello"])
        self.assertTrue(stop_event.is_set())

    def test_warm_up_logs_and_calls_components(self) -> None:
        runtime = object.__new__(BotRuntime)
        calls: list[str] = []
        runtime.transcriber = WarmupComponent("stt", calls)
        runtime.llm = WarmupComponent("llm", calls)
        runtime.voicevox = WarmupComponent("tts", calls)
        logs: list[str] = []

        runtime.warm_up(logger=logs.append)

        self.assertEqual(calls, ["stt", "llm", "tts"])
        self.assertEqual(
            logs,
            [
                "[warmup] loading STT model...",
                "[warmup] loading LLM model...",
                "[warmup] priming TTS engine...",
                "[warmup] ready",
            ],
        )


if __name__ == "__main__":
    unittest.main()
