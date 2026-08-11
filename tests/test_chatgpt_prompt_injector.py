from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from vrchat_ai_tool.chatgpt_prompt_injector import (
    ComposerNotReady,
    PromptInjectionError,
    find_prompt_target,
    load_prompt,
    parse_rectangle,
    run_prompt_injector,
    wait_for_prompt_target,
)
from vrchat_ai_tool.chatgpt_ui_diagnostic import UiElementRecord, UiScanResult


def record(
    locator: str,
    *,
    process_id: int = 10,
    window_handle: int = 20,
    name: str = "何でもどうぞ",
    control_type: str = "Edit",
    class_name: str = "ProseMirror ProseMirror-focused",
    rectangle: str = "100,200,500,260",
    is_enabled: bool = True,
    is_offscreen: bool = False,
) -> UiElementRecord:
    return UiElementRecord(
        locator=locator,
        process_id=process_id,
        window_handle=window_handle,
        window_title="ChatGPT",
        control_type=control_type,
        name=name,
        automation_id="",
        class_name=class_name,
        is_enabled=is_enabled,
        is_offscreen=is_offscreen,
        rectangle=rectangle,
    )


def result(*records: UiElementRecord) -> UiScanResult:
    return UiScanResult(
        process_ids=(10,),
        window_count=1,
        elements={item.locator: item for item in records},
    )


class FakeProvider:
    def __init__(self, scans: list[UiScanResult]) -> None:
        self.scans = scans
        self.calls = 0

    def scan(self) -> UiScanResult:
        index = min(self.calls, len(self.scans) - 1)
        self.calls += 1
        return self.scans[index]


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, str]] = []

    def send(self, target, prompt: str, submit_key: str) -> None:
        self.calls.append((target, prompt, submit_key))


class PromptInjectorTests(unittest.TestCase):
    def test_batch_launcher_is_ascii_compatible(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]

        batch = (repository_root / "apply_voice_prompt.bat").read_bytes()

        self.assertTrue(batch)
        batch.decode("ascii")

    def test_load_prompt_accepts_utf8_bom_without_returning_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system_prompt.txt"
            path.write_bytes(b"\xef\xbb\xbf" + "ラズリ".encode())

            self.assertEqual(load_prompt(path), "ラズリ")

    def test_load_prompt_rejects_empty_and_oversized_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            empty = Path(temporary_directory) / "empty.txt"
            empty.write_text(" \n", encoding="utf-8")
            with self.assertRaises(PromptInjectionError):
                load_prompt(empty)

            large = Path(temporary_directory) / "large.txt"
            large.write_text("12345", encoding="utf-8")
            with self.assertRaises(PromptInjectionError):
                load_prompt(large, max_bytes=4)

    def test_parse_rectangle_requires_a_nonempty_rectangle(self) -> None:
        self.assertEqual(parse_rectangle("1,2,10,20"), (1, 2, 10, 20))
        with self.assertRaises(PromptInjectionError):
            parse_rectangle("1,2,1,20")
        with self.assertRaises(PromptInjectionError):
            parse_rectangle("not,a,rectangle")

    def test_find_target_accepts_prosemirror_and_ignores_hidden_editors(self) -> None:
        target = find_prompt_target(
            result(
                record("hidden", is_offscreen=True),
                record("composer", name="Localized placeholder"),
            )
        )

        self.assertEqual(target.window_handle, 20)
        self.assertEqual(target.rectangle, (100, 200, 500, 260))

    def test_find_target_fails_if_no_or_multiple_composers_are_safe(self) -> None:
        with self.assertRaises(ComposerNotReady):
            find_prompt_target(result(record("text", control_type="Text")))
        with self.assertRaises(PromptInjectionError):
            find_prompt_target(
                result(
                    record("one"),
                    record("two", window_handle=21, rectangle="600,200,900,260"),
                )
            )

    def test_wait_retries_until_the_composer_appears(self) -> None:
        provider = FakeProvider(
            [
                result(record("text", control_type="Text")),
                result(record("composer")),
            ]
        )
        now = [0.0]

        target = wait_for_prompt_target(
            provider,
            wait_seconds=2.0,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

        self.assertEqual(target.locator, "composer")
        self.assertEqual(provider.calls, 2)

    def test_run_dry_run_does_not_call_sender_or_print_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system_prompt.txt"
            path.write_text("TOP SECRET PROMPT", encoding="utf-8")
            provider = FakeProvider([result(record("composer"))])
            sender = FakeSender()

            output = StringIO()
            with redirect_stdout(output):
                exit_code = run_prompt_injector(
                    prompt_path=path,
                    wait_seconds=0,
                    dry_run=True,
                    provider=provider,
                    sender=sender,
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(sender.calls, [])
            self.assertNotIn("TOP SECRET PROMPT", output.getvalue())

    def test_run_sends_exact_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system_prompt.txt"
            path.write_bytes(b"line 1\nline 2\n")
            provider = FakeProvider([result(record("composer"))])
            sender = FakeSender()

            exit_code = run_prompt_injector(
                prompt_path=path,
                wait_seconds=0,
                submit_key="ctrl-enter",
                provider=provider,
                sender=sender,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(sender.calls[0][1:], ("line 1\nline 2\n", "ctrl-enter"))


if __name__ == "__main__":
    unittest.main()
