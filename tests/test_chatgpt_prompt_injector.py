from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vrchat_ai_tool.chatgpt_prompt_injector import (
    CodexModeNotReady,
    ComposerNotReady,
    PromptInjectionError,
    VoiceStartNotReady,
    WindowsPromptSender,
    find_new_chat_target,
    find_prompt_target,
    find_voice_start_target,
    load_prompt,
    parse_rectangle,
    require_codex_mode,
    run_prompt_injector,
    wait_for_prompt_target,
    wait_for_voice_ready_after_click,
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


class FakeClicker:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def click(self, target) -> None:
        self.calls.append(target)


class PromptInjectorTests(unittest.TestCase):
    def test_clipboard_restore_failure_clears_prompt_without_raising(self) -> None:
        sender = WindowsPromptSender()
        pythoncom = SimpleNamespace(
            OleSetClipboard=Mock(side_effect=RuntimeError("restore failed")),
            OleFlushClipboard=Mock(),
        )
        win32clipboard = SimpleNamespace(
            OpenClipboard=Mock(),
            EmptyClipboard=Mock(),
            CloseClipboard=Mock(),
        )

        restored = sender._restore_or_clear_clipboard(
            pythoncom,
            win32clipboard,
            object(),
            True,
        )

        self.assertFalse(restored)
        win32clipboard.EmptyClipboard.assert_called_once_with()
        win32clipboard.CloseClipboard.assert_called_once_with()

    def test_batch_launcher_is_ascii_compatible(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]

        batch = (repository_root / "controls" / "apply_voice_prompt.bat").read_bytes()

        self.assertTrue(batch)
        batch.decode("ascii")

    def test_batch_applies_the_standalone_session_prompt(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        batch = (repository_root / "controls" / "apply_voice_prompt.bat").read_text(
            encoding="ascii"
        )
        prompt = (repository_root / "system_prompt.txt").read_text(encoding="utf-8")

        self.assertIn('--prompt-file "%REPO_ROOT%\\system_prompt.txt"', batch)
        self.assertIn("--start-voice", batch)
        self.assertIn("--require-codex", batch)
        self.assertIn("--voice-stabilization-seconds 5", batch)
        self.assertIn("この音声セッション全体に適用する会話設定", prompt)
        self.assertIn("ほかのファイルや事前設定の読み込みは必要ありません", prompt)
        self.assertNotIn("AGENTS.md", prompt)

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

    def test_find_target_accepts_voice_work_mode_composer(self) -> None:
        target = find_prompt_target(
            result(
                record(
                    "voice-composer",
                    name="Work モードで作成",
                    class_name="voice-chat-composer",
                )
            )
        )

        self.assertEqual(target.locator, "voice-composer")

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

    def test_find_new_chat_uses_only_the_exact_visible_button(self) -> None:
        target = find_new_chat_target(
            result(
                record("composer"),
                record(
                    "project-chat",
                    control_type="Button",
                    name="VRChat AI で新しいチャットを開始",
                    class_name="button",
                    rectangle="10,10,40,40",
                ),
                record(
                    "new-chat",
                    control_type="Button",
                    name="新しいチャット",
                    class_name="button",
                    rectangle="50,10,80,40",
                ),
            )
        )

        self.assertEqual(target.locator, "new-chat")

    def test_find_new_chat_prefers_the_sidebar_button_when_two_are_visible(self) -> None:
        target = find_new_chat_target(
            result(
                record("composer", rectangle="300,200,800,260"),
                record(
                    "main-new-chat",
                    control_type="Button",
                    name="New chat",
                    class_name="button",
                    rectangle="500,20,600,60",
                ),
                record(
                    "sidebar-new-chat",
                    control_type="Button",
                    name="New chat",
                    class_name="button",
                    rectangle="10,20,180,60",
                ),
            )
        )

        self.assertEqual(target.locator, "sidebar-new-chat")

    def test_require_codex_mode_accepts_product_text_or_codex_sidebar_marker(self) -> None:
        composer = record("composer", rectangle="300,200,800,260")
        codex = record(
            "codex-mode",
            control_type="Text",
            name="Codex",
            class_name="product-switcher",
            rectangle="10,10,100,40",
        )
        pull_requests = record(
            "pull-requests",
            control_type="Button",
            name="プルリクエスト",
            class_name="sidebar-item",
            rectangle="10,50,180,80",
        )

        require_codex_mode(result(composer, codex))
        require_codex_mode(result(composer, pull_requests))
        with self.assertRaises(CodexModeNotReady):
            require_codex_mode(result(composer))
        with self.assertRaises(CodexModeNotReady):
            require_codex_mode(
                result(
                    composer,
                    record(
                        "chat-mode",
                        control_type="Button",
                        name="ChatGPT",
                        class_name="product-switcher",
                        rectangle="10,10,100,40",
                    ),
                )
            )

    def test_find_voice_start_ignores_dictation_and_busy_actions(self) -> None:
        voice = record(
            "voice",
            control_type="Button",
            name="音声モードを開始",
            class_name="cursor-interaction size-token-button-composer rounded-full",
            rectangle="470,270,500,300",
        )
        microphone = record(
            "dictation",
            control_type="Button",
            name="音声入力",
            class_name="button",
            rectangle="430,270,460,300",
        )

        target = find_voice_start_target(result(record("composer"), microphone, voice))

        self.assertEqual(target.locator, "voice")
        with self.assertRaises(VoiceStartNotReady):
            find_voice_start_target(
                result(
                    record("composer"),
                    record(
                        "stop",
                        control_type="Button",
                        name="停止",
                        class_name=("cursor-interaction size-token-button-composer rounded-full"),
                        rectangle="470,270,500,300",
                    ),
                )
            )

    def test_find_voice_start_accepts_chat_mode_right_edge_button(self) -> None:
        voice = record(
            "chat-mode-voice",
            control_type="Button",
            name="会話を開始",
            class_name=(
                "no-drag cursor-interaction items-center rounded-full "
                "bg-token-foreground"
            ),
            rectangle="470,270,500,300",
        )
        microphone = record(
            "dictation",
            control_type="Button",
            name="音声入力",
            class_name="no-drag cursor-interaction items-center rounded-full",
            rectangle="430,270,460,300",
        )

        target = find_voice_start_target(result(record("composer"), microphone, voice))

        self.assertEqual(target.locator, "chat-mode-voice")

    def test_find_voice_start_accepts_controls_beyond_chat_mode_editor(self) -> None:
        voice = record(
            "chat-mode-voice",
            control_type="Button",
            name="会話を開始",
            class_name="no-drag cursor-interaction rounded-full",
            rectangle="536,270,566,300",
        )
        microphone = record(
            "dictation",
            control_type="Button",
            name="音声入力",
            class_name="no-drag cursor-interaction rounded-full",
            rectangle="500,270,530,300",
        )

        target = find_voice_start_target(
            result(
                record("composer", rectangle="100,200,400,260"),
                microphone,
                voice,
            )
        )

        self.assertEqual(target.locator, "chat-mode-voice")

    def test_find_voice_start_accepts_unnamed_right_edge_button(self) -> None:
        voice = record(
            "unnamed-voice",
            control_type="Button",
            name="",
            class_name="no-drag cursor-interaction rounded-full",
            rectangle="470,270,500,300",
        )

        target = find_voice_start_target(result(record("composer"), voice))

        self.assertEqual(target.locator, "unnamed-voice")

    def test_find_voice_start_rejects_unknown_named_right_edge_button(self) -> None:
        with self.assertRaises(VoiceStartNotReady):
            find_voice_start_target(
                result(
                    record("composer"),
                    record(
                        "unknown",
                        control_type="Button",
                        name="Open settings",
                        class_name="no-drag cursor-interaction rounded-full",
                        rectangle="470,270,500,300",
                    ),
                )
            )

    def test_wait_for_voice_ready_requires_start_button_transition(self) -> None:
        voice = record(
            "voice",
            control_type="Button",
            name="Start voice",
            class_name="size-token-button-composer rounded-full",
            rectangle="470,270,500,300",
        )
        stop = record(
            "stop",
            control_type="Button",
            name="Stop",
            class_name="size-token-button-composer rounded-full",
            rectangle="470,270,500,300",
        )
        provider = FakeProvider(
            [
                result(record("composer"), voice),
                result(record("voice-composer"), stop),
            ]
        )
        now = [0.0]

        target = wait_for_voice_ready_after_click(
            provider,
            wait_seconds=3.0,
            transition_delay_seconds=0.5,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

        self.assertEqual(target.locator, "voice-composer")

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

    def test_run_can_open_new_chat_start_voice_and_send(self) -> None:
        codex = record(
            "codex-mode",
            control_type="Button",
            name="Codex",
            class_name="product-switcher",
            rectangle="10,10,90,40",
        )
        new_chat = record(
            "new-chat",
            control_type="Button",
            name="New chat",
            class_name="button",
            rectangle="10,10,40,40",
        )
        voice = record(
            "voice",
            control_type="Button",
            name="Start voice",
            class_name="size-token-button-composer rounded-full",
            rectangle="470,270,500,300",
        )
        stop = record(
            "stop",
            control_type="Button",
            name="Stop",
            class_name="size-token-button-composer rounded-full",
            rectangle="470,270,500,300",
        )
        provider = FakeProvider(
            [
                result(record("old-composer"), codex, new_chat, voice),
                result(record("old-composer"), codex, new_chat, voice),
                result(record("new-composer"), voice),
                result(
                    record(
                        "voice-composer",
                        name="Work モードで作成",
                        class_name="voice-chat-composer",
                    ),
                    stop,
                ),
            ]
        )
        sender = FakeSender()
        clicker = FakeClicker()
        now = [0.0]

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system_prompt.txt"
            path.write_text("persona", encoding="utf-8")
            exit_code = run_prompt_injector(
                prompt_path=path,
                start_voice=True,
                require_codex=True,
                wait_seconds=3.0,
                voice_wait_seconds=3.0,
                voice_stabilization_seconds=5.0,
                provider=provider,
                sender=sender,
                clicker=clicker,
                clock=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual([target.locator for target in clicker.calls], ["new-chat", "voice"])
        self.assertEqual(sender.calls[0][0].locator, "voice-composer")
        self.assertEqual(sender.calls[0][1:], ("persona", "enter"))
        self.assertGreaterEqual(now[0], 6.25)


if __name__ == "__main__":
    unittest.main()
