from __future__ import annotations

import platform
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .chatgpt_ui_diagnostic import (
    DEFAULT_PROCESS_NAMES,
    PywinautoSnapshotProvider,
    SnapshotProvider,
    UiElementRecord,
    UiScanResult,
    windows_com_apartment,
)

DEFAULT_PROMPT_PATH = Path("system_prompt.txt")
DEFAULT_WAIT_SECONDS = 15.0
DEFAULT_VOICE_WAIT_SECONDS = 45.0
DEFAULT_VOICE_STABILIZATION_SECONDS = 5.0
MAX_PROMPT_BYTES = 64 * 1024
COMPOSER_NAMES = (
    "何でもどうぞ",
    "chatgpt にメッセージ",
    "chatgptにメッセージ",
    "message chatgpt",
    "work モードで作成",
    "workモードで作成",
    "create in work mode",
)
NEW_CHAT_NAMES = (
    "新しいチャット",
    "new chat",
)
VOICE_NAME_MARKERS = (
    "音声",
    "会話",
    "話す",
    "ライブ",
    "voice",
    "live",
)
BLOCKED_COMPOSER_ACTION_NAMES = (
    "停止",
    "送信",
    "音声入力",
    "stop",
    "send",
    "voice input",
)
VOICE_BUTTON_CLASS_FRAGMENT = "size-token-button-composer"
COMPOSER_ACTION_MAX_RIGHT_GAP = 240


class PromptInjectionError(RuntimeError):
    """Raised when a prompt cannot be sent without risking the wrong target."""


class ComposerNotReady(PromptInjectionError):
    """Raised when ChatGPT is not yet exposing a usable composer."""


class VoiceStartNotReady(PromptInjectionError):
    """Raised when a safe GPT Live start button is not yet available."""


class CodexModeNotReady(PromptInjectionError):
    """Raised when the desktop app is not visibly in Codex mode."""


@dataclass(frozen=True, slots=True)
class PromptTarget:
    process_id: int
    window_handle: int
    window_title: str
    rectangle: tuple[int, int, int, int]
    locator: str


class PromptSender(Protocol):
    def send(self, target: PromptTarget, prompt: str, submit_key: str) -> None: ...


class UiClicker(Protocol):
    def click(self, target: PromptTarget) -> None: ...


def load_prompt(path: Path, *, max_bytes: int = MAX_PROMPT_BYTES) -> str:
    """Read a private prompt without ever printing or logging its contents."""
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except FileNotFoundError as exc:
        raise PromptInjectionError(f"Prompt file was not found: {resolved}") from exc
    if size > max_bytes:
        raise PromptInjectionError(
            f"Prompt file is too large ({size} bytes; maximum is {max_bytes} bytes)."
        )
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as prompt_file:
            prompt = prompt_file.read()
    except UnicodeDecodeError as exc:
        raise PromptInjectionError(f"Prompt file must be UTF-8: {resolved}") from exc
    if not prompt.strip():
        raise PromptInjectionError(f"Prompt file is empty: {resolved}")
    return prompt


def parse_rectangle(value: str) -> tuple[int, int, int, int]:
    try:
        rectangle = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise PromptInjectionError(f"Invalid UI rectangle: {value!r}") from exc
    if len(rectangle) != 4:
        raise PromptInjectionError(f"Invalid UI rectangle: {value!r}")
    left, top, right, bottom = rectangle
    if right <= left or bottom <= top:
        raise PromptInjectionError(f"Empty UI rectangle: {value!r}")
    return left, top, right, bottom


def _is_composer(record: UiElementRecord) -> bool:
    if record.control_type.casefold() != "edit":
        return False
    if record.is_enabled is False or record.is_offscreen is True:
        return False
    name = " ".join(record.name.casefold().split())
    class_name = record.class_name.casefold()
    return "prosemirror" in class_name or name in COMPOSER_NAMES


def _target_from_record(record: UiElementRecord) -> PromptTarget:
    return PromptTarget(
        process_id=record.process_id,
        window_handle=record.window_handle,
        window_title=record.window_title,
        rectangle=parse_rectangle(record.rectangle),
        locator=record.locator,
    )


def _unique_target(records: list[UiElementRecord], *, description: str) -> PromptTarget:
    targets: dict[tuple[int, str], PromptTarget] = {}
    for record in records:
        target = _target_from_record(record)
        targets[(target.window_handle, record.rectangle)] = target
    if not targets:
        raise VoiceStartNotReady(f"No safe {description} button is available yet.")
    if len(targets) != 1:
        raise PromptInjectionError(
            f"More than one {description} button is visible. Close the extra ChatGPT "
            "window before continuing."
        )
    return next(iter(targets.values()))


def find_prompt_target(result: UiScanResult) -> PromptTarget:
    candidates = [record for record in result.elements.values() if _is_composer(record)]
    if not candidates:
        if not result.process_ids:
            raise ComposerNotReady("ChatGPT.exe is not running.")
        if result.window_count == 0:
            raise ComposerNotReady("No visible ChatGPT window is available.")
        raise ComposerNotReady(
            "The ChatGPT message box is not ready. Open the GPT Live task and try again."
        )

    targets: dict[tuple[int, str], PromptTarget] = {}
    for record in candidates:
        target = _target_from_record(record)
        targets[(target.window_handle, record.rectangle)] = target

    if len(targets) != 1:
        raise PromptInjectionError(
            "More than one ChatGPT message box is visible. Close the extra window or tab "
            "before sending the private prompt."
        )
    return next(iter(targets.values()))


def find_new_chat_target(result: UiScanResult) -> PromptTarget:
    composer = find_prompt_target(result)
    records = [
        record
        for record in result.elements.values()
        if record.window_handle == composer.window_handle
        and record.control_type.casefold() == "button"
        and " ".join(record.name.casefold().split()) in NEW_CHAT_NAMES
        and record.is_enabled is not False
        and record.is_offscreen is not True
    ]

    sidebar_targets = [
        _target_from_record(record)
        for record in records
        if parse_rectangle(record.rectangle)[2] <= composer.rectangle[0]
    ]
    if sidebar_targets:
        return min(
            sidebar_targets,
            key=lambda target: (
                target.rectangle[1],
                target.rectangle[0],
                (target.rectangle[2] - target.rectangle[0])
                * (target.rectangle[3] - target.rectangle[1]),
            ),
        )

    return _unique_target(records, description="new chat")


def require_codex_mode(result: UiScanResult) -> None:
    composer = find_prompt_target(result)
    for record in result.elements.values():
        if record.window_handle != composer.window_handle:
            continue
        if record.control_type.casefold() != "button":
            continue
        if record.is_enabled is False or record.is_offscreen is True:
            continue
        if " ".join(record.name.casefold().split()) != "codex":
            continue
        try:
            _left, top, right, _bottom = parse_rectangle(record.rectangle)
        except PromptInjectionError:
            continue
        if right <= composer.rectangle[0] and top < composer.rectangle[1]:
            return
    raise CodexModeNotReady(
        "Codex mode was not detected. Select Codex from the upper-left product menu "
        "in the ChatGPT desktop app, then run the launcher again."
    )


def _is_near_composer(
    button_rectangle: tuple[int, int, int, int],
    composer_rectangle: tuple[int, int, int, int],
) -> bool:
    button_left, button_top, button_right, button_bottom = button_rectangle
    _composer_left, composer_top, composer_right, composer_bottom = composer_rectangle
    return (
        button_left >= composer_right - 100
        # Chat mode can expose the ProseMirror edit rectangle without the
        # model, dictation, and Voice controls that share its visual container.
        and button_right <= composer_right + COMPOSER_ACTION_MAX_RIGHT_GAP
        and button_top >= composer_top - 20
        and button_bottom <= composer_bottom + 80
    )


def _is_compact_right_edge_composer_action(
    button_rectangle: tuple[int, int, int, int],
    composer_rectangle: tuple[int, int, int, int],
) -> bool:
    button_left, button_top, button_right, button_bottom = button_rectangle
    _composer_left, composer_top, composer_right, composer_bottom = composer_rectangle
    width = button_right - button_left
    height = button_bottom - button_top
    return (
        16 <= width <= 64
        and 16 <= height <= 64
        and button_left >= composer_right - 80
        and button_right <= composer_right + COMPOSER_ACTION_MAX_RIGHT_GAP
        and button_top >= composer_top - 20
        and button_bottom <= composer_bottom + 80
    )


def find_voice_start_target(result: UiScanResult) -> PromptTarget:
    composer = find_prompt_target(result)
    records: list[UiElementRecord] = []
    blocked_action_found = False
    for record in result.elements.values():
        if record.window_handle != composer.window_handle:
            continue
        if record.control_type.casefold() != "button":
            continue
        if record.is_enabled is False or record.is_offscreen is True:
            continue
        try:
            rectangle = parse_rectangle(record.rectangle)
        except PromptInjectionError:
            continue
        if not _is_near_composer(rectangle, composer.rectangle):
            continue
        name = " ".join(record.name.casefold().split())
        if name in BLOCKED_COMPOSER_ACTION_NAMES:
            blocked_action_found = True
            continue
        class_name = record.class_name.casefold()
        has_voice_name = any(marker in name for marker in VOICE_NAME_MARKERS)
        has_known_class = VOICE_BUTTON_CLASS_FRAGMENT in class_name
        is_unnamed_right_edge_action = (
            not name
            and _is_compact_right_edge_composer_action(rectangle, composer.rectangle)
        )
        if not (
            has_voice_name
            or (has_known_class and not name)
            or is_unnamed_right_edge_action
        ):
            continue
        records.append(record)

    if not records and blocked_action_found:
        raise VoiceStartNotReady(
            "The composer action button is busy or is not in Voice-start state yet."
        )
    return _unique_target(records, description="GPT Live start")


class WindowsUiClicker:
    """Activate one verified ChatGPT window and click an exact screen rectangle."""

    @staticmethod
    def _activate_window(win32con, win32gui, window_handle: int) -> None:
        if not win32gui.IsWindow(window_handle):
            raise PromptInjectionError("The selected ChatGPT window was closed.")
        if win32gui.IsIconic(window_handle):
            win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
        try:
            win32gui.BringWindowToTop(window_handle)
            win32gui.SetForegroundWindow(window_handle)
        except Exception as exc:
            raise PromptInjectionError("Could not focus the ChatGPT window.") from exc

        deadline = time.monotonic() + 2.0
        while win32gui.GetForegroundWindow() != window_handle:
            if time.monotonic() >= deadline:
                raise PromptInjectionError("ChatGPT did not become the foreground window.")
            time.sleep(0.05)

    def click(self, target: PromptTarget) -> None:
        if platform.system() != "Windows":
            raise PromptInjectionError("ChatGPT UI control is available only on Windows.")
        try:
            import win32con
            import win32gui
            from pywinauto import mouse
        except ImportError as exc:  # pragma: no cover - exercised only on a broken install
            raise PromptInjectionError(
                "pywinauto is not installed. Run 'uv sync' in the project folder."
            ) from exc

        self._activate_window(win32con, win32gui, target.window_handle)
        left, top, right, bottom = target.rectangle
        point = ((left + right) // 2, (top + bottom) // 2)
        point_window = win32gui.WindowFromPoint(point)
        root_window = win32gui.GetAncestor(point_window, win32con.GA_ROOT)
        if root_window != target.window_handle:
            raise PromptInjectionError(
                "The selected ChatGPT control is covered by another window; nothing was clicked."
            )
        mouse.click(coords=point)
        time.sleep(0.15)
        if win32gui.GetForegroundWindow() != target.window_handle:
            raise PromptInjectionError("ChatGPT lost focus immediately after the click.")


class WindowsPromptSender:
    """Paste into one verified ChatGPT composer and optionally submit it."""

    @staticmethod
    def _open_clipboard(win32clipboard, deadline: float) -> None:
        while True:
            try:
                win32clipboard.OpenClipboard()
                return
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise PromptInjectionError("The Windows clipboard is busy.") from exc
                time.sleep(0.05)

    def _put_prompt_on_clipboard(self, win32clipboard, prompt: str) -> None:
        self._open_clipboard(win32clipboard, time.monotonic() + 2.0)
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, prompt)
        finally:
            win32clipboard.CloseClipboard()

    def _clear_clipboard(self, win32clipboard) -> None:
        self._open_clipboard(win32clipboard, time.monotonic() + 2.0)
        try:
            win32clipboard.EmptyClipboard()
        finally:
            win32clipboard.CloseClipboard()

    def _restore_or_clear_clipboard(
        self,
        pythoncom,
        win32clipboard,
        original_clipboard,
        original_available: bool,
    ) -> bool:
        if original_available:
            try:
                pythoncom.OleSetClipboard(original_clipboard)
                pythoncom.OleFlushClipboard()
                return True
            except Exception:  # noqa: BLE001 - clipboard providers can reject OLE restore
                self._clear_clipboard(win32clipboard)
                return False
        self._clear_clipboard(win32clipboard)
        return True

    def send(self, target: PromptTarget, prompt: str, submit_key: str) -> None:
        if platform.system() != "Windows":
            raise PromptInjectionError("Prompt injection is available only on Windows.")
        if submit_key not in {"enter", "ctrl-enter", "none"}:
            raise PromptInjectionError(f"Unsupported submit key: {submit_key}")

        try:
            import pythoncom
            import win32clipboard
            import win32gui
            from pywinauto import keyboard
        except ImportError as exc:  # pragma: no cover - exercised only on a broken install
            raise PromptInjectionError(
                "pywinauto is not installed. Run 'uv sync' in the project folder."
            ) from exc

        with windows_com_apartment():
            original_clipboard = None
            original_available = False
            try:
                try:
                    original_clipboard = pythoncom.OleGetClipboard()
                    original_available = original_clipboard is not None
                except Exception:  # noqa: BLE001 - an empty clipboard can raise a COM error
                    original_clipboard = None

                self._put_prompt_on_clipboard(win32clipboard, prompt)
                WindowsUiClicker().click(target)
                if win32gui.GetForegroundWindow() != target.window_handle:
                    raise PromptInjectionError(
                        "ChatGPT lost focus before pasting; nothing was submitted."
                    )
                keyboard.send_keys("^v", pause=0.02)
                time.sleep(0.35)

                if submit_key != "none":
                    if win32gui.GetForegroundWindow() != target.window_handle:
                        raise PromptInjectionError(
                            "ChatGPT lost focus after pasting; the prompt was not submitted."
                        )
                    keys = "{ENTER}" if submit_key == "enter" else "^{ENTER}"
                    keyboard.send_keys(keys, pause=0.02)
                    time.sleep(0.2)
            finally:
                restored = self._restore_or_clear_clipboard(
                    pythoncom,
                    win32clipboard,
                    original_clipboard,
                    original_available,
                )
                if original_available and not restored:
                    print(
                        "Warning: the prompt was submitted, but the previous clipboard "
                        "could not be restored. The clipboard was cleared for privacy."
                    )
                # Release the OLE proxy before leaving the initialized COM apartment.
                original_clipboard = None


def wait_for_prompt_target(
    provider: SnapshotProvider,
    *,
    wait_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PromptTarget:
    if wait_seconds < 0:
        raise PromptInjectionError("wait_seconds must be zero or greater.")
    deadline = clock() + wait_seconds
    last_error: ComposerNotReady | None = None
    while True:
        result = provider.scan()
        try:
            return find_prompt_target(result)
        except ComposerNotReady as exc:
            last_error = exc
        now = clock()
        if now >= deadline:
            assert last_error is not None
            raise last_error
        sleep(min(0.5, deadline - now))


def _wait_for_ui_target(
    provider: SnapshotProvider,
    finder: Callable[[UiScanResult], PromptTarget],
    *,
    wait_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PromptTarget:
    if wait_seconds < 0:
        raise PromptInjectionError("wait_seconds must be zero or greater.")
    deadline = clock() + wait_seconds
    last_error: PromptInjectionError | None = None
    while True:
        result = provider.scan()
        try:
            return finder(result)
        except (ComposerNotReady, VoiceStartNotReady) as exc:
            last_error = exc
        now = clock()
        if now >= deadline:
            assert last_error is not None
            raise last_error
        sleep(min(0.5, deadline - now))


def wait_for_voice_ready_after_click(
    provider: SnapshotProvider,
    *,
    wait_seconds: float,
    transition_delay_seconds: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PromptTarget:
    if wait_seconds <= 0:
        raise PromptInjectionError("voice_wait_seconds must be greater than zero.")
    started = clock()
    deadline = started + wait_seconds
    voice_transition_seen = False
    while True:
        result = provider.scan()
        now = clock()
        if now - started >= max(0.0, transition_delay_seconds):
            try:
                find_voice_start_target(result)
            except (ComposerNotReady, VoiceStartNotReady):
                voice_transition_seen = True

        if voice_transition_seen:
            try:
                return find_prompt_target(result)
            except ComposerNotReady:
                pass

        if now >= deadline:
            raise VoiceStartNotReady(
                "GPT Live did not become ready before the timeout. Check ChatGPT for a "
                "microphone permission or setup dialog."
            )
        sleep(min(0.5, deadline - now))


def run_prompt_injector(
    *,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    submit_key: str = "enter",
    process_names: Iterable[str] = DEFAULT_PROCESS_NAMES,
    dry_run: bool = False,
    start_voice: bool = False,
    require_codex: bool = False,
    voice_wait_seconds: float = DEFAULT_VOICE_WAIT_SECONDS,
    voice_stabilization_seconds: float = DEFAULT_VOICE_STABILIZATION_SECONDS,
    provider: SnapshotProvider | None = None,
    sender: PromptSender | None = None,
    clicker: UiClicker | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    if voice_stabilization_seconds < 0:
        raise PromptInjectionError(
            "voice_stabilization_seconds must be zero or greater."
        )
    prompt = load_prompt(prompt_path)
    scanner = provider or PywinautoSnapshotProvider(process_names, include_offscreen=False)

    print("ChatGPT Voice Prompt Injector")
    print(f"- Prompt file: {prompt_path.expanduser().resolve()}")
    print("- Privacy: prompt contents are not printed or logged.")
    print("- Target: the single visible ChatGPT message box")
    if start_voice:
        if require_codex:
            print("- Mode: require Codex, open a new task, start GPT Live, then apply the prompt")
        else:
            print("- Mode: open a new task, start GPT Live, then apply the prompt")
    else:
        print("- Mode: apply to an already-started GPT Live voice task")
    print()

    if start_voice:
        if require_codex:
            print("Checking for Codex mode...")
            require_codex_mode(scanner.scan())
        print("Looking for the New chat button...")
        new_chat_target = _wait_for_ui_target(
            scanner,
            find_new_chat_target,
            wait_seconds=wait_seconds,
            clock=clock,
            sleep=sleep,
        )
        if dry_run:
            initial_scan = scanner.scan()
            find_voice_start_target(initial_scan)
            print("Ready: New chat and GPT Live start controls were found. No input was sent.")
            return 0

        ui_clicker = clicker or WindowsUiClicker()
        ui_clicker.click(new_chat_target)
        sleep(0.75)
        print("Looking for the GPT Live start button...")
        voice_start_target = _wait_for_ui_target(
            scanner,
            find_voice_start_target,
            wait_seconds=wait_seconds,
            clock=clock,
            sleep=sleep,
        )
        ui_clicker.click(voice_start_target)
        print("GPT Live start requested. Waiting for the voice task...")
        target = wait_for_voice_ready_after_click(
            scanner,
            wait_seconds=voice_wait_seconds,
            clock=clock,
            sleep=sleep,
        )
        if voice_stabilization_seconds > 0:
            print(
                "GPT Live UI is visible. Waiting "
                f"{voice_stabilization_seconds:g} seconds for the session to stabilize..."
            )
            sleep(voice_stabilization_seconds)
    else:
        print("Looking for the ChatGPT message box...")
        target = wait_for_prompt_target(
            scanner,
            wait_seconds=wait_seconds,
            clock=clock,
            sleep=sleep,
        )
        if dry_run:
            print("Ready: one ChatGPT message box was found. No input was sent.")
            return 0

    (sender or WindowsPromptSender()).send(target, prompt, submit_key)
    if submit_key == "none":
        print("Prompt pasted but not submitted.")
    else:
        print("Prompt sent. Wait for ChatGPT's configured ready confirmation.")
    return 0
