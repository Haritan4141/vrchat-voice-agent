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
)

DEFAULT_PROMPT_PATH = Path("system_prompt.txt")
DEFAULT_WAIT_SECONDS = 15.0
MAX_PROMPT_BYTES = 64 * 1024
COMPOSER_NAMES = (
    "何でもどうぞ",
    "chatgpt にメッセージ",
    "chatgptにメッセージ",
    "message chatgpt",
)


class PromptInjectionError(RuntimeError):
    """Raised when a prompt cannot be sent without risking the wrong target."""


class ComposerNotReady(PromptInjectionError):
    """Raised when ChatGPT is not yet exposing a usable composer."""


@dataclass(frozen=True, slots=True)
class PromptTarget:
    process_id: int
    window_handle: int
    window_title: str
    rectangle: tuple[int, int, int, int]
    locator: str


class PromptSender(Protocol):
    def send(self, target: PromptTarget, prompt: str, submit_key: str) -> None: ...


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
        target = PromptTarget(
            process_id=record.process_id,
            window_handle=record.window_handle,
            window_title=record.window_title,
            rectangle=parse_rectangle(record.rectangle),
            locator=record.locator,
        )
        targets[(target.window_handle, record.rectangle)] = target

    if len(targets) != 1:
        raise PromptInjectionError(
            "More than one ChatGPT message box is visible. Close the extra window or tab "
            "before sending the private prompt."
        )
    return next(iter(targets.values()))


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

    def send(self, target: PromptTarget, prompt: str, submit_key: str) -> None:
        if platform.system() != "Windows":
            raise PromptInjectionError("Prompt injection is available only on Windows.")
        if submit_key not in {"enter", "ctrl-enter", "none"}:
            raise PromptInjectionError(f"Unsupported submit key: {submit_key}")

        try:
            import pythoncom
            import win32clipboard
            import win32con
            import win32gui
            from pywinauto import keyboard, mouse
        except ImportError as exc:  # pragma: no cover - exercised only on a broken install
            raise PromptInjectionError(
                "pywinauto is not installed. Run 'uv sync' in the project folder."
            ) from exc

        pythoncom.CoInitialize()
        original_clipboard = None
        original_available = False
        try:
            try:
                original_clipboard = pythoncom.OleGetClipboard()
                original_available = original_clipboard is not None
            except Exception:  # noqa: BLE001 - an empty clipboard can raise a COM error
                original_clipboard = None

            self._put_prompt_on_clipboard(win32clipboard, prompt)
            self._activate_window(win32con, win32gui, target.window_handle)

            left, top, right, bottom = target.rectangle
            point = ((left + right) // 2, (top + bottom) // 2)
            point_window = win32gui.WindowFromPoint(point)
            root_window = win32gui.GetAncestor(point_window, win32con.GA_ROOT)
            if root_window != target.window_handle:
                raise PromptInjectionError(
                    "The ChatGPT message box is covered by another window; nothing was pasted."
                )

            mouse.click(coords=point)
            time.sleep(0.15)
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
            try:
                if original_available:
                    pythoncom.OleSetClipboard(original_clipboard)
                    pythoncom.OleFlushClipboard()
                else:
                    self._clear_clipboard(win32clipboard)
            finally:
                pythoncom.CoUninitialize()


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


def run_prompt_injector(
    *,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    submit_key: str = "enter",
    process_names: Iterable[str] = DEFAULT_PROCESS_NAMES,
    dry_run: bool = False,
    provider: SnapshotProvider | None = None,
    sender: PromptSender | None = None,
) -> int:
    prompt = load_prompt(prompt_path)
    scanner = provider or PywinautoSnapshotProvider(process_names, include_offscreen=False)

    print("ChatGPT Voice Prompt Injector")
    print(f"- Prompt file: {prompt_path.expanduser().resolve()}")
    print("- Privacy: prompt contents are not printed or logged.")
    print("- Target: the single visible ChatGPT message box")
    print("- Use only after a new GPT Live voice task has started.")
    print()
    print("Looking for the ChatGPT message box...")
    target = wait_for_prompt_target(scanner, wait_seconds=wait_seconds)
    if dry_run:
        print("Ready: one ChatGPT message box was found. No input was sent.")
        return 0

    (sender or WindowsPromptSender()).send(target, prompt, submit_key)
    if submit_key == "none":
        print("Prompt pasted but not submitted.")
    else:
        print("Prompt sent. Wait for ChatGPT's configured ready confirmation.")
    return 0
