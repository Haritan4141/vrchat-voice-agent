from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import psutil

DEFAULT_PROCESS_NAMES = ("ChatGPT.exe",)
DEFAULT_KEYWORDS = (
    "検索",
    "調べ",
    "作業",
    "考え",
    "処理",
    "search",
    "brows",
    "working",
    "thinking",
    "processing",
    "stop",
    "cancel",
    "停止",
    "キャンセル",
)


def normalize_text(value: object, max_length: int = 240) -> str:
    """Make UI Automation text safe to print as a one-line log field."""
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 1)] + "…"


@dataclass(frozen=True, slots=True)
class UiElementRecord:
    locator: str
    process_id: int
    window_handle: int
    window_title: str
    control_type: str
    name: str
    automation_id: str
    class_name: str
    is_enabled: bool | None
    is_offscreen: bool | None
    rectangle: str

    def comparable(self) -> tuple[object, ...]:
        # The window title is intentionally omitted. ChatGPT may update the title without
        # changing every descendant control.
        return (
            self.control_type,
            self.name,
            self.automation_id,
            self.class_name,
            self.is_enabled,
            self.is_offscreen,
            self.rectangle,
        )


@dataclass(frozen=True, slots=True)
class UiScanResult:
    process_ids: tuple[int, ...]
    window_count: int
    elements: dict[str, UiElementRecord]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UiChange:
    kind: str
    before: UiElementRecord | None
    after: UiElementRecord | None

    @property
    def record(self) -> UiElementRecord:
        record = self.after or self.before
        if record is None:  # pragma: no cover - protected by construction
            raise RuntimeError("UI change has no element record")
        return record


class SnapshotProvider(Protocol):
    def scan(self) -> UiScanResult: ...


@contextmanager
def windows_com_apartment() -> Iterator[None]:
    """Keep COM initialized for one Windows UI Automation operation."""
    if platform.system() != "Windows":
        yield
        return

    try:
        import pythoncom
        import winerror
    except ImportError as exc:  # pragma: no cover - pywinauto installs pywin32
        raise RuntimeError("pywin32 is required for ChatGPT UI Automation.") from exc

    initialized = False
    try:
        try:
            mode = getattr(sys, "coinit_flags", pythoncom.COINIT_MULTITHREADED)
            pythoncom.CoInitializeEx(mode)
            initialized = True
        except pythoncom.com_error as exc:
            if exc.hresult != winerror.RPC_E_CHANGED_MODE:
                raise RuntimeError("Could not initialize Windows COM automation.") from exc
            # The calling thread already has another valid COM apartment model.
        yield
    finally:
        if initialized:
            pythoncom.CoUninitialize()


def diff_snapshots(
    before: dict[str, UiElementRecord],
    after: dict[str, UiElementRecord],
) -> tuple[UiChange, ...]:
    changes: list[UiChange] = []
    before_keys = set(before)
    after_keys = set(after)

    for locator in sorted(after_keys - before_keys):
        changes.append(UiChange("added", None, after[locator]))
    for locator in sorted(before_keys & after_keys):
        if before[locator].comparable() != after[locator].comparable():
            changes.append(UiChange("changed", before[locator], after[locator]))
    for locator in sorted(before_keys - after_keys):
        changes.append(UiChange("removed", before[locator], None))
    return tuple(changes)


def is_candidate(record: UiElementRecord, keywords: Iterable[str] = DEFAULT_KEYWORDS) -> bool:
    haystack = (
        f"{record.name} {record.automation_id} {record.control_type} {record.class_name}"
    ).casefold()
    return any(keyword.casefold() in haystack for keyword in keywords if keyword)


def _safe_value(callable_, default=None):
    try:
        return callable_()
    except Exception:  # noqa: BLE001 - UIA providers raise several COM exception types
        return default


def _safe_attr(value: object, attribute: str, default=None):
    return _safe_value(lambda: getattr(value, attribute), default)


def _rectangle_text(rectangle: object) -> str:
    if rectangle is None:
        return ""
    left = getattr(rectangle, "left", None)
    top = getattr(rectangle, "top", None)
    right = getattr(rectangle, "right", None)
    bottom = getattr(rectangle, "bottom", None)
    if None not in (left, top, right, bottom):
        return f"{left},{top},{right},{bottom}"
    return normalize_text(rectangle, 80)


def _runtime_id_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return normalize_text(value, 120)


class PywinautoSnapshotProvider:
    """Read ChatGPT's Windows accessibility tree without sending any input."""

    def __init__(
        self,
        process_names: Iterable[str] = DEFAULT_PROCESS_NAMES,
        *,
        include_offscreen: bool = False,
        name_max_length: int = 240,
    ) -> None:
        self.process_names = {name.casefold() for name in process_names if name.strip()}
        self.include_offscreen = include_offscreen
        self.name_max_length = max(80, int(name_max_length))

    def _process_ids(self) -> tuple[int, ...]:
        result: list[int] = []
        for process in psutil.process_iter(("pid", "name")):
            try:
                name = str(process.info.get("name") or "").casefold()
                if name in self.process_names:
                    result.append(int(process.info["pid"]))
            except (psutil.Error, OSError, ValueError):
                continue
        return tuple(sorted(result))

    @staticmethod
    def _top_level_windows(process_ids: tuple[int, ...]) -> tuple[tuple[int, int, str], ...]:
        """Find native top-level windows without walking the complete UIA desktop tree."""
        try:
            import win32gui
            import win32process
        except ImportError as exc:  # pragma: no cover - pywinauto installs pywin32
            raise RuntimeError("pywin32 is required for ChatGPT window discovery.") from exc

        wanted = set(process_ids)
        result: list[tuple[int, int, str]] = []

        def visit(handle: int, _extra: object) -> None:
            try:
                _thread_id, process_id = win32process.GetWindowThreadProcessId(handle)
                if process_id not in wanted:
                    return
                if win32gui.GetParent(handle):
                    return
                if not win32gui.IsWindowVisible(handle):
                    return
                title = normalize_text(win32gui.GetWindowText(handle))
                result.append((int(process_id), int(handle), title))
            except Exception:  # noqa: BLE001 - EnumWindows callback must not abort enumeration
                return

        win32gui.EnumWindows(visit, None)
        return tuple(sorted(set(result)))

    def scan(self) -> UiScanResult:
        if platform.system() != "Windows":
            raise RuntimeError("ChatGPT UI Automation diagnostic is available only on Windows.")
        with windows_com_apartment():
            return self._scan_with_com()

    def _scan_with_com(self) -> UiScanResult:
        try:
            from pywinauto.uia_defines import IUIA
        except ImportError as exc:  # pragma: no cover - exercised only on a broken install
            raise RuntimeError(
                "pywinauto is not installed. Run 'uv sync' in the project folder."
            ) from exc

        process_ids = self._process_ids()
        elements: dict[str, UiElementRecord] = {}
        errors: list[str] = []
        top_level_windows = self._top_level_windows(process_ids)
        uia = IUIA()
        dll = uia.UIA_dll
        cache_request = uia.iuia.CreateCacheRequest()
        for property_id in (
            dll.UIA_NamePropertyId,
            dll.UIA_ControlTypePropertyId,
            dll.UIA_AutomationIdPropertyId,
            dll.UIA_ClassNamePropertyId,
            dll.UIA_IsEnabledPropertyId,
            dll.UIA_IsOffscreenPropertyId,
            dll.UIA_BoundingRectanglePropertyId,
            dll.UIA_RuntimeIdPropertyId,
        ):
            cache_request.AddProperty(property_id)
        cache_request.TreeScope = dll.TreeScope_Subtree

        for process_id, window_handle, native_title in top_level_windows:
            try:
                window = uia.iuia.ElementFromHandle(window_handle)
                cached_elements = window.FindAllBuildCache(
                    dll.TreeScope_Subtree,
                    uia.true_condition,
                    cache_request,
                )
            except Exception as exc:  # noqa: BLE001 - preserve provider-specific COM errors
                errors.append(
                    f"PID {process_id} HWND {window_handle}: UIA subtree lookup failed: {exc}"
                )
                continue
            window_title = native_title

            for tree_index in range(int(cached_elements.Length)):
                info = _safe_value(
                    lambda index=tree_index, source=cached_elements: source.GetElement(index)
                )
                if info is None:
                    continue
                control_type_id = int(_safe_attr(info, "CachedControlType", 0) or 0)
                control_type = normalize_text(
                    uia.known_control_type_ids.get(control_type_id, str(control_type_id)),
                    80,
                )
                name = normalize_text(
                    _safe_attr(info, "CachedName", ""),
                    self.name_max_length,
                )
                automation_id = normalize_text(
                    _safe_attr(info, "CachedAutomationId", ""), 160
                )
                class_name = normalize_text(
                    _safe_attr(info, "CachedClassName", ""), 160
                )
                cached_enabled = _safe_attr(info, "CachedIsEnabled")
                cached_offscreen = _safe_attr(info, "CachedIsOffscreen")
                is_enabled = None if cached_enabled is None else bool(cached_enabled)
                is_offscreen = None if cached_offscreen is None else bool(cached_offscreen)
                if not self.include_offscreen and is_offscreen is True:
                    continue
                rectangle = _rectangle_text(_safe_attr(info, "CachedBoundingRectangle"))
                runtime_id = _runtime_id_text(
                    _safe_value(
                        lambda element=info, property_id=dll.UIA_RuntimeIdPropertyId: (
                            element.GetCachedPropertyValue(property_id)
                        )
                    )
                )
                if runtime_id:
                    locator_suffix = f"runtime:{runtime_id}"
                else:
                    locator_suffix = (
                        f"fallback:{tree_index}:{control_type}:{automation_id}:"
                        f"{class_name}:{rectangle}"
                    )
                locator = f"{process_id}:{window_handle}:{locator_suffix}"
                elements[locator] = UiElementRecord(
                    locator=locator,
                    process_id=process_id,
                    window_handle=window_handle,
                    window_title=window_title,
                    control_type=control_type,
                    name=name,
                    automation_id=automation_id,
                    class_name=class_name,
                    is_enabled=is_enabled,
                    is_offscreen=is_offscreen,
                    rectangle=rectangle,
                )

        return UiScanResult(
            process_ids=process_ids,
            window_count=len(top_level_windows),
            elements=elements,
            errors=tuple(errors),
        )


def default_output_path(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(timezone.utc).astimezone()).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    return Path("artifacts") / f"chatgpt-ui-diagnostic-{timestamp}.jsonl"


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _write_event(file, event: dict[str, object]) -> None:
    file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    file.flush()


def _console_change(change: UiChange, candidate: bool) -> str:
    marker = {"added": "+", "changed": "~", "removed": "-"}.get(change.kind, "?")
    record = change.record
    candidate_text = " [CANDIDATE]" if candidate else ""
    details = [f"type={record.control_type or '-'}"]
    if record.name:
        details.append(f"name={json.dumps(record.name, ensure_ascii=False)}")
    if record.automation_id:
        details.append(f"id={json.dumps(record.automation_id, ensure_ascii=False)}")
    if record.class_name:
        details.append(f"class={json.dumps(record.class_name, ensure_ascii=False)}")
    if (
        change.kind == "changed"
        and change.before is not None
        and change.after is not None
        and change.before.name != change.after.name
    ):
        details.append("before_name=" + json.dumps(change.before.name, ensure_ascii=False))
    return f"{marker}{candidate_text} " + " ".join(details)


def run_ui_diagnostic(
    *,
    duration_seconds: float = 180.0,
    interval_seconds: float = 0.5,
    output_path: Path | None = None,
    process_names: Iterable[str] = DEFAULT_PROCESS_NAMES,
    include_offscreen: bool = False,
    show_initial: bool = False,
    keywords: Iterable[str] = DEFAULT_KEYWORDS,
    provider: SnapshotProvider | None = None,
) -> int:
    if interval_seconds < 0.2:
        raise ValueError("interval_seconds must be at least 0.2")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be zero (unlimited) or greater")

    output = (output_path or default_output_path()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(process_names)
    keyword_list = tuple(keywords)
    scanner = provider or PywinautoSnapshotProvider(
        names,
        include_offscreen=include_offscreen,
    )

    print("ChatGPT UI Automation Diagnostic")
    print("- Read-only: no click, keyboard input, or ChatGPT operation is performed.")
    print(f"- Process: {', '.join(names)}")
    print(f"- Output: {output}")
    print("- Privacy: visible conversation text may be included in this local log.")
    if duration_seconds:
        print(f"- Duration: {duration_seconds:g} seconds (Ctrl+C to finish early)")
    else:
        print("- Duration: unlimited (Ctrl+C to finish)")
    print()
    print("Start Voice after this message, then try a normal reply and a Web search.")
    print("Candidate status elements are marked [CANDIDATE].")
    print()

    previous: dict[str, UiElementRecord] = {}
    previous_errors: tuple[str, ...] = ()
    baseline_ready = False
    found_window = False
    started = time.monotonic()
    next_notice = started
    next_heartbeat = started
    stop_reason = "duration_complete"

    # A reused --output path must fail instead of overwriting an earlier diagnosis or
    # allowing two diagnostic processes to interleave writes into invalid JSONL.
    with output.open("x", encoding="utf-8", newline="\n") as log_file:
        _write_event(
            log_file,
            {
                "event": "session_start",
                "timestamp": _timestamp(),
                "pid": os.getpid(),
                "process_names": names,
                "duration_seconds": duration_seconds,
                "interval_seconds": interval_seconds,
                "include_offscreen": include_offscreen,
                "show_initial": show_initial,
                "keywords": keyword_list,
            },
        )
        try:
            while duration_seconds == 0 or time.monotonic() - started < duration_seconds:
                loop_started = time.monotonic()
                result = scanner.scan()
                now = time.monotonic()

                if result.errors != previous_errors:
                    for error in result.errors:
                        print(f"! UIA warning: {error}")
                    _write_event(
                        log_file,
                        {
                            "event": "scan_errors",
                            "timestamp": _timestamp(),
                            "errors": result.errors,
                        },
                    )
                    previous_errors = result.errors

                if result.window_count == 0:
                    if now >= next_notice:
                        if result.process_ids:
                            print(
                                "Waiting for an accessible ChatGPT window "
                                f"(processes found: {result.process_ids})..."
                            )
                        else:
                            print("Waiting for ChatGPT.exe...")
                        next_notice = now + 5.0
                else:
                    if not found_window:
                        found_window = True
                        print(
                            f"Connected: {result.window_count} window(s), "
                            f"{len(result.elements)} element(s), PID {result.process_ids}"
                        )
                        _write_event(
                            log_file,
                            {
                                "event": "window_connected",
                                "timestamp": _timestamp(),
                                "process_ids": result.process_ids,
                                "window_count": result.window_count,
                                "element_count": len(result.elements),
                            },
                        )

                    if not baseline_ready:
                        if show_initial:
                            changes = diff_snapshots({}, result.elements)
                        else:
                            changes = ()
                        previous = result.elements
                        baseline_ready = True
                        print("Baseline captured. Start the verification actions now.")
                    else:
                        changes = diff_snapshots(previous, result.elements)
                        previous = result.elements

                    for change in changes:
                        candidate = is_candidate(change.record, keyword_list)
                        local_now = datetime.now(timezone.utc).astimezone()
                        print(
                            f"[{local_now.strftime('%H:%M:%S.%f')[:-3]}] "
                            + _console_change(change, candidate)
                        )
                        event: dict[str, object] = {
                            "event": change.kind,
                            "timestamp": _timestamp(),
                            "candidate": candidate,
                        }
                        if change.before is not None:
                            event["before"] = asdict(change.before)
                        if change.after is not None:
                            event["after"] = asdict(change.after)
                        _write_event(log_file, event)

                if now >= next_heartbeat:
                    _write_event(
                        log_file,
                        {
                            "event": "heartbeat",
                            "timestamp": _timestamp(),
                            "process_ids": result.process_ids,
                            "window_count": result.window_count,
                            "element_count": len(result.elements),
                        },
                    )
                    next_heartbeat = now + 10.0

                remaining = interval_seconds - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            print("\nDiagnostic stopped by user.")
        finally:
            _write_event(
                log_file,
                {
                    "event": "session_end",
                    "timestamp": _timestamp(),
                    "reason": stop_reason,
                    "found_window": found_window,
                },
            )

    print(f"Saved: {output}")
    if not found_window:
        print("No accessible ChatGPT window was found.")
        return 1
    return 0
