from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from .chatgpt_ui_diagnostic import (
    DEFAULT_PROCESS_NAMES,
    PywinautoSnapshotProvider,
    SnapshotProvider,
    UiScanResult,
)
from .voice_config import VoiceUiMonitorConfig

ACTIVITY_CLASS_FRAGMENT = "activitypillmaterial"
SEARCH_TEXT_MARKERS = (
    "ウェブを検索中",
    "webを検索中",
    "web検索中",
    "searching the web",
    "searching web",
)


class UiActivityState(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    SEARCHING = "searching"


@dataclass(frozen=True, slots=True)
class UiActivitySignals:
    activity: bool = False
    searching: bool = False


def detect_ui_activity(result: UiScanResult) -> UiActivitySignals:
    """Classify the current ChatGPT accessibility tree without reading conversation text."""
    activity_pill = False
    search_text = False
    for record in result.elements.values():
        name = record.name.casefold()
        class_name = record.class_name.casefold()
        if any(marker.casefold() in name for marker in SEARCH_TEXT_MARKERS):
            search_text = True
        if (
            record.control_type.casefold() == "statusbar"
            and ACTIVITY_CLASS_FRAGMENT in class_name
        ):
            activity_pill = True
    # The activity pill is required. This prevents old conversation text such as
    # "ウェブを検索中" or "作業中" from leaving the avatar permanently active.
    return UiActivitySignals(
        activity=activity_pill,
        searching=activity_pill and search_text,
    )


class UiActivityTracker:
    """Debounce ChatGPT UI re-renders while preserving a short visible tail."""

    def __init__(self, release_hold_sec: float, search_hold_sec: float) -> None:
        self.release_hold_sec = max(0.0, float(release_hold_sec))
        self.search_hold_sec = max(0.0, float(search_hold_sec))
        self._activity_until = 0.0
        self._search_until = 0.0

    def reset(self) -> UiActivityState:
        self._activity_until = 0.0
        self._search_until = 0.0
        return UiActivityState.IDLE

    def update(self, signals: UiActivitySignals, now: float) -> UiActivityState:
        if signals.activity:
            self._activity_until = max(
                self._activity_until,
                now + self.release_hold_sec,
            )
        if signals.searching:
            self._search_until = max(
                self._search_until,
                now + self.search_hold_sec,
            )
            self._activity_until = max(
                self._activity_until,
                now + self.release_hold_sec,
            )

        activity_active = signals.activity or now < self._activity_until
        if activity_active and (signals.searching or now < self._search_until):
            return UiActivityState.SEARCHING
        if activity_active:
            return UiActivityState.WORKING
        return UiActivityState.IDLE


class ChatGptUiStateMonitor:
    """Continuously observe ChatGPT's read-only Windows UI Automation state."""

    def __init__(
        self,
        config: VoiceUiMonitorConfig,
        on_state: Callable[[UiActivityState], None],
        on_error: Callable[[str], None] | None = None,
        *,
        process_names: Iterable[str] = DEFAULT_PROCESS_NAMES,
        provider: SnapshotProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.on_state = on_state
        self.on_error = on_error
        self.provider = provider or PywinautoSnapshotProvider(
            process_names,
            include_offscreen=config.include_offscreen,
        )
        self.clock = clock
        self.tracker = UiActivityTracker(
            config.release_hold_sec,
            config.search_hold_sec,
        )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = bool(config.enabled)
        self._state = UiActivityState.IDLE
        self._available = False
        self._window_count = 0
        self._element_count = 0
        self._last_error = ""
        self._last_scan_at = 0.0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="chatgpt-ui-state-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, self.config.interval_sec + 1.0))
        self._thread = None
        self._publish_state(self.tracker.reset())

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._lock:
            self._enabled = enabled
            self.config.enabled = enabled
            if not enabled:
                self._available = False
                self._window_count = 0
                self._element_count = 0
                self._last_error = ""
        if not enabled:
            self._publish_state(self.tracker.reset())

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "running": self.running,
                "available": self._available,
                "state": self._state.value,
                "thinking": self._state != UiActivityState.IDLE,
                "searching": self._state == UiActivityState.SEARCHING,
                "window_count": self._window_count,
                "element_count": self._element_count,
                "last_error": self._last_error,
                "last_scan_monotonic": self._last_scan_at,
            }

    def _publish_state(self, state: UiActivityState) -> None:
        with self._lock:
            if state == self._state:
                return
            self._state = state
        try:
            self.on_state(state)
        except Exception as exc:  # noqa: BLE001 - OSC failures must not kill UI monitoring
            self._report_error(f"ChatGPT UI state output failed: {exc}")

    def _report_error(self, detail: str) -> None:
        with self._lock:
            if detail == self._last_error:
                return
            self._last_error = detail
        if self.on_error is not None:
            self.on_error(detail)

    def _run(self) -> None:
        interval = max(0.2, float(self.config.interval_sec))
        while not self._stop_event.is_set():
            if not self.enabled:
                self._stop_event.wait(min(interval, 0.5))
                continue

            loop_started = self.clock()
            try:
                result = self.provider.scan()
                now = self.clock()
                signals = detect_ui_activity(result)
                next_state = self.tracker.update(signals, now)
                with self._lock:
                    self._available = result.window_count > 0
                    self._window_count = result.window_count
                    self._element_count = len(result.elements)
                    self._last_error = " | ".join(result.errors)
                    self._last_scan_at = now
                self._publish_state(next_state)
            except Exception as exc:  # noqa: BLE001 - COM exposes several exception classes
                with self._lock:
                    self._available = False
                    self._window_count = 0
                    self._element_count = 0
                self._publish_state(self.tracker.reset())
                self._report_error(f"ChatGPT UI monitor failed: {exc}")

            remaining = interval - (self.clock() - loop_started)
            self._stop_event.wait(max(0.0, remaining))
