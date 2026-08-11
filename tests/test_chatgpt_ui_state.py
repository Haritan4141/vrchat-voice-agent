from __future__ import annotations

import unittest

from vrchat_ai_tool.chatgpt_ui_diagnostic import (
    PywinautoSnapshotProvider,
    UiElementRecord,
    UiScanResult,
)
from vrchat_ai_tool.chatgpt_ui_state import (
    ChatGptUiStateMonitor,
    UiActivitySignals,
    UiActivityState,
    UiActivityTracker,
    detect_ui_activity,
)
from vrchat_ai_tool.voice_config import VoiceUiMonitorConfig


def record(
    locator: str,
    *,
    control_type: str = "Text",
    name: str = "",
    class_name: str = "",
) -> UiElementRecord:
    return UiElementRecord(
        locator=locator,
        process_id=1,
        window_handle=2,
        window_title="",
        control_type=control_type,
        name=name,
        automation_id="",
        class_name=class_name,
        is_enabled=True,
        is_offscreen=False,
        rectangle="0,0,10,10",
    )


class ChatGptUiStateTests(unittest.TestCase):
    def test_monitor_includes_offscreen_elements_when_configured(self) -> None:
        monitor = ChatGptUiStateMonitor(
            VoiceUiMonitorConfig(include_offscreen=True),
            lambda _state: None,
        )

        self.assertIsInstance(monitor.provider, PywinautoSnapshotProvider)
        self.assertTrue(monitor.provider.include_offscreen)

    def test_activity_pill_detects_unlabelled_work(self) -> None:
        result = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "pill": record(
                    "pill",
                    control_type="StatusBar",
                    class_name="_activityPillMaterial_abcd",
                )
            },
        )

        self.assertEqual(
            detect_ui_activity(result),
            UiActivitySignals(activity=True, searching=False),
        )

    def test_current_compact_status_bar_detects_work(self) -> None:
        result = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "pill": record(
                    "pill",
                    control_type="StatusBar",
                    class_name=(
                        "_Material_abcd _CompactMaterial_abcd "
                        "no-drag pointer-events-none"
                    ),
                )
            },
        )

        self.assertEqual(
            detect_ui_activity(result),
            UiActivitySignals(activity=True, searching=False),
        )

    def test_current_active_shimmer_detects_work_but_inactive_does_not(self) -> None:
        active = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "shimmer": record(
                    "shimmer",
                    control_type="Group",
                    class_name=(
                        "loading-shimmer-pure-text _cadencedShimmer_abcd "
                        "_cadencedShimmerActive_abcd"
                    ),
                )
            },
        )
        inactive = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "shimmer": record(
                    "shimmer",
                    control_type="Group",
                    class_name="loading-shimmer-pure-text _cadencedShimmer_abcd",
                )
            },
        )

        self.assertTrue(detect_ui_activity(active).activity)
        self.assertFalse(detect_ui_activity(inactive).activity)

    def test_current_activity_header_only_matches_live_thinking(self) -> None:
        thinking = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "header": record(
                    "header",
                    control_type="Button",
                    name="思考中",
                    class_name="group/activity-header inline-flex",
                )
            },
        )
        completed = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "header": record(
                    "header",
                    control_type="Button",
                    name="17s間作業しました",
                    class_name="group/activity-header inline-flex",
                )
            },
        )

        self.assertTrue(detect_ui_activity(thinking).activity)
        self.assertFalse(detect_ui_activity(completed).activity)

    def test_web_search_text_is_more_specific_than_working(self) -> None:
        result = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "pill": record(
                    "pill",
                    control_type="StatusBar",
                    class_name="_activityPillMaterial_abcd",
                ),
                "search": record("search", name="ウェブを検索中"),
            },
        )

        self.assertEqual(
            detect_ui_activity(result),
            UiActivitySignals(activity=True, searching=True),
        )

    def test_conversation_text_does_not_trigger_without_active_wording(self) -> None:
        result = UiScanResult(
            process_ids=(1,),
            window_count=1,
            elements={
                "history": record("history", name="3s作業しました"),
                "button": record("button", control_type="Button", name="検索"),
            },
        )

        self.assertEqual(detect_ui_activity(result), UiActivitySignals())

    def test_tracker_holds_search_across_short_ui_rerenders(self) -> None:
        tracker = UiActivityTracker(release_hold_sec=2.5, search_hold_sec=3.0)

        self.assertEqual(
            tracker.update(UiActivitySignals(True, True), 0.0),
            UiActivityState.SEARCHING,
        )
        self.assertEqual(
            tracker.update(UiActivitySignals(True, False), 2.0),
            UiActivityState.SEARCHING,
        )
        self.assertEqual(
            tracker.update(UiActivitySignals(True, False), 3.1),
            UiActivityState.WORKING,
        )
        self.assertEqual(
            tracker.update(UiActivitySignals(), 5.0),
            UiActivityState.WORKING,
        )
        self.assertEqual(
            tracker.update(UiActivitySignals(), 5.7),
            UiActivityState.IDLE,
        )


if __name__ == "__main__":
    unittest.main()
