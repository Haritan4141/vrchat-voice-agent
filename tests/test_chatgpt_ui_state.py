from __future__ import annotations

import unittest

from vrchat_ai_tool.chatgpt_ui_diagnostic import UiElementRecord, UiScanResult
from vrchat_ai_tool.chatgpt_ui_state import (
    UiActivitySignals,
    UiActivityState,
    UiActivityTracker,
    detect_ui_activity,
)


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
