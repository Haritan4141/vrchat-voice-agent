from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from vrchat_ai_tool.chatgpt_ui_diagnostic import (
    UiElementRecord,
    default_output_path,
    diff_snapshots,
    is_candidate,
    normalize_text,
    run_ui_diagnostic,
)


def element(
    locator: str,
    *,
    name: str = "",
    control_type: str = "Text",
    automation_id: str = "",
) -> UiElementRecord:
    return UiElementRecord(
        locator=locator,
        process_id=100,
        window_handle=200,
        window_title="ChatGPT",
        control_type=control_type,
        name=name,
        automation_id=automation_id,
        class_name="",
        is_enabled=True,
        is_offscreen=False,
        rectangle="0,0,10,10",
    )


class ChatGptUiDiagnosticTests(unittest.TestCase):
    def test_normalize_text_makes_one_line_and_truncates(self) -> None:
        self.assertEqual(normalize_text("  Web\n  search  "), "Web search")
        self.assertEqual(normalize_text("abcdef", 4), "abc…")

    def test_diff_reports_added_changed_and_removed(self) -> None:
        before = {
            "changed": element("changed", name="待機中"),
            "removed": element("removed", name="古い表示"),
        }
        after = {
            "changed": element("changed", name="検索中"),
            "added": element("added", name="停止", control_type="Button"),
        }

        changes = diff_snapshots(before, after)

        self.assertEqual([change.kind for change in changes], ["added", "changed", "removed"])
        changed = next(change for change in changes if change.kind == "changed")
        self.assertEqual(changed.before.name, "待機中")  # type: ignore[union-attr]
        self.assertEqual(changed.after.name, "検索中")  # type: ignore[union-attr]

    def test_window_title_change_does_not_change_every_descendant(self) -> None:
        before_record = element("same", name="Voice")
        after_record = UiElementRecord(
            locator=before_record.locator,
            process_id=before_record.process_id,
            window_handle=before_record.window_handle,
            window_title="Different task title",
            control_type=before_record.control_type,
            name=before_record.name,
            automation_id=before_record.automation_id,
            class_name=before_record.class_name,
            is_enabled=before_record.is_enabled,
            is_offscreen=before_record.is_offscreen,
            rectangle=before_record.rectangle,
        )
        self.assertEqual(diff_snapshots({"same": before_record}, {"same": after_record}), ())

    def test_candidate_matches_japanese_and_english_status_words(self) -> None:
        self.assertTrue(is_candidate(element("a", name="Webを検索しています")))
        self.assertTrue(is_candidate(element("b", name="Working")))
        self.assertFalse(is_candidate(element("c", name="こんにちは")))

    def test_default_output_path_includes_microseconds(self) -> None:
        now = datetime(2026, 8, 11, 23, 2, 15, 123456, tzinfo=timezone.utc)

        path = default_output_path(now)

        self.assertEqual(
            path,
            Path("artifacts/chatgpt-ui-diagnostic-20260811-230215-123456.jsonl"),
        )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "existing.jsonl"
            output.write_text("preserve me\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_ui_diagnostic(output_path=output, provider=None)

            self.assertEqual(output.read_text(encoding="utf-8"), "preserve me\n")


if __name__ == "__main__":
    unittest.main()
