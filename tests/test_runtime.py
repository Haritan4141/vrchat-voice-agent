from __future__ import annotations

import unittest

from vrchat_ai_tool.runtime import clean_reply_text


class RuntimeTests(unittest.TestCase):
    def test_clean_reply_text(self) -> None:
        self.assertEqual(clean_reply_text("  こんにちは   です  ", 20), "こんにちは です")
        self.assertEqual(clean_reply_text("abcdef", 4), "abcd")
        self.assertEqual(clean_reply_text("   ", 10), "")


if __name__ == "__main__":
    unittest.main()
