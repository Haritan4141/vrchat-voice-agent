from __future__ import annotations

import textwrap
import unittest

from vrchat_ai_tool.simple_toml import loads


class SimpleTomlTests(unittest.TestCase):
    def test_loads_basic_types_and_multiline_string(self) -> None:
        raw = textwrap.dedent(
            '''
            [audio.capture]
            sample_rate = 16000
            enabled = true

            [llm]
            temperature = 0.7
            system_prompt = """
            hello
            world
            """
            '''
        ).strip()

        data = loads(raw)

        self.assertEqual(data["audio"]["capture"]["sample_rate"], 16000)
        self.assertTrue(data["audio"]["capture"]["enabled"])
        self.assertEqual(data["llm"]["temperature"], 0.7)
        self.assertEqual(data["llm"]["system_prompt"], "hello\nworld")


if __name__ == "__main__":
    unittest.main()
