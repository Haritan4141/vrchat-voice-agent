import unittest
from pathlib import Path


class BatchLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[1]
        self.controls = self.repository_root / "controls"

    def test_daily_launchers_are_grouped_under_controls(self) -> None:
        for name in (
            "apply_voice_prompt.bat",
            "launch_voice_control.bat",
            "run_chatgpt_voice_production.bat",
        ):
            self.assertTrue((self.controls / name).is_file(), name)
            self.assertFalse((self.repository_root / name).exists(), name)

    def test_moved_launchers_resolve_the_repository_root(self) -> None:
        for name in ("apply_voice_prompt.bat", "launch_voice_control.bat"):
            content = (self.controls / name).read_text(encoding="ascii")
            self.assertIn('for %%R in ("%~dp0..")', content)
            self.assertIn('cd /d "%REPO_ROOT%"', content)

    def test_production_launcher_calls_the_sibling_control_launcher(self) -> None:
        content = (self.controls / "run_chatgpt_voice_production.bat").read_text(
            encoding="ascii"
        )
        self.assertIn('call "%~dp0launch_voice_control.bat"', content)


if __name__ == "__main__":
    unittest.main()
