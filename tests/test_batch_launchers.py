import unittest
from pathlib import Path


class BatchLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[1]
        self.controls = self.repository_root / "controls"

    def test_daily_launchers_are_grouped_under_controls(self) -> None:
        for name in (
            "apply_voice_prompt.bat",
            "ensure_voice_monitor.bat",
            "run_chatgpt_voice_production.bat",
        ):
            self.assertTrue((self.controls / name).is_file(), name)
            self.assertFalse((self.repository_root / name).exists(), name)

        self.assertTrue((self.repository_root / "launch_voice_control.bat").is_file())
        self.assertFalse((self.controls / "launch_voice_control.bat").exists())

    def test_moved_launchers_resolve_the_repository_root(self) -> None:
        content = (self.controls / "apply_voice_prompt.bat").read_text(
            encoding="ascii"
        )
        self.assertIn('for %%R in ("%~dp0..")', content)
        self.assertIn('cd /d "%REPO_ROOT%"', content)

    def test_production_launcher_calls_the_sibling_control_launcher(self) -> None:
        content = (self.controls / "run_chatgpt_voice_production.bat").read_text(
            encoding="ascii"
        )
        self.assertIn('call "%~dp0..\\launch_voice_control.bat"', content)

    def test_prompt_launcher_ensures_production_monitor_is_running(self) -> None:
        prompt_launcher = (self.controls / "apply_voice_prompt.bat").read_text(
            encoding="ascii"
        )
        monitor_launcher = (self.controls / "ensure_voice_monitor.bat").read_text(
            encoding="ascii"
        )

        self.assertIn(
            'call "%REPO_ROOT%\\controls\\ensure_voice_monitor.bat"',
            prompt_launcher,
        )
        self.assertIn("if errorlevel 1 goto :monitor_failed", prompt_launcher)
        self.assertIn('set "VOICE_CONTROL_PORT=18765"', monitor_launcher)
        self.assertIn("netstat.exe", monitor_launcher)
        self.assertIn("run_chatgpt_voice_production.bat", monitor_launcher)
        self.assertIn("for /l %%N in (1,1,45)", monitor_launcher)

    def test_control_launcher_installs_production_dependencies(self) -> None:
        content = (self.repository_root / "launch_voice_control.bat").read_text(
            encoding="ascii"
        )
        self.assertIn("sync --quiet", content)
        self.assertIn("run vrchat-voice-control", content)
        self.assertNotIn("--extra", content)

        project = (self.repository_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"faster-whisper>=1.2,<2",', project)


if __name__ == "__main__":
    unittest.main()
