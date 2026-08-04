from __future__ import annotations

import unittest
from pathlib import Path

from vrchat_ai_tool.voice_config import (
    ChatGPTVoiceConfig,
    VoiceAudioConfig,
    VoiceControlConfig,
    VoiceLoopGuardConfig,
    VoiceOscConfig,
    VoiceParsecConfig,
    VoiceProcessConfig,
)
from vrchat_ai_tool.voice_doctor import _is_legacy_runtime, build_doctor_report
from vrchat_ai_tool.windows_audio import (
    AudioEndpointInfo,
    AudioSessionInfo,
    WindowsAudioSnapshot,
)


def session(name: str, pid: int) -> AudioSessionInfo:
    return AudioSessionInfo(pid, name, "active", False, 1.0)


class VoiceDoctorTests(unittest.TestCase):
    def test_legacy_runtime_detection_does_not_match_uv_doctor(self) -> None:
        self.assertTrue(_is_legacy_runtime("python -m vrchat_ai_tool run --config settings.toml"))
        self.assertTrue(_is_legacy_runtime("vrchat-ai-tool.exe run --config settings.toml"))
        self.assertFalse(
            _is_legacy_runtime(
                "uv run chatgpt-voice-doctor --config C:\\repo\\vrchat_ai_tool\\voice.toml"
            )
        )

    def test_unnamed_stale_endpoint_is_ignored(self) -> None:
        snapshot = WindowsAudioSnapshot(
            endpoints=(AudioEndpointInfo("stale", None, "render", "active"),)  # type: ignore[arg-type]
        )
        self.assertEqual(snapshot.find("CABLE-A Input", "render"), [])

    def test_expected_routes_are_all_ok(self) -> None:
        config = ChatGPTVoiceConfig(
            audio=VoiceAudioConfig(),
            processes=VoiceProcessConfig(),
            osc=VoiceOscConfig(),
            control=VoiceControlConfig(),
            loop_guard=VoiceLoopGuardConfig(),
            parsec=VoiceParsecConfig(require_admin_mute_disabled=False),
            source_path=Path("voice.toml"),
        )
        snapshot = WindowsAudioSnapshot(
            endpoints=(
                AudioEndpointInfo(
                    "a-in", "CABLE-A Input (VB-Audio Virtual Cable A)", "render", "active",
                    sessions=(session("VRChat.exe", 10),),
                ),
                AudioEndpointInfo(
                    "a-out", "CABLE-A Output (VB-Audio Virtual Cable A)", "capture", "active",
                    is_default_multimedia=True, is_default_communications=True,
                    sessions=(session("ChatGPT.exe", 20),),
                ),
                AudioEndpointInfo(
                    "b-in", "CABLE-B Input (VB-Audio Virtual Cable B)", "render", "active",
                    is_default_multimedia=True,
                    sessions=(session("ChatGPT.exe", 20), AudioSessionInfo(0, "System Sounds", "active", True, 1.0)),
                ),
                AudioEndpointInfo(
                    "b-out", "CABLE-B Output (VB-Audio Virtual Cable B)", "capture", "active",
                    sessions=(session("VRChat.exe", 10),),
                ),
            )
        )
        report = build_doctor_report(
            config,
            snapshot_provider=lambda: snapshot,
            process_provider=lambda: [("chatgpt.exe", ""), ("vrchat.exe", "")],
        )
        self.assertEqual(report.exit_code, 0)
        self.assertTrue(all(check.level.value == "OK" for check in report.checks))


if __name__ == "__main__":
    unittest.main()
