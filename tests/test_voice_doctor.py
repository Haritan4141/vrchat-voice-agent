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
from vrchat_ai_tool.voice_doctor import build_doctor_report
from vrchat_ai_tool.windows_audio import (
    AudioEndpointInfo,
    AudioSessionInfo,
    WindowsAudioSnapshot,
)


def session(name: str, pid: int) -> AudioSessionInfo:
    return AudioSessionInfo(pid, name, "active", False, 1.0)


class VoiceDoctorTests(unittest.TestCase):
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
