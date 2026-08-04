from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import psutil

from .audio import WaveInRecorder, find_device_id, pcm16le_rms
from .voice_config import ChatGPTVoiceConfig, split_names
from .windows_audio import WindowsAudioSnapshot, collect_windows_audio_snapshot


class CheckLevel(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(slots=True)
class VoiceCheck:
    level: CheckLevel
    code: str
    title: str
    detail: str


@dataclass(slots=True)
class DoctorReport:
    checks: list[VoiceCheck]
    live_levels: dict[str, float]

    @property
    def exit_code(self) -> int:
        if any(check.level is CheckLevel.ERROR for check in self.checks):
            return 2
        if any(check.level is CheckLevel.WARN for check in self.checks):
            return 1
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "checks": [asdict(check) for check in self.checks],
            "live_levels": self.live_levels,
        }


def _process_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (process.info.get("name") or "").casefold()
            command = " ".join(process.info.get("cmdline") or []).casefold()
            rows.append((name, command))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rows


def _matches_endpoint(snapshot: WindowsAudioSnapshot, name: str, direction: str):
    return [endpoint for endpoint in snapshot.find(name, direction) if endpoint.state == "active"]


def _check_endpoint(
    checks: list[VoiceCheck],
    snapshot: WindowsAudioSnapshot,
    code: str,
    label: str,
    name: str,
    direction: str,
) -> None:
    matches = _matches_endpoint(snapshot, name, direction)
    if len(matches) == 1:
        checks.append(VoiceCheck(CheckLevel.OK, code, label, matches[0].name))
    elif not matches:
        checks.append(VoiceCheck(CheckLevel.ERROR, code, label, f"有効な端点が見つかりません: {name}"))
    else:
        checks.append(
            VoiceCheck(CheckLevel.ERROR, code, label, f"複数の端点が一致しました: {name}")
        )


def _check_default(
    checks: list[VoiceCheck],
    snapshot: WindowsAudioSnapshot,
    code: str,
    label: str,
    expected_name: str,
    direction: str,
    role_attr: str,
    level: CheckLevel,
) -> None:
    defaults = [
        endpoint
        for endpoint in snapshot.endpoints
        if endpoint.direction == direction and getattr(endpoint, role_attr)
    ]
    expected = snapshot.find(expected_name, direction)
    if defaults and expected and defaults[0].id == expected[0].id:
        checks.append(VoiceCheck(CheckLevel.OK, code, label, defaults[0].name))
    else:
        actual = defaults[0].name if defaults else "未設定"
        checks.append(
            VoiceCheck(level, code, label, f"現在: {actual} / 期待: {expected_name}")
        )


def _check_process_route(
    checks: list[VoiceCheck],
    snapshot: WindowsAudioSnapshot,
    process_names: tuple[str, ...],
    expected_endpoint: str,
    direction: str,
    code: str,
    label: str,
    process_running: bool,
) -> None:
    sessions = [
        (endpoint, session)
        for endpoint, session in snapshot.sessions_for(process_names)
        if endpoint.direction == direction and session.state == "active"
    ]
    if not sessions:
        reason = "音声セッションがまだありません" if process_running else "アプリが起動していません"
        checks.append(VoiceCheck(CheckLevel.WARN, code, label, reason))
        return

    expected_ids = {endpoint.id for endpoint in snapshot.find(expected_endpoint, direction)}
    wrong = [endpoint.name for endpoint, _session in sessions if endpoint.id not in expected_ids]
    if wrong:
        checks.append(
            VoiceCheck(
                CheckLevel.ERROR,
                code,
                label,
                f"誤った端点で使用中: {', '.join(sorted(set(wrong)))} / 期待: {expected_endpoint}",
            )
        )
    else:
        checks.append(VoiceCheck(CheckLevel.OK, code, label, expected_endpoint))


def _parsec_paths(config: ChatGPTVoiceConfig) -> list[Path]:
    if config.parsec.config_paths.strip():
        return [Path(value.strip()) for value in config.parsec.config_paths.split(",") if value.strip()]
    candidates: list[Path] = []
    for env_name in ("APPDATA", "PROGRAMDATA"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "Parsec" / "config.txt")
    return candidates


def _check_parsec(checks: list[VoiceCheck], config: ChatGPTVoiceConfig) -> None:
    if not config.parsec.require_admin_mute_disabled:
        return
    existing = [path for path in _parsec_paths(config) if path.exists()]
    if not existing:
        checks.append(
            VoiceCheck(
                CheckLevel.WARN,
                "parsec.admin_mute",
                "Parsecの自動ミュート",
                "config.txtを確認できませんでした",
            )
        )
        return
    for path in existing:
        try:
            values: dict[str, str] = {}
            for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                key, separator, value = raw_line.partition("=")
                if separator:
                    values[key.strip().casefold()] = value.strip()
            if values.get("server_admin_mute") == "0":
                checks.append(
                    VoiceCheck(CheckLevel.OK, "parsec.admin_mute", "Parsecの自動ミュート", str(path))
                )
                return
        except OSError:
            continue
    checks.append(
        VoiceCheck(
            CheckLevel.ERROR,
            "parsec.admin_mute",
            "Parsecの自動ミュート",
            "server_admin_mute=0 が見つかりません",
        )
    )


def sample_live_levels(config: ChatGPTVoiceConfig, seconds: float) -> dict[str, float]:
    if seconds <= 0:
        return {}
    targets = {
        "CABLE-A (VRChat→ChatGPT)": config.audio.chatgpt_input,
        "CABLE-B (ChatGPT→VRChat)": config.audio.vrchat_microphone,
    }
    recorders: dict[str, WaveInRecorder] = {}
    peaks = {label: 0.0 for label in targets}
    try:
        for label, device_name in targets.items():
            device_id = find_device_id("input", device_name)
            recorder = WaveInRecorder(
                device_id,
                config.audio.sample_rate,
                config.audio.channels,
                config.audio.chunk_ms,
            )
            recorder.open()
            recorders[label] = recorder
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            for label, recorder in recorders.items():
                frame = recorder.read_chunk(timeout=0.15)
                if frame:
                    peaks[label] = max(peaks[label], pcm16le_rms(frame))
    finally:
        for recorder in recorders.values():
            recorder.close()
    return {key: round(value, 1) for key, value in peaks.items()}


def build_doctor_report(
    config: ChatGPTVoiceConfig,
    live_seconds: float = 0,
    snapshot_provider: Callable[[], WindowsAudioSnapshot] = collect_windows_audio_snapshot,
    process_provider: Callable[[], list[tuple[str, str]]] = _process_rows,
) -> DoctorReport:
    checks: list[VoiceCheck] = []
    snapshot = snapshot_provider()
    process_rows = process_provider()

    _check_endpoint(checks, snapshot, "device.a.render", "VRChat出力端点", config.audio.vrchat_output, "render")
    _check_endpoint(checks, snapshot, "device.a.capture", "ChatGPT入力端点", config.audio.chatgpt_input, "capture")
    _check_endpoint(checks, snapshot, "device.b.render", "ChatGPT出力端点", config.audio.chatgpt_output, "render")
    _check_endpoint(checks, snapshot, "device.b.capture", "VRChatマイク端点", config.audio.vrchat_microphone, "capture")

    _check_default(
        checks, snapshot, "default.render", "Windows既定出力", config.audio.chatgpt_output,
        "render", "is_default_multimedia", CheckLevel.WARN,
    )
    _check_default(
        checks, snapshot, "default.capture", "Windows既定入力", config.audio.chatgpt_input,
        "capture", "is_default_multimedia", CheckLevel.WARN,
    )
    _check_default(
        checks, snapshot, "default.communications.capture", "既定の通信入力", config.audio.chatgpt_input,
        "capture", "is_default_communications", CheckLevel.WARN,
    )

    running_names = {name for name, _command in process_rows}
    chatgpt_names = split_names(config.processes.chatgpt)
    vrchat_names = split_names(config.processes.vrchat)
    _check_process_route(
        checks, snapshot, chatgpt_names, config.audio.chatgpt_output, "render",
        "route.chatgpt.render", "ChatGPTの出力先", bool(running_names.intersection(chatgpt_names)),
    )
    _check_process_route(
        checks, snapshot, chatgpt_names, config.audio.chatgpt_input, "capture",
        "route.chatgpt.capture", "ChatGPTの入力元", bool(running_names.intersection(chatgpt_names)),
    )
    _check_process_route(
        checks, snapshot, vrchat_names, config.audio.vrchat_output, "render",
        "route.vrchat.render", "VRChatの出力先", bool(running_names.intersection(vrchat_names)),
    )
    _check_process_route(
        checks, snapshot, vrchat_names, config.audio.vrchat_microphone, "capture",
        "route.vrchat.capture", "VRChatのマイク元", bool(running_names.intersection(vrchat_names)),
    )

    b_render_ids = {endpoint.id for endpoint in snapshot.find(config.audio.chatgpt_output, "render")}
    system_leaks = [
        endpoint.name
        for endpoint in snapshot.endpoints
        for session in endpoint.sessions
        if endpoint.id in b_render_ids
        and session.process_id == 0
        and session.muted is False
        and (session.volume or 0) > 0
    ]
    if system_leaks:
        checks.append(
            VoiceCheck(
                CheckLevel.ERROR,
                "system_sounds.leak",
                "システム音のVRChat混入",
                "CABLE-B上のシステム音がミュートされていません",
            )
        )
    else:
        checks.append(
            VoiceCheck(CheckLevel.OK, "system_sounds.leak", "システム音のVRChat混入", "検出なし")
        )

    forbidden = split_names(config.processes.forbidden)
    active_forbidden = sorted({name for name in running_names if name in forbidden})
    legacy_runtime = any("vrchat_ai_tool" in command and " run" in command for _name, command in process_rows)
    if active_forbidden or legacy_runtime:
        detail = ", ".join(active_forbidden + (["python -m vrchat_ai_tool run"] if legacy_runtime else []))
        checks.append(VoiceCheck(CheckLevel.ERROR, "process.double_reply", "二重応答の危険", detail))
    else:
        checks.append(VoiceCheck(CheckLevel.OK, "process.double_reply", "二重応答の危険", "検出なし"))

    _check_parsec(checks, config)
    if config.osc.status_parameter != "VoiceAgentStatus":
        checks.append(
            VoiceCheck(
                CheckLevel.WARN,
                "osc.status_parameter",
                "アバター状態パラメーター",
                f"現在: {config.osc.status_parameter} / 推奨: VoiceAgentStatus",
            )
        )
    else:
        checks.append(
            VoiceCheck(CheckLevel.OK, "osc.status_parameter", "アバター状態パラメーター", "VoiceAgentStatus")
        )

    levels = sample_live_levels(config, live_seconds) if live_seconds > 0 else {}
    return DoctorReport(checks=checks, live_levels=levels)


def print_doctor_report(report: DoctorReport, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    print("ChatGPT Voice Doctor")
    print("=" * 72)
    for check in report.checks:
        print(f"[{check.level.value:5}] {check.title}: {check.detail}")
    if report.live_levels:
        print("\nLive peak RMS")
        for label, value in report.live_levels.items():
            print(f"- {label}: {value:.1f}")
    errors = sum(check.level is CheckLevel.ERROR for check in report.checks)
    warnings = sum(check.level is CheckLevel.WARN for check in report.checks)
    print(f"\nResult: {errors} error(s), {warnings} warning(s)")
