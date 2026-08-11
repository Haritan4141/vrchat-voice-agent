from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(slots=True)
class VoiceAudioConfig:
    vrchat_output: str = "CABLE-A Input"
    chatgpt_input: str = "CABLE-A Output"
    chatgpt_output: str = "CABLE-B Input"
    vrchat_microphone: str = "CABLE-B Output"
    sample_rate: int = 48000
    channels: int = 1
    chunk_ms: int = 100


@dataclass(slots=True)
class VoiceProcessConfig:
    chatgpt: str = "ChatGPT.exe"
    vrchat: str = "VRChat.exe"
    parsec: str = "parsecd.exe,parsec.exe"
    forbidden: str = "ollama.exe,voicevox.exe"


@dataclass(slots=True)
class VoiceOscConfig:
    target_host: str = "127.0.0.1"
    input_port: int = 9000
    listen_host: str = "127.0.0.1"
    output_port: int = 9001
    status_parameter: str = "VoiceAgentStatus"
    motion_enabled_parameter: str = "VoiceAgentMotionEnabled"
    motion_activity_parameter: str = "VoiceAgentActivity"
    motion_energy_parameter: str = "VoiceAgentEnergy"
    motion_gesture_parameter: str = "VoiceAgentGesture"
    motion_expression_parameter: str = "VoiceAgentExpression"
    thinking_parameter: str = "VoiceAgentThinking"
    probe_parameter: str = "VoiceAgentOscProbe"
    voice_input_mode: str = "toggle"
    mute_confirm_timeout_sec: float = 2.0
    mute_retry_count: int = 2
    avatar_reload_timeout_sec: float = 10.0
    avatar_reload_settle_sec: float = 1.5


@dataclass(slots=True)
class VoiceControlConfig:
    bind_host: str = "0.0.0.0"
    port: int = 18765
    token_file: str = "control-token.txt"
    allowed_client_ips: str = ""


@dataclass(slots=True)
class VoiceLoopGuardConfig:
    enabled: bool = True
    auto_mute: bool = True
    rms_threshold: float = 250.0
    correlation_threshold: float = 0.95
    min_consecutive_matches: int = 5
    feature_ms: int = 20
    comparison_window_ms: int = 1500
    min_delay_ms: int = 100
    max_delay_ms: int = 1800
    reliable_max_delay_ms: int = 1800
    delay_tolerance_ms: int = 160
    min_match_duration_ms: int = 1500


@dataclass(slots=True)
class VoiceMotionConfig:
    enabled: bool = True
    speech_on_rms: float = 350.0
    speech_off_rms: float = 180.0
    energy_floor_rms: float = 220.0
    energy_ceiling_rms: float = 4000.0
    attack_ms: int = 250
    release_ms: int = 700
    settling_ms: int = 550
    energy_smoothing: float = 0.28
    idle_gesture_min_sec: float = 8.0
    idle_gesture_max_sec: float = 18.0
    speaking_gesture_min_sec: float = 2.8
    speaking_gesture_max_sec: float = 5.8
    gesture_sync_hold_sec: float = 1.5
    speaking_expression_min_sec: float = 2.0
    speaking_expression_max_sec: float = 4.0


@dataclass(slots=True)
class VoiceParsecConfig:
    require_admin_mute_disabled: bool = True
    config_paths: str = ""


@dataclass(slots=True)
class VoiceUiMonitorConfig:
    enabled: bool = True
    include_offscreen: bool = True
    interval_sec: float = 0.75
    release_hold_sec: float = 2.5
    search_hold_sec: float = 3.0


@dataclass(slots=True)
class VoiceCaptionConfig:
    # off: no chatbox captions, uia: ChatGPT accessibility text, stt: CABLE-B STT
    mode: str = "off"
    prefix: str = "AI: "
    max_chars: int = 144
    min_send_interval_sec: float = 1.5
    # Keep early UI changes pending long enough for the assistant response to
    # appear below the user's just-finished voice transcript.
    uia_initial_hold_sec: float = 1.0
    uia_post_speech_grace_sec: float = 2.5
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "ja"
    stt_beam_size: int = 1
    stt_vad_filter: bool = True
    stt_vad_min_silence_ms: int = 350
    stt_partial_interval_sec: float = 2.5
    stt_speech_on_rms: float = 350.0
    stt_speech_off_rms: float = 180.0
    stt_end_silence_ms: int = 700
    stt_min_audio_ms: int = 600
    stt_max_utterance_ms: int = 30000


@dataclass(slots=True)
class ChatGPTVoiceConfig:
    audio: VoiceAudioConfig
    processes: VoiceProcessConfig
    osc: VoiceOscConfig
    control: VoiceControlConfig
    loop_guard: VoiceLoopGuardConfig
    parsec: VoiceParsecConfig
    source_path: Path
    motion: VoiceMotionConfig = field(default_factory=VoiceMotionConfig)
    ui_monitor: VoiceUiMonitorConfig = field(default_factory=VoiceUiMonitorConfig)
    captions: VoiceCaptionConfig = field(default_factory=VoiceCaptionConfig)


T = TypeVar("T")


def _dataclass_from_table(cls: type[T], table: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"Unknown keys in [{cls.__name__}]: {', '.join(unknown)}")
    return cls(**table)


def load_voice_config(path: Path) -> ChatGPTVoiceConfig:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Voice config not found: {path}. Copy config/chatgpt_voice.example.toml "
            "to config/chatgpt_voice.toml first."
        )
    with path.open("rb") as file:
        raw = tomllib.load(file)

    valid_sections = {
        "audio",
        "processes",
        "osc",
        "control",
        "loop_guard",
        "parsec",
        "motion",
        "ui_monitor",
        "captions",
    }
    unknown_sections = sorted(set(raw) - valid_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")

    return ChatGPTVoiceConfig(
        audio=_dataclass_from_table(VoiceAudioConfig, raw.get("audio", {})),
        processes=_dataclass_from_table(VoiceProcessConfig, raw.get("processes", {})),
        osc=_dataclass_from_table(VoiceOscConfig, raw.get("osc", {})),
        control=_dataclass_from_table(VoiceControlConfig, raw.get("control", {})),
        loop_guard=_dataclass_from_table(VoiceLoopGuardConfig, raw.get("loop_guard", {})),
        parsec=_dataclass_from_table(VoiceParsecConfig, raw.get("parsec", {})),
        source_path=path,
        motion=_dataclass_from_table(VoiceMotionConfig, raw.get("motion", {})),
        ui_monitor=_dataclass_from_table(VoiceUiMonitorConfig, raw.get("ui_monitor", {})),
        captions=_dataclass_from_table(VoiceCaptionConfig, raw.get("captions", {})),
    )


def resolve_config_relative(config: ChatGPTVoiceConfig, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config.source_path.parent / path


def _save_boolean_setting(
    config: ChatGPTVoiceConfig,
    section: str,
    key: str,
    enabled: bool,
) -> None:
    """Persist one boolean without rewriting unrelated TOML settings."""
    path = config.source_path
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    section_start: int | None = None
    section_end = len(lines)

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"[{section}]":
            section_start = index
            continue
        if section_start is not None and stripped.startswith("[") and stripped.endswith("]"):
            section_end = index
            break

    value = "true" if enabled else "false"
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend((f"[{section}]", f"{key} = {value}"))
    else:
        for index in range(section_start + 1, section_end):
            candidate = lines[index].lstrip()
            if candidate.startswith(key) and candidate[len(key) :].lstrip().startswith("="):
                indent = lines[index][: len(lines[index]) - len(candidate)]
                lines[index] = f"{indent}{key} = {value}"
                break
        else:
            lines.insert(section_start + 1, f"{key} = {value}")

    updated = newline.join(lines) + newline
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(updated, encoding="utf-8", newline="")
    temporary.replace(path)


def _save_string_setting(
    config: ChatGPTVoiceConfig,
    section: str,
    key: str,
    value: str,
) -> None:
    """Persist one simple string while preserving comments and unrelated TOML."""
    if any(character in value for character in ('"', "\r", "\n")):
        raise ValueError(f"unsupported characters in {section}.{key}")
    path = config.source_path
    text = path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    section_start: int | None = None
    section_end = len(lines)

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"[{section}]":
            section_start = index
            continue
        if section_start is not None and stripped.startswith("[") and stripped.endswith("]"):
            section_end = index
            break

    encoded = f'"{value}"'
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend((f"[{section}]", f"{key} = {encoded}"))
    else:
        for index in range(section_start + 1, section_end):
            candidate = lines[index].lstrip()
            if candidate.startswith(key) and candidate[len(key) :].lstrip().startswith("="):
                indent = lines[index][: len(lines[index]) - len(candidate)]
                lines[index] = f"{indent}{key} = {encoded}"
                break
        else:
            lines.insert(section_start + 1, f"{key} = {encoded}")

    updated = newline.join(lines) + newline
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(updated, encoding="utf-8", newline="")
    temporary.replace(path)


def save_loop_guard_enabled(config: ChatGPTVoiceConfig, enabled: bool) -> None:
    """Persist the loop guard switch without rewriting unrelated TOML settings."""
    _save_boolean_setting(config, "loop_guard", "enabled", enabled)
    config.loop_guard.enabled = enabled


def save_motion_enabled(config: ChatGPTVoiceConfig, enabled: bool) -> None:
    """Persist the avatar motion switch without rewriting unrelated TOML settings."""
    _save_boolean_setting(config, "motion", "enabled", enabled)
    config.motion.enabled = enabled


def save_ui_monitor_enabled(config: ChatGPTVoiceConfig, enabled: bool) -> None:
    """Persist the ChatGPT UI monitor switch without rewriting unrelated settings."""
    _save_boolean_setting(config, "ui_monitor", "enabled", enabled)
    config.ui_monitor.enabled = enabled


def save_caption_mode(config: ChatGPTVoiceConfig, mode: str) -> None:
    """Persist the selected caption source without rewriting unrelated settings."""
    normalized = mode.strip().casefold()
    if normalized not in {"off", "uia", "stt"}:
        raise ValueError("caption mode must be off, uia, or stt")
    _save_string_setting(config, "captions", "mode", normalized)
    config.captions.mode = normalized


def split_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip().casefold() for item in value.split(",") if item.strip())
