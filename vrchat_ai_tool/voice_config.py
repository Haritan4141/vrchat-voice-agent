from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
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
    voice_input_mode: str = "toggle"
    mute_confirm_timeout_sec: float = 2.0
    mute_retry_count: int = 2


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
    correlation_threshold: float = 0.88
    min_consecutive_matches: int = 3
    feature_ms: int = 20
    comparison_window_ms: int = 1000
    min_delay_ms: int = 100
    max_delay_ms: int = 5000


@dataclass(slots=True)
class VoiceParsecConfig:
    require_admin_mute_disabled: bool = True
    config_paths: str = ""


@dataclass(slots=True)
class ChatGPTVoiceConfig:
    audio: VoiceAudioConfig
    processes: VoiceProcessConfig
    osc: VoiceOscConfig
    control: VoiceControlConfig
    loop_guard: VoiceLoopGuardConfig
    parsec: VoiceParsecConfig
    source_path: Path


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

    valid_sections = {"audio", "processes", "osc", "control", "loop_guard", "parsec"}
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
    )


def resolve_config_relative(config: ChatGPTVoiceConfig, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config.source_path.parent / path


def split_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip().casefold() for item in value.split(",") if item.strip())
