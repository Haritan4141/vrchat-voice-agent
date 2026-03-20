from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    tomllib = None

from . import simple_toml


@dataclass(slots=True)
class AudioCaptureConfig:
    mode: str
    input_device: str
    sample_rate: int
    channels: int
    chunk_ms: int
    silence_timeout_ms: int
    rms_threshold: float
    min_speech_ms: int
    max_utterance_ms: int


@dataclass(slots=True)
class AudioOutputConfig:
    tts_output_device: str
    monitor_output_device: str


@dataclass(slots=True)
class SttConfig:
    backend: str
    model: str
    device: str
    compute_type: str
    language: str
    timeout_sec: int


@dataclass(slots=True)
class LlmConfig:
    backend: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    timeout_sec: int


@dataclass(slots=True)
class TtsConfig:
    backend: str
    base_url: str
    speaker: int
    speed_scale: float
    timeout_sec: int


@dataclass(slots=True)
class ConversationConfig:
    max_response_chars: int
    min_reply_interval_sec: int
    allow_topic_suggestions: bool
    pause_listening_while_speaking: bool


@dataclass(slots=True)
class AppConfig:
    audio_capture: AudioCaptureConfig
    audio_output: AudioOutputConfig
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    conversation: ConversationConfig


def _table(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = data
    path = []
    for key in keys:
        path.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise KeyError(f"Missing config table: {joined}")
        current = current[key]
    if not isinstance(current, dict):
        joined = ".".join(keys)
        raise TypeError(f"Config entry is not a table: {joined}")
    return current


def load_config(path: Path) -> AppConfig:
    config_text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        raw = tomllib.loads(config_text)
    else:
        raw = simple_toml.loads(config_text)

    audio_capture = _table(raw, "audio", "capture")
    audio_output = _table(raw, "audio", "output")
    stt = _table(raw, "stt")
    llm = _table(raw, "llm")
    tts = _table(raw, "tts")
    conversation = _table(raw, "conversation")

    return AppConfig(
        audio_capture=AudioCaptureConfig(
            mode=str(audio_capture["mode"]),
            input_device=str(audio_capture["input_device"]),
            sample_rate=int(audio_capture["sample_rate"]),
            channels=int(audio_capture["channels"]),
            chunk_ms=int(audio_capture["chunk_ms"]),
            silence_timeout_ms=int(audio_capture["silence_timeout_ms"]),
            rms_threshold=float(audio_capture.get("rms_threshold", 500.0)),
            min_speech_ms=int(audio_capture.get("min_speech_ms", 400)),
            max_utterance_ms=int(audio_capture.get("max_utterance_ms", 9000)),
        ),
        audio_output=AudioOutputConfig(
            tts_output_device=str(audio_output["tts_output_device"]),
            monitor_output_device=str(audio_output["monitor_output_device"]),
        ),
        stt=SttConfig(
            backend=str(stt["backend"]),
            model=str(stt["model"]),
            device=str(stt["device"]),
            compute_type=str(stt["compute_type"]),
            language=str(stt["language"]),
            timeout_sec=int(stt.get("timeout_sec", 8)),
        ),
        llm=LlmConfig(
            backend=str(llm["backend"]),
            base_url=str(llm["base_url"]),
            model=str(llm["model"]),
            temperature=float(llm["temperature"]),
            max_tokens=int(llm["max_tokens"]),
            system_prompt=str(llm["system_prompt"]).strip(),
            timeout_sec=int(llm.get("timeout_sec", 60)),
        ),
        tts=TtsConfig(
            backend=str(tts["backend"]),
            base_url=str(tts["base_url"]),
            speaker=int(tts["speaker"]),
            speed_scale=float(tts["speed_scale"]),
            timeout_sec=int(tts.get("timeout_sec", 30)),
        ),
        conversation=ConversationConfig(
            max_response_chars=int(conversation["max_response_chars"]),
            min_reply_interval_sec=int(conversation["min_reply_interval_sec"]),
            allow_topic_suggestions=bool(conversation["allow_topic_suggestions"]),
            pause_listening_while_speaking=bool(conversation["pause_listening_while_speaking"]),
        ),
    )


def probe_http_endpoint(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
