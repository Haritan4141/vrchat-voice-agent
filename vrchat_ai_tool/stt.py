from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import os
from pathlib import Path
import subprocess
from typing import Protocol
import wave

from .config import SttConfig


POWERSHELL_TRANSCRIBE_SCRIPT = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Speech

$culture = $env:VRCHAT_AI_TOOL_STT_CULTURE
$wavePath = $env:VRCHAT_AI_TOOL_STT_WAVE_PATH
$timeoutSec = [int]$env:VRCHAT_AI_TOOL_STT_TIMEOUT_SEC

$recognizerInfo = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() |
    Where-Object {
        $_.Culture.Name -eq $culture -or $_.Culture.Name.StartsWith("$culture-")
    } |
    Select-Object -First 1

if ($null -eq $recognizerInfo) {
    Write-Error "No recognizer installed for culture $culture"
    exit 2
}

try {
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($recognizerInfo.Id)
} catch {
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($recognizerInfo.Culture)
}

$engine.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$engine.EndSilenceTimeout = [TimeSpan]::FromMilliseconds(700)
$engine.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromMilliseconds(1200)
$engine.BabbleTimeout = [TimeSpan]::FromSeconds($timeoutSec)
$engine.SetInputToWaveFile($wavePath)

$results = New-Object System.Collections.Generic.List[string]
while ($true) {
    $result = $engine.Recognize([TimeSpan]::FromSeconds($timeoutSec))
    if ($null -eq $result) {
        break
    }
    if (-not [string]::IsNullOrWhiteSpace($result.Text)) {
        $results.Add($result.Text)
    }
}

$joined = ($results -join ' ').Trim()
if (-not [string]::IsNullOrWhiteSpace($joined)) {
    Write-Output $joined
}
"""


class Transcriber(Protocol):
    def warm_up(self) -> None:
        ...

    def transcribe_wav(self, wave_path: Path) -> str:
        ...


def normalize_whisper_language(language: str) -> str | None:
    cleaned = language.strip()
    if not cleaned or cleaned.casefold() == "auto":
        return None
    return cleaned.split("-", 1)[0].casefold()


@dataclass(slots=True)
class SystemSpeechTranscriber:
    culture: str
    timeout_sec: int

    def warm_up(self) -> None:
        return None

    def transcribe_wav(self, wave_path: Path) -> str:
        env = os.environ.copy()
        env["VRCHAT_AI_TOOL_STT_CULTURE"] = self.culture
        env["VRCHAT_AI_TOOL_STT_WAVE_PATH"] = str(wave_path)
        env["VRCHAT_AI_TOOL_STT_TIMEOUT_SEC"] = str(self.timeout_sec)

        with wave.open(str(wave_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate() or 1
            frame_count = wav_file.getnframes()
        duration_sec = frame_count / frame_rate
        process_timeout = max(self.timeout_sec + 5, int(duration_sec * 3) + 5)

        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "-"],
            input=POWERSHELL_TRANSCRIBE_SCRIPT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=process_timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"System.Speech transcription failed: {detail}")
        return " ".join(completed.stdout.split())


@dataclass(slots=True)
class FasterWhisperTranscriber:
    model_name: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    vad_filter: bool
    vad_min_silence_ms: int
    _model: object | None = field(default=None, init=False, repr=False)

    def warm_up(self) -> None:
        self._get_model()

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            module = import_module("faster_whisper")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run `python -m pip install faster-whisper`."
            ) from exc

        model_class = getattr(module, "WhisperModel", None)
        if model_class is None:
            raise RuntimeError("faster-whisper import succeeded, but WhisperModel was not found.")

        self._model = model_class(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def transcribe_wav(self, wave_path: Path) -> str:
        model = self._get_model()
        language = normalize_whisper_language(self.language)
        transcribe_kwargs: dict[str, object] = {
            "beam_size": self.beam_size,
            "task": "transcribe",
        }
        if language is not None:
            transcribe_kwargs["language"] = language
        if self.vad_filter:
            transcribe_kwargs["vad_filter"] = True
            transcribe_kwargs["vad_parameters"] = {
                "min_silence_duration_ms": self.vad_min_silence_ms,
            }

        segments, _info = model.transcribe(str(wave_path), **transcribe_kwargs)
        return "".join(getattr(segment, "text", "") for segment in segments).strip()


def create_transcriber(config: SttConfig) -> Transcriber:
    backend = config.backend.casefold()

    if backend == "system_speech":
        return SystemSpeechTranscriber(
            culture=config.language,
            timeout_sec=config.timeout_sec,
        )

    if backend == "faster_whisper":
        return FasterWhisperTranscriber(
            model_name=config.model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            vad_min_silence_ms=config.vad_min_silence_ms,
        )

    raise RuntimeError(f"Unsupported STT backend: {config.backend}")
