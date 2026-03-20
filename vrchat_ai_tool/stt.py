from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import wave


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


@dataclass(slots=True)
class SystemSpeechTranscriber:
    culture: str
    timeout_sec: int

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
