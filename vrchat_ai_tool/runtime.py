from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
import time

from .audio import (
    WaveInRecorder,
    find_device_id,
    list_input_devices,
    list_output_devices,
    play_wav_to_devices,
    save_pcm_as_wav,
)
from .config import AppConfig
from .services import OllamaClient, VoicevoxClient
from .stt import create_transcriber


def clean_reply_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _normalize_route_name(name: str) -> str:
    normalized = " ".join(name.casefold().split())
    for token in ("input", "output", "speakers", "speaker", "microphone", "mic", "マイク", "スピーカー"):
        normalized = normalized.replace(token, "")
    return " ".join(normalized.split())


def is_probably_same_virtual_route(capture_name: str, output_name: str) -> bool:
    if not capture_name or not output_name:
        return False

    normalized_capture = _normalize_route_name(capture_name)
    normalized_output = _normalize_route_name(output_name)
    if "cable" not in normalized_capture or "cable" not in normalized_output:
        return False
    return normalized_capture == normalized_output


def resolve_output_device_ids(config: AppConfig) -> tuple[int, int | None]:
    output_device_id = find_device_id("output", config.audio_output.tts_output_device)
    monitor_device_id = (
        find_device_id("output", config.audio_output.monitor_output_device)
        if config.audio_output.monitor_output_device
        else None
    )
    return output_device_id, monitor_device_id


def speak_with_config(config: AppConfig, text: str, save_audio: bool = False) -> Path | None:
    base_dir = Path.cwd()
    recordings_dir = base_dir / "recordings"
    voicevox = VoicevoxClient(
        base_url=config.tts.base_url,
        speaker=config.tts.speaker,
        speed_scale=config.tts.speed_scale,
        timeout_sec=config.tts.timeout_sec,
    )
    wav_bytes = voicevox.synthesize(text)
    wav_path: Path | None = None
    if save_audio:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recordings_dir.mkdir(parents=True, exist_ok=True)
        wav_path = recordings_dir / f"{timestamp}_output.wav"
        wav_path.write_bytes(wav_bytes)

    output_device_id, monitor_device_id = resolve_output_device_ids(config)
    output_ids = [output_device_id]
    if monitor_device_id is not None:
        output_ids.append(monitor_device_id)
    play_wav_to_devices(wav_bytes, output_ids)
    return wav_path


@dataclass(slots=True)
class HeardUtterance:
    text: str
    wav_path: Path | None


class BotRuntime:
    def __init__(self, config: AppConfig, base_dir: Path | None = None) -> None:
        self.config = config
        self.base_dir = base_dir or Path.cwd()
        self.recordings_dir = self.base_dir / "recordings"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.input_device_id = find_device_id("input", config.audio_capture.input_device)
        self.output_device_id = find_device_id("output", config.audio_output.tts_output_device)
        self.monitor_device_id = (
            find_device_id("output", config.audio_output.monitor_output_device)
            if config.audio_output.monitor_output_device
            else None
        )

        self.recorder = WaveInRecorder(
            device_id=self.input_device_id,
            sample_rate=config.audio_capture.sample_rate,
            channels=config.audio_capture.channels,
            chunk_ms=config.audio_capture.chunk_ms,
        )
        self.transcriber = create_transcriber(config.stt)
        self.ollama = OllamaClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout_sec=config.llm.timeout_sec,
        )
        self.voicevox = VoicevoxClient(
            base_url=config.tts.base_url,
            speaker=config.tts.speaker,
            speed_scale=config.tts.speed_scale,
            timeout_sec=config.tts.timeout_sec,
        )
        self.history: deque[dict[str, str]] = deque(maxlen=8)
        self.last_reply_at = 0.0

    def is_probably_same_virtual_route(self) -> bool:
        return is_probably_same_virtual_route(
            self.config.audio_capture.input_device,
            self.config.audio_output.tts_output_device,
        )

    def start(self) -> None:
        self.recorder.open()

    def stop(self) -> None:
        self.recorder.close()

    def __enter__(self) -> "BotRuntime":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.stop()

    def capture_and_transcribe_once(
        self,
        max_wait_sec: float | None = None,
        save_audio: bool = False,
    ) -> HeardUtterance:
        pcm_data = self.recorder.record_until_silence(
            rms_threshold=self.config.audio_capture.rms_threshold,
            min_speech_ms=self.config.audio_capture.min_speech_ms,
            silence_timeout_ms=self.config.audio_capture.silence_timeout_ms,
            max_utterance_ms=self.config.audio_capture.max_utterance_ms,
            max_wait_sec=max_wait_sec,
        )
        if not pcm_data:
            return HeardUtterance(text="", wav_path=None)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_audio:
            input_path = self.recordings_dir / f"{timestamp}_input.wav"
            save_pcm_as_wav(
                input_path,
                pcm_data=pcm_data,
                sample_rate=self.config.audio_capture.sample_rate,
                channels=self.config.audio_capture.channels,
            )
        else:
            temp_file = tempfile.NamedTemporaryFile(
                prefix="input_",
                suffix=".wav",
                dir=self.artifacts_dir,
                delete=False,
            )
            temp_file.close()
            input_path = Path(temp_file.name)
            save_pcm_as_wav(
                input_path,
                pcm_data=pcm_data,
                sample_rate=self.config.audio_capture.sample_rate,
                channels=self.config.audio_capture.channels,
            )

        try:
            text = self.transcriber.transcribe_wav(input_path)
        finally:
            if not save_audio and input_path.exists():
                input_path.unlink(missing_ok=True)

        return HeardUtterance(text=text, wav_path=input_path if save_audio else None)

    def speak_text(self, text: str, save_audio: bool = False) -> Path | None:
        should_pause_listening = self.config.conversation.pause_listening_while_speaking
        if should_pause_listening:
            self.recorder.close()

        wav_bytes = self.voicevox.synthesize(text)
        wav_path: Path | None = None
        if save_audio:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            wav_path = self.recordings_dir / f"{timestamp}_output.wav"
            wav_path.write_bytes(wav_bytes)

        try:
            output_ids = [self.output_device_id]
            if self.monitor_device_id is not None:
                output_ids.append(self.monitor_device_id)
            play_wav_to_devices(wav_bytes, output_ids)
        finally:
            if should_pause_listening:
                self.recorder.open()
        return wav_path

    def build_messages(self, transcript: str) -> list[dict[str, str]]:
        user_prompt = [
            "以下はVRChatで今聞こえた発話です。",
            transcript,
            "",
            f"{self.config.conversation.max_response_chars}文字以内で自然に短く返答してください。",
            "相手の話を優先して拾い、AIっぽい説明はしないでください。",
        ]
        if self.config.conversation.allow_topic_suggestions:
            user_prompt.append("会話が止まりそうなら、軽い話題を1つだけ添えても構いません。")
        else:
            user_prompt.append("話題を広げすぎず、返答だけに留めてください。")

        messages = [{"role": "system", "content": self.config.llm.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": "\n".join(user_prompt)})
        return messages

    def generate_reply(self, transcript: str) -> str:
        messages = self.build_messages(transcript)
        reply = self.ollama.chat(messages)
        reply = clean_reply_text(reply, self.config.conversation.max_response_chars)
        if not reply:
            reply = "うん、なるほど。"
        return reply

    def run_forever(self, save_audio: bool = False) -> None:
        while True:
            heard = self.capture_and_transcribe_once(save_audio=save_audio)
            transcript = heard.text.strip()
            if not transcript:
                continue

            print(f"[heard] {transcript}")

            now = time.monotonic()
            if now - self.last_reply_at < self.config.conversation.min_reply_interval_sec:
                print("[skip] cooldown active")
                self.history.append({"role": "user", "content": transcript})
                continue

            reply = self.generate_reply(transcript)
            print(f"[reply] {reply}")
            self.speak_text(reply, save_audio=save_audio)
            self.last_reply_at = time.monotonic()
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": reply})


def describe_devices() -> str:
    lines = ["Input devices:"]
    for device in list_input_devices():
        lines.append(f"- [{device.id}] {device.name}")
    lines.append("")
    lines.append("Output devices:")
    for device in list_output_devices():
        lines.append(f"- [{device.id}] {device.name}")
    return "\n".join(lines)
