from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config, probe_http_endpoint
from .audio import find_device_id
from .runtime import BotRuntime, describe_devices, is_probably_same_virtual_route, speak_with_config
from .services import OllamaClient, VoicevoxClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrchat-ai-tool",
        description="Local VRChat voice assistant for desktop mode.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Load config and print a sanity summary.")
    doctor_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.example.toml"),
        help="Path to a TOML config file.",
    )
    doctor_parser.add_argument(
        "--check-services",
        action="store_true",
        help="Probe configured Ollama and VOICEVOX endpoints.",
    )
    doctor_parser.add_argument(
        "--check-devices",
        action="store_true",
        help="Resolve configured input and output devices.",
    )

    subparsers.add_parser("devices", help="List Windows waveIn/waveOut devices.")

    listen_parser = subparsers.add_parser("listen-once", help="Capture one utterance and transcribe it.")
    listen_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.example.toml"),
        help="Path to a TOML config file.",
    )
    listen_parser.add_argument(
        "--max-wait-sec",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for speech before giving up.",
    )
    listen_parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save the captured WAV into recordings/.",
    )

    speak_parser = subparsers.add_parser("speak", help="Synthesize text with VOICEVOX and play it.")
    speak_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.example.toml"),
        help="Path to a TOML config file.",
    )
    speak_parser.add_argument("--text", required=True, help="Text to synthesize.")
    speak_parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save the generated WAV into recordings/.",
    )

    run_parser = subparsers.add_parser("run", help="Run the realtime capture and reply loop.")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.example.toml"),
        help="Path to a TOML config file.",
    )
    run_parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save input and output WAV files into recordings/.",
    )

    return parser


def print_summary(config) -> None:
    print("VRChat AI Tool doctor")
    print(f"- capture.mode: {config.audio_capture.mode}")
    print(f"- capture.input_device: {config.audio_capture.input_device}")
    print(f"- output.tts_output_device: {config.audio_output.tts_output_device}")
    print(f"- stt.backend: {config.stt.backend}")
    print(f"- stt.model: {config.stt.model}")
    print(f"- stt.language: {config.stt.language}")
    print(f"- llm.backend: {config.llm.backend}")
    print(f"- llm.model: {config.llm.model}")
    print(f"- tts.backend: {config.tts.backend}")
    print(f"- tts.speaker: {config.tts.speaker}")


def run_doctor(config_path: Path, check_services: bool, check_devices: bool) -> int:
    config = load_config(config_path)
    print_summary(config)

    if check_services:
        print("")
        print("HTTP probe")
        ollama = OllamaClient(
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout_sec=config.llm.timeout_sec,
        )
        voicevox = VoicevoxClient(
            base_url=config.tts.base_url,
            speaker=config.tts.speaker,
            speed_scale=config.tts.speed_scale,
            timeout_sec=config.tts.timeout_sec,
        )
        for name, url in {
            "llm": ollama.healthcheck_url(),
            "tts": voicevox.healthcheck_url(),
        }.items():
            ok, detail = probe_http_endpoint(url)
            status = "ok" if ok else "fail"
            print(f"- {name}: {status} ({detail})")

    if check_devices:
        print("")
        print("Device resolution")
        input_device_id = find_device_id("input", config.audio_capture.input_device)
        output_device_id = find_device_id("output", config.audio_output.tts_output_device)
        print(f"- input: {input_device_id}")
        print(f"- output: {output_device_id}")
        if config.audio_output.monitor_output_device:
            monitor_device_id = find_device_id("output", config.audio_output.monitor_output_device)
            print(f"- monitor: {monitor_device_id}")
        if is_probably_same_virtual_route(
            config.audio_capture.input_device,
            config.audio_output.tts_output_device,
        ):
            print("- warning: capture and TTS output look like the same VB-CABLE route")
            print("  use a different virtual route for bot output, or VRChat audio may echo back into its mic")

    return 0


def run_devices() -> int:
    print(describe_devices())
    return 0


def run_listen_once(config_path: Path, max_wait_sec: float, save_audio: bool) -> int:
    config = load_config(config_path)
    with BotRuntime(config) as runtime:
        heard = runtime.capture_and_transcribe_once(max_wait_sec=max_wait_sec, save_audio=save_audio)
    if not heard.text:
        print("No speech detected.")
        return 1
    print(heard.text)
    if heard.wav_path is not None:
        print(f"saved: {heard.wav_path}")
    return 0


def run_speak(config_path: Path, text: str, save_audio: bool) -> int:
    config = load_config(config_path)
    wav_path = speak_with_config(config, text, save_audio=save_audio)
    if wav_path is not None:
        print(f"saved: {wav_path}")
    return 0


def run_pipeline(config_path: Path, save_audio: bool) -> int:
    config = load_config(config_path)
    with BotRuntime(config) as runtime:
        runtime.run_forever(save_audio=save_audio)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return run_doctor(args.config, args.check_services, args.check_devices)
        if args.command == "devices":
            return run_devices()
        if args.command == "listen-once":
            return run_listen_once(args.config, args.max_wait_sec, args.save_audio)
        if args.command == "speak":
            return run_speak(args.config, args.text, args.save_audio)
        if args.command == "run":
            return run_pipeline(args.config, args.save_audio)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2
