from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .chatgpt_ui_diagnostic import UiElementRecord, UiScanResult
from .chatgpt_ui_state import UiActivityState
from .stt import FasterWhisperTranscriber
from .voice_config import VoiceCaptionConfig

CAPTION_MODES = {"off", "uia", "stt"}
STT_QUALITY_PRESETS = {
    "standard": {
        "model": "small",
        "beam_size": 1,
        "partial_interval_sec": 2.5,
        "end_silence_ms": 700,
    },
    "accuracy": {
        "model": "medium",
        "beam_size": 5,
        "partial_interval_sec": 4.0,
        "end_silence_ms": 900,
    },
}
_TEXT_CONTENT_PATTERN = re.compile(r"[0-9A-Za-zぁ-んァ-ヶ一-龯]")
_UI_TEXT_MARKERS = (
    "ウェブを検索中",
    "webを検索中",
    "web検索中",
    "searching the web",
    "searching web",
    "思考中",
    "thinking",
    "作業中",
    "working",
    "停止",
    "stop",
    "キャンセル",
    "cancel",
)


class CaptionOsc(Protocol):
    def send_chatbox(self, text: str, *, notify: bool = False) -> None: ...

    def send_chatbox_typing(self, typing: bool) -> None: ...


class CaptionTranscriber(Protocol):
    def warm_up(self) -> None: ...

    def transcribe_wav(self, wave_path: Path) -> str: ...


def normalize_caption_text(value: object) -> str:
    """Collapse UIA/STT whitespace while retaining intentional Japanese punctuation."""
    return " ".join(str(value or "").replace("\u200b", "").split()).strip()


def format_chatbox_text(text: str, prefix: str, max_chars: int) -> str:
    """Fit a rolling caption into VRChat's documented 144-character chatbox limit."""
    limit = max(1, min(144, int(max_chars)))
    clean_prefix = normalize_caption_text(prefix)
    if clean_prefix and str(prefix).endswith(" "):
        clean_prefix += " "
    clean_text = normalize_caption_text(text)
    available = max(1, limit - len(clean_prefix))
    if len(clean_text) > available:
        clean_text = "…" + clean_text[-max(1, available - 1) :]
    return (clean_prefix + clean_text)[:limit]


def _rectangle_key(record: UiElementRecord) -> tuple[int, int]:
    try:
        left, top, _right, _bottom = (int(value) for value in record.rectangle.split(","))
    except (TypeError, ValueError):
        return (-1, -1)
    return (top, left)


def _is_possible_caption(record: UiElementRecord) -> bool:
    if record.control_type.casefold() != "text":
        return False
    text = normalize_caption_text(record.name)
    if len(text) < 2 or not _TEXT_CONTENT_PATTERN.search(text):
        return False
    folded = text.casefold()
    return not any(marker.casefold() == folded for marker in _UI_TEXT_MARKERS)


@dataclass(slots=True)
class _UiCandidate:
    record: UiElementRecord
    first_seen_order: int
    update_count: int = 0
    prefix_growth_count: int = 0


class UiaCaptionExtractor:
    """Select text created during one CABLE-B utterance from a flat UIA snapshot.

    The baseline prevents historical conversation text from being broadcast. Dynamic
    candidates are preferred over static additions, then the visually lowest element,
    which is normally the newest response in ChatGPT's conversation view.
    """

    def __init__(self) -> None:
        self._baseline_by_locator: dict[str, str] = {}
        self._baseline_texts: set[str] = set()
        self._candidates: dict[str, _UiCandidate] = {}
        self._order = 0

    def begin(self, baseline: UiScanResult | None) -> None:
        self._baseline_by_locator.clear()
        self._baseline_texts.clear()
        self._candidates.clear()
        self._order = 0
        if baseline is None:
            return
        for locator, record in baseline.elements.items():
            if not _is_possible_caption(record):
                continue
            text = normalize_caption_text(record.name)
            self._baseline_by_locator[locator] = text
            self._baseline_texts.add(text)

    def reset(self) -> None:
        self.begin(None)

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    def update(self, result: UiScanResult) -> str | None:
        current_locators = set(result.elements)
        for locator in tuple(self._candidates):
            if locator not in current_locators:
                self._candidates.pop(locator, None)

        for locator, record in result.elements.items():
            if not _is_possible_caption(record):
                continue
            text = normalize_caption_text(record.name)
            baseline_text = self._baseline_by_locator.get(locator)
            candidate = self._candidates.get(locator)
            changed_from_baseline = baseline_text is not None and text != baseline_text
            new_after_baseline = baseline_text is None and text not in self._baseline_texts
            if candidate is None and not (changed_from_baseline or new_after_baseline):
                continue
            if candidate is None:
                self._order += 1
                self._candidates[locator] = _UiCandidate(record, self._order)
                continue
            previous = normalize_caption_text(candidate.record.name)
            if previous != text:
                candidate.update_count += 1
                if text.startswith(previous) and len(text) > len(previous):
                    candidate.prefix_growth_count += 1
            candidate.record = record

        if not self._candidates:
            return None

        candidate = max(
            self._candidates.values(),
            key=lambda item: (
                item.prefix_growth_count > 0,
                item.update_count,
                _rectangle_key(item.record),
                item.first_seen_order,
                len(item.record.name),
            ),
        )
        return normalize_caption_text(candidate.record.name) or None


@dataclass(frozen=True, slots=True)
class _SttJob:
    sequence: int
    utterance_id: int
    pcm: bytes
    final: bool
    submitted_at: float
    warmup_only: bool = False
    reload_model: bool = False


@dataclass(frozen=True, slots=True)
class _PendingCaption:
    text: str
    source: str
    generation: int


class CaptionService:
    """Publish UIA or local CABLE-B speech captions to the VRChat chatbox."""

    def __init__(
        self,
        config: VoiceCaptionConfig,
        osc: CaptionOsc,
        *,
        transcriber_factory: Callable[[], CaptionTranscriber] | None = None,
        sample_rate: int = 48000,
        channels: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        # Existing standard configurations may intentionally tune segmentation
        # thresholds. Preserve those values; the accuracy preset deliberately
        # replaces them with its longer-context settings.
        self._apply_stt_quality(
            config.stt_quality,
            force=config.stt_quality.strip().casefold() == "accuracy",
        )
        self.osc = osc
        self.clock = clock
        self.transcriber_factory = transcriber_factory or self._create_transcriber
        self.sample_rate = max(1, int(sample_rate))
        self.channels = max(1, int(channels))
        self._lock = threading.RLock()
        self._output_condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._output_thread: threading.Thread | None = None
        self._stt_thread: threading.Thread | None = None
        self._stt_queue: queue.Queue[_SttJob] = queue.Queue(maxsize=1)
        self._transcriber: CaptionTranscriber | None = None
        self._mode = self._validate_mode(config.mode)
        self._generation = 0
        self._pending: _PendingCaption | None = None
        self._last_send_at = 0.0
        self._last_text = ""
        self._last_source = ""
        self._last_error = ""
        self._send_count = 0
        self._typing = False
        self._speaking = False
        self._last_ui_scan: UiScanResult | None = None
        self._ui_state = UiActivityState.IDLE
        self._uia = UiaCaptionExtractor()
        self._uia_not_before = 0.0
        self._uia_grace_until = 0.0
        self._uia_candidate_count = 0
        self._stt_state = "not_loaded"
        self._last_stt_latency_ms: float | None = None
        # LoopGuard supplies 100 ms chunks by default. Three chunks keep enough
        # lead-in to avoid clipping the first syllable without adding much latency.
        self._pre_roll: deque[bytes] = deque(maxlen=3)
        self._utterance_frames: list[bytes] = []
        self._utterance_id = 0
        self._utterance_started_at = 0.0
        self._silence_ms = 0
        self._last_partial_at = 0.0
        self._job_sequence = 0
        self._latest_job_by_utterance: dict[int, int] = {}

    @staticmethod
    def _validate_mode(mode: str) -> str:
        normalized = str(mode).strip().casefold()
        if normalized not in CAPTION_MODES:
            raise ValueError("caption mode must be off, uia, or stt")
        return normalized

    @staticmethod
    def _validate_stt_quality(quality: str) -> str:
        normalized = str(quality).strip().casefold()
        if normalized not in STT_QUALITY_PRESETS:
            raise ValueError("STT quality must be standard or accuracy")
        return normalized

    def _apply_stt_quality(self, quality: str, *, force: bool = True) -> str:
        normalized = self._validate_stt_quality(quality)
        preset = STT_QUALITY_PRESETS[normalized]
        self.config.stt_quality = normalized
        if force:
            self.config.stt_model = str(preset["model"])
            self.config.stt_beam_size = int(preset["beam_size"])
            self.config.stt_partial_interval_sec = float(preset["partial_interval_sec"])
            self.config.stt_end_silence_ms = int(preset["end_silence_ms"])
        return normalized

    @property
    def running(self) -> bool:
        thread = self._output_thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._output_thread = threading.Thread(
            target=self._output_loop,
            name="voice-caption-output",
            daemon=True,
        )
        self._stt_thread = threading.Thread(
            target=self._stt_loop,
            name="voice-caption-stt",
            daemon=True,
        )
        self._output_thread.start()
        self._stt_thread.start()
        if self._mode == "stt":
            self._request_stt_warmup()

    def stop(self) -> None:
        self._stop_event.set()
        with self._output_condition:
            self._pending = None
            self._output_condition.notify_all()
        try:
            self.osc.send_chatbox_typing(False)
        except Exception as exc:  # noqa: BLE001 - shutdown remains best effort
            with self._lock:
                self._last_error = f"Failed to clear chatbox typing state: {exc}"
        for thread in (self._output_thread, self._stt_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self._output_thread = None
        self._stt_thread = None

    def set_mode(self, mode: str) -> str:
        normalized = self._validate_mode(mode)
        with self._output_condition:
            self._mode = normalized
            self.config.mode = normalized
            self._generation += 1
            self._pending = None
            # A caption from the previous source is not evidence that the newly
            # selected source is working. Clear it so the LAN GUI reflects only
            # captions produced after this mode switch.
            self._last_text = ""
            self._last_source = ""
            self._last_error = ""
            self._reset_utterance_locked()
            self._uia.reset()
            self._uia_not_before = 0.0
            self._uia_grace_until = 0.0
            self._uia_candidate_count = 0
            self._set_typing_locked(False)
            self._output_condition.notify_all()
        if normalized == "stt":
            self._request_stt_warmup()
        return normalized

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": self._mode,
                "running": self.running,
                "speaking": self._speaking,
                "typing": self._typing,
                "last_text": self._last_text[-180:],
                "last_source": self._last_source,
                "last_error": self._last_error,
                "send_count": self._send_count,
                "uia_candidate_count": self._uia_candidate_count,
                "stt_state": self._stt_state,
                "stt_quality": self.config.stt_quality,
                "stt_model": self.config.stt_model,
                "stt_beam_size": self.config.stt_beam_size,
                "stt_latency_ms": self._last_stt_latency_ms,
            }

    def set_stt_quality(self, quality: str) -> str:
        """Apply and warm a new STT preset without restarting the control server."""
        with self._output_condition:
            normalized = self._apply_stt_quality(quality)
            self._generation += 1
            self._pending = None
            self._latest_job_by_utterance.clear()
            self._reset_utterance_locked()
            self._last_text = ""
            self._last_source = ""
            self._last_error = ""
            self._stt_state = "loading"
            self._last_stt_latency_ms = None
            self._set_typing_locked(False)
            self._job_sequence += 1
            job = _SttJob(
                self._job_sequence,
                self._utterance_id,
                b"",
                False,
                self.clock(),
                warmup_only=True,
                reload_model=True,
            )
            self._replace_stt_job(job)
            self._output_condition.notify_all()
        return normalized

    def on_ui_scan(self, result: UiScanResult) -> None:
        now = self.clock()
        with self._lock:
            self._last_ui_scan = result
            if self._mode != "uia" or (not self._speaking and now > self._uia_grace_until):
                return
            text = self._uia.update(result)
            self._uia_candidate_count = self._uia.candidate_count
            if not text:
                return
            # A user's completed voice transcript can be exposed by UIA just
            # before the assistant starts speaking. Keep the first changes
            # pending so a later/lower assistant response wins the extractor's
            # ranking instead of broadcasting the user's words.
            if now < self._uia_not_before:
                return
        self._publish(text, "uia")

    def on_ui_state(self, state: UiActivityState) -> None:
        """Freeze the last idle UI snapshot while ChatGPT builds a response."""
        with self._lock:
            self._ui_state = state

    def on_audio_chunk(self, pcm: bytes, rms: float) -> None:
        if not pcm:
            return
        now = self.clock()
        bytes_per_second = self.sample_rate * self.channels * 2
        chunk_ms = max(1, round(len(pcm) * 1000 / bytes_per_second))
        with self._lock:
            if not self._speaking:
                if rms < self.config.stt_speech_on_rms:
                    self._pre_roll.append(pcm)
                    return
                self._speaking = True
                self._utterance_id += 1
                self._utterance_started_at = now
                self._last_partial_at = now
                self._silence_ms = 0
                self._utterance_frames = [*self._pre_roll, pcm]
                self._pre_roll.clear()
                # Baseline the most recent scan at the exact CABLE-B onset. The
                # user's transcript normally appears before this point, while
                # assistant text appears during playback. An older idle
                # baseline cannot tell those two roles apart.
                self._uia.begin(self._last_ui_scan)
                self._uia_not_before = now + max(
                    0.0, self.config.uia_initial_hold_sec
                )
                self._uia_grace_until = 0.0
                self._uia_candidate_count = 0
                if self._mode != "off":
                    self._set_typing_locked(True)
                return

            self._utterance_frames.append(pcm)
            if rms > self.config.stt_speech_off_rms:
                self._silence_ms = 0
            else:
                self._silence_ms += chunk_ms

            duration_ms = int((now - self._utterance_started_at) * 1000)
            if (
                self._mode == "stt"
                and duration_ms >= self.config.stt_min_audio_ms
                and now - self._last_partial_at >= self.config.stt_partial_interval_sec
            ):
                self._submit_stt_locked(final=False, now=now)
                self._last_partial_at = now

            if (
                self._silence_ms >= self.config.stt_end_silence_ms
                or duration_ms >= self.config.stt_max_utterance_ms
            ):
                self._finish_utterance_locked(now)

    def send_test_caption(self) -> None:
        text = format_chatbox_text("字幕表示テストです", self.config.prefix, self.config.max_chars)
        self.osc.send_chatbox(text, notify=False)
        with self._lock:
            self._last_text = text
            self._last_source = "test"
            self._send_count += 1
            self._last_send_at = self.clock()

    def report_input_error(self, detail: str) -> None:
        with self._lock:
            self._last_error = f"CABLE-B caption input failed: {detail}"

    def _finish_utterance_locked(self, now: float) -> None:
        if self._mode == "stt" and self._utterance_frames:
            self._submit_stt_locked(final=True, now=now)
        if self._mode == "uia":
            self._uia_grace_until = now + max(0.0, self.config.uia_post_speech_grace_sec)
        self._set_typing_locked(False)
        self._speaking = False
        self._utterance_frames = []
        self._silence_ms = 0
        self._pre_roll.clear()

    def _reset_utterance_locked(self) -> None:
        self._speaking = False
        self._utterance_frames = []
        self._pre_roll.clear()
        self._silence_ms = 0
        self._uia_not_before = 0.0
        self._uia_candidate_count = 0

    def _set_typing_locked(self, typing: bool) -> None:
        typing = bool(typing)
        if typing == self._typing:
            return
        self.osc.send_chatbox_typing(typing)
        self._typing = typing

    def _publish(self, text: str, source: str) -> None:
        formatted = format_chatbox_text(text, self.config.prefix, self.config.max_chars)
        if not formatted:
            return
        with self._output_condition:
            if formatted == self._last_text and source == self._last_source:
                return
            self._pending = _PendingCaption(formatted, source, self._generation)
            self._output_condition.notify_all()

    def _output_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._output_condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._output_condition.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                wait_seconds = max(
                    0.0,
                    self.config.min_send_interval_sec - (self.clock() - self._last_send_at),
                )
                if wait_seconds > 0:
                    self._output_condition.wait(timeout=wait_seconds)
                    continue
                pending = self._pending
                self._pending = None
            if pending is None or pending.generation != self._generation:
                continue
            try:
                self.osc.send_chatbox(pending.text, notify=False)
                with self._lock:
                    self._last_text = pending.text
                    self._last_source = pending.source
                    self._last_send_at = self.clock()
                    self._send_count += 1
                    self._last_error = ""
            except Exception as exc:  # noqa: BLE001 - optional captions must not stop safety
                with self._lock:
                    self._last_error = f"OSC字幕送信に失敗しました: {exc}"

    def _request_stt_warmup(self) -> None:
        with self._lock:
            if self._transcriber is not None or self._stt_state == "loading":
                return
            self._stt_state = "loading"
            self._job_sequence += 1
            job = _SttJob(
                self._job_sequence,
                self._utterance_id,
                b"",
                False,
                self.clock(),
                warmup_only=True,
            )
        self._replace_stt_job(job)

    def _submit_stt_locked(self, *, final: bool, now: float) -> None:
        pcm = b"".join(self._utterance_frames)
        if not pcm:
            return
        duration_ms = len(pcm) * 1000 // (self.sample_rate * self.channels * 2)
        if duration_ms < self.config.stt_min_audio_ms:
            return
        self._job_sequence += 1
        sequence = self._job_sequence
        utterance_id = self._utterance_id
        self._latest_job_by_utterance[utterance_id] = sequence
        self._replace_stt_job(
            _SttJob(sequence, utterance_id, pcm, final, now)
        )

    def _replace_stt_job(self, job: _SttJob) -> None:
        try:
            self._stt_queue.put_nowait(job)
            return
        except queue.Full:
            pass
        try:
            self._stt_queue.get_nowait()
        except queue.Empty:
            pass
        self._stt_queue.put_nowait(job)

    def _stt_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._stt_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job.reload_model:
                    with self._lock:
                        self._transcriber = None
                transcriber = self._get_transcriber()
                if job.warmup_only:
                    continue
                text = self._transcribe_pcm(transcriber, job.pcm)
                latency_ms = max(0.0, (self.clock() - job.submitted_at) * 1000.0)
                with self._lock:
                    latest = self._latest_job_by_utterance.get(job.utterance_id)
                    current_mode = self._mode
                    self._last_stt_latency_ms = round(latency_ms, 1)
                if latest != job.sequence or current_mode != "stt":
                    continue
                if text:
                    self._publish(text, "stt")
            except Exception as exc:  # noqa: BLE001 - keep production monitor alive
                with self._lock:
                    self._stt_state = "error"
                    self._last_error = f"字幕STTに失敗しました: {exc}"

    def _get_transcriber(self) -> CaptionTranscriber:
        with self._lock:
            existing = self._transcriber
        if existing is not None:
            return existing
        transcriber = self.transcriber_factory()
        transcriber.warm_up()
        with self._lock:
            self._transcriber = transcriber
            self._stt_state = "ready"
            self._last_error = ""
        return transcriber

    def _create_transcriber(self) -> CaptionTranscriber:
        return FasterWhisperTranscriber(
            model_name=self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
            language=self.config.stt_language,
            beam_size=self.config.stt_beam_size,
            vad_filter=self.config.stt_vad_filter,
            vad_min_silence_ms=self.config.stt_vad_min_silence_ms,
        )

    def _transcribe_pcm(self, transcriber: CaptionTranscriber, pcm: bytes) -> str:
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                path = Path(temporary.name)
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm)
            return normalize_caption_text(transcriber.transcribe_wav(path))
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
