from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from .audio import WaveInRecorder, find_device_id, pcm16le_rms
from .voice_config import ChatGPTVoiceConfig, VoiceLoopGuardConfig


@dataclass(slots=True)
class LoopDetection:
    score: float = 0.0
    delay_ms: int | None = None
    consecutive_matches: int = 0
    candidate_duration_ms: int = 0
    triggered: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LoopDetector:
    """Compares the CABLE-B level envelope with later CABLE-A audio."""

    def __init__(self, config: VoiceLoopGuardConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.feature_samples = max(1, sample_rate * config.feature_ms // 1000)
        self.window_bins = max(4, config.comparison_window_ms // config.feature_ms)
        self.min_lag_bins = max(1, config.min_delay_ms // config.feature_ms)
        effective_max_delay_ms = min(config.max_delay_ms, config.reliable_max_delay_ms)
        self.max_lag_bins = max(
            self.min_lag_bins,
            effective_max_delay_ms // config.feature_ms,
        )
        self.delay_tolerance_bins = max(1, config.delay_tolerance_ms // config.feature_ms)
        self.max_history_bins = self.window_bins + self.max_lag_bins + 10
        self._a_features = np.empty(0, dtype=np.float32)
        self._b_features = np.empty(0, dtype=np.float32)
        self._consecutive = 0
        self._candidate_lag_bins: int | None = None
        self._candidate_duration_bins = 0
        self._latched = False
        self._latched_score = 0.0
        self._latched_delay_ms: int | None = None
        self.last = LoopDetection()

    def reset(self) -> None:
        self._a_features = np.empty(0, dtype=np.float32)
        self._b_features = np.empty(0, dtype=np.float32)
        self._consecutive = 0
        self._candidate_lag_bins = None
        self._candidate_duration_bins = 0
        self._latched = False
        self._latched_score = 0.0
        self._latched_delay_ms = None
        self.last = LoopDetection()

    def _features(self, pcm: bytes) -> np.ndarray:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        usable = samples.size - (samples.size % self.feature_samples)
        if usable <= 0:
            return np.empty(0, dtype=np.float32)
        frames = samples[:usable].reshape(-1, self.feature_samples)
        return np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64)).astype(np.float32)

    def add_pcm(self, cable_a_pcm: bytes, cable_b_pcm: bytes) -> LoopDetection:
        return self.add_features(self._features(cable_a_pcm), self._features(cable_b_pcm))

    def add_features(self, cable_a: np.ndarray, cable_b: np.ndarray) -> LoopDetection:
        length = min(cable_a.size, cable_b.size)
        if length == 0:
            return self.last
        self._a_features = np.concatenate((self._a_features, cable_a[:length]))[
            -self.max_history_bins :
        ]
        self._b_features = np.concatenate((self._b_features, cable_b[:length]))[
            -self.max_history_bins :
        ]
        available = min(self._a_features.size, self._b_features.size)
        required = self.window_bins + self.min_lag_bins
        if available < required:
            self.last = LoopDetection(
                score=self._latched_score if self._latched else 0.0,
                delay_ms=self._latched_delay_ms if self._latched else None,
                consecutive_matches=self._consecutive,
                candidate_duration_ms=self._candidate_duration_bins * self.config.feature_ms,
                triggered=self._latched,
            )
            return self.last

        a_window = self._a_features[-self.window_bins :].astype(np.float64)
        a_rms = float(np.sqrt(np.mean(a_window * a_window)))
        best_score = -1.0
        best_lag: int | None = None
        upper_lag = min(self.max_lag_bins, available - self.window_bins)
        for lag in range(self.min_lag_bins, upper_lag + 1):
            b_end = available - lag
            b_start = b_end - self.window_bins
            if b_start < 0:
                continue
            b_window = self._b_features[b_start:b_end].astype(np.float64)
            b_rms = float(np.sqrt(np.mean(b_window * b_window)))
            if a_rms < self.config.rms_threshold or b_rms < self.config.rms_threshold:
                continue
            a_centered = a_window - a_window.mean()
            b_centered = b_window - b_window.mean()
            denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
            if denominator <= 1e-9:
                continue
            score = float(np.dot(a_centered, b_centered) / denominator)
            if score > best_score:
                best_score = score
                best_lag = lag

        matched = best_lag is not None and best_score >= self.config.correlation_threshold
        stable_lag = (
            matched
            and self._candidate_lag_bins is not None
            and abs(best_lag - self._candidate_lag_bins) <= self.delay_tolerance_bins
        )
        if stable_lag:
            self._consecutive += 1
            self._candidate_duration_bins += length
        elif matched:
            self._consecutive = 1
            self._candidate_lag_bins = best_lag
            self._candidate_duration_bins = length
        else:
            self._consecutive = 0
            self._candidate_lag_bins = None
            self._candidate_duration_bins = 0

        candidate_duration_ms = self._candidate_duration_bins * self.config.feature_ms
        if (
            not self._latched
            and self._consecutive >= self.config.min_consecutive_matches
            and candidate_duration_ms >= self.config.min_match_duration_ms
        ):
            self._latched = True
            self._latched_score = round(max(0.0, best_score), 4)
            self._latched_delay_ms = (
                best_lag * self.config.feature_ms if best_lag is not None else None
            )

        visible_score = self._latched_score if self._latched else round(max(0.0, best_score), 4)
        visible_delay_ms = (
            self._latched_delay_ms
            if self._latched
            else (best_lag * self.config.feature_ms if best_lag is not None else None)
        )
        self.last = LoopDetection(
            score=visible_score,
            delay_ms=visible_delay_ms,
            consecutive_matches=self._consecutive,
            candidate_duration_ms=candidate_duration_ms,
            triggered=self._latched,
        )
        return self.last


class LoopGuardService:
    def __init__(
        self,
        config: ChatGPTVoiceConfig,
        on_trigger: Callable[[LoopDetection], None],
        on_error: Callable[[str], None],
    ) -> None:
        self.config = config
        self.detector = LoopDetector(config.loop_guard, config.audio.sample_rate)
        self.on_trigger = on_trigger
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_a_rms = 0.0
        self._last_b_rms = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "enabled": self.config.loop_guard.enabled,
                "running": self.running,
                "auto_mute": self.config.loop_guard.auto_mute,
                "cable_a_rms": round(self._last_a_rms, 1),
                "cable_b_rms": round(self._last_b_rms, 1),
                **self.detector.last.to_dict(),
            }

    def reset(self) -> None:
        with self._lock:
            self.detector.reset()

    def set_enabled(self, enabled: bool) -> None:
        self.config.loop_guard.enabled = enabled
        if enabled:
            self.reset()
            self.start()
        else:
            self.stop()
            self.reset()

    def start(self) -> None:
        if self.running or not self.config.loop_guard.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="voice-loop-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def _run(self) -> None:
        recorders: list[WaveInRecorder] = []
        try:
            a_id = find_device_id("input", self.config.audio.chatgpt_input)
            b_id = find_device_id("input", self.config.audio.vrchat_microphone)
            a_recorder = WaveInRecorder(
                a_id,
                self.config.audio.sample_rate,
                self.config.audio.channels,
                self.config.audio.chunk_ms,
            )
            b_recorder = WaveInRecorder(
                b_id,
                self.config.audio.sample_rate,
                self.config.audio.channels,
                self.config.audio.chunk_ms,
            )
            recorders = [a_recorder, b_recorder]
            for recorder in recorders:
                recorder.open()
            was_triggered = False
            while not self._stop.is_set():
                a_pcm = a_recorder.read_chunk(timeout=0.5)
                b_pcm = b_recorder.read_chunk(timeout=0.5)
                if not a_pcm or not b_pcm:
                    continue
                with self._lock:
                    self._last_a_rms = pcm16le_rms(a_pcm)
                    self._last_b_rms = pcm16le_rms(b_pcm)
                    detection = self.detector.add_pcm(a_pcm, b_pcm)
                if detection.triggered and not was_triggered:
                    was_triggered = True
                    self.on_trigger(detection)
                if not detection.triggered:
                    was_triggered = False
        except Exception as exc:  # noqa: BLE001 - monitoring thread reports all device failures
            self.on_error(str(exc))
        finally:
            for recorder in recorders:
                recorder.close()
