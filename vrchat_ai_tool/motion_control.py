from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Protocol

from .osc_control import AgentStatus
from .voice_config import VoiceMotionConfig


class MotionActivity(IntEnum):
    IDLE = 0
    SPEAKING = 1
    SETTLING = 2


class MotionOsc(Protocol):
    @property
    def mute_state(self) -> bool | None: ...

    @property
    def status(self) -> AgentStatus: ...

    @property
    def motion_enabled(self) -> bool: ...

    def send_motion_enabled(self, enabled: bool) -> None: ...

    def send_motion_activity(self, activity: int) -> None: ...

    def send_motion_energy(self, energy: float) -> None: ...

    def send_motion_gesture(self, gesture: int) -> None: ...


@dataclass(slots=True)
class MotionSnapshot:
    enabled: bool = True
    running: bool = False
    activity: int = int(MotionActivity.IDLE)
    activity_name: str = MotionActivity.IDLE.name
    energy: float = 0.0
    input_rms: float = 0.0
    last_gesture: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MotionService:
    """Turns CABLE-B output level into restrained avatar motion parameters."""

    def __init__(
        self,
        config: VoiceMotionConfig,
        osc: MotionOsc,
        *,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.osc = osc
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._running = False
        self._activity = MotionActivity.IDLE
        self._energy = 0.0
        self._input_rms = 0.0
        self._last_sent_energy = -1.0
        self._last_energy_sent_at = 0.0
        self._above_since: float | None = None
        self._below_since: float | None = None
        self._settling_until = 0.0
        self._next_gesture_at = 0.0
        self._last_gesture = 0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self.osc.motion_enabled if self._running else self.config.enabled

    def start(self) -> None:
        now = self._clock()
        with self._lock:
            self._running = True
            self._activity = MotionActivity.IDLE
            self._energy = 0.0
            self._above_since = None
            self._below_since = None
            self._settling_until = 0.0
            self._last_gesture = 0
            self._next_gesture_at = now + self._gesture_interval(MotionActivity.IDLE)
        self.osc.send_motion_enabled(self.config.enabled)
        self.osc.send_motion_activity(int(MotionActivity.IDLE))
        self.osc.send_motion_energy(0.0)
        self.osc.send_motion_gesture(0)

    def stop(self) -> None:
        with self._lock:
            was_running = self._running
            self._running = False
            self._activity = MotionActivity.IDLE
            self._energy = 0.0
        if was_running:
            self.osc.send_motion_activity(int(MotionActivity.IDLE))
            self.osc.send_motion_energy(0.0)
            self.osc.send_motion_gesture(0)
            self.osc.send_motion_enabled(False)

    def set_enabled(self, enabled: bool) -> None:
        now = self._clock()
        with self._lock:
            self.config.enabled = enabled
            self._above_since = None
            self._below_since = None
            self._activity = MotionActivity.IDLE
            self._energy = 0.0
            self._last_sent_energy = -1.0
            self._last_gesture = 0
            self._next_gesture_at = now + self._gesture_interval(MotionActivity.IDLE)
        self.osc.send_motion_enabled(enabled)
        self.osc.send_motion_activity(int(MotionActivity.IDLE))
        self.osc.send_motion_energy(0.0)
        self.osc.send_motion_gesture(0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            value = MotionSnapshot(
                enabled=self.osc.motion_enabled if self._running else self.config.enabled,
                running=self._running,
                activity=int(self._activity),
                activity_name=self._activity.name,
                energy=round(self._energy, 3),
                input_rms=round(self._input_rms, 1),
                last_gesture=self._last_gesture,
            )
        return value.to_dict()

    def on_audio_level(self, rms: float) -> None:
        now = self._clock()
        sends: list[tuple[str, object]] = []
        with self._lock:
            if not self._running:
                return
            self._input_rms = max(0.0, float(rms))
            blocked = self.osc.mute_state is True or self.osc.status == AgentStatus.ERROR
            motion_enabled = self.osc.motion_enabled
            effective_rms = 0.0 if blocked or not motion_enabled else self._input_rms

            next_activity = self._update_activity(effective_rms, now)
            if next_activity != self._activity:
                self._activity = next_activity
                self._next_gesture_at = now + self._gesture_interval(next_activity)
                sends.append(("activity", int(next_activity)))

            target_energy = self._normalise_energy(effective_rms)
            if self._activity != MotionActivity.SPEAKING:
                target_energy = 0.0
            smoothing = min(1.0, max(0.01, self.config.energy_smoothing))
            self._energy += (target_energy - self._energy) * smoothing
            if self._energy < 0.003:
                self._energy = 0.0
            if (
                abs(self._energy - self._last_sent_energy) >= 0.02
                or now - self._last_energy_sent_at >= 0.5
            ):
                self._last_sent_energy = self._energy
                self._last_energy_sent_at = now
                sends.append(("energy", round(self._energy, 3)))

            if (
                motion_enabled
                and not blocked
                and now >= self._next_gesture_at
                and self._activity != MotionActivity.SETTLING
            ):
                gesture = self._choose_gesture(self._activity)
                self._last_gesture = gesture
                self._next_gesture_at = now + self._gesture_interval(self._activity)
                sends.append(("gesture", gesture))

        for kind, value in sends:
            if kind == "activity":
                self.osc.send_motion_activity(int(value))
            elif kind == "energy":
                self.osc.send_motion_energy(float(value))
            else:
                self.osc.send_motion_gesture(int(value))

    def _update_activity(self, rms: float, now: float) -> MotionActivity:
        if self._activity == MotionActivity.IDLE:
            self._below_since = None
            if rms >= self.config.speech_on_rms:
                if self._above_since is None:
                    self._above_since = now
                if (now - self._above_since) * 1000 + 1e-6 >= self.config.attack_ms:
                    self._above_since = None
                    return MotionActivity.SPEAKING
            else:
                self._above_since = None
            return MotionActivity.IDLE

        if self._activity == MotionActivity.SPEAKING:
            self._above_since = None
            if rms <= self.config.speech_off_rms:
                if self._below_since is None:
                    self._below_since = now
                if (now - self._below_since) * 1000 + 1e-6 >= self.config.release_ms:
                    self._below_since = None
                    self._settling_until = now + self.config.settling_ms / 1000
                    return MotionActivity.SETTLING
            else:
                self._below_since = None
            return MotionActivity.SPEAKING

        if rms >= self.config.speech_on_rms:
            self._settling_until = 0.0
            return MotionActivity.SPEAKING
        if now >= self._settling_until:
            return MotionActivity.IDLE
        return MotionActivity.SETTLING

    def _normalise_energy(self, rms: float) -> float:
        span = max(1.0, self.config.energy_ceiling_rms - self.config.energy_floor_rms)
        return min(1.0, max(0.0, (rms - self.config.energy_floor_rms) / span))

    def _gesture_interval(self, activity: MotionActivity) -> float:
        if activity == MotionActivity.SPEAKING:
            low = self.config.speaking_gesture_min_sec
            high = self.config.speaking_gesture_max_sec
        else:
            low = self.config.idle_gesture_min_sec
            high = self.config.idle_gesture_max_sec
        low, high = sorted((max(0.1, low), max(0.1, high)))
        return self._rng.uniform(low, high)

    def _choose_gesture(self, activity: MotionActivity) -> int:
        choices = [1, 1, 2, 3] if activity == MotionActivity.IDLE else [1, 2, 3, 4]
        available = [value for value in choices if value != self._last_gesture] or choices
        return self._rng.choice(available)
