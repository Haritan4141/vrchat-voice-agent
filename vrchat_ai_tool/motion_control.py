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


# Repeated values are deliberate weights. The immediately previous gesture is
# removed before drawing so the avatar never repeats the same accent back-to-back.
IDLE_GESTURE_CHOICES = (1, 1, 1, 5, 8, 8, 8, 9)
SPEAKING_GESTURE_CHOICES = (1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7)
SPEAKING_EXPRESSION_CHOICES = (0, 1, 2, 3, 4, 5, 6)
DIAGNOSTIC_GESTURES = tuple(range(1, 10))
DIAGNOSTIC_EXPRESSIONS = tuple(range(0, 7))
DIAGNOSTIC_STEP_SEC = 5.0


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

    def send_motion_expression(self, expression: int) -> None: ...


@dataclass(slots=True)
class MotionSnapshot:
    enabled: bool = True
    running: bool = False
    activity: int = int(MotionActivity.IDLE)
    activity_name: str = MotionActivity.IDLE.name
    energy: float = 0.0
    input_rms: float = 0.0
    last_gesture: int = 0
    last_expression: int = 0
    diagnostic_running: bool = False
    diagnostic_label: str = ""

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
        self._gesture_reset_at = 0.0
        self._next_expression_at = 0.0
        self._last_expression = 0
        self._diagnostic_running = False
        self._diagnostic_label = ""
        self._diagnostic_generation = 0
        self._diagnostic_gesture_generation = 0

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
            self._gesture_reset_at = 0.0
            self._next_gesture_at = now + self._gesture_interval(MotionActivity.IDLE)
            self._last_expression = 0
            self._next_expression_at = 0.0
            self._diagnostic_running = False
            self._diagnostic_label = ""
            self._diagnostic_generation += 1
            self._diagnostic_gesture_generation += 1
        self.osc.send_motion_enabled(self.config.enabled)
        self.osc.send_motion_activity(int(MotionActivity.IDLE))
        self.osc.send_motion_energy(0.0)
        self.osc.send_motion_gesture(0)
        self.osc.send_motion_expression(0)

    def stop(self) -> None:
        with self._lock:
            was_running = self._running
            self._running = False
            self._activity = MotionActivity.IDLE
            self._energy = 0.0
            self._gesture_reset_at = 0.0
            self._last_expression = 0
            self._next_expression_at = 0.0
            self._diagnostic_running = False
            self._diagnostic_label = ""
            self._diagnostic_generation += 1
            self._diagnostic_gesture_generation += 1
        if was_running:
            self.osc.send_motion_activity(int(MotionActivity.IDLE))
            self.osc.send_motion_energy(0.0)
            self.osc.send_motion_gesture(0)
            self.osc.send_motion_expression(0)
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
            self._gesture_reset_at = 0.0
            self._next_gesture_at = now + self._gesture_interval(MotionActivity.IDLE)
            self._last_expression = 0
            self._next_expression_at = 0.0
            self._diagnostic_running = False
            self._diagnostic_label = ""
            self._diagnostic_generation += 1
            self._diagnostic_gesture_generation += 1
        self.osc.send_motion_enabled(enabled)
        self.osc.send_motion_activity(int(MotionActivity.IDLE))
        self.osc.send_motion_energy(0.0)
        self.osc.send_motion_gesture(0)
        self.osc.send_motion_expression(0)

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
                last_expression=self._last_expression,
                diagnostic_running=self._diagnostic_running,
                diagnostic_label=self._diagnostic_label,
            )
        return value.to_dict()

    def start_diagnostic_test(self) -> None:
        """Run every synced accent and expression at a remotely visible pace.

        Audio-driven updates are temporarily held so this test can distinguish an
        OSC/Animator/sync problem from a CABLE-B level-detection problem. Each
        accent step lasts five seconds so another VRChat client can observe it.
        """
        with self._lock:
            self._require_diagnostic_available()
            self._diagnostic_generation += 1
            generation = self._diagnostic_generation
            self._diagnostic_gesture_generation += 1
            self._diagnostic_running = True
            self._diagnostic_label = "全動作テスト: 準備中"
            self._activity = MotionActivity.SPEAKING
            self._energy = 0.72
            self._last_gesture = 0
            self._gesture_reset_at = 0.0
            self._last_expression = 0

        self.osc.send_motion_activity(int(MotionActivity.SPEAKING))
        self.osc.send_motion_energy(0.72)
        self.osc.send_motion_gesture(0)
        self.osc.send_motion_expression(0)

        thread = threading.Thread(
            target=self._run_diagnostic_test,
            args=(generation,),
            name="voice-agent-motion-test",
            daemon=True,
        )
        thread.start()

    def stop_diagnostic_test(self) -> None:
        now = self._clock()
        with self._lock:
            self._diagnostic_generation += 1
            was_running = self._diagnostic_running
            self._diagnostic_running = False
            self._diagnostic_label = ""
            self._diagnostic_gesture_generation += 1
            self._activity = MotionActivity.IDLE
            self._energy = 0.0
            self._last_gesture = 0
            self._gesture_reset_at = 0.0
            self._last_expression = 0
            self._above_since = None
            self._below_since = None
            self._settling_until = 0.0
            self._next_gesture_at = now + self._gesture_interval(MotionActivity.IDLE)
            self._next_expression_at = 0.0
        if was_running:
            self.osc.send_motion_activity(int(MotionActivity.IDLE))
            self.osc.send_motion_energy(0.0)
            self.osc.send_motion_gesture(0)
            self.osc.send_motion_expression(0)

    def set_diagnostic_activity(self, activity: int) -> None:
        try:
            selected = MotionActivity(int(activity))
        except ValueError as exc:
            raise ValueError("activity must be 0 (idle), 1 (speaking), or 2 (settling)") from exc

        with self._lock:
            started = self._begin_manual_diagnostic()
            self._activity = selected
            self._energy = 0.72 if selected == MotionActivity.SPEAKING else 0.0
            self._diagnostic_label = f"状態確認: {selected.name}"
            energy = self._energy
        self.osc.send_motion_activity(int(selected))
        self.osc.send_motion_energy(energy)
        if started:
            self.osc.send_motion_gesture(0)

    def play_diagnostic_gesture(self, gesture: int) -> None:
        gesture = int(gesture)
        if gesture not in DIAGNOSTIC_GESTURES:
            raise ValueError("gesture must be between 1 and 9")

        with self._lock:
            started = self._begin_manual_diagnostic()
            if started:
                self._activity = MotionActivity.SPEAKING
                self._energy = 0.72
            self._last_gesture = gesture
            self._diagnostic_label = f"アクセント確認: {gesture}/9"
            activity = self._activity
            energy = self._energy
        if started:
            self.osc.send_motion_activity(int(activity))
            self.osc.send_motion_energy(energy)
        self._send_diagnostic_gesture(gesture)

    def set_diagnostic_expression(self, expression: int) -> None:
        expression = int(expression)
        if expression not in DIAGNOSTIC_EXPRESSIONS:
            raise ValueError("expression must be between 0 and 6")

        with self._lock:
            started = self._begin_manual_diagnostic()
            # Speaking expressions are intentionally gated by Activity=1 in FX.
            # Selecting a face from the GUI must therefore enter speaking mode
            # even when the previous manual check was idle or settling.
            self._activity = MotionActivity.SPEAKING
            self._energy = 0.72
            self._last_expression = expression
            self._diagnostic_label = f"表情確認: {expression}/6"
            activity = self._activity
            energy = self._energy
        self.osc.send_motion_activity(int(activity))
        self.osc.send_motion_energy(energy)
        if started:
            self.osc.send_motion_gesture(0)
        self.osc.send_motion_expression(expression)

    def _begin_manual_diagnostic(self) -> bool:
        self._require_diagnostic_available()
        started = not self._diagnostic_running
        self._diagnostic_generation += 1
        self._diagnostic_running = True
        if started:
            self._diagnostic_gesture_generation += 1
            self._gesture_reset_at = 0.0
            self._last_gesture = 0
        return started

    def _require_diagnostic_available(self) -> None:
        if not self._running:
            raise RuntimeError("Avatar motion service is not running")
        if not self.osc.motion_enabled:
            raise RuntimeError("Enable avatar motion before running the test")

    def _send_diagnostic_gesture(
        self,
        gesture: int,
        *,
        diagnostic_generation: int | None = None,
    ) -> bool:
        hold_sec = self._gesture_sync_hold_sec()
        with self._lock:
            if (
                not self._diagnostic_running
                or (
                    diagnostic_generation is not None
                    and diagnostic_generation != self._diagnostic_generation
                )
            ):
                return False
            self._diagnostic_gesture_generation += 1
            gesture_generation = self._diagnostic_gesture_generation
            self._gesture_reset_at = self._clock() + hold_sec
        self.osc.send_motion_gesture(gesture)
        threading.Thread(
            target=self._reset_diagnostic_gesture_after,
            args=(gesture_generation, hold_sec),
            name="voice-agent-gesture-reset",
            daemon=True,
        ).start()
        return True

    def _reset_diagnostic_gesture_after(self, generation: int, hold_sec: float) -> None:
        time.sleep(hold_sec)
        with self._lock:
            should_reset = (
                self._diagnostic_running
                and generation == self._diagnostic_gesture_generation
            )
            if should_reset:
                self._gesture_reset_at = 0.0
        if should_reset:
            self.osc.send_motion_gesture(0)

    def _run_diagnostic_test(self, generation: int) -> None:
        try:
            for index, gesture in enumerate(DIAGNOSTIC_GESTURES, start=1):
                expression = DIAGNOSTIC_EXPRESSIONS[(index - 1) % len(DIAGNOSTIC_EXPRESSIONS)]
                with self._lock:
                    if (
                        not self._diagnostic_running
                        or generation != self._diagnostic_generation
                    ):
                        return
                    self._last_gesture = gesture
                    self._last_expression = expression
                    self._diagnostic_label = (
                        f"全動作テスト: {index}/9 "
                        f"(アクセント{gesture}・表情{expression})"
                    )
                self.osc.send_motion_expression(expression)
                if not self._send_diagnostic_gesture(
                    gesture,
                    diagnostic_generation=generation,
                ):
                    return
                time.sleep(DIAGNOSTIC_STEP_SEC)

            with self._lock:
                if (
                    not self._diagnostic_running
                    or generation != self._diagnostic_generation
                ):
                    return
                self._activity = MotionActivity.SETTLING
                self._energy = 0.0
                self._last_expression = 0
                self._diagnostic_label = "全動作テスト: 収束確認"
            self.osc.send_motion_activity(int(MotionActivity.SETTLING))
            self.osc.send_motion_energy(0.0)
            self.osc.send_motion_expression(0)
            time.sleep(2.0)

            with self._lock:
                if (
                    not self._diagnostic_running
                    or generation != self._diagnostic_generation
                ):
                    return
                self._activity = MotionActivity.IDLE
                self._diagnostic_label = "全動作テスト: 待機確認"
            self.osc.send_motion_activity(int(MotionActivity.IDLE))
            time.sleep(2.0)
        finally:
            with self._lock:
                should_stop = (
                    self._diagnostic_running
                    and generation == self._diagnostic_generation
                )
            if should_stop:
                self.stop_diagnostic_test()

    def on_audio_level(self, rms: float) -> None:
        now = self._clock()
        sends: list[tuple[str, object]] = []
        with self._lock:
            if not self._running:
                return
            self._input_rms = max(0.0, float(rms))
            if self._diagnostic_running:
                return

            gesture_reset_sent = False
            if self._gesture_reset_at > 0.0 and now >= self._gesture_reset_at:
                self._gesture_reset_at = 0.0
                gesture_reset_sent = True
                sends.append(("gesture", 0))
            blocked = self.osc.mute_state is True or self.osc.status == AgentStatus.ERROR
            motion_enabled = self.osc.motion_enabled
            effective_rms = 0.0 if blocked or not motion_enabled else self._input_rms

            next_activity = self._update_activity(effective_rms, now)
            if next_activity != self._activity:
                self._activity = next_activity
                self._next_gesture_at = now + self._gesture_interval(next_activity)
                sends.append(("activity", int(next_activity)))
                if next_activity == MotionActivity.SPEAKING and motion_enabled and not blocked:
                    expression = self._choose_expression()
                    self._last_expression = expression
                    self._next_expression_at = now + self._expression_interval()
                    sends.append(("expression", expression))
                else:
                    self._next_expression_at = 0.0
                    if self._last_expression != 0:
                        self._last_expression = 0
                        sends.append(("expression", 0))

            if (blocked or not motion_enabled) and self._last_expression != 0:
                self._last_expression = 0
                self._next_expression_at = 0.0
                sends.append(("expression", 0))

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
                and not gesture_reset_sent
                and self._gesture_reset_at <= 0.0
                and now >= self._next_gesture_at
                and self._activity != MotionActivity.SETTLING
            ):
                gesture = self._choose_gesture(self._activity)
                self._last_gesture = gesture
                self._gesture_reset_at = now + self._gesture_sync_hold_sec()
                self._next_gesture_at = now + self._gesture_interval(self._activity)
                sends.append(("gesture", gesture))

            if (
                motion_enabled
                and not blocked
                and self._activity == MotionActivity.SPEAKING
                and now >= self._next_expression_at
            ):
                expression = self._choose_expression()
                self._last_expression = expression
                self._next_expression_at = now + self._expression_interval()
                sends.append(("expression", expression))

        for kind, value in sends:
            if kind == "activity":
                self.osc.send_motion_activity(int(value))
            elif kind == "energy":
                self.osc.send_motion_energy(float(value))
            elif kind == "gesture":
                self.osc.send_motion_gesture(int(value))
            else:
                self.osc.send_motion_expression(int(value))

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

    def _gesture_sync_hold_sec(self) -> float:
        # Custom synced parameters use VRChat's Playable sync, which can take up
        # to about one second. Keep the non-zero accent value alive long enough
        # for remote clients to observe it before returning to neutral.
        return max(0.1, float(self.config.gesture_sync_hold_sec))

    def _choose_gesture(self, activity: MotionActivity) -> int:
        choices = (
            IDLE_GESTURE_CHOICES
            if activity == MotionActivity.IDLE
            else SPEAKING_GESTURE_CHOICES
        )
        available = [value for value in choices if value != self._last_gesture] or choices
        return self._rng.choice(available)

    def _expression_interval(self) -> float:
        low = max(0.1, self.config.speaking_expression_min_sec)
        high = max(0.1, self.config.speaking_expression_max_sec)
        low, high = sorted((low, high))
        return self._rng.uniform(low, high)

    def _choose_expression(self) -> int:
        available = [
            value
            for value in SPEAKING_EXPRESSION_CHOICES
            if value != self._last_expression
        ]
        return self._rng.choice(available or list(SPEAKING_EXPRESSION_CHOICES))
