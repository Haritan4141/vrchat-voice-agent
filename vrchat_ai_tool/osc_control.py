from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import IntEnum

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from .voice_config import VoiceOscConfig


class AgentStatus(IntEnum):
    STOPPED = 0
    ONLINE = 1
    ERROR = 2
    MAINTENANCE = 3


class VRChatOscController:
    """Confirmed VRChat mute control and generic avatar status output."""

    def __init__(
        self,
        config: VoiceOscConfig,
        client_factory: Callable[[str, int], SimpleUDPClient] = SimpleUDPClient,
        server_factory: Callable[..., ThreadingOSCUDPServer] = ThreadingOSCUDPServer,
    ) -> None:
        if config.voice_input_mode.casefold() != "toggle":
            raise ValueError("Remote mute currently requires osc.voice_input_mode='toggle'.")
        self.config = config
        self._client = client_factory(config.target_host, config.input_port)
        self._server_factory = server_factory
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._mute_state: bool | None = None
        self._mute_version = 0
        self._status = AgentStatus.STOPPED
        self._observed_status: AgentStatus | None = None
        self._status_version = 0
        self._motion_enabled = True
        self._observed_motion_enabled: bool | None = None
        self._motion_activity = 0
        self._observed_motion_activity: int | None = None
        self._motion_energy = 0.0
        self._observed_motion_energy: float | None = None
        self._motion_gesture = 0
        self._observed_motion_gesture: int | None = None
        self._motion_expression = 0
        self._observed_motion_expression: int | None = None
        self._thinking = False
        self._observed_thinking: bool | None = None
        self._probe = False
        self._observed_probe: bool | None = None
        self._probe_version = 0
        self._probe_feedback_at = 0.0
        self._probe_lock = threading.Lock()
        self._avatar_id: str | None = None
        self._last_feedback_at = 0.0
        self._avatar_replay_generation = 0

    @property
    def mute_state(self) -> bool | None:
        with self._condition:
            return self._mute_state

    @property
    def status(self) -> AgentStatus:
        with self._condition:
            return self._status

    @property
    def motion_enabled(self) -> bool:
        with self._condition:
            return self._motion_enabled

    def start(self) -> None:
        if self._server is not None:
            return
        dispatcher = Dispatcher()
        dispatcher.map("/avatar/parameters/MuteSelf", self._on_mute_self)
        dispatcher.map(
            f"/avatar/parameters/{self.config.status_parameter}",
            self._on_status,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.motion_enabled_parameter}",
            self._on_motion_enabled,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.motion_activity_parameter}",
            self._on_motion_activity,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.motion_energy_parameter}",
            self._on_motion_energy,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.motion_gesture_parameter}",
            self._on_motion_gesture,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.motion_expression_parameter}",
            self._on_motion_expression,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.thinking_parameter}",
            self._on_thinking,
        )
        dispatcher.map(
            f"/avatar/parameters/{self.config.probe_parameter}",
            self._on_probe,
        )
        dispatcher.map("/avatar/change", self._on_avatar_change)
        self._server = self._server_factory(
            (self.config.listen_host, self.config.output_port), dispatcher
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="vrchat-osc-listener",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None

    def _on_mute_self(self, _address: str, value: object) -> None:
        state = bool(value)
        with self._condition:
            self._mute_state = state
            self._mute_version += 1
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_avatar_change(self, _address: str, *values: object) -> None:
        # VRChat reports the avatar change before the new playable is always ready
        # to consume custom parameters. Delay and replay the complete state instead
        # of relying on a single immediate UDP packet.
        with self._condition:
            self._observed_status = None
            self._observed_motion_enabled = None
            self._observed_motion_activity = None
            self._observed_motion_energy = None
            self._observed_motion_gesture = None
            self._observed_motion_expression = None
            self._observed_thinking = None
            self._observed_probe = None
            self._avatar_id = str(values[0]) if values else None
            self._last_feedback_at = time.monotonic()
            self._avatar_replay_generation += 1
            generation = self._avatar_replay_generation
        threading.Thread(
            target=self._replay_after_avatar_change,
            args=(generation,),
            name="vrchat-osc-avatar-replay",
            daemon=True,
        ).start()

    def _replay_after_avatar_change(self, generation: int) -> None:
        time.sleep(0.75)
        with self._condition:
            if generation != self._avatar_replay_generation or self._server is None:
                return
        # A new avatar resets parameters, so replay the service's current state.
        self.send_status(self.status)
        self.send_motion_enabled(self.motion_enabled)
        with self._condition:
            activity = self._motion_activity
            energy = self._motion_energy
            gesture = self._motion_gesture
            expression = self._motion_expression
            thinking = self._thinking
        self.send_motion_activity(activity)
        self.send_motion_energy(energy)
        self.send_motion_gesture(gesture)
        self.send_motion_expression(expression)
        self.send_thinking(thinking)
        self.send_probe(False)

    def _on_status(self, _address: str, value: object) -> None:
        try:
            status = AgentStatus(int(value))
        except (TypeError, ValueError):
            return
        with self._condition:
            self._observed_status = status
            self._status_version += 1
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_motion_enabled(self, _address: str, value: object) -> None:
        with self._condition:
            self._motion_enabled = bool(value)
            self._observed_motion_enabled = bool(value)
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_motion_activity(self, _address: str, value: object) -> None:
        try:
            activity = max(0, min(2, int(value)))
        except (TypeError, ValueError):
            return
        with self._condition:
            self._observed_motion_activity = activity
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_motion_energy(self, _address: str, value: object) -> None:
        try:
            energy = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        with self._condition:
            self._observed_motion_energy = energy
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_motion_gesture(self, _address: str, value: object) -> None:
        try:
            gesture = max(0, min(9, int(value)))
        except (TypeError, ValueError):
            return
        with self._condition:
            self._observed_motion_gesture = gesture
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_motion_expression(self, _address: str, value: object) -> None:
        try:
            expression = max(0, min(6, int(value)))
        except (TypeError, ValueError):
            return
        with self._condition:
            self._observed_motion_expression = expression
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_thinking(self, _address: str, value: object) -> None:
        with self._condition:
            self._observed_thinking = bool(value)
            self._last_feedback_at = time.monotonic()
            self._condition.notify_all()

    def _on_probe(self, _address: str, value: object) -> None:
        with self._condition:
            self._observed_probe = bool(value)
            self._probe_version += 1
            self._probe_feedback_at = time.monotonic()
            self._last_feedback_at = self._probe_feedback_at
            self._condition.notify_all()

    def feedback_snapshot(self) -> dict[str, object]:
        with self._condition:
            feedback_age = (
                None
                if self._last_feedback_at <= 0.0
                else round(max(0.0, time.monotonic() - self._last_feedback_at), 2)
            )
            return {
                "status": None if self._observed_status is None else int(self._observed_status),
                "status_target": int(self._status),
                "status_confirmed": self._observed_status == self._status,
                "motion_enabled": self._observed_motion_enabled,
                "motion_enabled_target": self._motion_enabled,
                "activity": self._observed_motion_activity,
                "activity_target": self._motion_activity,
                "energy": self._observed_motion_energy,
                "energy_target": self._motion_energy,
                "gesture": self._observed_motion_gesture,
                "gesture_target": self._motion_gesture,
                "expression": self._observed_motion_expression,
                "expression_target": self._motion_expression,
                "thinking": self._observed_thinking,
                "thinking_target": self._thinking,
                "probe": self._observed_probe,
                "probe_target": self._probe,
                "avatar_id": self._avatar_id,
                "avatar_generation": self._avatar_replay_generation,
                "osc_listener_running": self._server is not None,
                "last_feedback_age_sec": feedback_age,
            }

    def send_status(self, status: AgentStatus | int) -> None:
        status = AgentStatus(int(status))
        with self._condition:
            self._status = status
        address = f"/avatar/parameters/{self.config.status_parameter}"
        self._send_message_reliably(address, int(status))

    def set_status_confirmed(self, status: AgentStatus | int) -> AgentStatus:
        status = AgentStatus(int(status))
        if self._server is None:
            raise RuntimeError("OSC listener is not running")
        for _attempt in range(max(2, self.config.mute_retry_count + 1)):
            with self._condition:
                if self._observed_status == status:
                    self._status = status
                    return status
                before_version = self._status_version
            self.send_status(status)
            deadline = time.monotonic() + self.config.mute_confirm_timeout_sec
            with self._condition:
                while self._status_version == before_version:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._observed_status == status:
                    return status
        actual = self.feedback_snapshot()["status"]
        raise RuntimeError(
            "VRChat did not confirm VoiceAgentStatus "
            f"{int(status)} (actual: {actual if actual is not None else 'unknown'}). "
            "Check that VRChat OSC is enabled and the current avatar is the AI avatar."
        )

    def send_motion_enabled(self, enabled: bool) -> None:
        with self._condition:
            self._motion_enabled = bool(enabled)
        self._send_parameter_reliably(self.config.motion_enabled_parameter, bool(enabled))

    def send_motion_activity(self, activity: int) -> None:
        activity = max(0, min(2, int(activity)))
        with self._condition:
            self._motion_activity = activity
        self._send_parameter_reliably(self.config.motion_activity_parameter, activity)

    def send_motion_energy(self, energy: float) -> None:
        energy = max(0.0, min(1.0, float(energy)))
        with self._condition:
            self._motion_energy = energy
        self._send_parameter(self.config.motion_energy_parameter, energy)

    def send_motion_gesture(self, gesture: int) -> None:
        gesture = max(0, min(9, int(gesture)))
        with self._condition:
            self._motion_gesture = gesture
        self._send_parameter_reliably(self.config.motion_gesture_parameter, gesture)

    def send_motion_expression(self, expression: int) -> None:
        expression = max(0, min(6, int(expression)))
        with self._condition:
            self._motion_expression = expression
        self._send_parameter_reliably(self.config.motion_expression_parameter, expression)

    def send_thinking(self, thinking: bool) -> None:
        with self._condition:
            self._thinking = bool(thinking)
        self._send_parameter_reliably(self.config.thinking_parameter, bool(thinking))

    def send_probe(self, value: bool) -> None:
        with self._condition:
            self._probe = bool(value)
        self._send_parameter_reliably(self.config.probe_parameter, bool(value))

    def confirm_probe_roundtrip(self) -> float:
        """Toggle the dedicated avatar Bool and return the worst OSC RTT in ms."""
        if self._server is None:
            raise RuntimeError("OSC listener is not running")

        with self._probe_lock:
            # Establish a known OFF edge first. It is intentionally not required to
            # produce feedback because an already-OFF avatar may not emit duplicates.
            self.send_probe(False)
            time.sleep(0.12)
            on_rtt = self._set_probe_confirmed(True)
            off_rtt = self._set_probe_confirmed(False)
            return round(max(on_rtt, off_rtt), 1)

    def _set_probe_confirmed(self, desired: bool) -> float:
        for _attempt in range(max(2, self.config.mute_retry_count + 1)):
            with self._condition:
                before_version = self._probe_version
            started_at = time.monotonic()
            with self._condition:
                self._probe = bool(desired)
            self._send_parameter(self.config.probe_parameter, bool(desired))
            deadline = started_at + self.config.mute_confirm_timeout_sec
            with self._condition:
                while (
                    self._probe_version == before_version
                    or self._observed_probe is not desired
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._probe_version != before_version and self._observed_probe is desired:
                    return max(0.0, (self._probe_feedback_at - started_at) * 1000.0)

        actual = self.feedback_snapshot()["probe"]
        raise RuntimeError(
            "VRChat did not return VoiceAgentOscProbe "
            f"{str(desired).upper()} (actual: {actual if actual is not None else 'unknown'}). "
            "Enable VRChat OSC and upload the avatar version containing VoiceAgentOscProbe."
        )

    def _send_parameter(self, parameter: str, value: object) -> None:
        self._client.send_message(f"/avatar/parameters/{parameter}", value)

    def _send_parameter_reliably(self, parameter: str, value: object) -> None:
        self._send_message_reliably(f"/avatar/parameters/{parameter}", value)

    def _send_message_reliably(self, address: str, value: object) -> None:
        # OSC uses UDP. Three closely spaced copies make static state changes much
        # less likely to disappear while keeping normal energy updates single-shot.
        for attempt in range(3):
            self._client.send_message(address, value)
            if attempt < 2:
                time.sleep(0.035)

    def _pulse_voice(self) -> None:
        self._client.send_message("/input/Voice", True)
        time.sleep(0.06)
        self._client.send_message("/input/Voice", False)

    def set_muted(self, desired: bool) -> bool:
        """Set mute state using a toggle input and MuteSelf feedback.

        If initial state is unknown, one pulse discovers it. A second pulse corrects
        the state when needed, so the method never assumes that a blind toggle equals
        mute or unmute.
        """

        if self._server is None:
            raise RuntimeError("OSC listener is not running")
        for _attempt in range(max(2, self.config.mute_retry_count + 1)):
            with self._condition:
                if self._mute_state is desired:
                    return desired
                before_version = self._mute_version
            self._pulse_voice()
            deadline = time.monotonic() + self.config.mute_confirm_timeout_sec
            with self._condition:
                while self._mute_version == before_version:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if self._mute_state is desired:
                    return desired
        actual = self.mute_state
        actual_text = "unknown" if actual is None else ("muted" if actual else "unmuted")
        raise RuntimeError(
            f"VRChat did not confirm the requested mute state (actual: {actual_text}). "
            "Enable OSC and set VRChat microphone behavior to Toggle."
        )
