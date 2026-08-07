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
        self._motion_enabled = True
        self._motion_activity = 0
        self._motion_energy = 0.0
        self._motion_gesture = 0

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
            f"/avatar/parameters/{self.config.motion_enabled_parameter}",
            self._on_motion_enabled,
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
            self._condition.notify_all()

    def _on_avatar_change(self, _address: str, *_values: object) -> None:
        # A new avatar resets parameters, so replay the service's current state.
        self.send_status(self.status)
        self.send_motion_enabled(self.motion_enabled)
        with self._condition:
            activity = self._motion_activity
            energy = self._motion_energy
            gesture = self._motion_gesture
        self.send_motion_activity(activity)
        self.send_motion_energy(energy)
        self.send_motion_gesture(gesture)

    def _on_motion_enabled(self, _address: str, value: object) -> None:
        with self._condition:
            self._motion_enabled = bool(value)
            self._condition.notify_all()

    def send_status(self, status: AgentStatus | int) -> None:
        status = AgentStatus(int(status))
        with self._condition:
            self._status = status
        address = f"/avatar/parameters/{self.config.status_parameter}"
        self._client.send_message(address, int(status))

    def send_motion_enabled(self, enabled: bool) -> None:
        with self._condition:
            self._motion_enabled = bool(enabled)
        self._send_parameter(self.config.motion_enabled_parameter, bool(enabled))

    def send_motion_activity(self, activity: int) -> None:
        activity = max(0, min(2, int(activity)))
        with self._condition:
            self._motion_activity = activity
        self._send_parameter(self.config.motion_activity_parameter, activity)

    def send_motion_energy(self, energy: float) -> None:
        energy = max(0.0, min(1.0, float(energy)))
        with self._condition:
            self._motion_energy = energy
        self._send_parameter(self.config.motion_energy_parameter, energy)

    def send_motion_gesture(self, gesture: int) -> None:
        gesture = max(0, min(4, int(gesture)))
        with self._condition:
            self._motion_gesture = gesture
        self._send_parameter(self.config.motion_gesture_parameter, gesture)

    def _send_parameter(self, parameter: str, value: object) -> None:
        self._client.send_message(f"/avatar/parameters/{parameter}", value)

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
