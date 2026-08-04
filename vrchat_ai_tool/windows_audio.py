from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass(slots=True)
class AudioSessionInfo:
    process_id: int
    process_name: str
    state: str
    muted: bool | None
    volume: float | None


@dataclass(slots=True)
class AudioEndpointInfo:
    id: str
    name: str
    direction: str
    state: str
    is_default_console: bool = False
    is_default_multimedia: bool = False
    is_default_communications: bool = False
    sessions: tuple[AudioSessionInfo, ...] = ()


@dataclass(slots=True)
class WindowsAudioSnapshot:
    endpoints: tuple[AudioEndpointInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"endpoints": [asdict(endpoint) for endpoint in self.endpoints]}

    def find(self, name: str, direction: str | None = None) -> list[AudioEndpointInfo]:
        query = normalize_name(name)
        result: list[AudioEndpointInfo] = []
        for endpoint in self.endpoints:
            if direction is not None and endpoint.direction != direction:
                continue
            candidate = normalize_name(endpoint.name)
            if not candidate:
                continue
            if query == candidate or query in candidate or candidate in query:
                result.append(endpoint)
        return result

    def sessions_for(self, process_names: tuple[str, ...]) -> list[tuple[AudioEndpointInfo, AudioSessionInfo]]:
        wanted = {name.casefold() for name in process_names}
        return [
            (endpoint, session)
            for endpoint in self.endpoints
            for session in endpoint.sessions
            if normalize_name(session.process_name) in wanted
        ]


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.casefold().replace("(vb-audio virtual cable a)", "").replace(
        "(vb-audio virtual cable b)", ""
    ).split())


def collect_windows_audio_snapshot() -> WindowsAudioSnapshot:
    """Collect Core Audio endpoints, defaults and per-endpoint sessions.

    Imports are intentionally local so the pure doctor/report code remains testable
    on non-Windows CI runners.
    """

    import comtypes
    from pycaw.pycaw import (
        AudioSession,
        AudioUtilities,
        EDataFlow,
        ERole,
        IAudioSessionControl2,
    )

    comtypes.CoInitialize()
    try:
        default_ids: dict[tuple[str, str], str] = {}
        enumerator = AudioUtilities.GetDeviceEnumerator()
        for direction, flow in (("render", EDataFlow.eRender), ("capture", EDataFlow.eCapture)):
            for role_name, role in (
                ("console", ERole.eConsole),
                ("multimedia", ERole.eMultimedia),
                ("communications", ERole.eCommunications),
            ):
                try:
                    raw = enumerator.GetDefaultAudioEndpoint(flow.value, role.value)
                    default_ids[(direction, role_name)] = AudioUtilities.CreateDevice(raw).id
                except Exception:  # noqa: BLE001 - a role may have no default endpoint
                    default_ids.pop((direction, role_name), None)

        endpoints: list[AudioEndpointInfo] = []
        # Only active endpoints can participate in the current route. Stale
        # endpoint records may have incomplete property stores after removal.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"pycaw\.utils")
            devices = AudioUtilities.GetAllDevices(EDataFlow.eAll.value, 1)

        for device in devices:
            direction = "capture" if device.id.startswith("{0.0.1.") else "render"
            try:
                friendly_name = device.FriendlyName
            except Exception:  # noqa: BLE001 - an unreadable endpoint cannot be validated
                friendly_name = None
            if not friendly_name:
                continue
            sessions: list[AudioSessionInfo] = []
            try:
                session_enumerator = device.AudioSessionManager.GetSessionEnumerator()
                for index in range(session_enumerator.GetCount()):
                    control = session_enumerator.GetSession(index)
                    session = AudioSession(control.QueryInterface(IAudioSessionControl2))
                    process = session.Process
                    try:
                        process_name = process.name() if process is not None else "System Sounds"
                    except psutil.Error:
                        process_name = f"PID {session.ProcessId}"
                    process_name = process_name or f"PID {session.ProcessId}"
                    volume = session.SimpleAudioVolume
                    sessions.append(
                        AudioSessionInfo(
                            process_id=int(session.ProcessId),
                            process_name=process_name,
                            state={0: "inactive", 1: "active", 2: "expired"}.get(
                                int(session.State), str(session.State)
                            ),
                            muted=bool(volume.GetMute()),
                            volume=float(volume.GetMasterVolume()),
                        )
                    )
            except Exception:  # noqa: BLE001 - disabled endpoints reject session activation
                # Some disabled/not-present endpoints reject session activation.
                sessions = []

            endpoints.append(
                AudioEndpointInfo(
                    id=device.id,
                    name=friendly_name,
                    direction=direction,
                    state=str(getattr(device.state, "name", device.state)).casefold(),
                    is_default_console=default_ids.get((direction, "console")) == device.id,
                    is_default_multimedia=default_ids.get((direction, "multimedia")) == device.id,
                    is_default_communications=default_ids.get((direction, "communications")) == device.id,
                    sessions=tuple(sessions),
                )
            )
        return WindowsAudioSnapshot(endpoints=tuple(endpoints))
    finally:
        comtypes.CoUninitialize()
