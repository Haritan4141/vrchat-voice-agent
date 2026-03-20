from __future__ import annotations

import ctypes
import io
import math
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ctypes import wintypes

MAXPNAMELEN = 32
WAVE_FORMAT_PCM = 0x0001
CALLBACK_FUNCTION = 0x00030000
WIM_DATA = 0x03C0
WHDR_DONE = 0x00000001
WAVE_MAPPER = 0xFFFFFFFF

DWORD_PTR = ctypes.c_size_t
MMRESULT = wintypes.UINT
HWAVEIN = wintypes.HANDLE
HWAVEOUT = wintypes.HANDLE

winmm = ctypes.WinDLL("winmm")


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class WAVEHDR(ctypes.Structure):
    _fields_ = [
        ("lpData", ctypes.c_void_p),
        ("dwBufferLength", wintypes.DWORD),
        ("dwBytesRecorded", wintypes.DWORD),
        ("dwUser", DWORD_PTR),
        ("dwFlags", wintypes.DWORD),
        ("dwLoops", wintypes.DWORD),
        ("lpNext", ctypes.c_void_p),
        ("reserved", DWORD_PTR),
    ]


class WAVEINCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * MAXPNAMELEN),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
    ]


class WAVEOUTCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * MAXPNAMELEN),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("dwSupport", wintypes.DWORD),
    ]


winmm.waveInGetNumDevs.restype = wintypes.UINT
winmm.waveOutGetNumDevs.restype = wintypes.UINT
winmm.waveInGetDevCapsW.argtypes = [DWORD_PTR, ctypes.POINTER(WAVEINCAPSW), wintypes.UINT]
winmm.waveOutGetDevCapsW.argtypes = [DWORD_PTR, ctypes.POINTER(WAVEOUTCAPSW), wintypes.UINT]
winmm.waveInOpen.argtypes = [
    ctypes.POINTER(HWAVEIN),
    wintypes.UINT,
    ctypes.POINTER(WAVEFORMATEX),
    DWORD_PTR,
    DWORD_PTR,
    wintypes.DWORD,
]
winmm.waveInOpen.restype = MMRESULT
winmm.waveInPrepareHeader.argtypes = [HWAVEIN, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveInPrepareHeader.restype = MMRESULT
winmm.waveInAddBuffer.argtypes = [HWAVEIN, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveInAddBuffer.restype = MMRESULT
winmm.waveInStart.argtypes = [HWAVEIN]
winmm.waveInStart.restype = MMRESULT
winmm.waveInStop.argtypes = [HWAVEIN]
winmm.waveInStop.restype = MMRESULT
winmm.waveInReset.argtypes = [HWAVEIN]
winmm.waveInReset.restype = MMRESULT
winmm.waveInUnprepareHeader.argtypes = [HWAVEIN, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveInUnprepareHeader.restype = MMRESULT
winmm.waveInClose.argtypes = [HWAVEIN]
winmm.waveInClose.restype = MMRESULT
winmm.waveInGetErrorTextW.argtypes = [MMRESULT, wintypes.LPWSTR, wintypes.UINT]

winmm.waveOutOpen.argtypes = [
    ctypes.POINTER(HWAVEOUT),
    wintypes.UINT,
    ctypes.POINTER(WAVEFORMATEX),
    DWORD_PTR,
    DWORD_PTR,
    wintypes.DWORD,
]
winmm.waveOutOpen.restype = MMRESULT
winmm.waveOutPrepareHeader.argtypes = [HWAVEOUT, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveOutPrepareHeader.restype = MMRESULT
winmm.waveOutWrite.argtypes = [HWAVEOUT, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveOutWrite.restype = MMRESULT
winmm.waveOutUnprepareHeader.argtypes = [HWAVEOUT, ctypes.POINTER(WAVEHDR), wintypes.UINT]
winmm.waveOutUnprepareHeader.restype = MMRESULT
winmm.waveOutClose.argtypes = [HWAVEOUT]
winmm.waveOutClose.restype = MMRESULT
winmm.waveOutGetErrorTextW.argtypes = [MMRESULT, wintypes.LPWSTR, wintypes.UINT]


@dataclass(slots=True)
class AudioDevice:
    id: int
    name: str
    direction: str


def _normalize_device_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _wave_in_error_text(code: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    winmm.waveInGetErrorTextW(code, buffer, len(buffer))
    return buffer.value or f"MMRESULT {code}"


def _wave_out_error_text(code: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    winmm.waveOutGetErrorTextW(code, buffer, len(buffer))
    return buffer.value or f"MMRESULT {code}"


def _raise_wave_in(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed: {_wave_in_error_text(code)}")


def _raise_wave_out(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed: {_wave_out_error_text(code)}")


def list_input_devices() -> list[AudioDevice]:
    devices: list[AudioDevice] = []
    for index in range(winmm.waveInGetNumDevs()):
        caps = WAVEINCAPSW()
        code = winmm.waveInGetDevCapsW(index, ctypes.byref(caps), ctypes.sizeof(caps))
        _raise_wave_in(code, "waveInGetDevCapsW")
        devices.append(AudioDevice(id=index, name=caps.szPname.rstrip("\x00"), direction="input"))
    return devices


def list_output_devices() -> list[AudioDevice]:
    devices: list[AudioDevice] = []
    for index in range(winmm.waveOutGetNumDevs()):
        caps = WAVEOUTCAPSW()
        code = winmm.waveOutGetDevCapsW(index, ctypes.byref(caps), ctypes.sizeof(caps))
        _raise_wave_out(code, "waveOutGetDevCapsW")
        devices.append(AudioDevice(id=index, name=caps.szPname.rstrip("\x00"), direction="output"))
    return devices


def find_device_id(direction: str, query: str) -> int:
    if not query:
        return WAVE_MAPPER

    devices = list_input_devices() if direction == "input" else list_output_devices()
    normalized_query = _normalize_device_name(query)

    exact_matches = [device for device in devices if _normalize_device_name(device.name) == normalized_query]
    if len(exact_matches) == 1:
        return exact_matches[0].id

    fuzzy_matches = [
        device
        for device in devices
        if normalized_query in _normalize_device_name(device.name)
        or _normalize_device_name(device.name) in normalized_query
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0].id
    if len(exact_matches) > 1 or len(fuzzy_matches) > 1:
        names = ", ".join(device.name for device in (exact_matches or fuzzy_matches))
        raise RuntimeError(f"Multiple {direction} devices matched '{query}': {names}")

    available = ", ".join(device.name for device in devices)
    raise RuntimeError(f"No {direction} device matched '{query}'. Available: {available}")


def pcm16le_rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = memoryview(frame).cast("h")
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples))


def save_pcm_as_wav(path: Path, pcm_data: bytes, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def _load_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Unsupported sample width: {sample_width * 8}bit")
    return frames, sample_rate, channels


class WaveOutPlayer:
    def __init__(self, device_id: int) -> None:
        self.device_id = device_id

    def play_wav_bytes(self, wav_bytes: bytes) -> None:
        pcm_data, sample_rate, channels = _load_wav_bytes(wav_bytes)
        self.play_pcm(pcm_data=pcm_data, sample_rate=sample_rate, channels=channels)

    def play_pcm(self, pcm_data: bytes, sample_rate: int, channels: int) -> None:
        handle = HWAVEOUT()
        wave_format = WAVEFORMATEX(
            wFormatTag=WAVE_FORMAT_PCM,
            nChannels=channels,
            nSamplesPerSec=sample_rate,
            nAvgBytesPerSec=sample_rate * channels * 2,
            nBlockAlign=channels * 2,
            wBitsPerSample=16,
            cbSize=0,
        )
        code = winmm.waveOutOpen(
            ctypes.byref(handle),
            self.device_id,
            ctypes.byref(wave_format),
            0,
            0,
            0,
        )
        _raise_wave_out(code, "waveOutOpen")

        buffer = ctypes.create_string_buffer(pcm_data, len(pcm_data))
        header = WAVEHDR(
            lpData=ctypes.addressof(buffer),
            dwBufferLength=len(pcm_data),
            dwBytesRecorded=0,
            dwUser=0,
            dwFlags=0,
            dwLoops=0,
            lpNext=0,
            reserved=0,
        )

        try:
            _raise_wave_out(
                winmm.waveOutPrepareHeader(handle, ctypes.byref(header), ctypes.sizeof(header)),
                "waveOutPrepareHeader",
            )
            _raise_wave_out(
                winmm.waveOutWrite(handle, ctypes.byref(header), ctypes.sizeof(header)),
                "waveOutWrite",
            )
            while not (header.dwFlags & WHDR_DONE):
                time.sleep(0.01)
        finally:
            if handle:
                winmm.waveOutUnprepareHeader(handle, ctypes.byref(header), ctypes.sizeof(header))
                winmm.waveOutClose(handle)


def play_wav_to_devices(wav_bytes: bytes, device_ids: list[int]) -> None:
    unique_ids: list[int] = []
    for device_id in device_ids:
        if device_id not in unique_ids:
            unique_ids.append(device_id)

    threads = [
        threading.Thread(target=WaveOutPlayer(device_id).play_wav_bytes, args=(wav_bytes,), daemon=False)
        for device_id in unique_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


_wave_in_recorders: dict[int, "WaveInRecorder"] = {}

WAVEINPROC = ctypes.WINFUNCTYPE(
    None,
    HWAVEIN,
    wintypes.UINT,
    DWORD_PTR,
    DWORD_PTR,
    DWORD_PTR,
)


def _wave_in_callback(
    _handle: HWAVEIN,
    message: int,
    instance: int,
    param1: int,
    _param2: int,
) -> None:
    recorder = _wave_in_recorders.get(int(instance))
    if recorder is None:
        return
    recorder._on_message(message, param1)


_WAVE_IN_CALLBACK = WAVEINPROC(_wave_in_callback)


class WaveInRecorder:
    _buffer_count: ClassVar[int] = 4

    def __init__(self, device_id: int, sample_rate: int, channels: int, chunk_ms: int) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.bytes_per_second = sample_rate * channels * 2
        self.buffer_size = max(1, self.bytes_per_second * chunk_ms // 1000)
        self.handle = HWAVEIN()
        self._headers: list[WAVEHDR] = []
        self._buffers: list[ctypes.Array[ctypes.c_char]] = []
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._instance_id = id(self)
        self._running = False
        self._opened = False

    def open(self) -> None:
        if self._opened:
            return

        wave_format = WAVEFORMATEX(
            wFormatTag=WAVE_FORMAT_PCM,
            nChannels=self.channels,
            nSamplesPerSec=self.sample_rate,
            nAvgBytesPerSec=self.bytes_per_second,
            nBlockAlign=self.channels * 2,
            wBitsPerSample=16,
            cbSize=0,
        )
        _wave_in_recorders[self._instance_id] = self
        try:
            code = winmm.waveInOpen(
                ctypes.byref(self.handle),
                self.device_id,
                ctypes.byref(wave_format),
                ctypes.cast(_WAVE_IN_CALLBACK, ctypes.c_void_p).value,
                self._instance_id,
                CALLBACK_FUNCTION,
            )
            _raise_wave_in(code, "waveInOpen")

            for _ in range(self._buffer_count):
                buffer = ctypes.create_string_buffer(self.buffer_size)
                header = WAVEHDR(
                    lpData=ctypes.addressof(buffer),
                    dwBufferLength=self.buffer_size,
                    dwBytesRecorded=0,
                    dwUser=0,
                    dwFlags=0,
                    dwLoops=0,
                    lpNext=0,
                    reserved=0,
                )
                _raise_wave_in(
                    winmm.waveInPrepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header)),
                    "waveInPrepareHeader",
                )
                _raise_wave_in(
                    winmm.waveInAddBuffer(self.handle, ctypes.byref(header), ctypes.sizeof(header)),
                    "waveInAddBuffer",
                )
                self._buffers.append(buffer)
                self._headers.append(header)

            _raise_wave_in(winmm.waveInStart(self.handle), "waveInStart")
            self._running = True
            self._opened = True
        except Exception:
            _wave_in_recorders.pop(self._instance_id, None)
            raise

    def close(self) -> None:
        if not self._opened:
            return

        self._running = False
        try:
            winmm.waveInStop(self.handle)
            winmm.waveInReset(self.handle)
            for header in self._headers:
                winmm.waveInUnprepareHeader(self.handle, ctypes.byref(header), ctypes.sizeof(header))
            winmm.waveInClose(self.handle)
        finally:
            _wave_in_recorders.pop(self._instance_id, None)
            self._headers.clear()
            self._buffers.clear()
            self._opened = False

    def __enter__(self) -> "WaveInRecorder":
        self.open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _push_frame(self, data: bytes) -> None:
        if not data:
            return
        try:
            self._frames.put_nowait(data)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(data)

    def _on_message(self, message: int, param1: int) -> None:
        if message != WIM_DATA or not param1:
            return

        header_ptr = ctypes.cast(param1, ctypes.POINTER(WAVEHDR))
        header = header_ptr.contents
        data = ctypes.string_at(header.lpData, header.dwBytesRecorded)
        self._push_frame(data)

        if self._running:
            header_ptr.contents.dwBytesRecorded = 0
            winmm.waveInAddBuffer(self.handle, header_ptr, ctypes.sizeof(WAVEHDR))

    def read_chunk(self, timeout: float = 1.0) -> bytes | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_until_silence(
        self,
        rms_threshold: float,
        min_speech_ms: int,
        silence_timeout_ms: int,
        max_utterance_ms: int,
        max_wait_sec: float | None = None,
    ) -> bytes:
        if not self._opened:
            self.open()

        started = False
        utterance_ms = 0
        speech_ms = 0
        silence_ms = 0
        chunks: list[bytes] = []
        deadline = time.monotonic() + max_wait_sec if max_wait_sec is not None else None

        while True:
            timeout = 1.0
            if deadline is not None:
                timeout = max(0.1, deadline - time.monotonic())
            chunk = self.read_chunk(timeout=timeout)

            if chunk is None:
                if deadline is not None and time.monotonic() >= deadline and not started:
                    return b""
                continue

            rms = pcm16le_rms(chunk)
            is_speech = rms >= rms_threshold

            if not started:
                if not is_speech:
                    if deadline is not None and time.monotonic() >= deadline:
                        return b""
                    continue
                started = True

            chunks.append(chunk)
            utterance_ms += self.chunk_ms
            if is_speech:
                speech_ms += self.chunk_ms
                silence_ms = 0
            else:
                silence_ms += self.chunk_ms

            if utterance_ms >= max_utterance_ms:
                break
            if speech_ms >= min_speech_ms and silence_ms >= silence_timeout_ms:
                break

        if speech_ms < min_speech_ms:
            return b""

        if silence_ms > 0:
            trailing_chunks = min(len(chunks) - 1, silence_ms // self.chunk_ms)
            if trailing_chunks > 0:
                chunks = chunks[:-trailing_chunks]

        return b"".join(chunks)
