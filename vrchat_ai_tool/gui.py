from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .audio import list_input_devices, list_output_devices
from .config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    AudioCaptureConfig,
    AudioOutputConfig,
    ConversationConfig,
    EXAMPLE_CONFIG_PATH,
    LlmConfig,
    SttConfig,
    TtsConfig,
    config_base_dir,
    ensure_config_file,
    load_config,
    save_config,
)
from .runtime import BotRuntime
from .services import create_llm_client


MODEL_CHOICES = ["small", "medium", "large-v3", "turbo"]
STT_BACKENDS = ["faster_whisper", "system_speech"]
STT_DEVICES = ["cuda", "cpu"]
COMPUTE_TYPES = ["float16", "int8", "int8_float16", "int8_float32", "float32"]
LLM_BACKENDS = ["ollama", "lm_studio"]
LLM_DEFAULT_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "lm_studio": "http://127.0.0.1:1234/v1",
}


@dataclass(slots=True)
class WorkerState:
    thread: threading.Thread | None = None
    stop_event: threading.Event | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class GuiApp:
    def __init__(self, root: tk.Tk, config_path: Path) -> None:
        self.root = root
        self.root.title("VRChat AI Tool")
        self.root.geometry("1280x860")
        self.root.minsize(1080, 720)

        self.current_config_path = config_path
        self.worker = WorkerState()
        self.pending_close = False
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()

        self.input_device_names: list[str] = []
        self.output_device_names: list[str] = []
        self.llm_model_names: list[str] = []
        self._suppress_llm_backend_callback = False

        self.file_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self.capture_mode_var = tk.StringVar()
        self.input_device_var = tk.StringVar()
        self.sample_rate_var = tk.StringVar()
        self.channels_var = tk.StringVar()
        self.chunk_ms_var = tk.StringVar()
        self.silence_timeout_ms_var = tk.StringVar()
        self.rms_threshold_var = tk.StringVar()
        self.min_speech_ms_var = tk.StringVar()
        self.max_utterance_ms_var = tk.StringVar()

        self.tts_output_device_var = tk.StringVar()
        self.monitor_output_device_var = tk.StringVar()

        self.stt_backend_var = tk.StringVar()
        self.stt_model_var = tk.StringVar()
        self.stt_device_var = tk.StringVar()
        self.compute_type_var = tk.StringVar()
        self.stt_language_var = tk.StringVar()
        self.stt_timeout_var = tk.StringVar()
        self.beam_size_var = tk.StringVar()
        self.vad_filter_var = tk.BooleanVar()
        self.vad_min_silence_ms_var = tk.StringVar()

        self.llm_backend_var = tk.StringVar()
        self.llm_base_url_var = tk.StringVar()
        self.llm_model_var = tk.StringVar()
        self.temperature_var = tk.StringVar()
        self.max_tokens_var = tk.StringVar()
        self.llm_timeout_var = tk.StringVar()

        self.tts_backend_var = tk.StringVar()
        self.tts_base_url_var = tk.StringVar()
        self.tts_speaker_var = tk.StringVar()
        self.speed_scale_var = tk.StringVar()
        self.tts_timeout_var = tk.StringVar()

        self.max_response_chars_var = tk.StringVar()
        self.min_reply_interval_var = tk.StringVar()
        self.allow_topic_suggestions_var = tk.BooleanVar()
        self.pause_listening_var = tk.BooleanVar()

        self.llm_backend_var.trace_add("write", self._on_llm_backend_changed)

        self._build_ui()
        self._bind_hotkeys()
        self._refresh_devices()
        self._load_into_form(config_path)
        self._set_running_state(False)
        self._poll_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        controls = ttk.Frame(header)
        controls.grid(row=0, column=0, columnspan=2, sticky="w")

        self.load_button = ttk.Button(controls, text="Load TOML", command=self._load_other_file)
        self.load_button.grid(row=0, column=0, padx=(0, 8))
        self.save_button = ttk.Button(controls, text="Save", command=self._save_current_file)
        self.save_button.grid(row=0, column=1, padx=(0, 8))
        self.start_button = ttk.Button(controls, text="Start (F1)", command=self._start_runtime)
        self.start_button.grid(row=0, column=2, padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Stop (F2)", command=self._stop_runtime)
        self.stop_button.grid(row=0, column=3, padx=(0, 8))
        self.refresh_devices_button = ttk.Button(
            controls,
            text="Refresh Devices",
            command=self._refresh_devices,
        )
        self.refresh_devices_button.grid(row=0, column=4)

        ttk.Label(header, text="Current file:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(header, textvariable=self.file_path_var).grid(row=1, column=1, sticky="w", pady=(10, 0))
        ttk.Label(header, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        audio_tab = ttk.Frame(notebook, padding=12)
        stt_tab = ttk.Frame(notebook, padding=12)
        llm_tab = ttk.Frame(notebook, padding=12)
        conversation_tab = ttk.Frame(notebook, padding=12)
        notebook.add(audio_tab, text="Audio")
        notebook.add(stt_tab, text="STT")
        notebook.add(llm_tab, text="LLM / TTS")
        notebook.add(conversation_tab, text="Conversation")

        self._build_audio_tab(audio_tab)
        self._build_stt_tab(stt_tab)
        self._build_llm_tab(llm_tab)
        self._build_conversation_tab(conversation_tab)

        logs_frame = ttk.LabelFrame(self.root, text="Logs", padding=12)
        logs_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap="word", height=12)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _build_audio_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        ttk.Label(
            parent,
            text="Recommended: VRChat output -> CABLE-A Input, Tool input -> CABLE-A Output, Tool TTS -> CABLE-B Input, VRChat mic -> CABLE-B Output",
            wraplength=980,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        self.input_device_combo = self._add_combo(parent, "Input device", self.input_device_var, 1, 0)
        self.tts_output_device_combo = self._add_combo(parent, "TTS output", self.tts_output_device_var, 1, 2)
        self.monitor_output_device_combo = self._add_combo(
            parent,
            "Monitor output",
            self.monitor_output_device_var,
            2,
            0,
        )
        self._add_combo(parent, "Capture mode", self.capture_mode_var, 2, 2, values=["virtual_device"])

        self._add_entry(parent, "Sample rate", self.sample_rate_var, 3, 0)
        self._add_entry(parent, "Channels", self.channels_var, 3, 2)
        self._add_entry(parent, "Chunk ms", self.chunk_ms_var, 4, 0)
        self._add_entry(parent, "Silence timeout ms", self.silence_timeout_ms_var, 4, 2)
        self._add_entry(parent, "RMS threshold", self.rms_threshold_var, 5, 0)
        self._add_entry(parent, "Min speech ms", self.min_speech_ms_var, 5, 2)
        self._add_entry(parent, "Max utterance ms", self.max_utterance_ms_var, 6, 0)

    def _build_stt_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        self._add_combo(parent, "Backend", self.stt_backend_var, 0, 0, values=STT_BACKENDS)
        self._add_combo(parent, "Model", self.stt_model_var, 0, 2, values=MODEL_CHOICES)
        self._add_combo(parent, "Device", self.stt_device_var, 1, 0, values=STT_DEVICES)
        self._add_combo(parent, "Compute type", self.compute_type_var, 1, 2, values=COMPUTE_TYPES)
        self._add_entry(parent, "Language", self.stt_language_var, 2, 0)
        self._add_entry(parent, "Timeout sec", self.stt_timeout_var, 2, 2)
        self._add_entry(parent, "Beam size", self.beam_size_var, 3, 0)
        self._add_entry(parent, "VAD min silence ms", self.vad_min_silence_ms_var, 3, 2)

        ttk.Checkbutton(parent, text="Use VAD filter", variable=self.vad_filter_var).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

    def _build_llm_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        parent.columnconfigure(4, weight=0)
        parent.rowconfigure(5, weight=1)

        ttk.Label(
            parent,
            text="Ollama default: http://127.0.0.1:11434   LM Studio default: http://127.0.0.1:1234/v1",
            wraplength=980,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        self._add_combo(parent, "LLM backend", self.llm_backend_var, 1, 0, values=LLM_BACKENDS)
        self._add_entry(parent, "LLM base URL", self.llm_base_url_var, 1, 2)
        self.llm_model_combo = self._add_combo(parent, "LLM model", self.llm_model_var, 2, 0)
        self.refresh_models_button = ttk.Button(
            parent,
            text="Refresh Models",
            command=self._refresh_llm_models,
        )
        self.refresh_models_button.grid(row=2, column=4, sticky="w", pady=4, padx=(8, 0))
        self._add_entry(parent, "Temperature", self.temperature_var, 2, 2)
        self._add_entry(parent, "Max tokens", self.max_tokens_var, 3, 0)
        self._add_entry(parent, "LLM timeout sec", self.llm_timeout_var, 3, 2)
        self._add_entry(parent, "TTS backend", self.tts_backend_var, 4, 0)
        self._add_entry(parent, "TTS base URL", self.tts_base_url_var, 4, 2)
        self._add_entry(parent, "Speaker", self.tts_speaker_var, 5, 0)
        self._add_entry(parent, "Speed scale", self.speed_scale_var, 5, 2)
        self._add_entry(parent, "TTS timeout sec", self.tts_timeout_var, 6, 0)

        ttk.Label(parent, text="System prompt").grid(row=7, column=0, sticky="nw", pady=(12, 4))
        self.system_prompt_text = scrolledtext.ScrolledText(parent, wrap="word", height=12)
        self.system_prompt_text.grid(row=8, column=0, columnspan=4, sticky="nsew")
        parent.rowconfigure(8, weight=1)

    def _build_conversation_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

        self._add_entry(parent, "Max response chars", self.max_response_chars_var, 0, 0)
        self._add_entry(parent, "Min reply interval sec", self.min_reply_interval_var, 0, 2)

        ttk.Checkbutton(
            parent,
            text="Allow topic suggestions",
            variable=self.allow_topic_suggestions_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Checkbutton(
            parent,
            text="Pause listening while speaking",
            variable=self.pause_listening_var,
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(10, 0))

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=column + 1, sticky="ew", pady=4)
        return entry

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
        values: list[str] | None = None,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values or [])
        combo.grid(row=row, column=column + 1, sticky="ew", pady=4)
        return combo

    def _bind_hotkeys(self) -> None:
        self.root.bind_all("<F1>", lambda _event: self._start_runtime())
        self.root.bind_all("<F2>", lambda _event: self._stop_runtime())

    def _on_llm_backend_changed(self, *_args) -> None:
        if self._suppress_llm_backend_callback:
            return

        backend = self.llm_backend_var.get().strip()
        default_url = LLM_DEFAULT_BASE_URLS.get(backend)
        if default_url is None:
            return

        current_url = self.llm_base_url_var.get().strip()
        known_urls = set(LLM_DEFAULT_BASE_URLS.values())
        if not current_url or current_url in known_urls:
            self.llm_base_url_var.set(default_url)
        self._refresh_llm_models(show_errors=False)

    def _refresh_llm_models(self, show_errors: bool = True) -> None:
        backend = self.llm_backend_var.get().strip()
        base_url = self.llm_base_url_var.get().strip()
        model = self.llm_model_var.get().strip() or "placeholder"

        if not backend or not base_url:
            self.llm_model_names = []
            self.llm_model_combo["values"] = []
            return

        try:
            client = create_llm_client(
                backend=backend,
                base_url=base_url,
                model=model,
                temperature=0.0,
                max_tokens=1,
                timeout_sec=5,
            )
            names = client.list_models()
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Model refresh failed", str(exc))
            return

        self.llm_model_names = names
        self.llm_model_combo["values"] = names

        current_model = self.llm_model_var.get().strip()
        if not current_model and names:
            self.llm_model_var.set(names[0])
        self.status_var.set("Models refreshed")
        self._append_log(f"[info] refreshed {backend} models: {len(names)} found")

    def _refresh_devices(self) -> None:
        try:
            self.input_device_names = [device.name for device in list_input_devices()]
            self.output_device_names = [device.name for device in list_output_devices()]
        except Exception as exc:
            messagebox.showerror("Device refresh failed", str(exc))
            return

        self.input_device_combo["values"] = self.input_device_names
        output_choices = [""] + self.output_device_names
        self.tts_output_device_combo["values"] = output_choices
        self.monitor_output_device_combo["values"] = output_choices

    def _load_into_form(self, path: Path) -> None:
        config = load_config(path)
        self.current_config_path = path
        self.file_path_var.set(str(path.resolve()))

        self._suppress_llm_backend_callback = True

        self.capture_mode_var.set(config.audio_capture.mode)
        self.input_device_var.set(config.audio_capture.input_device)
        self.sample_rate_var.set(str(config.audio_capture.sample_rate))
        self.channels_var.set(str(config.audio_capture.channels))
        self.chunk_ms_var.set(str(config.audio_capture.chunk_ms))
        self.silence_timeout_ms_var.set(str(config.audio_capture.silence_timeout_ms))
        self.rms_threshold_var.set(f"{config.audio_capture.rms_threshold:g}")
        self.min_speech_ms_var.set(str(config.audio_capture.min_speech_ms))
        self.max_utterance_ms_var.set(str(config.audio_capture.max_utterance_ms))

        self.tts_output_device_var.set(config.audio_output.tts_output_device)
        self.monitor_output_device_var.set(config.audio_output.monitor_output_device)

        self.stt_backend_var.set(config.stt.backend)
        self.stt_model_var.set(config.stt.model)
        self.stt_device_var.set(config.stt.device)
        self.compute_type_var.set(config.stt.compute_type)
        self.stt_language_var.set(config.stt.language)
        self.stt_timeout_var.set(str(config.stt.timeout_sec))
        self.beam_size_var.set(str(config.stt.beam_size))
        self.vad_filter_var.set(config.stt.vad_filter)
        self.vad_min_silence_ms_var.set(str(config.stt.vad_min_silence_ms))

        self.llm_backend_var.set(config.llm.backend)
        self.llm_base_url_var.set(config.llm.base_url)
        self.llm_model_var.set(config.llm.model)
        self.temperature_var.set(f"{config.llm.temperature:g}")
        self.max_tokens_var.set(str(config.llm.max_tokens))
        self.llm_timeout_var.set(str(config.llm.timeout_sec))

        self.tts_backend_var.set(config.tts.backend)
        self.tts_base_url_var.set(config.tts.base_url)
        self.tts_speaker_var.set(str(config.tts.speaker))
        self.speed_scale_var.set(f"{config.tts.speed_scale:g}")
        self.tts_timeout_var.set(str(config.tts.timeout_sec))

        self.max_response_chars_var.set(str(config.conversation.max_response_chars))
        self.min_reply_interval_var.set(str(config.conversation.min_reply_interval_sec))
        self.allow_topic_suggestions_var.set(config.conversation.allow_topic_suggestions)
        self.pause_listening_var.set(config.conversation.pause_listening_while_speaking)

        self.system_prompt_text.delete("1.0", tk.END)
        self.system_prompt_text.insert("1.0", config.llm.system_prompt)

        self._suppress_llm_backend_callback = False
        self._refresh_llm_models(show_errors=False)
        self.status_var.set("Loaded")
        self._append_log(f"[info] loaded config: {path}")

    def _load_other_file(self) -> None:
        if self.worker.running:
            messagebox.showinfo("Busy", "Stop the runtime before loading another TOML file.")
            return

        selected = filedialog.askopenfilename(
            title="Load TOML config",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialdir=str(self.current_config_path.parent),
        )
        if not selected:
            return

        path = Path(selected)
        try:
            self._load_into_form(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _save_current_file(self) -> None:
        if self.worker.running:
            messagebox.showinfo("Busy", "Stop the runtime before saving the config.")
            return

        try:
            config = self._build_config_from_form()
            save_config(config, self.current_config_path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status_var.set("Saved")
        self._append_log(f"[info] saved config: {self.current_config_path}")

    def _build_config_from_form(self) -> AppConfig:
        return AppConfig(
            audio_capture=AudioCaptureConfig(
                mode=self.capture_mode_var.get().strip() or "virtual_device",
                input_device=self.input_device_var.get().strip(),
                sample_rate=int(self.sample_rate_var.get()),
                channels=int(self.channels_var.get()),
                chunk_ms=int(self.chunk_ms_var.get()),
                silence_timeout_ms=int(self.silence_timeout_ms_var.get()),
                rms_threshold=float(self.rms_threshold_var.get()),
                min_speech_ms=int(self.min_speech_ms_var.get()),
                max_utterance_ms=int(self.max_utterance_ms_var.get()),
            ),
            audio_output=AudioOutputConfig(
                tts_output_device=self.tts_output_device_var.get().strip(),
                monitor_output_device=self.monitor_output_device_var.get().strip(),
            ),
            stt=SttConfig(
                backend=self.stt_backend_var.get().strip(),
                model=self.stt_model_var.get().strip(),
                device=self.stt_device_var.get().strip(),
                compute_type=self.compute_type_var.get().strip(),
                language=self.stt_language_var.get().strip(),
                timeout_sec=int(self.stt_timeout_var.get()),
                beam_size=int(self.beam_size_var.get()),
                vad_filter=bool(self.vad_filter_var.get()),
                vad_min_silence_ms=int(self.vad_min_silence_ms_var.get()),
            ),
            llm=LlmConfig(
                backend=self.llm_backend_var.get().strip(),
                base_url=self.llm_base_url_var.get().strip(),
                model=self.llm_model_var.get().strip(),
                temperature=float(self.temperature_var.get()),
                max_tokens=int(self.max_tokens_var.get()),
                system_prompt=self.system_prompt_text.get("1.0", tk.END).strip(),
                timeout_sec=int(self.llm_timeout_var.get()),
            ),
            tts=TtsConfig(
                backend=self.tts_backend_var.get().strip(),
                base_url=self.tts_base_url_var.get().strip(),
                speaker=int(self.tts_speaker_var.get()),
                speed_scale=float(self.speed_scale_var.get()),
                timeout_sec=int(self.tts_timeout_var.get()),
            ),
            conversation=ConversationConfig(
                max_response_chars=int(self.max_response_chars_var.get()),
                min_reply_interval_sec=int(self.min_reply_interval_var.get()),
                allow_topic_suggestions=bool(self.allow_topic_suggestions_var.get()),
                pause_listening_while_speaking=bool(self.pause_listening_var.get()),
            ),
        )

    def _start_runtime(self) -> None:
        if self.worker.running:
            return

        try:
            config = self._build_config_from_form()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        self.worker = WorkerState(thread=None, stop_event=threading.Event())
        base_dir = config_base_dir(self.current_config_path)
        self._append_log(f"[info] starting runtime with {self.current_config_path}")
        self.status_var.set("Warming up")
        self._set_running_state(True)

        thread = threading.Thread(
            target=self._run_worker,
            args=(config, base_dir, self.worker.stop_event),
            daemon=True,
        )
        self.worker.thread = thread
        thread.start()

    def _run_worker(self, config: AppConfig, base_dir: Path, stop_event: threading.Event) -> None:
        try:
            with BotRuntime(config, base_dir=base_dir) as runtime:
                runtime.warm_up(logger=self._enqueue_log)
                if stop_event.is_set():
                    return
                self.events.put(("status", "Running"))
                runtime.run_forever(
                    stop_event=stop_event,
                    logger=self._enqueue_log,
                    listen_timeout_sec=1.0,
                )
        except Exception as exc:
            self.events.put(("log", f"[error] {exc}"))
        finally:
            self.events.put(("stopped", ""))

    def _stop_runtime(self) -> None:
        if not self.worker.running or self.worker.stop_event is None:
            return
        self.worker.stop_event.set()
        self.status_var.set("Stopping")
        self._append_log("[info] stop requested")

    def _set_running_state(self, is_running: bool) -> None:
        self.start_button.state(["disabled"] if is_running else ["!disabled"])
        self.stop_button.state(["!disabled"] if is_running else ["disabled"])
        self.load_button.state(["disabled"] if is_running else ["!disabled"])
        self.save_button.state(["disabled"] if is_running else ["!disabled"])
        self.refresh_devices_button.state(["disabled"] if is_running else ["!disabled"])
        self.refresh_models_button.state(["disabled"] if is_running else ["!disabled"])

    def _enqueue_log(self, message: str) -> None:
        self.events.put(("log", message))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _poll_events(self) -> None:
        while True:
            try:
                event_name, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if event_name == "log":
                self._append_log(payload)
            elif event_name == "status":
                self.status_var.set(payload)
            elif event_name == "stopped":
                self.status_var.set("Stopped")
                self._set_running_state(False)
                self._append_log("[info] runtime stopped")
                if self.pending_close:
                    self.root.destroy()
                    return

        self.root.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.worker.running:
            self.pending_close = True
            self._stop_runtime()
            return
        self.root.destroy()


def run_gui(config_path: Path | None = None) -> int:
    if config_path is None:
        config_path = ensure_config_file(DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH)
    elif not config_path.exists():
        if config_path == DEFAULT_CONFIG_PATH:
            config_path = ensure_config_file(DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH)
        else:
            raise FileNotFoundError(f"Config file was not found: {config_path}")

    root = tk.Tk()
    GuiApp(root, config_path)
    root.mainloop()
    return 0
