from __future__ import annotations

import hmac
import json
import secrets
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .loop_guard import LoopDetection, LoopGuardService
from .osc_control import AgentStatus, VRChatOscController
from .voice_config import ChatGPTVoiceConfig, resolve_config_relative, split_names

CONTROL_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VRChat Voice Agent Control</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#081018;color:#e8f5ff}
body{max-width:760px;margin:0 auto;padding:24px}h1{font-size:1.35rem;margin:0 0 18px}
.card{background:#101d2a;border:1px solid #24445c;border-radius:14px;padding:16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
button,input{font:inherit;border-radius:9px;padding:11px;border:1px solid #3a617c}
button{cursor:pointer;background:#173149;color:#fff}button:hover{background:#214763}
.danger{background:#6f1e2a}.ok{background:#12613d}.warn{background:#765b13}
input{box-sizing:border-box;width:100%;background:#07121d;color:#fff}.row{display:flex;gap:10px}
.row input{flex:1}.pill{display:inline-block;border-radius:999px;padding:4px 10px;background:#20364a}
#message{min-height:1.4em;color:#9dd9ff}.muted{color:#ffabb6}.online{color:#7ff0b2}
small{color:#aac0cf}dl{display:grid;grid-template-columns:150px 1fr;gap:6px;margin:8px 0}dd{margin:0}
</style></head><body>
<h1>VRChat Voice Agent Control</h1>
<div class="card"><div class="row"><input id="token" type="password" placeholder="control-token.txt のトークン">
<button onclick="saveToken()">接続</button></div><small>トークンはこのタブ内だけに保存されます。</small></div>
<div class="card"><h2>現在の状態</h2><dl>
<dt>アバター表示</dt><dd id="agentStatus">—</dd><dt>VRChatマイク</dt><dd id="mic">—</dd>
<dt>ループ監視</dt><dd id="loop">—</dd><dt>CABLE-A / B</dt><dd id="levels">—</dd></dl>
<div id="message"></div></div>
<div class="card"><h2>緊急ミュート</h2><div class="grid">
<button class="danger" onclick="post('/api/mic/mute')">ミュート</button>
<button class="ok" onclick="post('/api/mic/unmute')">ミュート解除</button>
<button class="warn" onclick="post('/api/loop/reset')">ループ警報リセット</button></div>
<small>ループ警報中は、先に警報をリセットしないとミュート解除できません。</small></div>
<div class="card"><h2>アバター状態表示</h2><div class="grid">
<button onclick="setStatus(0)">0 STOPPED</button><button class="ok" onclick="setStatus(1)">1 ONLINE</button>
<button class="danger" onclick="setStatus(2)">2 ERROR</button><button class="warn" onclick="setStatus(3)">3 MAINTENANCE</button>
</div></div>
<script>
const names=['STOPPED','ONLINE','ERROR','MAINTENANCE']; let timer=null;
function token(){return sessionStorage.getItem('voiceAgentToken')||''}
function saveToken(){sessionStorage.setItem('voiceAgentToken',document.getElementById('token').value); refresh(); if(!timer)timer=setInterval(refresh,1500)}
async function request(path,opts={}){opts.headers={...(opts.headers||{}),Authorization:'Bearer '+token(),'Content-Type':'application/json'};
 const response=await fetch(path,opts); const data=await response.json(); if(!response.ok)throw new Error(data.error||response.statusText); return data}
async function post(path,body={}){try{const d=await request(path,{method:'POST',body:JSON.stringify(body)}); message(d.message||'OK');refresh()}catch(e){message(e.message,true)}}
function setStatus(value){post('/api/status',{value})} function message(value,error=false){const e=document.getElementById('message');e.textContent=value;e.style.color=error?'#ff9cab':'#9dd9ff'}
async function refresh(){if(!token())return;try{const d=await request('/api/status');document.getElementById('agentStatus').textContent=d.status+' '+names[d.status];
 const m=d.muted===null?'未確認':(d.muted?'MUTED':'OPEN');document.getElementById('mic').textContent=m;
 document.getElementById('loop').textContent=d.loop.triggered?`LOOP DETECTED (${d.loop.score}, ${d.loop.delay_ms}ms)`:(d.loop.running?'監視中':'停止中');
 document.getElementById('levels').textContent=`${d.loop.cable_a_rms} / ${d.loop.cable_b_rms}`;if(d.last_error)message(d.last_error,true)}catch(e){message(e.message,true)}}
document.getElementById('token').value=token();if(token()){refresh();timer=setInterval(refresh,1500)}
</script></body></html>"""


def load_or_create_token(config: ChatGPTVoiceConfig) -> tuple[str, Path, bool]:
    path = resolve_config_relative(config, config.control.token_file)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise ValueError(f"Control token is too short: {path}")
        return token, path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    return token, path, True


class VoiceControlService:
    def __init__(self, config: ChatGPTVoiceConfig) -> None:
        self.config = config
        self.osc = VRChatOscController(config.osc)
        self.loop_guard = LoopGuardService(config, self._on_loop, self._on_loop_error)
        self._lock = threading.RLock()
        self.last_error = ""

    def start(self) -> None:
        self.osc.start()
        self.osc.send_status(AgentStatus.STOPPED)
        self.loop_guard.start()

    def stop(self) -> None:
        self.loop_guard.stop()
        try:
            self.osc.send_status(AgentStatus.STOPPED)
        finally:
            self.osc.stop()

    def _on_loop(self, detection: LoopDetection) -> None:
        detail = f"Self-loop detected (score={detection.score}, delay={detection.delay_ms}ms)"
        with self._lock:
            self.last_error = detail
        self.osc.send_status(AgentStatus.ERROR)
        if self.config.loop_guard.auto_mute:
            try:
                self.osc.set_muted(True)
            except Exception as exc:  # noqa: BLE001 - safety boundary for OSC/UDP failures
                with self._lock:
                    self.last_error = f"{detail}; automatic mute failed: {exc}"

    def _on_loop_error(self, detail: str) -> None:
        with self._lock:
            self.last_error = f"Loop guard stopped: {detail}"
        try:
            self.osc.send_status(AgentStatus.ERROR)
        except Exception as exc:  # noqa: BLE001 - preserve the original monitor failure
            with self._lock:
                self.last_error = f"{self.last_error}; OSC status failed: {exc}"

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "status": int(self.osc.status),
                "muted": self.osc.mute_state,
                "loop": self.loop_guard.snapshot(),
                "last_error": self.last_error,
            }

    def mute(self) -> None:
        self.osc.set_muted(True)

    def unmute(self) -> None:
        if self.loop_guard.detector.last.triggered:
            raise RuntimeError("ループ警報中です。原因を確認してから警報をリセットしてください。")
        self.osc.set_muted(False)

    def set_status(self, value: int) -> None:
        if value == int(AgentStatus.ONLINE) and self.loop_guard.detector.last.triggered:
            raise RuntimeError("ループ警報中はONLINEへ戻せません。先に原因を確認して警報をリセットしてください。")
        self.osc.send_status(AgentStatus(value))
        with self._lock:
            if value != int(AgentStatus.ERROR):
                self.last_error = ""

    def reset_loop(self) -> None:
        self.loop_guard.reset()
        with self._lock:
            self.last_error = ""


def make_handler(
    service: VoiceControlService,
    token: str,
    allowed_ips: tuple[str, ...],
) -> type[BaseHTTPRequestHandler]:
    class ControlHandler(BaseHTTPRequestHandler):
        server_version = "VRChatVoiceControl/1.0"

        def _allowed(self) -> bool:
            return not allowed_ips or self.client_address[0].casefold() in allowed_ips or self.client_address[0] in {"127.0.0.1", "::1"}

        def _authorized(self) -> bool:
            provided = self.headers.get("Authorization", "")
            prefix = "Bearer "
            return provided.startswith(prefix) and hmac.compare_digest(provided[len(prefix) :], token)

        def _json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _reject_if_needed(self) -> bool:
            if not self._allowed():
                self._json(403, {"error": "この接続元IPは許可されていません"})
                return True
            if not self._authorized():
                self._json(401, {"error": "操作トークンが正しくありません"})
                return True
            return False

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("request body too large")
            if not length:
                return {}
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError("JSON object required")
            return value

        def do_GET(self) -> None:
            if self.path == "/":
                if not self._allowed():
                    self._json(403, {"error": "この接続元IPは許可されていません"})
                    return
                body = CONTROL_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/status":
                if self._reject_if_needed():
                    return
                self._json(200, service.snapshot())
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self._reject_if_needed():
                return
            try:
                body = self._body()
                if self.path == "/api/mic/mute":
                    service.mute()
                    message = "VRChatマイクをミュートしました"
                elif self.path == "/api/mic/unmute":
                    service.unmute()
                    message = "VRChatマイクのミュートを解除しました"
                elif self.path == "/api/status":
                    service.set_status(int(body["value"]))
                    message = "アバター状態を変更しました"
                elif self.path == "/api/loop/reset":
                    service.reset_loop()
                    message = "ループ警報をリセットしました（ミュートは解除していません）"
                else:
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, {"ok": True, "message": message, **service.snapshot()})
            except (KeyError, TypeError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._json(409, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - HTTP service must return JSON, not crash
                self._json(500, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            print(f"[{self.client_address[0]}] {format % args}")

    return ControlHandler


def run_control_server(
    config: ChatGPTVoiceConfig,
    service_factory: Callable[[ChatGPTVoiceConfig], VoiceControlService] = VoiceControlService,
) -> int:
    token, token_path, created = load_or_create_token(config)
    allowed_ips = split_names(config.control.allowed_client_ips)
    service = service_factory(config)
    handler = make_handler(service, token, allowed_ips)
    server = ThreadingHTTPServer((config.control.bind_host, config.control.port), handler)
    server.daemon_threads = True
    try:
        service.start()
        print(f"Voice control: http://{config.control.bind_host}:{config.control.port}/")
        print(f"Token file: {token_path}")
        if created:
            print(f"New token: {token}")
        print("LAN内だけで使用し、ルーターのポート開放はしないでください。")
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        service.stop()
    return 0
