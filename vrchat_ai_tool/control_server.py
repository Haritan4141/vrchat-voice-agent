from __future__ import annotations

import hmac
import json
import secrets
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .chatgpt_ui_state import ChatGptUiStateMonitor, UiActivityState
from .loop_guard import LoopDetection, LoopGuardService
from .motion_control import MotionActivity, MotionService
from .osc_control import AgentStatus, VRChatOscController
from .voice_config import (
    ChatGPTVoiceConfig,
    resolve_config_relative,
    save_loop_guard_enabled,
    save_motion_enabled,
    save_ui_monitor_enabled,
    split_names,
)

CONTROL_HTML = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VRChat Voice Agent Control</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#081018;color:#e8f5ff}
body{max-width:760px;margin:0 auto;padding:24px}h1{font-size:1.35rem;margin:0 0 18px}
.card{background:#101d2a;border:1px solid #24445c;border-radius:14px;padding:16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
button,input{font:inherit;border-radius:9px;padding:11px;border:1px solid #3a617c}
button{cursor:pointer;background:#173149;color:#fff}button:hover{background:#214763}button:disabled{cursor:wait;opacity:.65}
.danger{background:#6f1e2a}.ok{background:#12613d}.warn{background:#765b13}
input{box-sizing:border-box;width:100%;background:#07121d;color:#fff}.row{display:flex;gap:10px}
.row input{flex:1}.pill{display:inline-block;border-radius:999px;padding:4px 10px;background:#20364a}
#message{min-height:1.4em;color:#9dd9ff}.muted{color:#ffabb6}.online{color:#7ff0b2}
small{color:#aac0cf}dl{display:grid;grid-template-columns:150px 1fr;gap:6px;margin:8px 0}dd{margin:0}
h3{font-size:1rem;margin:16px 0 8px}.wide{grid-column:1/-1}
</style></head><body>
<h1>VRChat Voice Agent Control</h1>
<div class="card"><div class="row"><input id="token" type="password" placeholder="control-token.txt のトークン">
<button onclick="saveToken()">接続・保存</button><button onclick="forgetToken()">保存解除</button></div>
<small>初回接続後はこのブラウザに保存され、次回から自動接続します。</small></div>
<div class="card"><h2>現在の状態</h2><dl>
<dt>アバター表示</dt><dd id="agentStatus">—</dd><dt>VRChatマイク</dt><dd id="mic">—</dd>
<dt>ループ監視</dt><dd id="loop">—</dd><dt>CABLE-A / B</dt><dd id="levels">—</dd>
<dt>ChatGPT状態</dt><dd id="chatgptState">—</dd><dt>考え中表示</dt><dd id="thinking">—</dd>
<dt>自動モーション</dt><dd id="motion">—</dd><dt>発話レベル</dt><dd id="motionLevel">—</dd>
<dt>アクセント</dt><dd id="motionGesture">—</dd><dt>発話表情</dt><dd id="motionExpression">—</dd></dl>
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
<div class="card"><h2>開始前OSC同期</h2>
<div class="grid"><button id="preflightButton" class="ok wide" onclick="preflightStart()">同期確認して開始</button></div>
<dl><dt>総合</dt><dd id="preflightState">未実行</dd>
<dt>VRChat OSC</dt><dd id="preflightOsc">—</dd>
<dt>アバター</dt><dd id="preflightAvatar">—</dd>
<dt>専用プローブ</dt><dd id="preflightProbe">—</dd>
<dt>往復時間</dt><dd id="preflightRtt">—</dd>
<dt>初期状態</dt><dd id="preflightBaseline">—</dd></dl>
<small>同期中はMAINTENANCE、成功後はONLINEになります。マイクのミュート状態は安全のため自動変更しません。アバターを変更した場合は再確認してください。</small></div>
<div class="card"><h2>自己ループ対策</h2><div class="grid">
<button class="ok" onclick="setLoopGuard(true)">監視を有効化</button>
<button class="danger" onclick="setLoopGuard(false)">監視を無効化</button></div>
<small>設定はサブPCへ保存され、次回起動時にも引き継がれます。無効中も手動ミュートは使えます。</small></div>
<div class="card"><h2>ChatGPT画面状態監視</h2><div class="grid">
<button class="ok" onclick="setUiMonitor(true)">監視を有効化</button>
<button class="danger" onclick="setUiMonitor(false)">監視を無効化</button>
<button class="warn" onclick="setThinkingTest(true)">考え中表示テスト</button>
<button onclick="setThinkingTest(false)">テスト終了・自動へ</button></div>
<small>ChatGPTのアクセシビリティUIを読み取り、作業中・Web検索中だけアバターへ「考え中」を表示します。クリックや文字入力は行いません。</small></div>
<div class="card"><h2>アバター自動モーション</h2><div class="grid">
<button class="ok" onclick="setMotion(true)">モーションを有効化</button>
<button class="danger" onclick="setMotion(false)">モーションを停止</button></div></div>
<div class="card"><h2>遠隔視点の動作確認</h2>
<div class="grid"><button class="warn" onclick="post('/api/motion/test')">全動作テスト（約49秒）</button>
<button class="danger" onclick="post('/api/motion/diagnostic/stop')">診断終了・自動へ戻す</button></div>
<h3>動作状態</h3><div class="grid">
<button onclick="diagnosticActivity(0)">待機中</button>
<button class="ok" onclick="diagnosticActivity(1)">発話中</button>
<button onclick="diagnosticActivity(2)">収束中</button></div>
<h3>アクセント動作</h3><div class="grid">
<button onclick="diagnosticGesture(1)">1 大きく頷く</button>
<button onclick="diagnosticGesture(2)">2 左手を胸元へ</button>
<button onclick="diagnosticGesture(3)">3 右手をお腹へ</button>
<button onclick="diagnosticGesture(4)">4 両手を胸前へ</button>
<button onclick="diagnosticGesture(5)">5 髪の毛くるくる</button>
<button onclick="diagnosticGesture(6)">6 口に指</button>
<button onclick="diagnosticGesture(7)">7 前傾姿勢</button>
<button onclick="diagnosticGesture(8)">8 かわいい待機2</button>
<button onclick="diagnosticGesture(9)">9 うたた寝01</button></div>
<h3>表情</h3><div class="grid">
<button onclick="diagnosticExpression(0)">0 通常</button>
<button onclick="diagnosticExpression(1)">1 Open</button>
<button onclick="diagnosticExpression(2)">2 FingerPoint</button>
<button onclick="diagnosticExpression(3)">3 Victory</button>
<button onclick="diagnosticExpression(4)">4 Rock&amp;Roll</button>
<button onclick="diagnosticExpression(5)">5 Gun</button>
<button onclick="diagnosticExpression(6)">6 ThumbsUp</button></div>
<small>診断中はCABLE-Bによる自動判定を一時停止します。全動作テストは9アクセントを5秒ずつ再生し、その間に全7表情も切り替えます。個別の表情と状態は診断終了まで保持され、表情を選ぶと発話中へ切り替わります。</small></div>
<script>
const names=['STOPPED','ONLINE','ERROR','MAINTENANCE'];
const activityNames=['待機中','発話中','収束中'];
const uiStateNames={idle:'待機中',working:'作業中',searching:'Web検索中'};
const faceNames=['通常','Open','FingerPoint','Victory','Rock&Roll','Gun','ThumbsUp']; let timer=null;
const gestureNames=['なし','大きく頷く','左手を胸元へ','右手をお腹へ','両手を胸前へ','髪の毛くるくる','口に指','前傾姿勢','かわいい待機2','うたた寝01'];
function token(){let value=localStorage.getItem('voiceAgentToken')||'';if(!value){value=sessionStorage.getItem('voiceAgentToken')||'';if(value)localStorage.setItem('voiceAgentToken',value)}return value}
function saveToken(){const value=document.getElementById('token').value.trim();if(!value){message('トークンを入力してください',true);return}localStorage.setItem('voiceAgentToken',value);sessionStorage.removeItem('voiceAgentToken');refresh();if(!timer)timer=setInterval(refresh,1500)}
function forgetToken(){localStorage.removeItem('voiceAgentToken');sessionStorage.removeItem('voiceAgentToken');document.getElementById('token').value='';if(timer){clearInterval(timer);timer=null}message('保存したトークンを削除しました')}
async function request(path,opts={}){opts.headers={...(opts.headers||{}),Authorization:'Bearer '+token(),'Content-Type':'application/json'};
 const response=await fetch(path,opts); const data=await response.json(); if(!response.ok)throw new Error(data.error||response.statusText); return data}
async function post(path,body={}){try{const d=await request(path,{method:'POST',body:JSON.stringify(body)}); message(d.message||'OK');refresh()}catch(e){message(e.message,true)}}
function setStatus(value){post('/api/status',{value})} function setLoopGuard(enabled){post('/api/loop/enabled',{enabled})} function setMotion(enabled){post('/api/motion/enabled',{enabled})} function setUiMonitor(enabled){post('/api/ui-monitor/enabled',{enabled})} function setThinkingTest(enabled){post('/api/thinking/test',{enabled})}
function diagnosticActivity(value){post('/api/motion/diagnostic/activity',{value})} function diagnosticGesture(value){post('/api/motion/diagnostic/gesture',{value})} function diagnosticExpression(value){post('/api/motion/diagnostic/expression',{value})}
async function preflightStart(){const button=document.getElementById('preflightButton');button.disabled=true;message('OSC同期を確認しています…');try{const d=await request('/api/preflight/start',{method:'POST',body:'{}'});message(d.message||'同期確認が完了しました');await refresh()}catch(e){message(e.message,true);await refresh()}finally{button.disabled=false}}
function message(value,error=false){const e=document.getElementById('message');e.textContent=value;e.style.color=error?'#ff9cab':'#9dd9ff'}
function confirmedValue(actual,target,labeler){if(actual===null||actual===undefined)return `未確認（送信目標: ${labeler(target)}）`;if(actual!==target)return `${labeler(actual)} / 未反映→ ${labeler(target)}`;return `${labeler(actual)} ✓`}
async function refresh(){if(!token())return;try{const d=await request('/api/status');const a=d.avatar||{};document.getElementById('agentStatus').textContent=confirmedValue(a.status,a.status_target,v=>`${v} ${names[v]||'—'}`);
 const m=d.muted===null?'未確認':(d.muted?'MUTED':'OPEN');document.getElementById('mic').textContent=m;
 document.getElementById('loop').textContent=d.loop.enabled===false?'無効':(d.loop.triggered?`LOOP DETECTED (${d.loop.score}, ${d.loop.delay_ms}ms)`:(d.loop.running?'監視中':'停止中'));
 document.getElementById('levels').textContent=`${d.loop.cable_a_rms} / ${d.loop.cable_b_rms}`;
 const ui=d.ui_monitor||{};document.getElementById('chatgptState').textContent=(ui.enabled===false?'監視無効':(ui.last_error?`エラー: ${ui.last_error}`:(!ui.available?'ChatGPT未検出':`${uiStateNames[ui.state]||ui.state}（${ui.element_count||0}要素）`)))+(ui.test_override?' / 表示テスト中':'');
 document.getElementById('thinking').textContent=confirmedValue(a.thinking,a.thinking_target,v=>v?'考え中 ON':'OFF');
 const enabled=confirmedValue(a.motion_enabled,a.motion_enabled_target,v=>v?'ON':'OFF');const activity=confirmedValue(a.activity,a.activity_target,v=>activityNames[v]||String(v));document.getElementById('motion').textContent=`${activity}${d.motion.diagnostic_running?` / TEST: ${d.motion.diagnostic_label||'手動確認'}`:''} / ${enabled}`;
 const actualEnergy=a.energy===null||a.energy===undefined?'未確認':Number(a.energy).toFixed(2);document.getElementById('motionLevel').textContent=`RMS ${d.motion.input_rms} / ENERGY ${d.motion.energy}（VRChat ${actualEnergy}）`;
 document.getElementById('motionGesture').textContent=confirmedValue(a.gesture,a.gesture_target,v=>`${v} ${gestureNames[v]||'—'}`);
 document.getElementById('motionExpression').textContent=confirmedValue(a.expression,a.expression_target,v=>`${v} ${faceNames[v]||'—'}`);
 const p=d.preflight||{};const stateNames={not_run:'未実行',running:'確認中',ready:'READY ✓',stale:'要再確認',error:'ERROR'};document.getElementById('preflightState').textContent=`${stateNames[p.state]||p.state||'未実行'}${p.message?` — ${p.message}`:''}`;
 const osc=d.osc||{};document.getElementById('preflightOsc').textContent=p.osc_ok?`${osc.target||'—'} → ${osc.listen||'—'} ✓`:`${osc.target||'—'} → ${osc.listen||'—'}`;
 document.getElementById('preflightAvatar').textContent=p.avatar_id?`${p.avatar_id}${p.avatar_generation!==undefined?` / 世代${p.avatar_generation}`:''}`:(p.probe_ok?'現在のAIアバターを確認 ✓':'未確認');
 document.getElementById('preflightProbe').textContent=p.probe_ok?'OFF→ON→OFF ✓':(a.probe===null||a.probe===undefined?'未確認':`実値 ${a.probe?'ON':'OFF'}`);
 document.getElementById('preflightRtt').textContent=p.probe_rtt_ms===null||p.probe_rtt_ms===undefined?'—':`${Number(p.probe_rtt_ms).toFixed(1)} ms`;
 document.getElementById('preflightBaseline').textContent=p.baseline_ok?'安全な初期値を再送済み ✓':'未確認';if(d.last_error)message(d.last_error,true)}catch(e){message(e.message,true)}}
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


def should_show_thinking(
    ui_state: UiActivityState,
    motion_activity: int,
) -> bool:
    """Show work/search state only while ChatGPT Voice is not speaking."""
    return (
        ui_state != UiActivityState.IDLE
        and motion_activity != int(MotionActivity.SPEAKING)
    )


THINKING_REASSERT_SEC = 1.0


class VoiceControlService:
    def __init__(self, config: ChatGPTVoiceConfig) -> None:
        self.config = config
        self.osc = VRChatOscController(config.osc)
        self.motion = MotionService(config.motion, self.osc)
        self._ui_state = UiActivityState.IDLE
        self._thinking_output = False
        self._thinking_test_override = False
        # UI Automation and audio-level callbacks run on different threads. Keep
        # the complete decide/send/update sequence ordered so an older ON packet
        # cannot overtake a newer speech-suppression OFF packet.
        self._thinking_output_lock = threading.RLock()
        self._thinking_last_sent_at = 0.0
        self.ui_monitor = ChatGptUiStateMonitor(
            config.ui_monitor,
            self._on_ui_state,
            self._on_ui_monitor_error,
            process_names=split_names(config.processes.chatgpt),
        )
        self.loop_guard = LoopGuardService(
            config,
            self._on_loop,
            self._on_loop_error,
            on_cable_b_level=self._on_cable_b_level,
            on_cable_b_level_error=self._on_motion_error,
        )
        self._lock = threading.RLock()
        self._preflight_lock = threading.Lock()
        self._preflight: dict[str, object] = {
            "state": "not_run",
            "message": "同期確認を実行してください",
            "osc_ok": False,
            "probe_ok": False,
            "baseline_ok": False,
            "probe_rtt_ms": None,
            "avatar_id": None,
            "avatar_generation": 0,
            "completed_at": None,
        }
        self.last_error = ""

    def start(self) -> None:
        self.osc.start()
        self.osc.send_status(AgentStatus.STOPPED)
        self.osc.send_thinking(False)
        self.motion.start()
        self.loop_guard.start()
        self.ui_monitor.start()

    def stop(self) -> None:
        with self._lock:
            self._thinking_test_override = False
        self.ui_monitor.stop()
        self.loop_guard.stop()
        self.motion.stop()
        try:
            self.osc.send_thinking(False)
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

    def _on_motion_error(self, detail: str) -> None:
        with self._lock:
            self.last_error = f"Avatar motion update failed: {detail}"
        try:
            self.osc.send_status(AgentStatus.ERROR)
        except Exception as exc:  # noqa: BLE001 - keep loop monitoring alive on OSC failure
            with self._lock:
                self.last_error = f"{self.last_error}; OSC status failed: {exc}"

    def _on_ui_state(self, state: UiActivityState) -> None:
        with self._lock:
            self._ui_state = state
        self._update_thinking_output()

    def _on_cable_b_level(self, rms: float) -> None:
        self.motion.on_audio_level(rms)
        self._update_thinking_output()

    def _update_thinking_output(self, *, force: bool = False) -> None:
        with self._thinking_output_lock:
            activity = int(self.motion.snapshot()["activity"])
            now = time.monotonic()
            with self._lock:
                visible = self._thinking_test_override or should_show_thinking(
                    self._ui_state,
                    activity,
                )
                unchanged_and_recent = (
                    visible == self._thinking_output
                    and now - self._thinking_last_sent_at < THINKING_REASSERT_SEC
                )
                if not force and unchanged_and_recent:
                    return

            # UDP delivery and VRChat parameter propagation are best effort. A
            # one-second reassertion keeps the visible avatar state converged even
            # if one packet is lost, without producing meaningful traffic.
            self.osc.send_thinking(visible)
            with self._lock:
                self._thinking_output = visible
                self._thinking_last_sent_at = now

    def _on_ui_monitor_error(self, detail: str) -> None:
        with self._lock:
            self.last_error = detail

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            motion = self.motion.snapshot()
            ui_monitor = self.ui_monitor.snapshot()
            avatar = self.osc.feedback_snapshot()
            ui_monitor["suppressed_by_speech"] = (
                ui_monitor["thinking"]
                and int(motion["activity"]) == int(MotionActivity.SPEAKING)
            )
            ui_monitor["test_override"] = self._thinking_test_override
            return {
                "status": int(self.osc.status),
                "avatar": avatar,
                "muted": self.osc.mute_state,
                "loop": self.loop_guard.snapshot(),
                "motion": motion,
                "ui_monitor": ui_monitor,
                "osc": {
                    "target": f"{self.config.osc.target_host}:{self.config.osc.input_port}",
                    "listen": f"{self.config.osc.listen_host}:{self.config.osc.output_port}",
                },
                "preflight": self._preflight_snapshot(avatar),
                "last_error": self.last_error,
            }

    def _preflight_snapshot(self, avatar: dict[str, object]) -> dict[str, object]:
        result = dict(self._preflight)
        checked_generation = int(result.get("avatar_generation", 0))
        current_generation = int(avatar.get("avatar_generation", 0))
        if result.get("state") == "ready" and checked_generation != current_generation:
            result["state"] = "stale"
            result["message"] = "アバター変更後のため再確認が必要です"
            result["probe_ok"] = False
            result["baseline_ok"] = False
        return result

    def preflight_and_start(self) -> dict[str, object]:
        """Confirm OSC/avatar round-trip, restore safe values, then enter ONLINE."""
        if not self._preflight_lock.acquire(blocking=False):
            raise RuntimeError("OSC同期確認はすでに実行中です")
        try:
            with self._lock:
                self._preflight = {
                    "state": "running",
                    "message": "VRChatとアバターを確認中",
                    "osc_ok": True,
                    "probe_ok": False,
                    "baseline_ok": False,
                    "probe_rtt_ms": None,
                    "avatar_id": None,
                    "avatar_generation": 0,
                    "completed_at": None,
                }
                self.last_error = ""

            try:
                if self.loop_guard.detector.last.triggered:
                    raise RuntimeError(
                        "ループ警報中は開始できません。原因を確認して警報をリセットしてください。"
                    )
                self.osc.set_status_confirmed(AgentStatus.MAINTENANCE)
                probe_rtt_ms = self.osc.confirm_probe_roundtrip()

                # Clear manual diagnostics and stale transient values before the
                # agent is advertised as ONLINE. Muting is deliberately untouched.
                self.motion.stop_diagnostic_test()
                self.motion.set_enabled(self.config.motion.enabled)
                with self._thinking_output_lock:
                    with self._lock:
                        self._thinking_test_override = False
                    self.osc.send_thinking(False)
                    with self._lock:
                        self._thinking_output = False
                        self._thinking_last_sent_at = time.monotonic()

                self.osc.set_status_confirmed(AgentStatus.ONLINE)
                avatar = self.osc.feedback_snapshot()
                with self._lock:
                    self._preflight = {
                        "state": "ready",
                        "message": "OSC同期済み・ONLINE",
                        "osc_ok": bool(avatar.get("osc_listener_running")),
                        "probe_ok": avatar.get("probe") is False,
                        "baseline_ok": True,
                        "probe_rtt_ms": probe_rtt_ms,
                        "avatar_id": avatar.get("avatar_id"),
                        "avatar_generation": int(avatar.get("avatar_generation", 0)),
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.last_error = ""
                return dict(self._preflight)
            except Exception as exc:
                detail = f"OSC同期確認に失敗しました: {exc}"
                try:
                    self.osc.send_status(AgentStatus.ERROR)
                except Exception:  # noqa: BLE001, S110 - preserve the primary error
                    pass
                avatar = self.osc.feedback_snapshot()
                with self._lock:
                    self._preflight = {
                        "state": "error",
                        "message": str(exc),
                        "osc_ok": bool(avatar.get("osc_listener_running")),
                        "probe_ok": False,
                        "baseline_ok": False,
                        "probe_rtt_ms": None,
                        "avatar_id": avatar.get("avatar_id"),
                        "avatar_generation": int(avatar.get("avatar_generation", 0)),
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.last_error = detail
                raise RuntimeError(detail) from exc
        finally:
            self._preflight_lock.release()

    def mute(self) -> None:
        self.osc.set_muted(True)

    def unmute(self) -> None:
        if self.loop_guard.detector.last.triggered:
            raise RuntimeError("ループ警報中です。原因を確認してから警報をリセットしてください。")
        self.osc.set_muted(False)

    def set_status(self, value: int) -> None:
        if value == int(AgentStatus.ONLINE) and self.loop_guard.detector.last.triggered:
            raise RuntimeError("ループ警報中はONLINEへ戻せません。先に原因を確認して警報をリセットしてください。")
        self.osc.set_status_confirmed(AgentStatus(value))
        with self._lock:
            if value != int(AgentStatus.ERROR):
                self.last_error = ""

    def reset_loop(self) -> None:
        self.loop_guard.reset()
        with self._lock:
            self.last_error = ""

    def set_loop_guard_enabled(self, enabled: bool) -> None:
        save_loop_guard_enabled(self.config, enabled)
        self.loop_guard.set_enabled(enabled)
        with self._lock:
            if self.last_error.startswith(("Self-loop detected", "Loop guard stopped")):
                self.last_error = ""

    def set_motion_enabled(self, enabled: bool) -> None:
        save_motion_enabled(self.config, enabled)
        self.motion.set_enabled(enabled)

    def set_ui_monitor_enabled(self, enabled: bool) -> None:
        save_ui_monitor_enabled(self.config, enabled)
        self.ui_monitor.set_enabled(enabled)
        if not enabled:
            with self._lock:
                self._ui_state = UiActivityState.IDLE
            self._update_thinking_output()
        with self._lock:
            if self.last_error.startswith("ChatGPT UI monitor"):
                self.last_error = ""

    def set_thinking_test(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._thinking_output_lock:
            with self._lock:
                self._thinking_test_override = enabled
            if enabled:
                # A display test should remain visibly testable even when the previous
                # target/feedback is already True. Pulse OFF before ON so the avatar
                # Animator must traverse both transitions instead of receiving the same
                # unchanged Bool again.
                self.osc.send_thinking(False)
                time.sleep(0.15)
                self.osc.send_thinking(True)
                with self._lock:
                    self._thinking_output = True
                    self._thinking_last_sent_at = time.monotonic()
                return
            self._update_thinking_output(force=True)

    def start_motion_diagnostic_test(self) -> None:
        self.motion.start_diagnostic_test()

    def stop_motion_diagnostic(self) -> None:
        self.motion.stop_diagnostic_test()

    def set_motion_diagnostic_activity(self, value: int) -> None:
        self.motion.set_diagnostic_activity(value)

    def play_motion_diagnostic_gesture(self, value: int) -> None:
        self.motion.play_diagnostic_gesture(value)

    def set_motion_diagnostic_expression(self, value: int) -> None:
        self.motion.set_diagnostic_expression(value)


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
                elif self.path == "/api/preflight/start":
                    service.preflight_and_start()
                    message = "OSC同期を確認し、アバターをONLINEにしました"
                elif self.path == "/api/loop/reset":
                    service.reset_loop()
                    message = "ループ警報をリセットしました（ミュートは解除していません）"
                elif self.path == "/api/motion/enabled":
                    enabled = body["enabled"]
                    if not isinstance(enabled, bool):
                        raise TypeError("enabled must be true or false")
                    service.set_motion_enabled(enabled)
                    message = (
                        "自動モーションを有効にしました"
                        if enabled
                        else "自動モーションを停止しました"
                    )
                elif self.path == "/api/ui-monitor/enabled":
                    enabled = body["enabled"]
                    if not isinstance(enabled, bool):
                        raise TypeError("enabled must be true or false")
                    service.set_ui_monitor_enabled(enabled)
                    message = (
                        "ChatGPT画面状態監視を有効にしました"
                        if enabled
                        else "ChatGPT画面状態監視を無効にしました"
                    )
                elif self.path == "/api/thinking/test":
                    enabled = body["enabled"]
                    if not isinstance(enabled, bool):
                        raise TypeError("enabled must be true or false")
                    service.set_thinking_test(enabled)
                    message = (
                        "考え中表示テストを開始しました"
                        if enabled
                        else "考え中表示テストを終了し、自動判定へ戻しました"
                    )
                elif self.path == "/api/motion/test":
                    service.start_motion_diagnostic_test()
                    message = "全アクセント・全表情の遠隔表示テストを開始しました"
                elif self.path == "/api/motion/diagnostic/stop":
                    service.stop_motion_diagnostic()
                    message = "動作診断を終了し、自動モーションへ戻しました"
                elif self.path == "/api/motion/diagnostic/activity":
                    service.set_motion_diagnostic_activity(int(body["value"]))
                    message = "診断用の動作状態を変更しました"
                elif self.path == "/api/motion/diagnostic/gesture":
                    service.play_motion_diagnostic_gesture(int(body["value"]))
                    message = "診断用アクセントを再生しました"
                elif self.path == "/api/motion/diagnostic/expression":
                    service.set_motion_diagnostic_expression(int(body["value"]))
                    message = "診断用の表情を変更しました"
                elif self.path == "/api/loop/enabled":
                    enabled = body["enabled"]
                    if not isinstance(enabled, bool):
                        raise TypeError("enabled must be true or false")
                    service.set_loop_guard_enabled(enabled)
                    message = "自己ループ監視を有効にしました" if enabled else "自己ループ監視を無効にしました"
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
        print(f"Voice control (this PC): http://127.0.0.1:{config.control.port}/")
        if config.control.bind_host not in {"127.0.0.1", "localhost"}:
            try:
                addresses = socket.gethostbyname_ex(socket.gethostname())[2]
            except OSError:
                addresses = []
            for address in sorted({value for value in addresses if not value.startswith("127.")}):
                print(f"Voice control (LAN):     http://{address}:{config.control.port}/")
        print(f"Listening on: {config.control.bind_host}:{config.control.port}")
        print(f"Token file: {token_path}")
        if created:
            print(f"New token: {token}")
        print("LAN内だけで使用し、ルーターのポート開放はしないでください。")
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        service.stop()
    return 0
