# VRChat Voice Agent

VRChatの音声とChatGPTデスクトップアプリのVoiceを、2本のVB-CABLEで接続するための補助ツールです。会話そのものはChatGPT Voiceが担当し、このPythonツールは診断、安全停止、VRChat OSC、メインPCからの操作だけを担当します。

## 確定している音声経路

```text
VRChat出力
  → CABLE-A Input
  → CABLE-A Output
  → ChatGPT Voice入力

ChatGPT Voice出力
  → CABLE-B Input
  → CABLE-B Output
  → VRChatマイク
```

## 追加した機能

- `chatgpt-voice-doctor`: 4端点、Windows既定値、アプリ別ルート、システム音、Parsec設定、二重応答プロセスを検査
- 自己ループ監視: CABLE-Bの音声包絡が遅れてCABLE-Aへ戻った状態を検出
- 緊急ミュート: VRChat OSCの`MuteSelf`を確認しながら安全にミュート／解除
- LAN操作画面: メインPCのブラウザからミュートと状態表示を操作
- アバター状態: 汎用Intパラメーター`VoiceAgentStatus`へ0〜3を送信
- 考え中表示: ChatGPTの読み取り専用UI Automation状態とCABLE-Bの無音判定から、同期Bool `VoiceAgentThinking`を自動制御
- キャラクター指示: GPT Live開始後に`system_prompt.txt`を安全に送信し、必要時に再適用

## 初回準備

```powershell
Copy-Item config/chatgpt_voice.example.toml config/chatgpt_voice.toml
uv sync
uv run chatgpt-voice-doctor --config config/chatgpt_voice.toml
```

動作中のレベルも見る場合:

```powershell
uv run chatgpt-voice-doctor --config config/chatgpt_voice.toml --live-seconds 8
```

操作サーバーを起動する場合:

```powershell
uv run vrchat-voice-control --config config/chatgpt_voice.toml
```

初回起動時だけ`config/control-token.txt`が生成されます。メインPCから`http://サブPCのIPv4:18765/`を開き、そのトークンを入力します。

詳細は[ChatGPT Voice運用ガイド](docs/chatgpt-voice-control.md)を参照してください。

キャラクター指示は`apply_voice_prompt.bat`から適用します。詳しい手順は[GPT Liveへのキャラクター指示の適用](docs/chatgpt-voice-prompt.md)を参照してください。

## 重要

ChatGPT Voiceと併用中は、旧ローカルAIランタイムを起動しないでください。

```powershell
# 実行しない
python -m vrchat_ai_tool run
```

このコマンドはfaster-whisper、Ollama、VOICEVOXによる別の応答系です。Doctorと操作サーバーはこれらを起動しません。

## 旧ローカルAI機能

既存の`doctor`、`devices`、`listen-once`、`speak`、`run`、`gui`は互換性のため残しています。現在のChatGPT Voice構成では`devices`と診断用途以外は使用しません。
