# VRChat Voice Agent

VRChatの音声とChatGPTデスクトップアプリのVoiceを、2本のVB-CABLEで接続するための補助ツールです。会話そのものはChatGPT Voiceが担当し、このPythonツールは診断、安全停止、VRChat OSC、メインPCからの操作、任意のAI発話字幕だけを担当します。

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
- 開始前OSC同期: 現在のアバターをOSCで再読み込みし、`VoiceAgentOscProbe`のOFF→ON→OFF往復と初期状態を確認してからONLINEへ移行
- アバター状態: 汎用Intパラメーター`VoiceAgentStatus`へ0〜3を送信
- 考え中表示: ChatGPTの読み取り専用UI Automation状態とCABLE-Bの無音判定から、同期Bool `VoiceAgentThinking`を自動制御
- AI発話字幕: UI AutomationまたはCABLE-BのローカルSTTを選び、接頭辞なしの短いフレーズとしてVRChat標準チャットボックスへ送信
- STT精度切替: GUIから標準（small）と高精度（medium）を切り替え、モデルを停止なしで再読込
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

本番運用を開始する場合は、次のバッチを実行します。停止するまで無期限で動作し、ChatGPT画面状態監視、OSC、LAN操作画面、自己ループ監視をまとめて起動します。

```powershell
.\controls\run_chatgpt_voice_production.bat
```

コマンドから直接起動する場合:

```powershell
uv run vrchat-voice-control --config config/chatgpt_voice.toml
```

終了する場合は、サブPC側のコンソールで`Ctrl+C`を押します。`run_chatgpt_ui_diagnostic.bat`は3分間の調査ログを取得する診断専用であり、本番運用では起動不要です。

初回起動時だけ`config/control-token.txt`が生成されます。メインPCから`http://サブPCのIPv4:18765/`を開き、そのトークンを入力します。

詳細は[ChatGPT Voice運用ガイド](docs/chatgpt-voice-control.md)を参照してください。

キャラクター指示は`controls\apply_voice_prompt.bat`から適用します。普段使う起動バッチは`controls`フォルダーへまとめています。詳しい手順は[GPT Liveへのキャラクター指示の適用](docs/chatgpt-voice-prompt.md)を参照してください。

## 重要

ChatGPT Voiceと併用中は、旧ローカルAIランタイムを起動しないでください。

```powershell
# 実行しない
python -m vrchat_ai_tool run
```

このコマンドはfaster-whisper、Ollama、VOICEVOXによる別の応答系です。本番用操作サーバーはOllamaやVOICEVOXを起動しません。字幕を`STT`へ切り替えた場合だけ、応答生成を行わない字幕専用のfaster-whisperを使用します。

## 旧ローカルAI機能

既存の`doctor`、`devices`、`listen-once`、`speak`、`run`、`gui`は互換性のため残しています。現在のChatGPT Voice構成では`devices`と診断用途以外は使用しません。
