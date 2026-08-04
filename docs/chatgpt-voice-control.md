# ChatGPT Voice運用ガイド

## 1. サブPCの準備

リポジトリで次を実行します。

```powershell
Copy-Item config/chatgpt_voice.example.toml config/chatgpt_voice.toml
uv sync
```

VRChatのAction MenuでOSCを有効にしてください。標準ポートのままなら、このツールはVRChatへUDP 9000で送信し、UDP 9001で状態を受信します。

VRChatの「マイクの動作」は`切り替え（トグル）`にします。別の動作モードでは、確実なミュートON/OFFを保証できないためサーバーは起動時に拒否します。

## 2. Doctor

VRChatとChatGPT Voiceを起動し、Voiceを開始した状態で実行すると最も正確です。

```powershell
uv run chatgpt-voice-doctor --config config/chatgpt_voice.toml --live-seconds 8
```

判定は次の3段階です。

- `OK`: 確認できた
- `WARN`: アプリ未起動などで確認できない、または推奨値と異なる
- `ERROR`: 端点不在、誤配線、システム音混入、二重応答、Parsec自動ミュートなど

`--json`を付けると診断結果を機械可読JSONで出力します。ChatGPT Voice自体には外部から接続状態を問い合わせる公開インターフェースがないため、DoctorはWindowsの実際の音声セッションとレベルから判定します。

## 3. メインPCから操作する

サブPC側で起動します。

```powershell
uv run vrchat-voice-control --config config/chatgpt_voice.toml
```

初回のみ、十分に長いランダムトークンが`config/control-token.txt`へ生成されます。`ipconfig`でサブPCのIPv4を確認し、メインPCのブラウザから次を開きます。

```text
http://サブPCのIPv4:18765/
```

Windows Defender Firewallの確認が出た場合は、プライベートネットワークだけを許可します。ルーターのポート開放は行わないでください。

より安全にするには、`config/chatgpt_voice.toml`の`allowed_client_ips`へメインPCのIPv4だけを指定します。

```toml
[control]
allowed_client_ips = "192.168.1.20"
```

IPが複数ある場合はカンマ区切りです。`127.0.0.1`は常に許可されます。

操作画面では以下を実行できます。

- 緊急ミュート／ミュート解除
- ループ警報の手動リセット
- `VoiceAgentStatus`の0〜3への切り替え
- CABLE-A/Bの直近RMSとループ判定の確認

ループ警報が残っている間はミュート解除を拒否します。「原因を直す → 警報リセット → ミュート解除」の順で操作してください。

## 4. 自己ループ自動停止

初期設定では、ループ確定時の自動ミュートが有効です。検出後のミュート解除は必ず手動です。閾値の調整中だけ自動ミュートを止めたい場合は、次を変更します。

```toml
[loop_guard]
auto_mute = false
```

調整する主な値:

- `correlation_threshold`: 高くすると誤検出しにくい（既定0.88）
- `min_consecutive_matches`: 高くすると確定まで慎重になる（既定3）
- `rms_threshold`: 小さなノイズを比較対象から外す（既定250）
- `min_delay_ms` / `max_delay_ms`: BからAへ戻るまでの探索範囲

検出後は自動解除しません。音声データはファイルへ保存しません。

## 5. アバター状態表示（Unity作業は未実施）

Unity側は別セッションで作業中のため、この実装では変更していません。次回のUnity作業で、インポート済みの`Assets/StatusHalo_for_PC`と次のパラメーターを接続します。

| 値 | 表示 | 用途 |
|---:|---|---|
| 0 | STOPPED | 停止中 |
| 1 | ONLINE | 動作中 |
| 2 | ERROR | エラー／ループ警報 |
| 3 | MAINTENANCE | 調整中（将来用） |

VRChat Expression Parametersに次を追加します。

```text
Name: VoiceAgentStatus
Type: Int
Default: 0
Saved: false
Synced: true
```

固有のAI名はパラメーター、コード、表示仕様に使用していません。`Synced=true`にすることで、ステータスヘイローの変化を他ユーザーにも同期できます。

Animator側では値0〜3をヘイローの表示・色・テキスト状態へ割り当てます。アセット固有の階層やAnimator構成は、Unityを直接操作できる次のセッションで確認して接続します。

## 6. 終了

サブPCの操作サーバーで`Ctrl+C`を押すと、終了前に`VoiceAgentStatus=0`を送ります。異常終了時は送れない場合があるため、操作画面には手動のSTOPPEDボタンも用意しています。

ChatGPT Voice利用中は、次を実行しないでください。

```powershell
python -m vrchat_ai_tool run
```
