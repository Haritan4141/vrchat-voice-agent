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

トークン入力は最初の1回だけです。「接続・保存」を押すと同じブラウザへ保存され、次回から自動接続します。ブラウザのサイトデータを削除した場合、接続URLを変更した場合、または`control-token.txt`を再生成した場合だけ再入力が必要です。「保存解除」でブラウザから削除できます。

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
- 自己ループ監視の有効化／無効化（設定はサブPCへ保存）
- アバター自動モーションの有効化／停止（設定はサブPCへ保存）
- `VoiceAgentStatus`の0〜3への切り替え
- CABLE-A/Bの直近RMS、ループ判定、モーション状態と発話強度の確認

ループ警報が残っている間はミュート解除を拒否します。「原因を直す → 警報リセット → ミュート解除」の順で操作してください。

## 4. 自己ループ自動停止

初期設定では、ループ確定時の自動ミュートが有効です。検出後のミュート解除は必ず手動です。監視自体はメインPCの操作画面から有効／無効を切り替えられ、選択は`config/chatgpt_voice.toml`へ保存されます。監視を無効にしても、その時点のミュートは自動解除されません。

監視は続けたまま、閾値の調整中だけ自動ミュートを止めたい場合は、次を変更します。

```toml
[loop_guard]
auto_mute = false
```

調整する主な値:

- `correlation_threshold`: 高くすると誤検出しにくい（既定0.95）
- `min_consecutive_matches`: 高くすると確定まで慎重になる（既定5）
- `rms_threshold`: 小さなノイズを比較対象から外す（既定250）
- `min_delay_ms` / `max_delay_ms`: BからAへ戻るまでの探索範囲
- `reliable_max_delay_ms`: 偶然の一致を避ける探索遅延の安全上限（既定1800ms）
- `delay_tolerance_ms`: 連続一致として許容する遅延の揺れ幅（既定160ms）
- `min_match_duration_ms`: 同じ遅延の一致が続くべき最低時間（既定1500ms）

既存の設定ファイルに新しい3項目がなくても、上記の既定値が自動的に適用されます。今回のような2秒を超える単発一致は、既定設定ではループ確定になりません。

検出後は自動解除しません。音声データはファイルへ保存しません。監視を無効にした場合も、CABLE-Bのレベル取得は自動モーション用に継続しますが、相関比較とループ確定は行いません。

## 5. アバター状態表示

Unityプロジェクトでは、`Assets/StatusHalo_for_PC`のParallax版を使った`VoiceAgentStatusHalo`をアバタールートへ設置済みです。表示は共通のIntパラメーター`VoiceAgentStatus`で切り替わります。

| 値 | 表示 | 用途 |
|---:|---|---|
| 0 | STOPPED | 停止中 |
| 1 | ONLINE | 動作中 |
| 2 | ERROR | エラー／ループ警報 |
| 3 | MAINTENANCE | 調整中（将来用） |

Modular Avatarがビルド時に次のVRChat Expression Parameterを追加します。

```text
Name: VoiceAgentStatus
Type: Int
Default: 0
Saved: false
Synced: true
```

固有のAI名はパラメーター、コード、表示仕様に使用していません。`Synced=true`のため、ステータスヘイローの変化は他ユーザーにも同期されます。

操作サーバーはOSCの`/avatar/parameters/VoiceAgentStatus`へ値を送ります。Expressions Menuには`AI STATUS`サブメニューが追加され、`0 STOPPED`、`1 ONLINE`、`2 ERROR`、`3 MAINTENANCE`から同じパラメーターを手動変更できます。メニュー用の内部Boolはローカル専用で、選択結果のIntだけが同期されます。

生成・接続用のEditorスクリプトと生成物はUnityプロジェクトの`Assets/VoiceAgentStatusHalo`にあります。再生成する場合はアバタールートを選択し、Unityメニューの`Tools > Voice Agent Status Halo > Install Or Rebuild On Selected Avatar`を実行します。

## 6. アバター自動モーション

Unityプロジェクトの`Assets/VoiceAgentMotion`に、Modular Avatar用のAdditive Animatorと生成スクリプトを追加しています。既存の口パク、瞬き、Eye Lookを残しながら、移動を伴わない次の動作を行います。

- 待機中: 呼吸、首、上半身、肩、腕、手の小さな微動
- 発話中: CABLE-B上のChatGPT Voice音量に応じた動作強度
- アクセント: 頷き、左手、右手、両手の4種類から非反復でランダム選択
- 終話後: 約0.55秒の収束状態を経て待機動作へ復帰
- 緊急ミュートまたはERROR中: 発話動作とアクセントを抑制

OSCで使うローカル専用パラメーターは次のとおりです。

| パラメーター | 型 | 値 |
|---|---|---|
| `VoiceAgentMotionEnabled` | Bool | 自動モーションON/OFF |
| `VoiceAgentActivity` | Int | 0 IDLE / 1 SPEAKING / 2 SETTLING |
| `VoiceAgentEnergy` | Float | 発話強度 0.0〜1.0 |
| `VoiceAgentGesture` | Int | 0なし / 1頷き / 2左 / 3右 / 4両手 |

Expressions Menuには`AIモーション > 自動モーション`が追加されます。メインPCの操作画面からも同じON/OFFを切り替えられます。設定値は`config/chatgpt_voice.toml`の`[motion]`へ保存されます。

既定の検出値は通常のChatGPT Voice向けに控えめにしています。反応しにくい場合は`[motion]`の`speech_on_rms`を下げ、環境音へ反応する場合は上げてください。動き自体の大きさはUnityの生成クリップ側で安全な小さめの値に固定しています。

CABLE-Bへ入った音を発話として扱うため、ChatGPT以外のアプリをCABLE-Bへ出すと、その音にも反応します。Windowsのシステム音やブラウザーは別の出力先へ分離してください。

初回の実機確認は次の順番で行います。

1. Unityから更新したアバターをBuild & Publishする
2. VRChatのOSCを有効にし、更新したアバターへ切り替える
3. サブPCで`launch_voice_control.bat`を起動する
4. 操作画面の自動モーションが`IDLE / ON`になることを確認する
5. 無音時に呼吸・首・上半身・腕の小さな待機動作が出ることを確認する
6. ChatGPT Voiceに発話させ、表示が`SPEAKING / ON`へ切り替わることを確認する
7. Expressions Menuの`AIモーション > 自動モーション`で停止・再開できることを確認する

動きが強すぎる・腕が衣装へ干渉する場合は、まずUnity側のクリップ振幅を下げます。音量判定の閾値は動きの強さではなく、発話状態へ切り替わる感度だけを調整します。

Unity側を再生成する場合は、メニューの`Tools > Voice Agent Motion > Install Or Rebuild`を実行します。

## 7. 終了

サブPCの操作サーバーで`Ctrl+C`を押すと、終了前に`VoiceAgentStatus=0`を送ります。異常終了時は送れない場合があるため、操作画面には手動のSTOPPEDボタンも用意しています。

ChatGPT Voice利用中は、次を実行しないでください。

```powershell
python -m vrchat_ai_tool run
```
