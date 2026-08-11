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

サブPC側で本番用バッチを起動します。このプロセスは時間制限なく動作し、`Ctrl+C`を押すかウィンドウを閉じるまで停止しません。

```powershell
.\controls\run_chatgpt_voice_production.bat
```

コマンドから直接起動する場合は次を実行します。

```powershell
uv run vrchat-voice-control --config config/chatgpt_voice.toml
```

本番用プロセスにはChatGPT画面状態監視が内蔵されています。`run_chatgpt_ui_diagnostic.bat`はUI変化をJSONLへ記録する3分間の調査専用ツールであり、本番運用中に併用する必要はありません。

監視は0.75秒ごとに対象プロセスとトップレベルウィンドウを再探索します。ChatGPT内で新しいチャット／タスクへ移動した場合やVoiceを開始し直した場合も、通常は本番用プロセスを再起動する必要はありません。ChatGPTアプリ自体を終了・再起動した場合も、起動後のウィンドウを自動的に再検出します。再起動中は一時的にGUIの画面検出が未検出／待機中になります。

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
- ChatGPT画面状態監視の有効化／無効化（設定はサブPCへ保存）
- AI発話字幕のOFF／UIA／STT切り替えとVRChatチャットボックス表示テスト（設定はサブPCへ保存）
- アバター自動モーションの有効化／停止（設定はサブPCへ保存）
- 現在アバターのOSC再読み込み、専用OSCプローブによる同期確認、安全な初期状態への復帰後にONLINEで開始
- `VoiceAgentStatus`の0〜3への切り替え
- CABLE-A/Bの直近RMS、ループ判定、モーション状態と発話強度の確認

アバター状態・考え中表示・自動モーション・動作状態・アクセント・表情は、GUI内部の送信値ではなくVRChatからOSCで返った実値を表示します。`✓`は送信目標と実値が一致、`未確認`はまだ返答がない状態、`未反映→`は実値と送信目標が違う状態です。静的な状態変更はUDPの取りこぼしを抑えるため短い間隔で3回送信し、`VoiceAgentStatus`はVRChatから一致した値が返るまで成功扱いにしません。

アバター切り替え時はPlayable Layerの初期化を待ってから、現在の状態・モーション・表情を再送します。GUIが`未確認`のままの場合は、VRChatのAction MenuでOSCが有効か、現在選択中のアバターがAIアバターかを確認してください。

### 開始前OSC同期

GUIの`同期確認して開始`は、最初に現在のアバターをOSCで再読み込みします。現在IDはVRChatから受信済みの値を優先し、未受信の場合だけ最新のVRChatログから特定します。同じIDを`/avatar/change`へ1回だけ送信し、VRChatからのアバター変更通知を待ってから、`VoiceAgentStatus=3 (MAINTENANCE)`の確認と専用Bool `VoiceAgentOscProbe`の`OFF → ON → OFF`往復確認へ進みます。成功すると診断モーション、発話強度、アクセント、表情、考え中表示を安全な初期値へ戻し、最後に`VoiceAgentStatus=1 (ONLINE)`を確認します。マイクのミュート状態は自動変更しません。

VRChatの公開OSC仕様には、Action Menuにある完全な「アバターリセット」専用アドレスはありません。そのため、この機能は現在アバターを再読み込みするOSC上の代替手段です。現在IDを安全に特定できない、再読み込み通知が返らない、途中で別アバターに変わった、のいずれかでは開始を中止し、ERRORのままONLINEへ進みません。

GUIにはOSCの送受信先、アバター再読み込み結果と所要時間、現在のアバターID、プローブ結果、往復時間、初期化結果を表示します。アバター変更通知を受けると以前の成功結果は`要再確認`になります。これはサブPC上のPythonサービスと現在のVRChatアバター間の同期確認であり、別ユーザーへのネットワーク表示確認は遠隔視点の動作確認を併用してください。

```text
Name: VoiceAgentOscProbe
Type: Bool
Default: false
Saved: false
Synced: true
```

このパラメーターを追加したUnity版へ更新後は、アバターの再アップロードが1回必要です。古いアバターでは同期確認が失敗し、GUIに`VoiceAgentOscProbe`の確認エラーが表示されます。

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
- 発話アクセント: 大きな頷き、左手を胸元、右手をお腹、両手を胸前、髪の毛くるくる、口に指を当てる、前傾姿勢から非反復でランダム選択
- 待機アクセント: 大きな頷き、髪の毛くるくる、かわいい待機2、うたた寝01から非反復でランダム選択
- 終話後: 約0.55秒の収束状態を経て待機動作へ復帰
- 緊急ミュートまたはERROR中: 発話動作とアクセントを抑制

VRChat Desktopの頭・両手IKにモーションを上書きされないよう、自動モーション有効中はHead、Left Hand、Right HandをAnimator制御へ切り替えます。EyesとMouthは変更しないため、瞬き、Eye Look、口パクは維持されます。自動モーションをOFFにすると、頭と両手はVRChatのTracking制御へ戻ります。

OSCで使う同期パラメーターは次のとおりです。発話状態・音量・アクセントを同期することで、AIアバター本人だけでなく周囲のVRChatユーザーにも同じ動作が表示されます。

| パラメーター | 型 | 値 |
|---|---|---|
| `VoiceAgentMotionEnabled` | Bool | 自動モーションON/OFF |
| `VoiceAgentActivity` | Int | 0 IDLE / 1 SPEAKING / 2 SETTLING |
| `VoiceAgentEnergy` | Float | 発話強度 0.0〜1.0 |
| `VoiceAgentGesture` | Int | 0なし / 1大きな頷き / 2左手を胸元 / 3右手をお腹 / 4両手を胸前 / 5髪の毛くるくる / 6口に指 / 7前傾姿勢 / 8かわいい待機2 / 9うたた寝01 |
| `VoiceAgentExpression` | Int | 0通常 / 1 Open / 2 FingerPoint / 3 Victory / 4 Rock&Roll / 5 Gun / 6 ThumbsUpの表情 |

Expressions Menuには`AIモーション > 自動モーション`が追加されます。メインPCの操作画面からも同じON/OFFを切り替えられます。設定値は`config/chatgpt_voice.toml`の`[motion]`へ保存されます。

アクセント値は、ほかのユーザーへ送られるPlayableパラメーター同期を確実にするため、`gesture_sync_hold_sec`（既定1.5秒）の間だけ非0のまま保持してから0へ戻します。短くしすぎるとローカルでは動いても、遠隔視点ではアクセントを取りこぼす場合があります。

既定の検出値は通常のChatGPT Voice向けに控えめにしています。反応しにくい場合は`[motion]`の`speech_on_rms`を下げ、環境音へ反応する場合は上げてください。Unityの生成クリップでは、待機動作を小さく保ちつつ、通常発話を基準値の2倍、強い発話を2.25倍、アクセント動作を1.7倍にしています。

アクセント抽選は、待機中が8〜18秒ごと（平均13秒）、発話中が2.8〜5.8秒ごと（平均4.3秒）です。待機中は頷き37.5%、髪12.5%、かわいい待機37.5%、うたた寝12.5%を基準にし、発話中は頷き21.4%、胸元14.3%、お腹14.3%、両手14.3%、髪7.1%、口に指14.3%、前傾14.3%を基準にします。同じ動作は連続させず、収束中、緊急ミュート中、ERROR中、自動モーションOFF中は抽選しません。状態が待機・発話へ切り替わるたびに次回までの時間を引き直すため、短い発話ではアクセントが出ない場合があります。

発話表情は発話開始時に1つ選び、その後2〜4秒ごとに通常、Open、FingerPoint、Victory、Rock&Roll、Gun、ThumbsUpの7種類から非反復で切り替えます。手のGestureパラメーターは変更せず、各ジェスチャーに対応する顔の表情だけを利用します。発話終了、緊急ミュート、ERROR、自動モーションOFFでは通常表情へ戻ります。瞬き、Eye Look、口パク用のBlendShapeは表情クリップから除外しています。

操作画面の`全動作テスト（約49秒）`は、CABLE-Bの音量判定を一時的に迂回し、発話状態のまま9種類のアクセントを5秒ずつ再生します。同時に7種類の表情も順番に切り替え、最後に収束状態と待機状態を確認してから自動モーションへ戻ります。同期先の別アカウントでも見分けられるよう、一瞬で全項目を切り替えず約49秒かけて確認します。

同じ画面から、待機中・発話中・収束中の状態、各アクセント、各表情を個別に指定することもできます。個別指定中は音声による自動判定を停止し、表情と状態は`診断終了・自動へ戻す`まで保持します。表情を選ぶと表示条件を満たすため自動的に発話中へ切り替わります。アクセントは選択するたび1回再生され、同期用パラメーターは`gesture_sync_hold_sec`の間保持されます。別アカウントから特定動作の見え方を確認する場合はこちらを使用してください。

CABLE-Bへ入った音を発話として扱うため、ChatGPT以外のアプリをCABLE-Bへ出すと、その音にも反応します。Windowsのシステム音やブラウザーは別の出力先へ分離してください。

初回の実機確認は次の順番で行います。

1. Unityから更新したアバターをBuild & Publishする
2. VRChatのOSCを有効にし、更新したアバターへ切り替える
3. サブPCで`controls\run_chatgpt_voice_production.bat`を起動する
4. 操作画面の自動モーションが`IDLE / ON`になることを確認する
5. 無音時に呼吸・首・上半身・腕の小さな待機動作が出ることを確認する
6. ChatGPT Voiceに発話させ、表示が`SPEAKING / ON`へ切り替わることを確認する
7. Expressions Menuの`AIモーション > 自動モーション`で停止・再開できることを確認する

動きが強すぎる・腕が衣装へ干渉する場合は、まずUnity側のクリップ振幅を下げます。音量判定の閾値は動きの強さではなく、発話状態へ切り替わる感度だけを調整します。

Unity側を再生成する場合は、メニューの`Tools > Voice Agent Motion > Install Or Rebuild`を実行します。

### Unity内ライブプレビュー

VRChatへアップロードせず動きを確認する場合は、Unityメニューの`Tools > Voice Agent Motion > Open Live Preview`を開き、`プレビュー開始`を押します。Modular Avatar処理後のBase LocomotionとAdditive Animatorを、VRChatと同様に独立したPlayableレイヤーとして合成した一時コピーがScene Viewに表示されます。元のアバターはプレビュー中だけ非表示になり、シーンや元アバターには変更を保存しません。

プレビュー画面では、待機中・発話中・収束中の切り替え、発話強度0.0〜1.0、7種類の発話表情、再生速度、9種類の動作の選択と単発再生を操作できます。`自動デモ`を有効にすると、待機から発話、表情と強度の変化、9種類の動作、収束までを繰り返します。プレビュー内ではVRChat Desktop相当の組み込み値（`Upright=1`、`TrackingType=3`、`VRMode=0`、`Grounded=true`）を補って立位を維持します。終了時は`プレビュー終了`を押してください。ウィンドウを閉じた場合やPlay Modeへ入る場合も一時コピーは自動削除されます。

## 7. 考え中表示

操作サーバーはChatGPTデスクトップアプリのWindows UI Automation要素を読み取り、作業中ステータスバー、アクティブなシマー表示、または`思考中`ヘッダーを検出します。通常作業は`WORKING`、検索文言も確認できた場合は`Web検索中`としてGUIへ表示します。これは読み取り専用であり、ChatGPTへのクリックや文字入力は行いません。

アバター表示条件は次のとおりです。

```text
VoiceAgentThinking = ChatGPT画面が作業中 AND CABLE-Bが発話中ではない
```

回答音声中は一度隠れ、発話終了後もChatGPT側が作業中なら再び表示されます。UI再描画による短い消失を避けるため、完了判定には既定2.5秒、検索状態には既定3秒のホールドを設けています。

設定は`config/chatgpt_voice.toml`の次の項目です。既存ファイルに節がなくても既定値で動作します。

```toml
[ui_monitor]
enabled = true
include_offscreen = true
interval_sec = 0.75
release_hold_sec = 2.5
search_hold_sec = 3.0
```

`include_offscreen = true`は、VRChatが前面にある場合やChatGPTを最小化している場合も、ChatGPTのアクセシビリティ要素を監視する設定です。

LAN操作GUIの`考え中表示テスト`を押すと、画面検出とは無関係に`VoiceAgentThinking`を一度OFFにしてからONに送信します。腰前のLexa表示が`AI試験中`から`考え中…`へ切り替わることを確認し、確認後は`テスト終了・自動へ`を押してください。テスト中にもう一度押してもOFF→ONを再生するため、表示が変わらない原因がChatGPT画面検出側か、OSC／アバター側かを切り分けられます。

Unity側は`Assets/VoiceAgentLexa`の生成スクリプトで、同期Bool `VoiceAgentThinking`と「考え中…」表示を追加します。Expressions Menuの`AI表示 > 考え中テスト`でも単独確認できます。再生成はUnityメニューの`Tools > Voice Agent > Configure Lexa AI Trial Sign`を実行します。通常の1行・2行表示より考え中表示が優先され、OFFになると元の表示へ戻ります。

```text
Name: VoiceAgentThinking
Type: Bool
Default: false
Saved: false
Synced: true
```

Unity側を変更した後はアバターの再アップロードが必要です。

## 8. AI発話字幕

LAN操作GUIの`AI発話字幕（VRChatチャットボックス）`から、次の3方式を即時に切り替えられます。選択は`config/chatgpt_voice.toml`へ保存され、次回起動時にも引き継がれます。

- `OFF`: 字幕を送信しない
- `UIA`: ChatGPTデスクトップアプリの読み取り専用UI Automationから、発話中に追加・更新された最新回答を抽出する
- `STT`: CABLE-Bへ出たAI音声だけをfaster-whisperでローカル文字起こしする

どちらもVRChatの標準OSC `/chatbox/input`へ最大144文字で送ります。長い回答は先頭へ`…`を付けて最新側を表示し、初期設定では`AI: `を付けます。`/chatbox/typing`もAI発話中だけONになります。チャットボックス表示はアバター改変を使わないため、この機能だけのための再アップロードは不要です。

UIAは高速で音声認識誤りがありませんが、ChatGPTアプリ更新後にアクセシビリティ構造が変わると抽出できない場合があります。履歴本文や利用者側の発言を誤送信しないよう、CABLE-BでAI音声を検出した瞬間のUIを基準にし、直後の候補を短時間保留してから新しい回答だけを送ります。GUIの`最新字幕`と`エラー`を見ながら実機で確認してください。画面構造に左右されない安定性を優先する場合はSTTを使用してください。

STTは画面構造に依存しません。初めてSTTへ切り替えたときだけ`small`モデルを取得して読み込むため、`字幕STT`が`loading`から`ready`になるまで待ちます。既定はCPU用`int8`です。音声は発話単位の一時WAVへ変換して文字起こし後すぐ削除し、会話ログとして保存しません。OpenAI API、Ollama、VOICEVOX、旧ローカル応答ランタイムは起動しません。

```toml
[captions]
mode = "off"
prefix = "AI: "
max_chars = 144
min_send_interval_sec = 1.5
uia_initial_hold_sec = 1.0
stt_model = "small"
stt_device = "cpu"
stt_compute_type = "int8"
stt_language = "ja"
```

初回の比較テストは次の順番で行います。

1. VRChatのOSCとチャットボックス表示を有効にし、GUIの`字幕表示テスト`で`AI: 字幕表示テストです`が見えることを確認する
2. `UIA`へ切り替え、ChatGPT Voiceへ短い回答を依頼し、GUIの`最新字幕`とVRChat表示を確認する
3. `STT`へ切り替え、`字幕STT=ready`を待って同じ依頼を試す
4. 別アカウントからもチャットボックスが見えるか確認する
5. 採用する方式を選ぶか、試験後に`OFF`へ戻す

## 9. 終了

サブPCの操作サーバーで`Ctrl+C`を押すと、終了前に`VoiceAgentStatus=0`を送ります。異常終了時は送れない場合があるため、操作画面には手動のSTOPPEDボタンも用意しています。

ChatGPT Voice利用中は、次を実行しないでください。

```powershell
python -m vrchat_ai_tool run
```
