# Architecture

## 責務の分離

```text
ChatGPT Desktop Voice ── 会話生成・音声認識・音声出力
Windows + VB-CABLE   ── A/Bの音声経路
vrchat-voice-agent   ── 診断・監視・安全停止・UI状態読取・任意の字幕STT・OSC・LAN操作
VRChat Avatar        ── VoiceAgentStatus / VoiceAgentThinkingの表示・VoiceAgentOscProbeの往復確認
```

PythonサービスはChatGPTの画面をWindows UI Automationで読み取りますが、クリック、文字入力、設定変更、会話生成は行いません。字幕をSTTへ切り替えた場合だけCABLE-Bをローカル文字起こししますが、旧ローカルLLM/TTSは起動しません。

## 制御経路

```text
メインPCのブラウザ
  → HTTP/TCP 18765（Bearerトークン）
  → サブPCのVoiceControlService
      ├─ UDP/9000 → VRChat /input/Voice
      ├─ UDP/9000 → /avatar/parameters/VoiceAgentStatus
      ├─ UDP/9000 → /avatar/parameters/VoiceAgentThinking
      ├─ UDP/9000 → /chatbox/input / /chatbox/typing
      ├─ UDP/9000 ↔ /avatar/parameters/VoiceAgentOscProbe
      └─ UDP/9001 ← VRChat /avatar/parameters/MuteSelf
```

ミュートは`/input/Voice`を押したあと、VRChatから返る`MuteSelf`を確認します。現在値が不明な場合でも、確認結果が希望状態と逆ならもう一度だけ切り替えるため、単純なブラインドトグルにはなりません。

開始前同期では、現在のアバターIDをOSC受信値または最新のVRChatログから特定し、同じIDを`/avatar/change`へ1回だけ送って再読み込みを要求します。VRChatから`/avatar/change`が戻った後、`VoiceAgentOscProbe`をOFF→ON→OFFへ変化させ、両方のエッジがOSC出力から戻ることを確認します。成功後だけ安全な初期値とONLINEを送るため、以前のUDP取りこぼしや別アバター選択を開始前に発見できます。

VRChatの公開OSC仕様にはAction Menuの「アバターリセット」専用入力がないため、これは現在アバターの再読み込みによるOSC上の代替手段です。IDを安全に特定できない場合、再読み込み通知が返らない場合、途中で別アバターへ変わった場合はONLINEへ進みません。

## 自己ループ検出

CABLE-AとCABLE-Bを同時に録音し、短時間ごとのRMS包絡を作ります。最近のCABLE-Aと、100〜5000ms前のCABLE-Bを正規化相関で比較します。閾値を連続して超えたときだけ警報をラッチします。

警報時は`VoiceAgentStatus=2`にします。`auto_mute=true`の場合はVRChatマイクもミュートします。誤って会話が再開しないよう、自動ミュート解除は行いません。

この方式は音声内容を保存せず、音量の時間変化だけをメモリ上で比較します。

## 考え中表示

ChatGPTのアクセシビリティツリーに現れる作業中ステータスバーを定期的に読み取り、Web検索中の文言を補助信号にします。過去の会話本文や検索ボタンだけでは作業中と判定しません。短いUI再描画による消失は2.5〜3秒のホールドで吸収します。

UIが作業中でもCABLE-Bが発話中なら`VoiceAgentThinking=false`とし、発話が止まった時だけ`true`にします。Lexa表示では考え中が通常のAI試験中表示より優先され、解除後は元の1行・2行・OFF設定へ戻ります。

## AI発話字幕

UIA方式は考え中表示と同じアクセシビリティスキャンを共有し、CABLE-Bの発話開始時点のスナップショットとの差分を短時間保留してから最新回答候補を抽出します。これにより、AI再生より前に表示された利用者側の発言を基準側へ含めます。STT方式はLoopGuardがすでに取得しているCABLE-B PCMを別ワーカーへ渡し、発話区間ごとにfaster-whisperで処理します。どちらも字幕出力だけをOSCチャットボックスへ送り、応答生成経路やVRChatマイク経路は変更しません。
