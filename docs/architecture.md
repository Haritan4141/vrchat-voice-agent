# Architecture

## 責務の分離

```text
ChatGPT Desktop Voice ── 会話生成・音声認識・音声出力
Windows + VB-CABLE   ── A/Bの音声経路
vrchat-voice-agent   ── 診断・監視・安全停止・OSC・LAN操作
VRChat Avatar        ── VoiceAgentStatusの表示
```

PythonサービスはChatGPTの画面操作や会話生成を行いません。旧ローカルSTT/LLM/TTSも起動しません。

## 制御経路

```text
メインPCのブラウザ
  → HTTP/TCP 18765（Bearerトークン）
  → サブPCのVoiceControlService
      ├─ UDP/9000 → VRChat /input/Voice
      ├─ UDP/9000 → /avatar/parameters/VoiceAgentStatus
      └─ UDP/9001 ← VRChat /avatar/parameters/MuteSelf
```

ミュートは`/input/Voice`を押したあと、VRChatから返る`MuteSelf`を確認します。現在値が不明な場合でも、確認結果が希望状態と逆ならもう一度だけ切り替えるため、単純なブラインドトグルにはなりません。

## 自己ループ検出

CABLE-AとCABLE-Bを同時に録音し、短時間ごとのRMS包絡を作ります。最近のCABLE-Aと、100〜5000ms前のCABLE-Bを正規化相関で比較します。閾値を連続して超えたときだけ警報をラッチします。

警報時は`VoiceAgentStatus=2`にします。`auto_mute=true`の場合はVRChatマイクもミュートします。誤って会話が再開しないよう、自動ミュート解除は行いません。

この方式は音声内容を保存せず、音量の時間変化だけをメモリ上で比較します。
