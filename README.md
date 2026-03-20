# VRChat AI Tool

PC版VRChat(デスクトップモード)で、周囲の会話を聞き取り、ローカルLLMで返答を作り、TTSを仮想マイクへ流すためのローカルツールです。

現時点の実装は、`Windowsの音声ルーティング + Windows内蔵STT + Ollama + VOICEVOX + 仮想マイク` です。Whisper系STTより精度は落ちますが、追加導入なしでローカル完結するMVPとして先に動かせる形にしています。

## 目標

- 周囲の会話を聞いて短く自然に返事する
- 必要に応じて話題提供も行う
- ローカルLLMを使える
- VOICEVOXなどのローカルTTSを使える
- VRChat側のアバター制御は当面やらず、音声入出力に絞る

## 推奨アーキテクチャ

1. `VRChat出力` を専用の仮想オーディオデバイスへ送る
2. このツールがそのデバイスを録音入力として監視する
3. 発話区間をVADで切り出す
4. `faster-whisper` などで文字起こしする
5. `Ollama` などのローカルLLMに会話文脈を渡す
6. `VOICEVOX Engine` で音声合成する
7. 合成音声を別の仮想オーディオデバイスへ再生する
8. VRChatのマイク入力にその仮想デバイスを指定する

## なぜこの構成か

- Windowsの音声は基本的に「ウィンドウ単位」ではなく「デバイス/オーディオセッション単位」で扱う方が安定します。
- Microsoftは特定プロセスの音だけを拾うアプリケーションループバックも提供していますが、要件と実装難度を考えると、最初のMVPは仮想デバイス分離の方が速いです。
- `システム全体の音をそのまま拾う` と、Discordやブラウザなど他アプリの音まで混ざりやすくなります。

## 構成候補

- STT: 現実装は `System.Speech`、将来的に `faster-whisper`
- LLM: `Ollama` もしくは `llama.cpp` サーバ
- TTS: `VOICEVOX Engine`
- 音声ルーティング: `VB-CABLE` または `VoiceMeeter`
- 実装言語: Python 3.12

## MVPの範囲

- VRChatの音声だけを聞き取る
- 1ターンごとに短く返答する
- Bot発話中は一時的に聞き取りを止め、自己回り込みを避ける
- 会話履歴は短く保つ
- テキストログを残してデバッグしやすくする

## 将来追加しやすいもの

- 物理マイクとBot音声のミックス
- VRChat OSCを使ったミュート/トグル制御
- アバターパラメータ連携
- 話者分離
- NGワード/暴走防止ルール

## ディレクトリ構成

```text
.
|-- .github/workflows/ci.yml
|-- config/settings.example.toml
|-- docs/architecture.md
|-- vrchat_ai_tool/
|   |-- __init__.py
|   |-- __main__.py
|   |-- cli.py
|   `-- config.py
`-- tests/test_config.py
```

## すぐにやること

1. `config/settings.example.toml` を `config/settings.toml` にコピーする
2. `input_device` と `tts_output_device` を自分の環境に合わせる
3. `python -m vrchat_ai_tool devices` でデバイス名を確認する
4. `python -m vrchat_ai_tool doctor --config config/settings.toml --check-devices --check-services` を実行する
5. `python -m vrchat_ai_tool listen-once --config config/settings.toml` で聞き取り確認をする
6. `python -m vrchat_ai_tool speak --config config/settings.toml --text "こんにちは"` で発声確認をする
7. 問題なければ `python -m vrchat_ai_tool run --config config/settings.toml` を実行する

`input_device` と `tts_output_device` は同じVB-CABLE系ルートにしない方が安全です。片方をVRChat音声の受け取り用、もう片方をBot音声のマイク注入用に分けてください。

## 起動確認

```powershell
python -m vrchat_ai_tool doctor --config config/settings.example.toml
```

インストールなしでそのまま起動できます。

## Git / GitHub 運用の開始例

```powershell
git add .
git commit -m "Initial scaffold"
gh repo create VRChat-AI-Tool --private --source . --remote origin --push
```

`gh` が未導入なら、GitHubで空リポジトリを作ってから `git remote add origin ...` でも問題ありません。

## 実装フェーズで追加する予定の主要ライブラリ

- `faster-whisper`
- `sounddevice`

現段階のMVP本体は標準ライブラリとWindows標準APIで動作します。
