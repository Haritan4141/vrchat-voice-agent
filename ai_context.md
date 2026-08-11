# AI引き継ぎコンテキスト

最終更新: 2026-08-11

## 1. プロジェクトの目的

VRChat内でChatGPTデスクトップアプリのChatGPT Voice（GPT-Live）をAIキャラクターとして
会話させるプロジェクトです。

- OpenAI APIのRealtimeモデルは使用しません。
- ChatGPTデスクトップアプリのVoiceを使用します。
- Python側はChatGPTの会話内容を生成しません。
- Python側の責務は音声診断、自己ループ対策、LAN操作、VRChat OSC、アバター自動モーションです。
- AIの名前は将来変更される可能性があります。`Lazuli`などの名前をコードや表示へ固定しないでください。

## 2. 重要なパス

Pythonリポジトリ（メインPC）:

```text
C:\Users\Haritan\Documents\vrchat-voice-agent
```

GitHub:

```text
https://github.com/Haritan4141/vrchat-voice-agent
```

5700X PCで使用するリポジトリ:

```text
C:\Users\Atsuki\Documents\vrchat-voice-agent
```

Unityプロジェクト:

```text
C:\Users\Haritan\AppData\Local\VRChatCreatorCompanion\VRChatProjects\vrchat-voice-agent
```

UnityプロジェクトはGitリポジトリではありません。PythonリポジトリをpushしてもUnity側の変更は
5700X PCへ自動同期されません。

## 3. 実行PCと音声経路

VRChatとChatGPT Voiceを実際に動かすのは5700X PCです。操作はParsec経由です。

確定している音声経路:

```text
VRChat出力
  -> CABLE-A Input
  -> CABLE-A Output
  -> ChatGPT Voice入力

ChatGPT Voice出力
  -> CABLE-B Input
  -> CABLE-B Output
  -> VRChatマイク
```

VRChatでは「システムの既定値」を使用せず、次を明示指定します。

- VRChat出力: `CABLE-A Input`
- VRChatマイク: `CABLE-B Output`

Windowsの初回テスト用設定:

- 既定の録音入力: `CABLE-A Output`
- 既定の再生出力: `CABLE-B Input`

ChatGPTアプリの個別ルート:

- 入力: `CABLE-A Output`
- 出力: `CABLE-B Input`

Parsecのホスト自動ミュートは無効化済みです。

## 4. 絶対に起動しないもの

ChatGPT Voice使用中は、既存のローカルAIランタイムを起動しないでください。

```powershell
# 実行禁止
python -m vrchat_ai_tool run
```

このコマンドはfaster-whisper、Ollama、VOICEVOXによる別応答を開始し、ChatGPT Voiceと
二重応答になる可能性があります。

使用してよいもの:

- `python -m vrchat_ai_tool devices`
- `chatgpt-voice-doctor`
- `vrchat-voice-control`
- `chatgpt-ui-diagnostic`
- ユニットテストと読み取り専用診断

## 5. 既存のPython機能

### ChatGPT Voice Doctor

音声端点、Windows既定値、アプリ別ルート、システム音、Parsec設定、二重応答プロセスを確認します。

```powershell
run_chatgpt_voice_doctor.bat
```

### LAN操作サーバー

```powershell
launch_voice_control.bat
```

主な機能:

- メインPCのブラウザからVRChatマイクをミュート・解除
- 自己ループ監視の有効・無効と警報リセット
- アバター状態表示の変更
- 自動モーションの有効・無効
- アクセント・表情・発話状態の診断操作
- VRChatから返されたOSC実値の表示

操作トークンは`config/control-token.txt`へ作成されます。トークン内容をログ、文書、Gitへ
含めないでください。

### 自己ループ監視

CABLE-BのChatGPT音声が遅延してCABLE-Aへ戻っていないかを相関で監視します。
ループ確定時は警報状態にして、設定によりVRChatマイクを自動ミュートします。

### 自動モーション

CABLE-Bの音量から待機・発話・収束を判定し、OSCでアバターを動かします。

現在のパラメーター:

| パラメーター | 用途 |
|---|---|
| `VoiceAgentStatus` | 0 STOPPED / 1 ONLINE / 2 ERROR / 3 MAINTENANCE |
| `VoiceAgentMotionEnabled` | 自動モーションON/OFF |
| `VoiceAgentActivity` | 0 IDLE / 1 SPEAKING / 2 SETTLING |
| `VoiceAgentEnergy` | 発話強度 |
| `VoiceAgentGesture` | アクセント動作0～9 |
| `VoiceAgentExpression` | 通常を含む表情0～6 |

アクセント動作:

1. 大きく頷く
2. 左手を胸元へ
3. 右手をお腹へ
4. 両手を胸前へ
5. 髪の毛くるくる
6. 口に指
7. 前傾姿勢
8. かわいい待機2
9. うたた寝01

表情:

- 0 通常
- 1 Open
- 2 FingerPoint
- 3 Victory
- 4 Rock&Roll
- 5 Gun
- 6 ThumbsUp

詳しくは`docs/chatgpt-voice-control.md`と`docs/architecture.md`を参照してください。

## 6. Unity側の現在の構成

Unity 2022.3系のVRChatアバタープロジェクトです。

### ステータスヘイロー

- `Assets/StatusHalo_for_PC`のParallax版を採用
- `VoiceAgentStatus`でSTOPPED / ONLINE / ERROR / MAINTENANCEを切り替え
- Expressions Menuからも変更可能

### Lexa表示

元アセット:

```text
Assets/Finnit. Lexa - Text Choice Interface
```

生成・編集用:

```text
Assets/VoiceAgentLexa
```

現在の表示:

- `AI AGENT / AI試験中`
- 2行版 `AI試験中 / 会話できます`
- ON/OFFおよび1行・2行の切り替えが可能

現在のUnityパラメーター:

- `VoiceAgentAiTrialSign`
- `VoiceAgentAiTrialTwoLine`

今回検討している「考え中」はまだUnityへ追加していません。

### 自動モーション用アセット

```text
Assets/VoiceAgentMotion
```

Unity内ライブプレビューと、LAN GUIからのVRChat実機診断が実装されています。

## 7. 現在進行中の作業: ChatGPT画面状態診断

目的は、ChatGPT VoiceがWeb検索・長い作業などを行っている間、発話していない時間に
アバターへ「考え中…」と表示することです。

ChatGPTデスクトップアプリから外部へ思考状態を通知する公開APIは確認できなかったため、
Windows UI AutomationでChatGPT画面のアクセシビリティ要素を読み取る診断を実装しました。

### 今回追加した未コミットファイル

```text
vrchat_ai_tool/chatgpt_ui_diagnostic.py
tests/test_chatgpt_ui_diagnostic.py
docs/chatgpt-ui-diagnostic.md
run_chatgpt_ui_diagnostic.bat
```

変更した既存ファイル:

```text
pyproject.toml
vrchat_ai_tool/cli.py
```

追加したCLI:

```powershell
uv run chatgpt-ui-diagnostic --duration-seconds 180
```

ダブルクリック用:

```text
run_chatgpt_ui_diagnostic.bat
```

依存関係:

```text
pywinauto>=0.6.9,<1; sys_platform == 'win32'
```

### 診断の動作

- `ChatGPT.exe`の可視トップレベルウィンドウだけを対象にする
- UI Automationのプロパティを一括キャッシュして読み取る
- 初回は基準状態だけを保存し、その後の追加・変更・消失をJSONLへ記録する
- 「検索」「作業」「考え」「search」「working」などを含む要素を`CANDIDATE`として表示する
- ChatGPTへのクリック、キー入力、設定変更は行わない
- ログはGit管理対象外の`artifacts/`へ保存する
- 既定ログ名はマイクロ秒まで含め、同時起動時のファイル名衝突を避ける
- `--output`で既存ファイルを指定した場合は上書きせず終了する

重要: 最初に試した`pywinauto`の`window.descendants()`による全走査は、Chromium系UIで
40秒以上停止しました。この方式へ戻さないでください。現在はUI Automationの
`FindAllBuildCache`を使った一括キャッシュ方式です。

### 現在までの実測結果

メインPC上の現在のChatGPTアプリで、714個の可視アクセシビリティ要素を取得しました。
次の変化を実際に検出できています。

```text
15m 6s作業中 -> 15m 8s作業中
```

コンソールとJSONLの両方で`candidate: true`になりました。

1回のUIA走査には、このPCで約1～2秒かかります。`--interval-seconds 0.5`を指定しても、
実際の間隔は走査時間より短くなりません。検索や長い作業の検出用途には許容範囲です。

### テスト結果

- 全ユニットテスト: 52件成功
- 新規診断ファイルのRuffチェック: 成功
- CLIヘルプ: 成功
- 実機UI Automationスモークテスト: 成功

`artifacts/chatgpt-ui-diagnostic-smoke.jsonl`は、同じ明示パスを使った2セッションの書き込みが
重なって一部JSONLが壊れているため、状態判定の材料には使用しないでください。この確認を受け、
既定名の一意化と既存ログの上書き拒否を追加しました。5700X PCではバッチファイルが生成する
新しいタイムスタンプ付きログを使用します。

リポジトリ全体へ最新Ruffルールを適用すると既存ファイル由来の警告がありますが、今回の
新規ファイルには警告はありません。無関係な既存ファイルを一括整形しないでください。

## 8. 次にユーザーが行う動作確認

現時点では今回の診断実装をまだコミット・pushしていません。5700X PCで試すには、ユーザーから
push指示を受けたあと、意図したファイルだけをコミットしてpushし、5700X PCでpullします。

5700X PCでの手順:

1. ChatGPTデスクトップアプリを起動する
2. `run_chatgpt_ui_diagnostic.bat`を実行する
3. `Baseline captured`が表示されるまで待つ
4. 空のVoiceタスクから通常の短い質問をする
5. Web検索が必要な質問をする
6. 少し長い検索をする
7. 操作の間を5秒ほど空ける
8. `artifacts/chatgpt-ui-diagnostic-*.jsonl`を確認する

試す質問例:

```text
1足す1は？
今日の東京の天気をWebで調べて
最近のVRChatの更新内容をいくつか調べてまとめて
```

可能なら次も比較します。

- ChatGPTが前面
- ChatGPTが別ウィンドウの背面
- ChatGPTが最小化

最小化時に要素が取得できない場合は、次も試せます。

```powershell
uv run chatgpt-ui-diagnostic --duration-seconds 180 --include-offscreen
```

ログには画面上の会話本文が含まれる可能性があります。空の診断用タスクを使い、共有前に
ログ内容を確認してください。トークン、個人情報、非公開会話をGitへ入れないでください。

## 9. 動作確認後の実装予定

診断ログから安定した状態要素を特定できたら、次の実装へ進みます。

1. UIA要素を監視する常駐サービスをLAN操作サーバーへ統合
2. 新しいVRChat同期Bool `VoiceAgentThinking`を追加
3. `ChatGPTが作業中`かつ`CABLE-Bが無音`のときだけTrueにする
4. GUIへ検出状態と手動ON/OFFを追加
5. Lexaへ`考え中…`表示を追加
6. 他ユーザー視点で表示同期を確認

推奨する表示条件:

```text
VoiceAgentThinking表示 =
    ChatGPT画面が作業中
    AND ChatGPT音声が発話中ではない
```

ChatGPTが話している最中は「考え中」を隠し、発話終了後も画面上で作業中なら再表示します。

`VoiceAgentStatus=3 MAINTENANCE`は流用しません。これはサービス状態用であり、短時間の
思考・検索状態とは別です。

Lexa側では現在のAI試験中表示より`VoiceAgentThinking`を優先し、Falseへ戻ったら元の
1行・2行・OFF設定へ復帰させるのが安全です。

UIAが検出できない場合の予備手段:

- CABLE-A発話終了後、CABLE-Bが一定時間無音なら「考え中」と推定
- LAN GUIで手動切り替え

UIAによる明示状態を最優先し、音声推定と手動操作をフォールバックにします。

## 10. Gitの現在状態

ブランチ:

```text
main
```

現在のHEAD:

```text
ad4364d Confirm remote avatar OSC state
```

`origin/main`とHEADは一致しています。今回のUI診断変更は未コミットです。

意図した変更:

```text
M  pyproject.toml
M  vrchat_ai_tool/cli.py
?? docs/chatgpt-ui-diagnostic.md
?? run_chatgpt_ui_diagnostic.bat
?? tests/test_chatgpt_ui_diagnostic.py
?? vrchat_ai_tool/chatgpt_ui_diagnostic.py
?? ai_context.md
```

ユーザー所有の未追跡ファイルが別にあります。削除、ステージ、コミットしないでください。

```text
5700X_PC/
config/settings.toml
```

`config/chatgpt_voice.toml`、`config/control-token.txt`、`artifacts/`は`.gitignore`対象です。

## 11. よく使う確認コマンド

```powershell
git status --short --branch
uv sync
uv run python -m unittest discover -s tests
uv run chatgpt-ui-diagnostic --help
uv run chatgpt-ui-diagnostic --duration-seconds 180
```

新規診断コードだけのRuff確認:

```powershell
uv run --with ruff ruff check `
  vrchat_ai_tool/chatgpt_ui_diagnostic.py `
  tests/test_chatgpt_ui_diagnostic.py
```

## 12. 新しいセッションで最初にすること

1. この`ai_context.md`を読む
2. `git status --short --branch`で変更を再確認する
3. ユーザー所有の未追跡ファイルを触らない
4. 今回のUI診断が未コミットか、すでにpush済みかを確認する
5. ユーザーが5700X PCで取得したJSONLログを確認する
6. ログの安定したUIA要素が分かるまで`VoiceAgentThinking`の自動判定を決め打ちしない
7. Unityを直接変更する前に、ほかのセッションがUnityを操作中でないかユーザーへ確認する
