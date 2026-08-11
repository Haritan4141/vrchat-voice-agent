# ChatGPT画面状態診断

ChatGPTデスクトップアプリのWindows UI Automation（アクセシビリティ）要素を読み取り、
Voiceの通常回答・Web検索・長い作業で画面部品がどう変化するかを記録する診断です。

この診断は読み取り専用です。ChatGPTをクリックしたり、文字入力や設定変更を行ったりしません。

## 注意

ChatGPT画面に表示されている会話本文がログに含まれる可能性があります。
個人情報を含まない空の診断用チャットで実行してください。

ログはGit管理対象外の`artifacts/`へ保存されます。外部へ共有する前に内容を確認してください。

## 5700X PCでの実行

1. ChatGPTデスクトップアプリを起動します。
2. プロジェクト直下の`run_chatgpt_ui_diagnostic.bat`を実行します。バッチは背面・最小化されたChatGPTの要素も記録します。
3. `Baseline captured`と表示されるまで待ちます。
4. 空のChatGPTタスクでVoiceを開始します。
5. 次の順番で試します。各操作の間を5秒ほど空けてください。

   - 何もせず待機
   - 「1足す1は？」など、検索不要の短い質問
   - 「今日の東京の天気をWebで調べて」など、検索が必要な質問
   - 「最近のVRChatの更新内容をいくつか調べてまとめて」など、少し長い検索

6. 可能なら、ChatGPTを前面・別ウィンドウの背面・最小化の3状態でも試します。
7. 180秒で自動終了します。途中で終える場合は`Ctrl+C`を押します。

候補になりそうな状態要素は、コンソールで`[CANDIDATE]`と表示されます。

## 出力

`artifacts/chatgpt-ui-diagnostic-YYYYMMDD-HHMMSS-ffffff.jsonl`へ、次のイベントを記録します。
末尾の`ffffff`は同時起動時の衝突を避けるためのマイクロ秒です。`--output`で既存ファイルを
指定した場合は、以前の診断ログを壊さないよう上書きせず終了します。

- `session_start`: 診断条件
- `window_connected`: ChatGPTウィンドウの検出
- `added`: 新しく現れた画面要素
- `changed`: 名前や状態が変わった画面要素
- `removed`: 消えた画面要素
- `heartbeat`: 10秒ごとの接続状態
- `session_end`: 終了理由

`candidate: true`のイベントを中心に、通常回答とWeb検索で安定して差が出るか確認します。

## 5700X PCで確認できた判定信号

通常回答とWeb検索の両方で、`StatusBar`かつクラス名に`activityPillMaterial`を含む無名要素が作業中だけ安定して現れました。Web検索中はさらに「ウェブを検索中」のテキストが現れます。常駐監視では前者を必須信号、後者を検索種別の補助信号として使用します。

過去の会話に残る「3秒作業しました」や通常の検索ボタンは、ステータスバーがない限り作業中とは判定しません。

常駐監視では、VRChatが前面にある通常運用を想定して、背面・最小化されたChatGPTの要素も読み取ります。`config/chatgpt_voice.toml`の`[ui_monitor]`で`include_offscreen = false`にすると、画面外として報告された要素を除外できます。

## コマンドから実行する場合

```powershell
uv sync
uv run chatgpt-ui-diagnostic --duration-seconds 180
```

終了時間を設けず実行する場合は、`--duration-seconds 0`を指定します。

```powershell
uv run chatgpt-ui-diagnostic --duration-seconds 0
```

診断開始時点の全要素も保存したい場合は`--show-initial`を追加します。ログ量と会話本文の記録量が
大幅に増えるため、通常の検証では指定しないでください。

```powershell
uv run chatgpt-ui-diagnostic --duration-seconds 180 --show-initial
```
