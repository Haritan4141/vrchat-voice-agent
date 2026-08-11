# GPT Liveへのキャラクター指示の適用

GPT Liveの会話人格は、音声タスクごとに`system_prompt.txt`を最初のメッセージとして送って設定します。`AGENTS.md`や別の事前設定は使用しません。送信内容はコンソールやログには表示しません。

`system_prompt.txt`は、pull後にそのまま使えるようリポジトリへ含めています。このリポジトリは公開されているため、APIキー、パスワード、住所などの秘密情報は書かないでください。

## 通常の使い方

1. `controls\apply_voice_prompt.bat`をダブルクリックします。
2. ツールがChatGPTの「新しいチャット」を開き、GPT Live開始ボタンを押します。
3. `Prompt sent`と表示され、GPT Liveが「準備できたよ」と答えれば適用完了です。
4. 「名前は？」と聞き、「ラズリだよ」と答えることを確認します。

ChatGPTデスクトップアプリは、あらかじめ起動して待機状態にしてください。初回のマイク許可やVoice設定画面が表示された場合は、その画面を手動で完了してから再実行してください。

`run_chatgpt_ui_diagnostic.bat`と`launch_voice_control.bat`は、この自動開始とプロンプト適用には不要です。前者はUI調査用、後者は本番用バッチから呼び出されるVRChat OSC・ミュート・考え中表示用の内部ランチャーです。

## 長時間会話と再適用

会話の先頭で送った文章が、長時間経過しても原文のまま無期限に保持される保証はありません。会話が長くなると、古い文脈が要約・圧縮されたり、応答が設定から少しずつずれたりする可能性があります。

名前や口調がずれ始めた場合は、同じ`controls\apply_voice_prompt.bat`をもう一度実行してください。現在の音声タスクに同じ指示を再送できます。会話を妨げるため、一定時間ごとの自動再送は行いません。

## 手動コマンドとトラブル対処

送信せず入力欄の検出だけ確認する場合:

```powershell
uv run chatgpt-voice-prompt --prompt-file system_prompt.txt --start-voice --dry-run
```

すでに手動で開始したGPT Liveへ適用する場合は、`--start-voice`を外してください。

Enterで送信されず、入力欄に文章が残る設定の場合:

```powershell
uv run chatgpt-voice-prompt --prompt-file system_prompt.txt --submit-key ctrl-enter
```

複数のChatGPTウィンドウや入力欄を検出した場合は、誤送信防止のため何も送らず終了します。余分なウィンドウを閉じてから再実行してください。
