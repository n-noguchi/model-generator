# Person Image-to-3D MVP

ローカル限定の人物特化 Image-to-3D ワークフローです。`docker compose up --build` で API/UI/mock生成/Blenderワーカーをまとめて起動します。UI は `http://localhost` にだけ公開されます。

開発時は既定の `BACKEND_MODE=mock` で、GPUなしでもアップロード→3候補→選択→ジョブの流れを確認できます。mock はブラウザとBlenderで読める有効なGLB（三角形）を生成するだけで、品質を偽装しません。Blenderの実行環境がある場合は optimize/export も動きます。

実生成では `BACKEND_MODE=sf3d` とします。ワーカーイメージには公式 `Stability-AI/stable-fast-3d` のコードと依存関係がbuild時に導入されるため、ホストのソースcheckoutは不要です。利用許諾済み・gated なHugging Face重みだけを事前にキャッシュし、`HF_TOKEN` はDockerfile、compose、リポジトリへ保存しません。8GB VRAM向けに公式CLIの `--texture-resolution 512 --batch_size 1 --remesh_option none` を使用します。出力GLBを正規形式とします。SF3Dは入力seed指定を公式CLIで提供しないため、候補の多様性は実モデル導入時に前処理差分等を追加検討してください。

## Windows: SF3D の事前取得とオフライン実行

SF3Dの公式コードとHugging Faceのモデル重みは別物です。公式コードと依存関係はワーカーイメージにbuild時導入されます。モデルはホスト側のプロジェクト配下 `./models/huggingface` にキャッシュし、コンテナは同じ場所をread-onlyで参照します。このキャッシュは `.gitignore` 対象です。

1. Hugging Faceで [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d) のアクセスを申請・承認します（gated modelです）。
2. PowerShellで `hf auth login` を実行し、read権限トークンをホストのHugging Face認証情報へ保存します。トークンを `.env`、compose、リポジトリへ書かないでください。
3. `powershell -ExecutionPolicy Bypass -File .\scripts\download-model.ps1` を実行します。これは `stabilityai/stable-fast-3d` に加え、SF3Dが内部で使う `facebook/dinov2-large` と `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` を `./models/huggingface/hub` へ、人物背景除去用の `u2net_human_seg.onnx` を `./models/rembg` へ取得します。
4. `.env` に `BACKEND_MODE=sf3d` を設定し、`docker compose -f docker-compose.yml -f docker-compose.sf3d.yml up --build` を実行します。追加composeファイルは `gpus: all` を有効にします。通常の `docker compose up --build` はGPU不要のmockモード用です。

composeは `HF_HOME=/models/huggingface`、`HUGGINGFACE_HUB_CACHE=/models/huggingface/hub`、`HF_HUB_OFFLINE=1` をsf3d-workerへ設定します。そのためコンテナ実行時にモデルダウンロードやトークンは不要です。キャッシュmountはTransformers 4.42.3の初回キャッシュ移行が書込みを行うためread-writeです。`HF_HUB_OFFLINE=1` とトークン非注入は維持されるため、コンテナから外部通信や秘密情報の使用は行いません。キャッシュがない場合はジョブに診断エラーを記録します。

## 人物の自動背景除去

入力は背景が均一でない人物画像をそのまま使えます。既定の「自動AI（人物）」は、実用的な透明度を持つPNGはその透明度を維持し、それ以外は `rembg` の `u2net_human_seg` で人物を切り抜きます。モデルは事前取得済みの `./models/rembg/u2net_human_seg.onnx` だけを使用し、実行中にネットワークから取得しません。単色背景画像には従来どおり「単色背景」を選んで背景色・許容差を指定できます。生成候補のプレビューはチェッカー背景で透過状態を確認できます。

`rembg` 本体はMITライセンスですが、モデル重みには別のライセンス・利用条件が適用される場合があります。配布・商用利用の前に、[rembgのモデル情報](https://github.com/danielgatis/rembg#models) と取得元の重みのライセンスを確認してください。

SQLiteはメタデータのみを保持し、大きなファイルは `./data` bind mount に保存します。コンテナ間に渡すのは相対パスだけで、APIはstorage外パスを拒否します。実モデルの導入、NVIDIA Docker GPU、Blenderコンテナのイメージ可用性はホスト依存です。

Blender workerは公式 Blender 4.2.0 Linux tarballをUbuntuベースのコンテナへ展開してheadless実行します。公式tarballに含まれるライセンスファイルは `/opt/blender` に保持されます。

テスト: `cd api; pip install -r requirements.txt; pytest -q`
