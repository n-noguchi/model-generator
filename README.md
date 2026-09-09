# Person Image-to-3D MVP

人物専用の高品質4方向 Image-to-3D ツールです。正面・背面・人物本人基準の左・右画像から Hunyuan3D-2mv で形状を作り、4枚の元画像を全身テクスチャとして投影します。GPU（NVIDIA）が必須です。

![screenshot](screenshot.png)

## 起動

Docker Desktopを起動した状態で、プロジェクトのPowerShellから実行します。

```powershell
docker compose up --build
```

停止する場合は次を実行します。

実行中の生成ジョブがあるときは、完了または失敗を確認してから停止してください。途中で停止するとジョブが取り残され、再実行が必要になります。

```powershell
docker compose down
```

UIは次で開きます。

```text
http://localhost/
```

起動確認では、次の結果でGPUが認識されている必要があります。

```powershell
docker compose exec sf3d-worker /opt/hunyuan-venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`True` とGPU名が表示されない場合、高品質生成は開始できません。Docker DesktopをWSL 2バックエンドで起動し、NVIDIAドライバーとGPU連携を確認してください。

## モデルの事前取得

Hugging Faceで各モデルの利用条件に同意してから認証します。トークンを`.env`やリポジトリへ保存しないでください。

```powershell
hf auth login
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download-model.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-hunyuan.ps1
```

モデルはホストの `./models` に保存され、コンテナから読み取り専用で利用されます。

## 使い方

1. プロジェクトを作成または選択します。
2. 正面・背面・人物本人基準の左・右の4画像を指定します。
3. 背景除去方式、2048pxまたは4096pxのテクスチャ、形状密度を選びます。
4. 「4方向から高品質モデルを生成」を実行します。

左右は画面を見る側の左右ではなく、**人物本人から見た左右**です。4枚は同じ人物・ポーズ・画角・照明に揃えるほど品質が上がります。

## 出力と注意事項

生成結果には最終 `mesh.glb`、形状のみの `mesh-shape.glb`、`texture.png`、`hunyuan.log`、投影診断画像が含まれます。ゲーム利用前にはメッシュ整理、LOD、リグ、スキニングなどの後工程が必要です。

4方向画像間で顔、服、照明、輪郭が不一致の場合、投影境界が残ることがあります。プレビューと診断画像で確認してください。
