# ローカル Text-to-Motion

STEP 9「カスタムモーション（ローカルAI）」で、短い英語の指示から動作を生成します。
生成ボタン・進捗・自動プレビュー・ダウンロードを同じSTEPに配置しています。
リグ付きモデル、またはUnity LOD0を先に作成してください。Unity LOD0があれば優先して使います。

## 初回準備

Docker DesktopのNVIDIA GPU連携が必要です。MDMの50-step HumanML3D encoderを使用します。
この環境のRTX 3070 8GBで推論・リターゲット・GLB/FBX再読込を確認しています。
初回のビルド・モデル取得にはインターネット接続が必要です。

```powershell
docker compose build motion-worker
docker compose run --rm --no-deps motion-worker python /worker/prepare_models.py
docker compose up -d --build
```

モデルは `models/mdm` に保存します。チェックポイント・統計値・CLIPは固定SHA256を検証し、
準備中断後も同じコマンドで不足・破損ファイルを修復できます。モデルはGit管理しません。
通常の推論は保存済みファイルだけを使い、文章や人物モデルを外部へ送信しません。
ローカルAPIとの通信は必要ですが、推論そのものは外部ネットワーク不要です。

## 指示の例

- `A person walks forward.`
- `A person raises both arms.`
- `A person waves with the right hand.`
- `A person sits down.`（椅子は生成されません）

長さは2〜8秒、20fpsです。シードを変えると別の動作になります。
MDM学習時の制限に合わせ、指示はCLIPの20トークン以内の短い英語にしてください。
長すぎる文章は切り捨てずエラーにします。日本語の自動翻訳は未実装です。
非ループ動作を想定してプレビューのループは初期OFFです。
プレビューは水平方向の移動に追従してモデルを画面に収めます。出力ファイルには移動を保持します。

## 出力・復旧

元のリグ・メッシュは変更せず、ジョブごとの別ディレクトリに以下を作成します。

- `custom-motion.glb` / `custom-motion.fbx`: 現在の17ボーンAutoRigへ適用したモーション
- `joints.npz`: MDMの22関節座標（HumanML3D Y-up、pickle不使用）
- `manifest.json`: 文章・秒数・シード・モデル版・入力成果物ID/SHA256・検証結果

入力リグは受付時に固定します。生成中に新しいリグを作っても入力がすり替わりません。
再生成中は直前に成功したモーションを表示します。各出力はダウンロードできます。

3D生成とText-to-MotionはGPUを共有するため順番に処理します。
中止要求後は子プロセスの停止確認までGPUを予約したままにします。
API一時停止時は終了通知を保持して再送し、ワーカー再起動時は中断ジョブを失敗または中止に確定します。
中断されたジョブはUIから再生成してください。時間だけを根拠に実行中のGPU予約を解除しません。

## 品質・利用条件

初期実装は全身の関節動作です。手指、物体との接触、足接地IK、完全なループ生成は未対応です。
複雑な複合指示や学習データに少ない動作は指示どおりにならない場合があります。
固定骨長へ適用するため体格によって足滑り・接触ずれが残り、ゲーム使用前の確認・調整が必要です。

使用コード: [MDM公式リポジトリ](https://github.com/GuyTevet/motion-diffusion-model)、
コミット `ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778`。
HumanML3Dの関節出力だけを使用するため、未使用のSMPL変換を遅延ロードに変更しています。
SMPLモデルの取得・同梱は行いません。
MDMコードのMITライセンスだけで学習データや重みの商用利用を保証することはできません。
[HumanML3D](https://github.com/EricGuo5513/HumanML3D) とその元データを含め、利用条件は別途確認してください。

## テスト

APIの入力固定・GPU排他・中止・登録の冪等性は `api/tests/test_text_motion.py`、
子プロセス停止・API復旧・中断ジョブ回収は `sf3d-worker/tests/test_gpu_runtime.py`、
モデル準備の再実行は `motion-worker/tests/test_prepare_models.py` で検証します。
実モデルでは生成したGLBとFBXをBlenderへ再読込し、骨・スキン・アニメーション・有限座標を検証します。

### この環境での検証結果（2026-09-12）

- API 10件、ワーカー・モデル準備35件が成功（Hunyuan venvでxatlasを含めスキップなし）。
- TypeScript型チェック、Web本番ビルド、差分の空白チェックが成功。
- character10で4秒・80フレームを生成し、GLB/FBX双方を再読込。MDM推論約5.8秒、PyTorchのGPU割当ピーク670MiB（プロセス全体のVRAM量や全処理時間ではありません）。
- 外部ネットワークを無効にしたコンテナでも同じモデルで生成成功。
- 実機の中止要求が `running → cancelling → cancelled` となり、その後の再生成も成功。
- 低い初期姿勢を実リグへ適用してGLB/FBXを出力。床基準の期待骨盤高 -0.309603079 と再読込後の実測 -0.309603095 が一致。
- ブラウザで自動読み込み・再生・全身の追従表示と、生成指示の表示を目視確認。

API一時停止・応答喪失・子孫プロセスの停止・中断ジョブ回収は自動テストで検証しており、PCの強制電源断は試していません。
低姿勢の統合チェックは `motion-worker/tests/retarget_low_pose.py`、座標変換の数値チェックは `motion-worker/tests/retarget_checks.py` です。
