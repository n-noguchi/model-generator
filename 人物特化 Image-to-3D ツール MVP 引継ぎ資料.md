# 人物特化 Image-to-3D ツール MVP 引継ぎ資料

## 1. 目的

人物画像から、できるだけ手間をかけずに品質の良い3Dモデルを生成するローカルツールを作成する。

MVPでは機能を増やしすぎず、以下の体験を成立させることを最優先とする。

```text
人物画像を用意
    ↓
Web UIへ投入
    ↓
3Dモデルを複数候補生成
    ↓
ブラウザ上で比較
    ↓
気に入ったモデルを選択
    ↓
ゲーム向けに最低限の最適化
    ↓
GLB / FBXとして保存
```

「高機能な3D編集ツール」を作ることが目的ではない。

最重要なのは、

**人物の画像から、少ない操作で、見た目の良い3Dモデルを作れること。**

---

# 2. 対象

MVPは人物・人型キャラクターに特化する。

対象例:

- 実写人物
- イラスト調人物
- ゲームキャラクター
- アニメ調キャラクター
- ファンタジー系人型キャラクター

MVPでは以下は対象外とする。

- 動物
- モンスター
- 車
- 家具
- 建物
- 一般的な静的オブジェクト

---

# 3. 実行環境

想定PC:

```text
RAM: 48GB
GPU: NVIDIA RTX 3070
VRAM: 8GB
```

VRAM 8GBで安定して動作することを重要要件とする。

NVIDIA GPU必須。

---

# 4. 実行方式

バックエンドはDockerコンテナとして実装する。

ホストOSへPythonやAIモデルの依存関係を直接インストールしない。

基本構成:

```text
docker compose

├─ api
├─ web
├─ sf3d-worker
└─ blender-worker
```

必要なファイルはホスト側ディレクトリをBind Mountして保存する。

---

# 5. 技術構成

## Backend API

```text
Python
FastAPI
Pydantic v2
SQLAlchemy 2.x
SQLite
Alembic
uv
```

## Frontend

```text
React
TypeScript
Vite
shadcn/ui
Tailwind CSS
React Three Fiber
Three.js
TanStack Query
Zustand
```

Web UIは、

```text
http://localhost
```

からのみアクセス可能とする。

認証は不要。

---

# 6. MVPワークフロー

```text
人物画像
  ↓
入力チェック
  ↓
背景色をAlpha Maskへ変換
  ↓
Stable Fast 3D
  ↓
3候補生成
  ↓
Web 3D Viewer
  ↓
ユーザーが1候補選択
  ↓
Blender後処理
  ↓
GLB / FBX
```

---

# 7. 入力画像

MVPはSingle Imageのみ対応する。

複数画像入力は実装しない。

対応形式:

```text
PNG
JPEG
```

PNG推奨。

---

# 8. 背景処理

背景切り抜きそのものは本ツールでは行わない。

ユーザーが事前にNano Banana等を利用して背景を単色化する。

デフォルト背景色:

```text
#FF00FF
```

純マゼンタ。

ただし背景色は設定で変更可能にする。

例:

```text
人物
+
#FF00FF の単色背景
```

を入力する。

Backendでは背景色に近い画素を検出してAlpha Maskへ変換する。

JPEG圧縮等による色ずれを考慮し、完全一致ではなく色差に許容値を持たせる。

---

# 9. 元画像の任意登録

背景加工前の元画像も任意で登録可能にする。

```text
Original Image
+
Background Removed Image
```

の2枚がある場合は、Nano Banana等による背景加工で人物自体が大きく変化していないか簡易比較する。

大きな差分がある場合:

```text
警告
↓
生成停止
↓
ユーザー確認
```

とする。

MVPでは高度なAIによる差分解析は不要。

画像差分・輪郭差分など、実装可能な簡易チェックから始める。

---

# 10. 入力チェック

人物全体が写っていることを確認する。

最低限チェックするもの:

```text
画像サイズ
人物が画像内に存在する
人物が大きく画像端から切れていない
背景色が十分存在する
```

明らかに不適切な場合は生成前に警告する。

MVPでは入力姿勢をT-Pose等へ変換しない。

入力画像の人物姿勢をそのまま利用する。

---

# 11. Image-to-3D Backend

MVPでは

```text
Stable Fast 3D
```

のみ使用する。

複数のImage-to-3D Backendは実装しない。

ただしコード上ではBackendを交換可能にする。

例:

```python
class ImageTo3DBackend:
    def generate(...):
        ...
```

実装:

```text
StableFast3DBackend
```

とする。

Image-to-3D固有コードをPipeline本体へ直接埋め込まない。

---

# 12. 候補生成

1枚の入力画像からデフォルト3候補生成する。

```text
Candidate 1
Candidate 2
Candidate 3
```

MVPでは同じStable Fast 3Dを使用し、Seed等を変える。

GPU処理は並列実行しない。

```text
Candidate 1生成
↓
GPUメモリ解放
↓
Candidate 2生成
↓
GPUメモリ解放
↓
Candidate 3生成
```

とする。

---

# 13. 追加候補

3候補に満足できない場合、

```text
[候補を1つ追加]
```

ボタンで1候補ずつ追加生成できるようにする。

---

# 14. VRAM管理

RTX 3070 8GBで動作させるため、GPU処理を同時実行しない。

Worker処理終了時にモデル・Tensor等を解放する。

必要なら以下を利用する。

```text
FP16
CPU offload
低VRAM設定
```

ただし、MVPでは過度な最適化を先に行わない。

まずStable Fast 3Dが安定動作する構成を作る。

OOM発生時はエラー内容をUIに表示する。

---

# 15. Web Viewer

MVPに3D Viewerを組み込む。

React Three Fiberを使用する。

最低限必要な機能:

```text
回転
Zoom
Pan

Solid表示
Wireframe表示
MatCap表示

Texture ON/OFF

正面
側面
背面
```

3候補を比較できるようにする。

可能ならカメラ位置を同期する。

高度な3D編集機能は実装しない。

---

# 16. 入力画像との比較

Viewer上で、

```text
入力画像
生成3Dモデル
```

を並べて確認できるようにする。

加えて可能なら、

```text
入力画像
+
3Dレンダリング
```

を半透明で重ねて比較できるようにする。

目的はユーザー自身が

```text
顔
体型
シルエット
衣服
髪
```

などの再現性を判断できるようにすること。

---

# 17. 簡易自動評価

最終判断はユーザーが行う。

MVPでは補助情報として簡易スコアのみ計算する。

候補:

```text
Silhouette一致度
輪郭一致度
正面レンダリングと入力画像の簡易類似度
```

重い評価モデルは必須としない。

自動スコアが高いからといって自動採用しない。

---

# 18. 人間による評価

候補ごとに以下を保存可能にする。

```text
Geometry   1～5
Texture    1～5
Overall    1～5

Accept / Reject

Comment
```

Rigging / Animation評価はMVPでは不要。

候補の最終選択はユーザーが行う。

---

# 19. 候補選択

3候補の中から1つ選択する。

選択した候補だけ後続のBlender処理へ進める。

未選択候補は履歴として残す。

---

# 20. Blender Worker

BlenderはDockerコンテナ内でHeadless実行する。

例:

```text
blender --background --python process.py
```

MVPで行う処理:

```text
Mesh cleanup
法線修正
重複頂点修正
小さなMesh破損修正
不要な明確な浮遊パーツ削除
Decimate
UV確認/必要な処理
Normal Bake
AO Bake
PBR Material整理
スケール補正
接地
Export
```

複雑な自動RetopologyはMVPでは実装しない。

---

# 21. Polygon Reduction

ユーザーは目標Triangle数を選択可能にする。

プリセット:

```text
5,000
10,000
20,000
50,000
Custom
```

初期値:

```text
20,000
```

Blender Decimateを利用する。

Decimate前後で簡易的に形状差を確認し、極端に壊れた場合は警告する。

---

# 22. Texture

選択可能:

```text
1024
2048
4096
```

デフォルト:

```text
2048
```

MVPで扱うPBR:

```text
BaseColor
Normal
Roughness
Metallic
AO
```

Stable Fast 3Dの出力を基本とし、不足する情報はBlender側で可能な範囲だけ補完する。

AIによるテクスチャ再生成は行わない。

---

# 23. Material

人物では可能な場合、

```text
Skin
Hair
Clothes
Accessories
```

など意味のあるMaterialを維持する。

ただしMVPでは高度な自動パーツ分離を必須にしない。

Stable Fast 3Dの生成結果を大きく作り直す処理は行わない。

---

# 24. Scale

内部単位はメートル。

ユーザーが、

```text
Height
Width
Depth
```

のいずれか1つを指定できる。

指定値に合わせ、縦横比を維持してモデル全体をスケーリングする。

人物の場合は基本的にHeight指定を想定する。

---

# 25. LOD

MVPではユーザーがON/OFF可能。

ONの場合のデフォルト:

```text
LOD0 100%
LOD1 50%
LOD2 25%
LOD3 10%
```

高度な自動LOD評価は不要。

---

# 26. Collision

MVPでは簡易対応のみ。

ユーザーがON/OFF可能。

生成候補:

```text
Convex Hull
Simplified Collision Mesh
```

人物ではゲーム側でCharacter Controllerを使用するケースも多いため、Collision生成を必須にしない。

---

# 27. Export

対応形式:

```text
GLB
FBX
```

両方Blenderから生成する。

プリセット:

```text
Generic
Unity
Unreal Engine
```

内部座標系を固定し、Export時に必要な座標変換・スケール変換を行う。

出力フォルダ:

```text
export/
├─ model.glb
├─ model.fbx
├─ textures/
├─ preview/
└─ metadata.json
```

ZIP Exportも可能にする。

---

# 28. Export検証

書き出し後、

```text
GLB
FBX
```

をBlenderで再読み込みし、

最低限以下を確認する。

```text
Meshが存在する
Materialが存在する
Texture参照が壊れていない
Scaleが異常でない
ファイルが読み込める
```

問題があればExport成功扱いにしない。

---

# 29. プロジェクト管理

プロジェクト単位で保存する。

例:

```text
projects/
└─ character-name/
   ├─ input/
   ├─ runs/
   ├─ exports/
   ├─ previews/
   └─ project.json
```

保存先はユーザーがホスト側フォルダとして指定する。

---

# 30. Generation Run

生成を実行するたびにRunとして保存する。

```text
Run 1
Run 2
Run 3
```

過去Runは上書きしない。

各Runに保存:

```text
入力画像
Seed
Stable Fast 3D設定
品質設定
Candidate
評価
選択Candidate
生成日時
ログ
```

---

# 31. SQLite

SQLiteにはメタデータのみ保存する。

例:

```text
Project
Run
Candidate
Job
Evaluation
Settings
Artifact
```

GLB、画像、Texture等の大きなデータをSQLiteへ保存しない。

実ファイルはホスト側プロジェクトフォルダへ保存する。

SQLiteを正とする。

---

# 32. Job管理

独自SQLite Job Queueを使用する。

ただしWorkerがSQLiteを直接操作しない。

```text
Worker
 ↓ HTTP
FastAPI
 ↓
SQLite
```

とする。

WorkerはAPIをポーリングしてJobを取得する。

Worker共通操作:

```text
claim
start
progress
complete
fail
cancel
```

GPU Jobは同時に1件だけ実行する。

---

# 33. ファイル連携

画像や3DデータをHTTP API経由で転送しない。

各コンテナから同じBind Mountを参照する。

APIでは相対パスだけを渡す。

---

# 34. ログ

各工程のログを保存する。

最低限:

```text
開始時刻
終了時刻
処理時間
成功/失敗
エラー内容
```

UIからログを確認できるようにする。

詳細な監視ダッシュボード等はMVPでは不要。

---

# 35. GPUチェック

アプリ起動時に以下を確認する。

```text
NVIDIA GPU
CUDA
VRAM容量
DockerコンテナからGPUが利用可能か
SF3Dモデルが存在するか
```

問題があればWeb UIに表示する。

---

# 36. UI

作り込みすぎない。

MVPで必要なのは以下。

## Project画面

```text
新規プロジェクト
既存プロジェクト一覧
```

## Input画面

```text
人物画像
元画像（任意）
背景色
品質プリセット
[生成]
```

## Progress画面

```text
Candidate 1  Completed
Candidate 2  Running
Candidate 3  Waiting
```

## Comparison画面

```text
Input

Candidate 1
Candidate 2
Candidate 3

3D Viewer

簡易Score

人間評価

[この候補を使う]
```

## Optimize画面

```text
Target triangles
Texture size
LOD ON/OFF
Collision ON/OFF

[最適化]
```

## Export画面

```text
GLB
FBX

Generic
Unity
Unreal

[Export]
```

これ以上のUI機能はMVPでは原則追加しない。

---

# 37. 品質プリセット

MVPでは以下を用意する。

```text
Fast
Standard
High
Custom
```

具体的な値はStable Fast 3Dの実測を行って決定する。

RTX 3070 8GBでOOMしない設定を優先する。

初期デフォルト:

```text
Standard
```

---

# 38. 非対象

MVPでは以下を実装しない。

```text
Rigging
Skeleton
Animation
Text-to-Motion
Lip Sync
表情生成
BlendShape
自動Retopology
COLMAP
Multi-view Image-to-3D
動画入力
動物
静的オブジェクト
クラウド推論
外部API依存
高度なAI Router
複数Image-to-3D Backend
完全自動品質判定
高度な3D編集機能
```

仕様を膨らませないこと。

---

# 39. MVPのDone条件

以下がすべて成立した時点でMVP完成とする。

1. `docker compose` でアプリを起動できる
2. `localhost` からWeb UIへアクセスできる
3. 人物画像をアップロードできる
4. `#FF00FF` 背景をMask化できる
5. RTX 3070 8GB上でStable Fast 3Dが動く
6. 同一画像から3候補生成できる
7. Web Viewerで3候補を確認できる
8. 入力画像と3Dモデルを比較できる
9. ユーザーが候補を1つ選択できる
10. Blender WorkerでMesh Cleanup / Decimateできる
11. Texture / PBR Materialを維持できる
12. Triangle数を指定できる
13. GLBを出力できる
14. FBXを出力できる
15. Unity / Unreal向けExportプリセットが動く
16. Exportしたモデルを再読み込み検証できる
17. Run・Candidate・設定・評価を履歴保存できる
18. エラー時にログを確認できる

---

# 40. 実装方針

実装では次を優先する。

```text
動くこと
↓
安定すること
↓
人物の見た目の品質
↓
操作が簡単であること
```

コードの抽象化や将来拡張のためにMVPを複雑化しない。

ただし、

```text
ImageTo3DBackend
Worker
Artifact
```

など、明らかに交換可能にすべき境界だけはインターフェース化する。

新機能はMVP Done条件を満たすまで追加しない。