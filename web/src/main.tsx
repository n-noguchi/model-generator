import React, { Component, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { clone as cloneSkeleton } from "three/examples/jsm/utils/SkeletonUtils.js";
import "./style.css";
const api = "/api";
const call = async (p: string, o?: RequestInit) => {
  const r = await fetch(api + p, o),
    x = await r.json().catch(() => ({ detail: "応答を読み取れません" }));
  if (!r.ok) throw Error(x.detail || "request failed");
  return x;
};
type Project = { id: string; name: string };
type Candidate = {
  id: string;
  number: number;
  status: string;
  preview_path?: string;
  model_path?: string;
  score?: number;
};
type Artifact = {
  id: string;
  kind: string;
  path: string;
  candidate_id: string;
};
type Job = {
  id: string;
  candidate_id?: string;
  kind: string;
  status: string;
  progress: number;
  log: string;
};
type Row = { id: string; status: string; created_at: string; warning?: string };
type Run = Row & {
  input_path: string;
  back_path?: string;
  left_path?: string;
  right_path?: string;
  bake_resolution: 512 | 1024 | 2048 | 4096;
  generation_mode?: "sf3d" | "multiview";
  inference_steps?: number;
  octree_resolution?: number;
  selected_candidate_id?: string;
};
type Detail = {
  run: Run;
  candidates: Candidate[];
  jobs: Job[];
  artifacts: Artifact[];
};
type ViewKey = "front" | "back" | "left" | "right";
type Files = Partial<Record<ViewKey, File>>;
const views: {
  key: ViewKey;
  title: string;
  help: string;
  required?: boolean;
}[] = [
  {
    key: "front",
    title: "正面",
    help: "人物がカメラを正面から見ている画像。",
    required: true,
  },
  {
    key: "back",
    title: "背面",
    help: "人物の背中側。",
  },
  {
    key: "left",
    title: "人物から見た左側面",
    help: "人物本人の左側です。画面を見る人の左ではありません。",
  },
  {
    key: "right",
    title: "人物から見た右側面",
    help: "人物本人の右側です。画面を見る人の右ではありません。",
  },
];
class ViewerBoundary extends Component<
  { children: React.ReactNode },
  { error?: Error }
> {
  state: { error?: Error } = {};
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    return this.state.error ? (
      <p className="error">
        3Dプレビューを読み込めません: {this.state.error.message}
      </p>
    ) : (
      this.props.children
    );
  }
}
function Model({
  path,
  mode,
  texture,
  showSkeleton = false,
  pose = "rest",
}: {
  path: string;
  mode: string;
  texture: boolean;
  showSkeleton?: boolean;
  pose?: string;
}) {
  const gltf = useGLTF(api + "/files/" + path);
  // SkinnedMesh needs SkeletonUtils: Object3D.clone leaves its bone references
  // pointing at the source scene, so a helper could move without deforming mesh.
  const scene = useMemo(() => cloneSkeleton(gltf.scene), [gltf.scene]);
  const skeleton = useMemo(() => showSkeleton ? new THREE.SkeletonHelper(scene) : undefined, [scene, showSkeleton]);
  useEffect(() => {
    const bones: Record<string, THREE.Bone> = {};
    scene.traverse((o: any) => {
      if (!o.isBone) return;
      bones[o.name] = o;
      if (!o.userData.restQuaternion) o.userData.restQuaternion = o.quaternion.clone();
      o.quaternion.copy(o.userData.restQuaternion);
    });
    const rotate = (name: string, axis: THREE.Vector3, degrees: number) => {
      const bone = bones[name];
      if (bone) bone.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(degrees)));
    };
    if (pose === "arms") {
      rotate("UpperArm.L", new THREE.Vector3(0, 0, 1), -55); rotate("UpperArm.R", new THREE.Vector3(0, 0, 1), 55);
      rotate("LowerArm.L", new THREE.Vector3(0, 1, 0), 70); rotate("LowerArm.R", new THREE.Vector3(0, 1, 0), -70);
    } else if (pose === "legs") {
      rotate("UpperLeg.L", new THREE.Vector3(1, 0, 0), 35); rotate("UpperLeg.R", new THREE.Vector3(1, 0, 0), -20);
      rotate("LowerLeg.L", new THREE.Vector3(1, 0, 0), -65); rotate("LowerLeg.R", new THREE.Vector3(1, 0, 0), -25);
    }
    scene.updateMatrixWorld(true);
  }, [scene, pose]);
  useEffect(() => {
    scene.traverse((o: any) => {
      if (!o.isMesh) return;
      // Keep an untouched material so restoring Texture after MatCap or
      // Texture OFF reinstates the GLB's original map.
      const original = o.userData.originalMaterial || o.material.clone();
      o.userData.originalMaterial = original;
      if (mode === "matcap")
        o.material = new THREE.MeshMatcapMaterial({ color: "#c98ca8" });
      else {
        const m = original.clone();
        m.wireframe = mode === "wire";
        if (!texture) m.map = null;
        m.needsUpdate = true;
        o.material = m;
      }
    });
  }, [scene, mode, texture]);
  return <><primitive object={scene} />{skeleton && <primitive object={skeleton} />}</>;
}
function Camera({ preset, multiview }: { preset: string; multiview: boolean }) {
  const { camera } = useThree();
  useEffect(() => {
    const p: [number, number, number] =
      preset === "front"
        ? [0, 1, multiview ? 3 : -3]
        : preset === "side"
          ? [3, 1, 0]
          : [0, 1, multiview ? -3 : 3];
    camera.position.set(...p);
    camera.lookAt(0, 0.5, 0);
    camera.updateProjectionMatrix();
  }, [camera, preset, multiview]);
  return null;
}
function Viewer({ candidate, multiview = false, modelPath, rigPreview = false }: { candidate?: Candidate; multiview?: boolean; modelPath?: string; rigPreview?: boolean }) {
  const [mode, setMode] = useState("solid"),
    [texture, setTexture] = useState(true),
    [preset, setPreset] = useState("front"),
    [pose, setPose] = useState("rest");
  const path = modelPath || candidate?.model_path;
  useEffect(() => setPose("rest"), [path]);
  if (!path)
    return (
      <div className="viewer empty">
        完了した候補を選択すると3Dプレビューを表示します。
      </div>
    );
  return (
    <ViewerBoundary>
      <div className="viewer-toolbar">
        <label>
          表示{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="solid">Solid</option>
            <option value="wire">Wireframe</option>
            <option value="matcap">MatCap</option>
          </select>
        </label>
        {rigPreview && <span className="toolbar-note">緑の線: スキニングされた骨格</span>}
        {rigPreview && <label>ポーズ確認 <select value={pose} onChange={(e) => setPose(e.target.value)}><option value="rest">レストポーズ</option><option value="arms">肩・肘テスト</option><option value="legs">股関節・膝テスト</option></select></label>}
        <label>
          <input
            type="checkbox"
            checked={texture}
            onChange={(e) => setTexture(e.target.checked)}
          />{" "}
          Texture
        </label>
        <span className="toolbar-note">
          3D表示方向（入力指定ではありません）
        </span>
        {["front", "side", "back"].map((x) => (
          <button
            className={preset === x ? "on" : ""}
            key={x}
            onClick={() => setPreset(x)}
          >
            {x === "front" ? "正面" : x === "side" ? "側面" : "背面"}
          </button>
        ))}
      </div>
      <div className="viewer">
        <Canvas>
          <ambientLight intensity={1.5} />
          <directionalLight position={[3, 4, 3]} />
          <Camera preset={preset} multiview={multiview} />
          <React.Suspense fallback={<></>}>
            <Model path={path} mode={mode} texture={texture} showSkeleton={rigPreview} pose={pose} />
          </React.Suspense>
          <OrbitControls />
        </Canvas>
      </div>
    </ViewerBoundary>
  );
}
function FileCard({
  view,
  file,
  onChange,
  onRemove,
  stored,
  readonly = false,
}: {
  view: (typeof views)[number];
  file?: File;
  onChange: (f?: File) => void;
  onRemove: () => void;
  stored?: string;
  readonly?: boolean;
}) {
  const objectUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : undefined),
    [file],
  );
  useEffect(
    () => () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    },
    [objectUrl],
  );
  const url = objectUrl ?? (stored ? api + "/files/" + stored : undefined);
  return (
    <article
      className={
        "file-card " + (view.required ? "generation-input" : "reference-input")
      }
    >
      <h3>
        {view.title}
        {view.required ? (
          <span className="badge primary">必須・生成に使用</span>
        ) : (
          <span className="badge">任意・参照のみ</span>
        )}
      </h3>
      <p>{view.help}</p>
      {url ? (
        <>
          <img
            className="view-thumb"
            src={url}
            alt={`${view.title}の画像プレビュー`}
          />
          <p className="filename">{file?.name || stored?.split("/").pop()}</p>
          {!readonly && (
            <div>
              <label className="button secondary">
                差し替え
                <input
                  aria-label={`${view.title}画像を差し替え`}
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(e) => onChange(e.target.files?.[0])}
                />
              </label>
              <button className="quiet" onClick={onRemove}>
                解除
              </button>
            </div>
          )}
        </>
      ) : (
        !readonly && (
          <label className="drop-target">
            画像を選択
            <input
              aria-label={`${view.title}画像を選択`}
              type="file"
              accept="image/png,image/jpeg"
              onChange={(e) => onChange(e.target.files?.[0])}
            />
            <span>PNG または JPEG</span>
          </label>
        )
      )}
      {readonly && !url && <p className="muted">登録なし</p>}
    </article>
  );
}
function App() {
  const [projects, setProjects] = useState<Project[]>([]),
    [project, setProject] = useState<Project>(),
    [name, setName] = useState("character"),
    [files, setFiles] = useState<Files>({}),
    [cutoutMode, setCutoutMode] = useState<"auto" | "color">("auto"),
    [background, setBackground] = useState("#FF00FF"),
    [tolerance, setTolerance] = useState(48),
    [bakeResolution, setBakeResolution] = useState<512 | 1024 | 2048 | 4096>(2048),
    [octreeResolution, setOctreeResolution] = useState<256 | 384>(384),
    [runs, setRuns] = useState<Row[]>([]),
    [run, setRun] = useState<Detail>(),
    [selected, setSelected] = useState<Candidate>(),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [tri, setTri] = useState(20000),
    [custom, setCustom] = useState("20000"),
    [tex, setTex] = useState(2048),
    [lod, setLod] = useState(false),
    [collision, setCollision] = useState(false),
    [height, setHeight] = useState("");
  const [rigPreviewPath, setRigPreviewPath] = useState<string>();
  useEffect(() => setRigPreviewPath(undefined), [selected?.id]);
  const fail = (e: unknown) =>
    setError(e instanceof Error ? e.message : "操作に失敗しました");
  const setFile = (key: ViewKey, file?: File) =>
    setFiles((x) => ({ ...x, [key]: file }));
  const loadRun = async (id: string) => {
    try {
      setLoading(true);
      const d = (await call("/runs/" + id)) as Detail;
      setRun(d);
      setSelected(
        d.candidates.find((c) => c.id === d.run.selected_candidate_id) ||
          d.candidates.find((c) => c.status === "completed"),
      );
      localStorage.setItem("modelgenerator.run", id);
    } catch (e) {
      fail(e);
    } finally {
      setLoading(false);
    }
  };
  const loadRuns = async (id: string) => {
    try {
      const rs = (await call("/projects/" + id + "/runs")) as Row[];
      setRuns(rs);
      const saved = localStorage.getItem("modelgenerator.run");
      const target = rs.find((x) => x.id === saved)?.id || rs[0]?.id;
      if (target) await loadRun(target);
      else {
        setRun(undefined);
        setSelected(undefined);
      }
    } catch (e) {
      fail(e);
    }
  };
  useEffect(() => {
    void (async () => {
      try {
        const ps = (await call("/projects")) as Project[];
        setProjects(ps);
        const p = ps.find(
          (x) => x.id === localStorage.getItem("modelgenerator.project"),
        );
        if (p) setProject(p);
      } catch (e) {
        fail(e);
      }
    })();
  }, []);
  useEffect(() => {
    if (project) {
      localStorage.setItem("modelgenerator.project", project.id);
      void loadRuns(project.id);
    }
  }, [project?.id]);
  useEffect(() => {
    if (!run) return;
    const t = window.setInterval(() => void loadRun(run.run.id), 1500);
    return () => clearInterval(t);
  }, [run?.run.id]);
  const upload = async (f: File) => {
    const d = new FormData();
    d.append("image", f);
    return (await call("/uploads", { method: "POST", body: d })).path;
  };
  const create = async () => {
    try {
      const p = await call("/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      setProjects((x) => [p, ...x]);
      setProject(p);
    } catch (e) {
      fail(e);
    }
  };
  const generate = async () => {
    if (!project || !files.front || !files.back || !files.left || !files.right) return;
    try {
      setLoading(true);
      const uploaded: Partial<Record<ViewKey, string>> = {};
      for (const key of ["front", "back", "left", "right"] as ViewKey[])
        if (files[key]) uploaded[key] = await upload(files[key]!);
      const body = {
        project_id: project.id,
        input_path: uploaded.front!,
        back_path: uploaded.back,
        left_path: uploaded.left,
        right_path: uploaded.right,
        cutout_mode: cutoutMode,
        background,
        tolerance,
        bake_resolution: bakeResolution,
        generation_mode: "multiview",
        inference_steps: 50,
        octree_resolution: octreeResolution,
        confirm_warning: false,
      };
      let d = await call("/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (d.requires_confirmation) {
        if (!window.confirm(d.warning + " 続行しますか？")) return;
        d = await call("/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...body, confirm_warning: true }),
        });
      }
      await loadRuns(project.id);
      await loadRun(d.run.id);
    } catch (e) {
      fail(e);
    } finally {
      setLoading(false);
    }
  };
  const action = async (p: string, o: RequestInit) => {
    try {
      await call(p, o);
      if (run) await loadRun(run.run.id);
    } catch (e) {
      fail(e);
    }
  };
  const candidates = run?.candidates || [],
    artifacts = run?.artifacts || [];
  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">PERSON IMAGE-TO-3D</p>
          <h1>人物モデル生成</h1>
          <p>
            4方向画像から全身の形状を推定し、元画像を全体へ投影します。
          </p>
        </div>
        <span className="status">
          <i /> API接続中
        </span>
      </header>
      {error && (
        <p className="error" role="alert">
          {error}
          <button onClick={() => setError("")}>閉じる</button>
        </p>
      )}
      <section>
        <div className="section-title">
          <div>
            <p className="step">STEP 1</p>
            <h2>プロジェクトと入力画像</h2>
          </div>
        </div>
        <div className="project-controls">
          <label>
            プロジェクト名
            <input
              aria-label="新規プロジェクト名"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <button
            onClick={() => void create()}
            disabled={!name.trim() || loading}
          >
            新規プロジェクト
          </button>
          <label>
            既存プロジェクト
            <select
              aria-label="既存プロジェクトを選択"
              value={project?.id || ""}
              onChange={(e) =>
                setProject(projects.find((p) => p.id === e.target.value))
              }
            >
              <option value="">選択してください</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="input-notice">
          <strong>方向の基準:</strong> 左右は画面を見る側ではなく、
          <strong>人物本人から見た左右</strong>です。
        </div>
        <div className="generation-mode">
          <strong>全身高品質（4方向）</strong>
          <span>正面・背面・人物本人基準の左・右の4枚すべてを、形状と全身テクスチャに使用します。</span>
        </div>
        <div className="input-grid">
          {views.map((view) => (
            <FileCard
              key={view.key}
              view={{...view, required: true, help: view.help + " 高品質生成に使用します。"}}
              file={files[view.key]}
              onChange={(file) => setFile(view.key, file)}
              onRemove={() => setFile(view.key)}
            />
          ))}
        </div>
        <div className="cutout-controls">
          <label>
            切り抜き方式{" "}
            <select
              value={cutoutMode}
              onChange={(e) =>
                setCutoutMode(e.target.value as "auto" | "color")
              }
            >
              <option value="auto">自動AI（人物）</option>
              <option value="color">単色背景</option>
            </select>
          </label>
          {cutoutMode === "color" && (
            <>
              <label>
                背景色{" "}
                <input
                  type="color"
                  value={background}
                  onChange={(e) => setBackground(e.target.value.toUpperCase())}
                />
                <code>{background}</code>
              </label>
              <label>
                許容差{" "}
                <input
                  type="range"
                  min="0"
                  max="441"
                  value={tolerance}
                  onChange={(e) => setTolerance(Number(e.target.value))}
                />
                {tolerance}
              </label>
            </>
          )}
        </div>
        <div className="generate-row">
          <div>
            <strong>全身高品質・4方向再構成</strong>
            <p>前後左右を形状生成に使い、元の4画像を全身テクスチャへ投影します。生成は1候補ずつ実行します。</p>
            <label>
              生成テクスチャ解像度{" "}
              <select
                value={bakeResolution}
                onChange={(e) =>
                  setBakeResolution(Number(e.target.value) as 512 | 1024 | 2048 | 4096)
                }
              >
                <><option value={2048}>2048px（推奨）</option><option value={4096}>4096px（最高精細・時間とメモリを使用）</option></>
              </select>
            </label>
            <label>形状密度 <select value={octreeResolution} onChange={(e) => setOctreeResolution(Number(e.target.value) as 256 | 384)}><option value={256}>256（省メモリ）</option><option value={384}>384（高品質・推奨）</option></select></label>
            <p>顔を含む品質は入力画像・視点整合性の影響を受けます。ゲーム用の整理、リグ、スキニングは後工程です。</p>
          </div>
          <button
            className="generate"
            disabled={!project || !files.front || !files.back || !files.left || !files.right || loading}
            onClick={() => void generate()}
          >
            {loading ? "準備中…" : "4方向から高品質モデルを生成"}
          </button>
        </div>
      </section>
      {project && (
        <section>
          <div className="section-title">
            <div>
              <p className="step">STEP 2</p>
              <h2>Run履歴</h2>
            </div>
          </div>
          {runs.length ? (
            <div className="run-list">
              {runs.map((x) => (
                <button
                  className={run?.run.id === x.id ? "on" : ""}
                  key={x.id}
                  onClick={() => void loadRun(x.id)}
                >
                  <span>{new Date(x.created_at).toLocaleString()}</span>
                  <b>{x.status}</b>
                </button>
              ))}
            </div>
          ) : (
            <p className="muted">まだRunはありません。</p>
          )}
        </section>
      )}
      {loading && <p className="loading">読み込み中…</p>}
      {run && (
        <>
          <section>
            <div className="section-title">
              <div>
                <p className="step">STEP 3</p>
                <h2>Runの入力と進捗</h2>
              </div>
              <span className={"badge state-" + run.run.status}>
                {run.run.status}
              </span>
            </div>
            {run.run.warning && <p className="warning">{run.run.warning}</p>}
            <p className="muted">
              {run.run.generation_mode === "multiview" ? `全身高品質（4方向・${run.run.inference_steps ?? 50}ステップ・形状密度 ${run.run.octree_resolution ?? 384}）` : "旧形式: 高速 SF3D（正面1枚）"}
              {" / テクスチャ解像度: "}{run.run.bake_resolution ?? 512}px
            </p>
            <div className="stored-grid">
              {views.map((v) => (
                <FileCard
                  key={v.key}
                  view={v}
                  stored={
                    v.key === "front"
                      ? run.run.input_path
                      : (run.run[`${v.key}_path` as keyof Run] as
                          | string
                          | undefined)
                  }
                  onChange={() => {}}
                  onRemove={() => {}}
                  readonly
                />
              ))}
            </div>
            <div className="job-list">
              {run.jobs.map((j) => (
                <div key={j.id}>
                  <b>
                    Candidate{" "}
                    {candidates.find((c) => c.id === j.candidate_id)?.number ||
                      "-"}{" "}
                    · {j.kind}
                  </b>
                  <span className={"badge state-" + j.status}>{j.status}</span>
                  <progress max="100" value={j.progress} />
                  <span>{j.progress}%</span>
                  {j.log && <small>{j.log}</small>}
                </div>
              ))}
            </div>
          </section>
          <section>
            <div className="section-title">
              <div>
                <p className="step">STEP 4</p>
                <h2>候補比較と3Dプレビュー</h2>
              </div>
            </div>
            <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} modelPath={rigPreviewPath} rigPreview={!!rigPreviewPath} />
            <div className="grid">
              {candidates.map((c) => (
                <article
                  className={
                    "candidate " + (selected?.id === c.id ? "selected" : "")
                  }
                  key={c.id}
                >
                  {c.preview_path && (
                    <>
                      <div className="checker">
                        <img
                          src={api + "/files/" + c.preview_path}
                          alt={`候補${c.number}の背景除去済み入力`}
                        />
                      </div>
                      <small>
                        背景除去済みの入力画像です。3Dレンダーではありません。
                      </small>
                    </>
                  )}
                  <h3>Candidate {c.number}</h3>
                  <p>
                    {c.status} / score {c.score ?? "-"}
                  </p>
                  <button
                    disabled={c.status !== "completed"}
                    onClick={() => {
                      setSelected(c);
                      void action("/candidates/" + c.id + "/select", {
                        method: "POST",
                      });
                    }}
                  >
                    表示・選択
                  </button>
                  <button
                    className="secondary"
                    disabled={c.status !== "completed"}
                    onClick={() =>
                      void action("/candidates/" + c.id + "/evaluation", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          geometry: 4,
                          texture: 4,
                          overall: 4,
                          verdict: "accept",
                          comment: "",
                        }),
                      })
                    }
                  >
                    評価: Accept
                  </button>
                </article>
              ))}
            </div>
            <button
              className="secondary"
              onClick={() =>
                void action("/runs/" + run.run.id + "/candidates", {
                  method: "POST",
                })
              }
            >
              候補を1つ追加
            </button>
          </section>
          {selected && (
            <section>
              <div className="section-title">
                <div>
                  <p className="step">STEP 5</p>
                  <h2>ゲーム向け整理・リグ・書き出し</h2>
                </div>
              </div>
              <p className="muted">「自動リグ・スキニング」は、別成果物としてメッシュの重複頂点・法線を整理し、指定ポリゴン数へ簡略化後に基本人体骨格と自動ウェイトを作成します。元の候補は変更しません。腕を自然に下げた直立全身モデル向けの試作機能です。</p>
              <div className="optimize">
                {[5000, 10000, 20000, 50000].map((n) => (
                  <button
                    key={n}
                    onClick={() => setTri(n)}
                    className={tri === n ? "on" : ""}
                  >
                    {n.toLocaleString()}
                  </button>
                ))}
                <label>
                  カスタム
                  <input
                    value={custom}
                    onChange={(e) => {
                      setCustom(e.target.value);
                      setTri(Number(e.target.value) || 20000);
                    }}
                    aria-label="custom triangles"
                  />
                </label>
                <label>
                  Texture
                  <select
                    value={tex}
                    onChange={(e) => setTex(Number(e.target.value))}
                  >
                    {[1024, 2048, 4096].map((n) => (
                      <option key={n}>{n}</option>
                    ))}
                  </select>
                </label>
                <label>
                  LOD
                  <input
                    type="checkbox"
                    checked={lod}
                    onChange={(e) => setLod(e.target.checked)}
                  />
                </label>
                <label>
                  Collision
                  <input
                    type="checkbox"
                    checked={collision}
                    onChange={(e) => setCollision(e.target.checked)}
                  />
                </label>
                <label>
                  Height m
                  <input
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    type="number"
                    min=".01"
                  />
                </label>
                <button
                  onClick={() =>
                    void action("/candidates/" + selected.id + "/optimize", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        triangles: tri,
                        texture_size: tex,
                        lod,
                        collision,
                        height_m: height ? Number(height) : null,
                      }),
                    })
                  }
                >
                  最適化
                </button>
                <button
                  disabled={selected.status !== "completed" || loading}
                  onClick={() =>
                    void action("/candidates/" + selected.id + "/rig", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ target_triangles: tri }),
                    })
                  }
                >
                  自動リグ・スキニング（別成果物）
                </button>
                <button
                  onClick={() =>
                    void action(
                      "/candidates/" + selected.id + "/export?preset=Generic",
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(["glb", "fbx"]),
                      },
                    )
                  }
                >
                  GLB / FBX / ZIP Export
                </button>
              </div>
              <div className="downloads">
                {artifacts
                  .filter((a) => a.candidate_id === selected.id)
                  .map((a) => (
                    <a key={a.id} href={api + "/files/" + a.path} download>
                      {a.kind.toUpperCase()} をダウンロード
                    </a>
                  ))}
                {artifacts
                  .filter((a) => a.candidate_id === selected.id && a.kind === "rigged_glb")
                  .map((a) => (
                    <button key={a.id} className={rigPreviewPath === a.path ? "on" : ""} onClick={() => setRigPreviewPath(rigPreviewPath === a.path ? undefined : a.path)}>
                      {rigPreviewPath === a.path ? "元モデルを表示" : "リグをプレビュー（骨格表示）"}
                    </button>
                  ))}
              </div>
            </section>
          )}
          <section>
            <h2>Logs</h2>
            <pre>{run.jobs.map((j) => j.log).join("\n")}</pre>
          </section>
        </>
      )}
    </main>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
