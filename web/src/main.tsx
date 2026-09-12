import React, { Component, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
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
  payload?: string;
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
  animation,
  followMotion = false,
  onAnimationNames,
}: {
  path: string;
  mode: string;
  texture: boolean;
  showSkeleton?: boolean;
  pose?: string;
  animation?: { clip: string; playing: boolean; loop: boolean; speed: number; restart: number };
  followMotion?: boolean;
  onAnimationNames?: (names: string[]) => void;
}) {
  const gltf = useGLTF(api + "/files/" + path);
  // SkinnedMesh needs SkeletonUtils: Object3D.clone leaves its bone references
  // pointing at the source scene, so a helper could move without deforming mesh.
  const scene = useMemo(() => cloneSkeleton(gltf.scene), [gltf.scene]);
  const skeleton = useMemo(() => showSkeleton ? new THREE.SkeletonHelper(scene) : undefined, [scene, showSkeleton]);
  const mixer = useMemo(() => new THREE.AnimationMixer(scene), [scene]);
  const actionRef = useRef<THREE.AnimationAction>();
  const displayGroup = useRef<THREE.Group>(null);
  const rootTracking = useMemo(() => {
    scene.updateMatrixWorld(true);
    const hips = scene.getObjectByName("Hips");
    return { hips, anchor: hips?.getWorldPosition(new THREE.Vector3()), current: new THREE.Vector3() };
  }, [scene]);
  useEffect(() => {
    if (!followMotion) return;
    // A skinned mesh's cached bounds do not follow its deformed vertices.
    // Display-only recentering must not cull a visible animated character.
    scene.traverse((object) => { if ((object as THREE.SkinnedMesh).isSkinnedMesh) object.frustumCulled = false; });
    if (skeleton) skeleton.frustumCulled = false;
  }, [followMotion, scene, skeleton]);
  useFrame((_, delta) => {
    mixer.update(delta);
    if (followMotion && displayGroup.current && rootTracking.hips && rootTracking.anchor) {
      scene.updateWorldMatrix(true, true);
      rootTracking.hips.getWorldPosition(rootTracking.current);
      // Display-only horizontal following. Keep actual vertical motion and
      // never modify the bones/clip or the downloadable root-motion tracks.
      displayGroup.current.position.x += rootTracking.anchor.x - rootTracking.current.x;
      displayGroup.current.position.z += rootTracking.anchor.z - rootTracking.current.z;
    }
  });
  useEffect(() => onAnimationNames?.(gltf.animations.map((clip) => clip.name)), [gltf.animations, onAnimationNames]);
  useEffect(() => {
    mixer.stopAllAction();
    actionRef.current = undefined;
    if (!animation?.clip) return;
    const clip = gltf.animations.find((candidate) => candidate.name === animation.clip);
    if (!clip) return;
    const action = mixer.clipAction(clip);
    actionRef.current = action;
    action.reset().play();
    return () => { action.stop(); mixer.uncacheAction(clip); };
  }, [animation?.clip, animation?.restart, gltf.animations, mixer]);
  useEffect(() => {
    const action = actionRef.current;
    if (!action || !animation?.clip) return;
    action.paused = !animation.playing;
    action.setLoop(animation.loop ? THREE.LoopRepeat : THREE.LoopOnce, animation.loop ? Infinity : 1);
    action.clampWhenFinished = !animation.loop;
    action.timeScale = animation.speed;
  }, [animation?.clip, animation?.loop, animation?.playing, animation?.speed, animation?.restart, gltf.animations, mixer]);
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
    // Manual pose checks and an AnimationMixer both modify pose bones; never
    // apply them together, or the preview would give misleading deformation.
    if (animation?.clip) return;
    if (pose === "arms") {
      rotate("UpperArm.L", new THREE.Vector3(0, 0, 1), -55); rotate("UpperArm.R", new THREE.Vector3(0, 0, 1), 55);
      rotate("LowerArm.L", new THREE.Vector3(0, 1, 0), 70); rotate("LowerArm.R", new THREE.Vector3(0, 1, 0), -70);
    } else if (pose === "legs") {
      rotate("UpperLeg.L", new THREE.Vector3(1, 0, 0), 35); rotate("UpperLeg.R", new THREE.Vector3(1, 0, 0), -20);
      rotate("LowerLeg.L", new THREE.Vector3(1, 0, 0), -65); rotate("LowerLeg.R", new THREE.Vector3(1, 0, 0), -25);
    }
    scene.updateMatrixWorld(true);
  }, [scene, pose, animation?.clip]);
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
  // SkeletonHelper already uses the root's world matrix; keep it outside the
  // display offset group to avoid applying the following translation twice.
  return <><group ref={displayGroup}><primitive object={scene} /></group>{skeleton && <primitive object={skeleton} />}</>;
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
function DiagnosticCamera({ path, preset, multiview }: { path: string; preset: string; multiview: boolean }) {
  const { scene } = useGLTF(api + "/files/" + path);
  const { camera, size, controls } = useThree();
  const box = useMemo(() => new THREE.Box3().setFromObject(scene), [scene]);
  useEffect(() => {
    if (!(camera instanceof THREE.PerspectiveCamera) || box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3()), extent = box.getSize(new THREE.Vector3());
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const width = preset === "side" ? extent.z : extent.x;
    const depth = preset === "side" ? extent.x : extent.z;
    const distance = Math.max(extent.y / (2 * Math.tan(halfFov)), width / (2 * Math.tan(halfFov) * size.width / Math.max(size.height, 1))) * 1.25 + depth / 2;
    const direction = preset === "side" ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, (preset === "front") === multiview ? 1 : -1);
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.near = Math.max(distance / 1000, .0001);
    camera.far = Math.max(distance * 100, 100);
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    const orbit = controls as unknown as { target?: THREE.Vector3; update?: () => void };
    if (orbit?.target) { orbit.target.copy(center); orbit.update?.(); }
  }, [box, camera, controls, preset, multiview, size.width, size.height]);
  return null;
}
function Viewer({ candidate, multiview = false, modelPath, rigPreview = false, motionPreview = false, defaultLoop = true, followMotion = false, diagnostic = false, fallbackToCandidate = true, emptyMessage }: { candidate?: Candidate; multiview?: boolean; modelPath?: string; rigPreview?: boolean; motionPreview?: boolean; defaultLoop?: boolean; followMotion?: boolean; diagnostic?: boolean; fallbackToCandidate?: boolean; emptyMessage?: string }) {
  const [mode, setMode] = useState("solid"),
    [texture, setTexture] = useState(true),
    [preset, setPreset] = useState("front"),
    [pose, setPose] = useState("rest"),
    [animationNames, setAnimationNames] = useState<string[]>([]),
    [clip, setClip] = useState(""),
    [playing, setPlaying] = useState(true),
    [loop, setLoop] = useState(defaultLoop),
    [speed, setSpeed] = useState(1),
    [restart, setRestart] = useState(0);
  const path = modelPath || (fallbackToCandidate ? candidate?.model_path : undefined);
  useEffect(() => { setPose("rest"); setAnimationNames([]); setClip(""); setPlaying(true); }, [path]);
  useEffect(() => { if (motionPreview && !clip && animationNames[0]) setClip(animationNames[0]); }, [motionPreview, clip, animationNames]);
  const animation = motionPreview && clip ? { clip, playing, loop, speed, restart } : undefined;
  if (!path)
    return (
      <div className="viewer empty">
        {emptyMessage || "完了した候補を選択すると3Dプレビューを表示します。"}
      </div>
    );
  return (
    <ViewerBoundary key={path}>
      <div className="viewer-toolbar">
        {!diagnostic && <label>
          表示{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="solid">Solid</option>
            <option value="wire">Wireframe</option>
            <option value="matcap">MatCap</option>
          </select>
        </label>}
        {rigPreview && <span className="toolbar-note">緑の線: スキニングされた骨格</span>}
        {rigPreview && !motionPreview && <label>ポーズ確認 <select value={pose} onChange={(e) => setPose(e.target.value)}><option value="rest">レストポーズ</option><option value="arms">肩・肘テスト</option><option value="legs">股関節・膝テスト</option></select></label>}
        {motionPreview && <>
          <label>クリップ <select value={clip} onChange={(e) => setClip(e.target.value)} disabled={!animationNames.length}>
            <option value="">選択してください</option>{animationNames.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <button disabled={!clip} onClick={() => setPlaying((value) => !value)}>{playing ? "一時停止" : "再生"}</button>
          <button disabled={!clip} onClick={() => { setRestart((value) => value + 1); setPlaying(true); }}>先頭へ戻す</button>
          <label><input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> ループ</label>
          <label>速度 <input aria-label="モーション再生速度" type="number" min="0.1" max="3" step="0.1" value={speed} onChange={(e) => setSpeed(Math.min(3, Math.max(.1, Number(e.target.value) || 1)))} /></label>
          {!animationNames.length && <span className="toolbar-note">モーションクリップを読み込み中です</span>}
        </>}
        {!diagnostic && <label>
          <input
            type="checkbox"
            checked={texture}
            onChange={(e) => setTexture(e.target.checked)}
          />{" "}
          Texture
        </label>}
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
        <Canvas gl={{ alpha: true }} style={{ background: "transparent" }}>
          <ambientLight intensity={1.5} />
          <directionalLight position={[3, 4, 3]} />
          {!diagnostic && !followMotion && <Camera preset={preset} multiview={multiview} />}
          <React.Suspense fallback={<></>}>
            {(diagnostic || followMotion) && <DiagnosticCamera path={path} preset={preset} multiview={multiview} />}
            <Model path={path} mode={mode} texture={texture} showSkeleton={rigPreview} pose={pose} animation={animation} followMotion={followMotion} onAnimationNames={motionPreview ? setAnimationNames : undefined} />
          </React.Suspense>
          <OrbitControls makeDefault={diagnostic || followMotion} />
        </Canvas>
      </div>
    </ViewerBoundary>
  );
}
type AttachmentReport = {
  status: "candidates_found" | "no_candidates" | "not_evaluated";
  source_file: string; candidate_faces: number; tested_poses: number; reason?: string;
  regions: { mesh: string; face_count: number; cross_branch_faces: number; max_stretch: number; pose: string; branches: string[] }[];
};
type SeparationReport = { status: "separated" | "not_separated" | "unchanged" | "not_evaluated"; before_faces: number; after_faces: number; reason: string };
function SeparationStatus({ source, artifacts }: { source?: Artifact; artifacts: Artifact[] }) {
  const directory = source?.path.slice(0, source.path.lastIndexOf("/") + 1);
  const file = directory ? artifacts.find((artifact) => artifact.kind === "unity_separation_report" && artifact.path.slice(0, artifact.path.lastIndexOf("/") + 1) === directory) : undefined;
  const [report, setReport] = useState<SeparationReport>();
  useEffect(() => {
    const controller = new AbortController(); setReport(undefined);
    if (file) void call("/files/" + file.path, { signal: controller.signal }).then((value: SeparationReport) => {
      if (!controller.signal.aborted) setReport(value);
    }).catch(() => { /* The download link remains available. */ });
    return () => controller.abort();
  }, [file?.path]);
  if (!file) return null;
  const summary = report?.status === "separated"
    ? `自動分離済み: 検出候補 ${report.before_faces}面 → ${report.after_faces}面。`
    : report ? `自動分離は安全条件により未変更です: ${report.reason}` : "自動分離の検証結果を読み込み中です…";
  return <p className={report?.status === "separated" ? "success" : "muted"}>{summary} <a href={api + "/files/" + file.path} download>検証結果</a></p>;
}
function AttachmentDiagnostics({ source, artifacts, prefix, multiview }: { source?: Artifact; artifacts: Artifact[]; prefix: string; multiview: boolean }) {
  // Source and diagnostics must belong to exactly the same job and LOD.
  const directory = source?.path.slice(0, source.path.lastIndexOf("/") + 1);
  const find = (suffix: string) => directory ? artifacts.find(a => a.kind === `${prefix}_attachment_${suffix}` && a.path.slice(0, a.path.lastIndexOf("/") + 1) === directory) : undefined;
  const reportFile = find("report"), preview = find("preview");
  const [state, setState] = useState<{ path?: string; report?: AttachmentReport; error?: string }>({});
  useEffect(() => {
    const controller = new AbortController();
    setState({ path: reportFile?.path });
    if (reportFile) {
      void call("/files/" + reportFile.path, { signal: controller.signal }).then((report: AttachmentReport) => {
        if (controller.signal.aborted) return;
        if (report.source_file !== source?.path.split("/").pop() || !Array.isArray(report.regions) || !["candidates_found", "no_candidates", "not_evaluated"].includes(report.status)) throw Error("診断結果と表示モデルが一致しません");
        setState({ path: reportFile.path, report });
      }).catch((error: Error) => { if (!controller.signal.aborted) setState({ path: reportFile.path, error: error.message }); });
    }
    return () => controller.abort();
  }, [reportFile?.path, source?.path]);
  const report = state.path === reportFile?.path ? state.report : undefined;
  const error = state.path === reportFile?.path ? state.error : undefined;
  const names: Record<string, string> = { arm_L: "左腕", arm_R: "右腕", leg_L: "左脚", leg_R: "右脚", trunk: "胴体" };
  return <article className="attachment-diagnostics">
    <h3>癒着・ウェイト異常の候補</h3>
    {!reportFile ? <p className="muted">未診断です。このSTEPを生成すると、候補箇所を自動検査します。</p> : error ? <p className="error">診断を読み込めません: {error}</p> : !report ? <p role="status">診断を読み込み中です…</p> : report.status === "not_evaluated" ? <p role="status">判定不可: {report.reason}</p> : <>
      <p role="status">{report.status === "candidates_found" ? `${report.regions.length}領域・${report.candidate_faces.toLocaleString()}面に異常変形の候補があります。` : "検査条件に該当する候補はありません。癒着がないことの保証ではありません。"} {report.tested_poses}ポーズを検査済み。</p>
      <p className="diagnostic-legend"><span className="risk-orange">橙: 異常な伸び</span> ／ <span className="risk-red">赤: 異常な伸び＋複数部位のウェイト</span> ／ 灰: 今回の検出対象外</p>
      <div className="diagnostic-layout">
        <Viewer key={preview?.path} modelPath={preview?.path} diagnostic multiview={multiview} fallbackToCandidate={false} emptyMessage="診断プレビューが見つかりません。再生成してください。" />
        {report.regions.length > 0 && <div className="diagnostic-regions"><h4>検出理由（伸びが大きい順）</h4><ol>{report.regions.slice(0, 10).map((region, index) => <li key={index}>
          <strong>領域 {index + 1}: 最大 {region.max_stretch.toFixed(1)}倍</strong>
          <div>{region.face_count}面・{region.branches.map(name => names[name] || name).join(" / ") || "部位不明"}</div>
          <div>{region.pose}</div>
          <small>接続した面で異常な伸び{region.cross_branch_faces > 0 ? `。うち${region.cross_branch_faces}面に複数部位の影響` : ""}。</small>
        </li>)}</ol>{report.regions.length > 10 && <p>残り{report.regions.length - 10}領域は診断JSONに記録しています。色付きプレビューは全候補を表示します。</p>}</div>}
      </div>
      <p className="muted">静止モデルに検出面を着色しています。癒着とウェイト誤りの確定・自動分離は行いません。画像との自動照合は未実施です。STEP 3の4方向画像と合わせて確認してください。</p>
    </>}
    {reportFile && <a href={api + "/files/" + reportFile.path} download>診断JSONをダウンロード</a>}
  </article>;
}
function JobProgress({ job, emptyMessage }: { job?: Job; emptyMessage: string }) {
  if (!job) return <p className="muted">{emptyMessage}</p>;
  return <div className="job-list"><div><b>{job.kind}</b><span className={"badge state-" + job.status}>{job.status}</span><progress max="100" value={job.progress} /><span>{job.progress}%</span>{job.log && <small>{job.log}</small>}</div></div>;
}
function ArtifactLinks({ artifacts, emptyMessage }: { artifacts: Artifact[]; emptyMessage: string }) {
  return artifacts.length ? <div className="artifact-links">{artifacts.map((artifact) => <a key={artifact.id} href={api + "/files/" + artifact.path} download>{artifact.kind.toUpperCase()}</a>)}</div> : <p className="muted">{emptyMessage}</p>;
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
  const [unityPreviewKind, setUnityPreviewKind] = useState("unity_lod0_glb");
  const [motionPrompt, setMotionPrompt] = useState("A person walks forward.");
  const [motionSeconds, setMotionSeconds] = useState(4);
  const [motionSeed, setMotionSeed] = useState(1234);
  const [localMotionStatus, setLocalMotionStatus] = useState({ ready: false, message: "ローカルモデルを確認中…" });
  useEffect(() => {
    let active = true;
    const refresh = () => void call("/text-motion/status").then(value => { if (active) setLocalMotionStatus(value); }).catch(() => { if (active) setLocalMotionStatus({ ready: false, message: "ローカルモーションワーカーに接続できません。" }); });
    refresh(); const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);
  useEffect(() => setUnityPreviewKind("unity_lod0_glb"), [selected?.id]);
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
  const selectedArtifacts = selected
    ? artifacts.filter((artifact) => artifact.candidate_id === selected.id)
    : [];
  const selectedJobs = selected
    ? (run?.jobs || []).filter((job) => job.candidate_id === selected.id && job.kind !== "generate")
    : [];
  // 同じ処理を再実行してもSTEP 5が過去の成果物で埋まらないよう、種類ごとに最新の1件だけを表示する。
  const latestSelectedJobs = selectedJobs.filter((job, index) => !selectedJobs.slice(index + 1).some((newer) => newer.kind === job.kind));
  // 実行中・失敗の再試行で、直前に成功した成果物が消えないよう、成果物は種類ごとの最新成功ジョブから選ぶ。
  const latestCompletedSelectedJobs = selectedJobs.filter((job, index) => job.status === "completed" && !selectedJobs.slice(index + 1).some((newer) => newer.kind === job.kind && newer.status === "completed"));
  const latestArtifacts = selectedArtifacts.filter((artifact) => latestCompletedSelectedJobs.some((job) => artifact.path.includes(`/${job.id}/`)));
  const latestRiggedGlb = latestArtifacts.find((artifact) => artifact.kind === "rigged_glb");
  const rigArtifacts = latestArtifacts.filter((artifact) => artifact.kind.startsWith("rigged_") || artifact.kind === "editable_blend" || artifact.kind === "rig_report");
  const unityPreviewArtifacts = ["unity_lod0_glb", "unity_lod1_glb", "unity_lod2_glb"]
    .map((kind) => latestArtifacts.find((artifact) => artifact.kind === kind))
    .filter((artifact): artifact is Artifact => !!artifact);
  const selectedUnityPreview = unityPreviewArtifacts.find((artifact) => artifact.kind === unityPreviewKind) || unityPreviewArtifacts[0];
  const latestMotionGlb = latestArtifacts.find((artifact) => artifact.kind === "template_motion_glb");
  const textMotionJob = latestSelectedJobs.find(job => job.kind === "text_motion");
  const textMotionArtifacts = latestArtifacts.filter(artifact => artifact.kind.startsWith("text_motion_"));
  const textMotionGlb = textMotionArtifacts.find(artifact => artifact.kind === "text_motion_glb");
  const displayedTextMotionJob = latestCompletedSelectedJobs.find(job => job.kind === "text_motion" && textMotionGlb?.path.includes(`/${job.id}/`));
  const displayedTextMotion = useMemo(() => {
    try { return JSON.parse(displayedTextMotionJob?.payload || "{}") as { prompt?: string; seconds?: number }; }
    catch { return {}; }
  }, [displayedTextMotionJob?.payload]);
  const textMotionBusy = !!textMotionJob && ["queued", "claimed", "running", "cancelling"].includes(textMotionJob.status);
  const unityArtifacts = latestArtifacts.filter((artifact) => artifact.kind.startsWith("unity_") && !artifact.kind.includes("_attachment_"));
  const motionArtifacts = latestArtifacts.filter((artifact) => artifact.kind.startsWith("template_motion_") || artifact.kind === "motion_manifest");
  const exportArtifacts = latestArtifacts.filter((artifact) => ["glb", "fbx", "zip"].includes(artifact.kind));
  const latestCompletedGameJob = latestCompletedSelectedJobs.find((job) => job.kind === "game_prepare");
  const latestUnityLod0 = latestCompletedGameJob
    ? latestArtifacts.find((artifact) => artifact.kind === "unity_lod0_glb" && artifact.path.includes(`/${latestCompletedGameJob.id}/`))
    : undefined;
  const canCreateMotion = !!latestUnityLod0 && selected?.status === "completed";
  const motionBlockedReason = !selected || selected.status !== "completed"
    ? "完了した候補を選択してください。"
    : !latestUnityLod0
      ? "先に「Unity用ゲームパッケージ」を完了すると、LOD0モデルからモーションを生成できます。"
      : "";
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
        <div className="input-notice pose-notice">
          <strong>入力ポーズ（必須）:</strong> 4方向すべて<strong>Tポーズ</strong>（腕を肩の高さで左右へまっすぐ伸ばし、脚を離して直立）にしてください。手・腕・胴体や脚が重ならない写真を使うと、形状生成と自動リグの精度が上がります。
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
              {run.jobs.filter((job) => job.kind === "generate").map((j) => (
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
                <h2>候補の選択</h2>
              </div>
            </div>
            <p className="muted">候補モデルを確認して選択します。以降の各STEPでは、開始・進捗・成果物プレビューをその場で確認できます。</p>
            <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} />
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
                  <h2>メッシュ最適化（任意）</h2>
                </div>
              </div>
              <p className="muted">選択中: Candidate {selected.number}。候補の表示モデルを指定ポリゴン数へ置き換えます。元の候補を上書きするため、必要な場合だけ実行してください。</p>
              <article className="game-card">
                <p className="setting-note"><strong>共通の目標ポリゴン数:</strong> ここで選ぶ数値は、次の自動リグとUnity LOD0の目標にも使われます。</p>
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
              </div>
              </article>
              <div className="process-layout">
                <div>
                  <h3>進捗</h3>
                  <JobProgress job={latestSelectedJobs.find((job) => job.kind === "optimize")} emptyMessage="まだ最適化を実行していません。" />
                </div>
                <article className="preview-card">
                  <h3>現在の候補プレビュー</h3>
                  <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} />
                </article>
              </div>
            </section>
          )}
          {selected && (
            <section>
              <div className="section-title"><div><p className="step">STEP 6</p><h2>リグ付きモデルを作成</h2></div></div>
              <p className="muted">基本人体骨格と自動ウェイトを別成果物として作成します。開始・進捗・プレビューをこのSTEPで確認できます。</p>
              <div className="process-layout">
                <div>
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
                  <h3>進捗</h3>
                  <JobProgress job={latestSelectedJobs.find((job) => job.kind === "rig")} emptyMessage="まだリグを作成していません。" />
                  <h3>ダウンロード</h3>
                  <ArtifactLinks artifacts={rigArtifacts} emptyMessage="完了後、ここにリグ成果物が表示されます。" />
                </div>
                <article className="preview-card">
                  <h3>リグ付きモデルのプレビュー</h3>
                  <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} modelPath={latestRiggedGlb?.path} rigPreview={!!latestRiggedGlb} fallbackToCandidate={false} emptyMessage="リグ付きモデルを作成すると、ここに表示されます。" />
                </article>
              </div>
              <AttachmentDiagnostics source={latestRiggedGlb} artifacts={latestArtifacts} prefix="rig" multiview={run.run.generation_mode === "multiview"} />
            </section>
          )}
          {selected && (
            <section>
              <div className="section-title"><div><p className="step">STEP 7</p><h2>Unity用ゲームパッケージ</h2></div></div>
              <p className="muted">LOD0 / 1 / 2、FBX、Humanoid対応表、Capsule Collider設定を出力します。</p>
              <div className="process-layout">
                <div>
                <button
                  disabled={selected.status !== "completed" || loading}
                  onClick={() =>
                    void action("/candidates/" + selected.id + "/game-prepare", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ lod0_triangles: tri }),
                    })
                  }
                >
                  Unity用ゲームパッケージ（LOD0/1/2）
                </button>
                  <h3>進捗</h3>
                  <JobProgress job={latestSelectedJobs.find((job) => job.kind === "game_prepare")} emptyMessage="まだUnity用ゲームパッケージを作成していません。" />
                  <h3>ダウンロード</h3>
                  <ArtifactLinks artifacts={unityArtifacts} emptyMessage="完了後、ここにUnity成果物が表示されます。" />
                </div>
                <article className="preview-card">
                  <h3>Unityモデルのプレビュー</h3>
                  {unityPreviewArtifacts.length > 0 && <div className="preview-tabs" role="group" aria-label="Unity LODを選択">{unityPreviewArtifacts.map((artifact) => <button key={artifact.id} aria-pressed={selectedUnityPreview?.id === artifact.id} className={selectedUnityPreview?.id === artifact.id ? "on" : ""} onClick={() => setUnityPreviewKind(artifact.kind)}>{artifact.kind.replace("unity_", "").replace("_glb", "").toUpperCase()}</button>)}</div>}
                  <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} modelPath={selectedUnityPreview?.path} rigPreview={!!selectedUnityPreview} fallbackToCandidate={false} emptyMessage="Unity用ゲームパッケージを作成すると、ここに表示されます。" />
                </article>
              </div>
              <SeparationStatus source={selectedUnityPreview} artifacts={latestArtifacts} />
              <AttachmentDiagnostics source={selectedUnityPreview} artifacts={latestArtifacts} prefix={selectedUnityPreview?.kind.replace("_glb", "") || "unity_lod0"} multiview={run.run.generation_mode === "multiview"} />
            </section>
          )}
          {selected && (
            <section>
              <div className="section-title"><div><p className="step">STEP 8</p><h2>テンプレートモーション</h2></div></div>
              <p className="muted">Idle / Walk / Run / Jump / Wave を、最新のUnity LOD0から生成します。</p>
              <div className="process-layout">
                <div>
                <button
                  disabled={!canCreateMotion}
                  onClick={() => {
                    if (latestUnityLod0) void action("/candidates/" + selected.id + "/motions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ artifact_id: latestUnityLod0.id }) });
                  }}
                >
                  テンプレートモーション生成（Idle / Walk / Run / Jump / Wave）
                </button>
                {!canCreateMotion && <p className="blocked-reason" role="status">生成できない理由: {motionBlockedReason}</p>}
                  <h3>進捗</h3>
                  <JobProgress job={latestSelectedJobs.find((job) => job.kind === "motion_prepare")} emptyMessage="まだテンプレートモーションを生成していません。" />
                  <h3>ダウンロード</h3>
                  <ArtifactLinks artifacts={motionArtifacts} emptyMessage="完了後、ここにモーション成果物が表示されます。" />
                </div>
                <article className="preview-card">
                  <h3>テンプレートモーションのプレビュー</h3>
                  <Viewer candidate={selected} multiview={run.run.generation_mode === "multiview"} modelPath={latestMotionGlb?.path} rigPreview={!!latestMotionGlb} motionPreview={!!latestMotionGlb} fallbackToCandidate={false} emptyMessage="テンプレートモーションを生成すると、ここに表示されます。" />
                </article>
              </div>
              <AttachmentDiagnostics source={latestMotionGlb} artifacts={latestArtifacts} prefix="motion" multiview={run.run.generation_mode === "multiview"} />
            </section>
          )}
          {selected && (
            <section>
              <div className="section-title"><div><p className="step">STEP 9</p><h2>カスタムモーション（ローカルAI）</h2></div></div>
              <p>短い英語の動作指示から、新しいモーションを生成します。モデルはこのPCで動作します。</p>
              <div className="process-layout">
                <div className="custom-motion-form">
                  <p role="status">{localMotionStatus.message}</p>
                  <label>動作の指示（英語）<textarea value={motionPrompt} maxLength={240} rows={3} onChange={e => setMotionPrompt(e.target.value)} placeholder="A person raises both arms." /></label>
                  <small>1つの動作を短く具体的に指定してください。日本語・長い文章・手指の細かな動作には未対応です。</small>
                  <div className="custom-motion-options">
                    <label>長さ（秒）<input type="number" min={2} max={8} step={.5} value={motionSeconds} onChange={e => setMotionSeconds(Number(e.target.value))} /></label>
                    <label>シード<input type="number" min={0} max={2147483647} step={1} value={motionSeed} onChange={e => setMotionSeed(Number(e.target.value))} /></label>
                  </div>
                  <small>シードを変えると別の動きを生成します。2〜8秒のクリップに対応します。</small>
                  <button disabled={loading || textMotionBusy || !localMotionStatus.ready || !(latestUnityLod0 || latestRiggedGlb) || !motionPrompt.trim() || !Number.isFinite(motionSeconds) || motionSeconds < 2 || motionSeconds > 8 || !Number.isInteger(motionSeed) || motionSeed < 0 || motionSeed > 2147483647}
                    onClick={() => { const source = latestUnityLod0 || latestRiggedGlb; if (source) void action("/candidates/" + selected.id + "/text-motions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ artifact_id: source.id, prompt: motionPrompt, seconds: motionSeconds, seed: motionSeed }) }); }}>
                    {textMotionBusy ? "カスタムモーションを生成中…" : "文章からモーションを生成"}
                  </button>
                  {!(latestUnityLod0 || latestRiggedGlb) && <p className="blocked-reason">先にリグ付きモデルまたはUnity用ゲームパッケージを作成してください。</p>}
                  <h3>進捗</h3>
                  <JobProgress job={textMotionJob} emptyMessage="まだカスタムモーションを生成していません。" />
                  {textMotionBusy && <button onClick={() => void action("/worker/jobs/" + textMotionJob!.id + "/cancel", { method: "POST" })} disabled={textMotionJob?.status === "cancelling"}>生成を中止</button>}
                  <h3>ダウンロード</h3><ArtifactLinks artifacts={textMotionArtifacts} emptyMessage="完了後、ここにGLB・FBXと生成条件を表示します。" />
                  <p className="muted">生成後は動作を確認してください。物体との接触・足接地IKには未対応です。モデル・学習データの利用条件は商用利用前に確認が必要です。</p>
                </div>
                <article className="preview-card">
                  <h3>カスタムモーションのプレビュー</h3>
                  {displayedTextMotion.prompt && <p className="muted">表示中: {displayedTextMotion.prompt}（{displayedTextMotion.seconds}秒）</p>}
                  <Viewer modelPath={textMotionGlb?.path} multiview={run.run.generation_mode === "multiview"} rigPreview={!!textMotionGlb} motionPreview={!!textMotionGlb} defaultLoop={false} followMotion fallbackToCandidate={false} emptyMessage="生成したカスタムモーションをここに自動表示します。" />
                  <p className="muted">プレビューは水平方向の移動に追従します。ダウンロードには元の移動を保持します。</p>
                  {textMotionGlb && displayedTextMotionJob?.id !== textMotionJob?.id && <p className="muted">直前に成功したモーションを表示しています。</p>}
                </article>
              </div>
            </section>
          )}
          {selected && (
            <section>
              <div className="section-title"><div><p className="step">STEP 10</p><h2>汎用エクスポート</h2></div></div>
              <p className="muted">選択中候補をGLB / FBX / ZIPとして書き出します。</p>
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
              <div className="export-status">
                <h3>進捗</h3>
                <JobProgress job={latestSelectedJobs.find((job) => job.kind === "export")} emptyMessage="まだ汎用エクスポートを実行していません。" />
                <h3>ダウンロード</h3>
                <ArtifactLinks artifacts={exportArtifacts} emptyMessage="完了後、ここにGLB / FBX / ZIPが表示されます。" />
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
