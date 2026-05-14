"use client";

import { Canvas } from "@react-three/fiber";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArenaScene } from "@/components/ArenaScene";
import { loadEpisode, loadEpisodeList, type EpisodeData } from "@/lib/episode";

const DEFAULT_EPISODE = "self_steer_scattered_nv2_seed0";
const STEP_MS = 280;

export default function ReplayViewer() {
  const [episodes, setEpisodes] = useState<string[]>([]);
  const [episodeId, setEpisodeId] = useState(DEFAULT_EPISODE);
  const [episode, setEpisode] = useState<EpisodeData | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.5);
  const [cinematic, setCinematic] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    loadEpisodeList().then((list) => {
      setEpisodes(list);
      if (list.length && !list.includes(episodeId)) {
        setEpisodeId(list[0]);
      }
    });
  }, [episodeId]);

  useEffect(() => {
    setError(null);
    setFrameIndex(0);
    loadEpisode(episodeId)
      .then(setEpisode)
      .catch((e: Error) => setError(e.message));
  }, [episodeId]);

  const maxFrame = episode ? episode.frames.length - 1 : 0;
  const frame = episode?.frames[frameIndex] ?? null;

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!playing || !episode) return;
    timerRef.current = setInterval(() => {
      setFrameIndex((i) => (i >= maxFrame ? 0 : i + 1));
    }, STEP_MS / speed);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, speed, episode, maxFrame]);

  const hero = useMemo(
    () => frame?.agents.find((a) => a.role === "hero") ?? null,
    [frame]
  );

  const onScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setFrameIndex(Number(e.target.value));
    setPlaying(false);
  }, []);

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
      <Canvas
        shadows
        camera={{ position: [0, 95, 120], fov: 42, near: 0.1, far: 800 }}
        style={{ background: "linear-gradient(180deg, #8ecae6 0%, #e8d5b5 45%, #95d5b2 100%)" }}
      >
        <Suspense fallback={null}>
          {episode && frame && (
            <ArenaScene
              episode={episode}
              frame={frame}
              frameIndex={frameIndex}
              cinematic={cinematic}
              heroId={hero?.id ?? "hero_1"}
            />
          )}
        </Suspense>
      </Canvas>

      {/* HUD */}
      <div
        style={{
          position: "absolute",
          left: 16,
          right: 16,
          bottom: 16,
          padding: "14px 18px",
          borderRadius: 14,
          background: "rgba(20, 24, 38, 0.82)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center", marginBottom: 10 }}>
          <select
            value={episodeId}
            onChange={(e) => setEpisodeId(e.target.value)}
            style={{
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid #3a4a6a",
              background: "#252b3d",
              color: "#fff",
            }}
          >
            {(episodes.length ? episodes : [episodeId]).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            style={btnStyle}
          >
            {playing ? "Pause" : "Play"}
          </button>

          <label style={{ fontSize: 13, color: "#a8b8d8" }}>
            Speed
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              style={{ width: 90, marginLeft: 8, verticalAlign: "middle" }}
            />
            {speed.toFixed(1)}×
          </label>

          <label style={{ fontSize: 13, color: "#a8b8d8", display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={cinematic}
              onChange={(e) => setCinematic(e.target.checked)}
            />
            Cinematic cam
          </label>

          {episode && (
            <span style={{ fontSize: 13, color: "#8fa8d0", marginLeft: "auto" }}>
              {episode.mapTemplate} · seed {episode.seed} · {episode.outcome?.replace(/_/g, " ")}
            </span>
          )}
        </div>

        <input
          type="range"
          min={0}
          max={maxFrame}
          value={frameIndex}
          onChange={onScrub}
          style={{ width: "100%" }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#7a90b8", marginTop: 4 }}>
          <span>Step {frame?.step ?? 0} / {episode?.steps ?? 0}</span>
          <span>t = {(frame?.time ?? 0).toFixed(1)}s</span>
        </div>
      </div>

      {/* Agent intent panel */}
      {frame && (
        <div
          style={{
            position: "absolute",
            top: 16,
            left: 16,
            padding: "12px 14px",
            borderRadius: 12,
            background: "rgba(20, 24, 38, 0.78)",
            backdropFilter: "blur(6px)",
            fontSize: 13,
            lineHeight: 1.6,
            minWidth: 200,
          }}
        >
          {frame.agents.map((a) => (
            <div key={a.id} style={{ marginBottom: 6 }}>
              <span style={{ color: a.role === "hero" ? "#7eb8ff" : "#ff8a7a", fontWeight: 600 }}>
                {a.id}
              </span>
              <span style={{ color: "#9aa8c4" }}> · </span>
              <span style={{ color: "#e8eeff" }}>{a.intent?.replace(/_/g, " ") ?? "—"}</span>
              {a.target && (
                <span style={{ color: "#6a7a98", fontSize: 11, display: "block" }}>
                  → ({a.target[0].toFixed(0)}, {a.target[1].toFixed(0)})
                </span>
              )}
              {a.messagesSent > 0 && (
                <span style={{ color: "#ffd166", fontSize: 11 }}> 📡 sent</span>
              )}
            </div>
          ))}
        </div>
      )}

      {error && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            padding: 24,
            background: "#2a1a1a",
            borderRadius: 12,
            border: "1px solid #ff5c4a",
            maxWidth: 420,
            textAlign: "center",
          }}
        >
          <p>{error}</p>
          <p style={{ fontSize: 13, color: "#aaa" }}>
            Run:{" "}
            <code style={{ color: "#7eb8ff" }}>
              PYTHONPATH=. python scripts/export_episode_for_viewer.py
            </code>
          </p>
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "6px 14px",
  borderRadius: 8,
  border: "none",
  background: "linear-gradient(135deg, #5b8def, #7eb8ff)",
  color: "#0d1526",
  fontWeight: 600,
};
