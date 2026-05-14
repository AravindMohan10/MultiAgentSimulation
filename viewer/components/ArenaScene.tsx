"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, SoftShadows } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { EpisodeData, EpisodeFrame, SlimAgent } from "@/lib/episode";
import { intentLabel, roleColor, worldToScene } from "@/lib/coords";

type Props = {
  episode: EpisodeData;
  frame: EpisodeFrame;
  frameIndex: number;
  cinematic: boolean;
  heroId: string;
};

export function ArenaScene({ episode, frame, frameIndex, cinematic, heroId }: Props) {
  const { camera } = useThree();
  const [ww, wh] = episode.worldSize;
  const groundW = ww + 8;
  const groundH = wh + 8;

  const hero = frame.agents.find((a) => a.id === heroId) ?? frame.agents[0];
  const heroPos = hero ? worldToScene(hero.x, hero.y, episode.worldSize) : [0, 0, 0];

  useFrame((_, delta) => {
    if (!cinematic || !hero) return;
    const target = new THREE.Vector3(heroPos[0], 0, heroPos[2]);
    const desired = new THREE.Vector3(heroPos[0] + 35, 72, heroPos[2] + 55);
    camera.position.lerp(desired, 1 - Math.pow(0.001, delta));
    camera.lookAt(target);
  });

  const trails = useMemo(() => {
    const start = Math.max(0, frameIndex - 12);
    const slice = episode.frames.slice(start, frameIndex + 1);
    const byAgent: Record<string, THREE.Vector3[]> = {};
    for (const f of slice) {
      for (const a of f.agents) {
        if (!byAgent[a.id]) byAgent[a.id] = [];
        const [x, , z] = worldToScene(a.x, a.y, episode.worldSize);
        byAgent[a.id].push(new THREE.Vector3(x, 0.35, z));
      }
    }
    return byAgent;
  }, [episode, frameIndex]);

  return (
    <>
      <ambientLight intensity={0.55} color="#fff5e6" />
      <directionalLight
        castShadow
        position={[60, 100, 40]}
        intensity={1.1}
        color="#fff8ee"
        shadow-mapSize={[1024, 1024]}
      />
      <hemisphereLight args={["#b8d4ff", "#8fbc8f", 0.35]} />
      <SoftShadows size={12} samples={8} focus={0.5} />

      {/* Ground — Ghibli-soft pastel */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} receiveShadow>
        <planeGeometry args={[groundW, groundH]} />
        <meshStandardMaterial color="#a8d5a2" roughness={0.92} metalness={0.02} />
      </mesh>

      {/* Arena border */}
      <lineSegments position={[0, 0.02, 0]}>
        <edgesGeometry args={[new THREE.PlaneGeometry(ww, wh)]} />
        <lineBasicMaterial color="#5a7a5a" transparent opacity={0.5} />
      </lineSegments>

      {/* Obstacles as rounded columns */}
      {episode.obstacles.map((o, i) => {
        const [x, , z] = worldToScene(o.x, o.y, episode.worldSize);
        const h = 2.5 + o.r * 0.8;
        return (
          <mesh key={i} position={[x, h / 2, z]} castShadow receiveShadow>
            <cylinderGeometry args={[o.r, o.r * 1.05, h, 12]} />
            <meshStandardMaterial color="#6b5b4a" roughness={0.85} />
          </mesh>
        );
      })}

      {/* Capture radius ring around hero */}
      {hero && (
        <mesh position={[heroPos[0], 0.03, heroPos[2]]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[episode.captureRadius * 0.92, episode.captureRadius, 32]} />
          <meshBasicMaterial color="#ff6b6b" transparent opacity={0.35} side={THREE.DoubleSide} />
        </mesh>
      )}

      {/* Trails */}
      {Object.entries(trails).map(([id, points]) => {
        if (points.length < 2) return null;
        const agent = frame.agents.find((a) => a.id === id);
        const color = roleColor(agent?.role ?? "villain");
        return (
          <Line
            key={id}
            points={points}
            color={color}
            lineWidth={1.5}
            transparent
            opacity={0.45}
          />
        );
      })}

      {/* Agents */}
      {frame.agents.map((agent) => (
        <AgentNode
          key={agent.id}
          agent={agent}
          worldSize={episode.worldSize}
          showTarget
        />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI / 2.1}
        minDistance={25}
        maxDistance={220}
        enabled={!cinematic}
      />
    </>
  );
}

function AgentNode({
  agent,
  worldSize,
  showTarget,
}: {
  agent: SlimAgent;
  worldSize: [number, number];
  showTarget: boolean;
}) {
  const [x, , z] = worldToScene(agent.x, agent.y, worldSize);
  const color = roleColor(agent.role);
  const scale = agent.role === "hero" ? 1.15 : 1.0;

  const targetLine = useMemo(() => {
    if (!showTarget || !agent.target) return null;
    const [tx, , tz] = worldToScene(agent.target[0], agent.target[1], worldSize);
    return [
      new THREE.Vector3(x, 1.2, z),
      new THREE.Vector3(tx, 0.5, tz),
    ];
  }, [agent.target, showTarget, x, z, worldSize]);

  return (
    <group position={[x, 0, z]}>
      {/* Body */}
      <mesh position={[0, 1.1 * scale, 0]} castShadow>
        <capsuleGeometry args={[0.55 * scale, 1.0 * scale, 6, 12]} />
        <meshStandardMaterial color={color} roughness={0.35} metalness={0.1} />
      </mesh>
      {/* Head */}
      <mesh position={[0, 2.1 * scale, 0]} castShadow>
        <sphereGeometry args={[0.42 * scale, 16, 16]} />
        <meshStandardMaterial color={color} roughness={0.3} />
      </mesh>

      {targetLine && (
        <Line points={targetLine} color={color} lineWidth={1} transparent opacity={0.7} />
      )}

      <Html position={[0, 2.8 * scale, 0]} center distanceFactor={28} style={{ pointerEvents: "none" }}>
        <div
          style={{
            background: "rgba(15,20,35,0.85)",
            color: "#fff",
            padding: "3px 8px",
            borderRadius: 6,
            fontSize: 11,
            whiteSpace: "nowrap",
            border: `1px solid ${color}`,
            fontFamily: "system-ui, sans-serif",
          }}
        >
          {agent.id.replace("_", " ")} · {intentLabel(agent.intent)}
        </div>
      </Html>
    </group>
  );
}
