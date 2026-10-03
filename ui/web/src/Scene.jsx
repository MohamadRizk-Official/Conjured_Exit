import React, { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { Html, Line, OrbitControls } from '@react-three/drei';
import { ACTIVE_COLOR, PATH_COLORS, positionAt, w2t } from './coords.js';

const DEFAULT_BOX = { xmin: -0.75, xmax: 0.75, ymin: -0.75, ymax: 0.75, zmin: 0.2, zmax: 1.2 };

function AxisLabel({ at, color, children }) {
  return (
    <Html position={at} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
      <span className="axis-label" style={{ color }}>
        {children}
      </span>
    </Html>
  );
}

/** Floor grid at world z = 0 (three Y = 0) plus the world axes. */
function Floor() {
  const o = [0, 0.002, 0];
  return (
    <group>
      <gridHelper args={[4, 20, '#3a4a5e', '#1e2733']} position={[0, 0, 0]} />
      {/* world +x forward = three +X (red) */}
      <Line points={[o, w2t([0.6, 0, 0])]} color="#ff4d4d" lineWidth={3} />
      <AxisLabel at={w2t([0.7, 0, 0])} color="#ff4d4d">x fwd</AxisLabel>
      {/* world +y left = three -Z (green) */}
      <Line points={[o, w2t([0, 0.6, 0])]} color="#4dff88" lineWidth={3} />
      <AxisLabel at={w2t([0, 0.7, 0])} color="#4dff88">y left</AxisLabel>
      {/* world +z up = three +Y (blue) */}
      <Line points={[o, w2t([0, 0, 0.6])]} color="#4da6ff" lineWidth={3} />
      <AxisLabel at={w2t([0, 0, 0.7])} color="#4da6ff">z up</AxisLabel>
      {/* ArUco world origin marker on the floor */}
      <mesh position={[0, 0.001, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.16, 0.16]} />
        <meshBasicMaterial color="#e8e8e8" side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

/** GEOFENCE box as a wireframe. */
function Geofence({ box }) {
  const b = box || DEFAULT_BOX;
  const geo = useMemo(() => {
    // three sizes: X = world x extent, Y = world z extent, Z = world y extent
    const g = new THREE.BoxGeometry(b.xmax - b.xmin, b.zmax - b.zmin, b.ymax - b.ymin);
    return new THREE.EdgesGeometry(g);
  }, [b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax]);
  const center = w2t([(b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, (b.zmin + b.zmax) / 2]);
  return (
    <lineSegments geometry={geo} position={center}>
      <lineBasicMaterial color="#39d98a" transparent opacity={0.65} />
    </lineSegments>
  );
}

/** One stored path as a polyline (highlighted when it is the active one). */
function PathLine({ path, active }) {
  const pts = useMemo(() => path.points.map(w2t), [path.points]);
  if (pts.length < 2) return null;
  const color = active ? ACTIVE_COLOR : PATH_COLORS[path.mode] || '#8899aa';
  return (
    <group>
      <Line points={pts} color={color} lineWidth={active ? 4 : 2} transparent opacity={active ? 1 : 0.55} />
      {/* start marker */}
      <mesh position={pts[0]}>
        <sphereGeometry args={[active ? 0.025 : 0.015, 12, 12]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}

/** The live recording polyline growing as samples arrive. */
function LiveRecording({ points, mode, active }) {
  const pts = useMemo(() => points.map(w2t), [points]);
  if (pts.length === 0) return null;
  const color = mode === 'spell' ? '#ff6bd6' : '#ffd93d';
  return (
    <group>
      {pts.length >= 2 && <Line points={pts} color={color} lineWidth={4} />}
      <mesh position={pts[pts.length - 1]}>
        <sphereGeometry args={[0.02, 12, 12]} />
        <meshBasicMaterial color={active ? color : '#888'} />
      </mesh>
    </group>
  );
}

/** Drone: sphere + heading arrow (from yaw) + drop line to the floor. */
function Drone({ pose, flightState, trackingOk }) {
  const p = w2t([pose.x, pose.y, pose.z]);
  const head = w2t([pose.x + 0.2 * Math.cos(pose.yaw), pose.y + 0.2 * Math.sin(pose.yaw), pose.z]);
  const color = !trackingOk ? '#ff4d4d' : flightState === 'estop' ? '#ff4d4d' : flightState === 'idle' ? '#4dd2ff' : '#00e5ff';
  return (
    <group>
      <mesh position={p}>
        <sphereGeometry args={[0.05, 20, 20]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} />
      </mesh>
      <Line points={[p, head]} color="#ffffff" lineWidth={4} />
      <mesh position={head}>
        <sphereGeometry args={[0.012, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      <Line points={[p, [p[0], 0, p[2]]]} color="#4dd2ff" lineWidth={1} transparent opacity={0.35} dashed dashSize={0.03} gapSize={0.03} />
      <mesh position={[p[0], 0.003, p[2]]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.03, 0.05, 24]} />
        <meshBasicMaterial color="#4dd2ff" transparent opacity={0.6} />
      </mesh>
    </group>
  );
}

/** Wand tip (spell mode) in a different colour, only while the tracker sees it. */
function WandDot({ wand }) {
  if (!wand || !wand.ok) return null;
  const p = w2t([wand.x, wand.y, wand.z]);
  return (
    <mesh position={p}>
      <sphereGeometry args={[0.035, 16, 16]} />
      <meshStandardMaterial color="#ff6bd6" emissive="#ff6bd6" emissiveIntensity={0.7} />
    </mesh>
  );
}

/**
 * Replay preview: a ghost dot moving along the selected path using the
 * path's own times.  While a real replay is running it follows replay.t;
 * otherwise it loops on a local clock so the operator can preview the flight.
 */
function Ghost({ path, replay }) {
  const ref = useRef();
  const start = useRef(null);
  useFrame(({ clock }) => {
    if (!ref.current || !path || path.points.length === 0) return;
    const dur = path.times[path.times.length - 1] || 0;
    let t;
    if (replay && replay.active) {
      t = replay.t;
      start.current = null;
    } else {
      if (start.current === null) start.current = clock.elapsedTime;
      const period = dur + 1.0; // pause a second at the end before looping
      t = dur > 0 ? (clock.elapsedTime - start.current) % period : 0;
    }
    const wp = positionAt(path, Math.min(t, dur));
    if (!wp) return;
    const p = w2t(wp);
    ref.current.position.set(p[0], p[1], p[2]);
  });
  if (!path || path.points.length === 0) return null;
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.035, 16, 16]} />
      <meshBasicMaterial color={ACTIVE_COLOR} transparent opacity={0.45} />
    </mesh>
  );
}

export default function Scene({ state, geoms, config }) {
  const box = config?.geofence || DEFAULT_BOX;
  const active = state?.active_path || null;
  const activeGeom = active ? geoms[active] : null;
  const live = state?.recording?.live_points || [];

  return (
    <Canvas
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: false }}
      // three coords: X = world x, Y = world z (up), Z = -world y.  Camera behind the
      // origin (world -x) and slightly to the right (world -y => three +Z), looking at
      // the middle of the flight volume.
      camera={{ position: [-2.1, 1.7, 2.3], fov: 42, near: 0.05, far: 50 }}
      onCreated={({ scene }) => {
        scene.background = new THREE.Color('#0b0e14');
      }}
    >
      <ambientLight intensity={0.8} />
      <directionalLight position={[2, 4, 3]} intensity={1.0} />
      <Floor />
      <Geofence box={box} />
      {Object.values(geoms).map((g) => (
        <PathLine key={g.name} path={g} active={g.name === active} />
      ))}
      <LiveRecording points={live} mode={state?.recording?.mode} active={!!state?.recording?.active} />
      {activeGeom && <Ghost path={activeGeom} replay={state?.replay} />}
      {state && <Drone pose={state.drone} flightState={state.flight?.state} trackingOk={state.tracking?.ok} />}
      {state && <WandDot wand={state.wand} />}
      <OrbitControls target={[0, 0.55, 0]} maxPolarAngle={Math.PI * 0.49} minDistance={0.8} maxDistance={8} makeDefault />
    </Canvas>
  );
}
