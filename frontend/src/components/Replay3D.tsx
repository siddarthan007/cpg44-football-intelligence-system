import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Analysis, AnalysisEvent } from '../lib/analysis';
import { ballAt, interpolateAt, TEAM_HEX, trackWindow } from '../lib/analysis';
import { Glyph } from './Glyph';

type View = 'director' | 'broadcast' | 'behind' | 'keeper' | 'above' | 'ball' | 'shooter';
const VIEWS: { key: View; label: string }[] = [
  { key: 'director', label: 'Director' }, { key: 'broadcast', label: 'Broadcast' }, { key: 'behind', label: 'Behind goal' }, { key: 'keeper', label: 'Keeper' },
  { key: 'above', label: 'Above' }, { key: 'ball', label: 'Follow ball' },
];
const REF = 3;

/* ---------------- textures ---------------- */
function pitchTexture(L: number, W: number, futsal = false) {
  const s = 16, pad = 8;
  const c = document.createElement('canvas');
  c.width = Math.round((L + 2 * pad) * s); c.height = Math.round((W + 2 * pad) * s);
  const g = c.getContext('2d')!;
  g.fillStyle = '#4f7a46'; g.fillRect(0, 0, c.width, c.height);
  const X = (v: number) => (v + pad) * s, Y = (v: number) => (v + pad) * s;
  for (let i = 0; i < 20; i++) { g.fillStyle = i % 2 ? '#5d8c51' : '#55824a'; g.fillRect(X((i * L) / 20), Y(-pad), (L / 20) * s, (W + 2 * pad) * s); }
  // grain so the surface does not look like plastic
  const img = g.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < img.data.length; i += 4) { const n = (Math.random() - 0.5) * 10; img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n * 0.6; }
  g.putImageData(img, 0, 0);
  g.strokeStyle = 'rgba(248,247,240,0.95)'; g.lineWidth = 0.12 * s; g.lineCap = 'round';
  const cy = W / 2;
  g.strokeRect(X(0), Y(0), L * s, W * s);
  g.beginPath(); g.moveTo(X(L / 2), Y(0)); g.lineTo(X(L / 2), Y(W)); g.stroke();
  g.beginPath(); g.arc(X(L / 2), Y(cy), (futsal ? 3 : 9.15) * s, 0, Math.PI * 2); g.stroke();
  g.fillStyle = 'rgba(248,247,240,0.95)';
  const spots = futsal ? [[L / 2, cy], [6, cy], [L - 6, cy], [10, cy], [L - 10, cy]] : [[L / 2, cy], [11, cy], [L - 11, cy]];
  for (const [x, y] of spots) { g.beginPath(); g.arc(X(x), Y(y), (futsal ? 0.12 : 0.25) * s, 0, Math.PI * 2); g.fill(); }
  if (futsal) {
    for (const [x0, d] of [[0, 1], [L, -1]] as const) {
      const seg = (draw: () => void) => { g.beginPath(); draw(); g.stroke(); };
      seg(() => g.arc(X(x0), Y(cy - 1.58), 6 * s, d > 0 ? -Math.PI / 2 : Math.PI, d > 0 ? 0 : 1.5 * Math.PI));
      seg(() => g.arc(X(x0), Y(cy + 1.58), 6 * s, d > 0 ? 0 : Math.PI / 2, d > 0 ? Math.PI / 2 : Math.PI));
      seg(() => { g.moveTo(X(x0 + d * 6), Y(cy - 1.58)); g.lineTo(X(x0 + d * 6), Y(cy + 1.58)); });
    }
  }
  for (const [x0, d] of (futsal ? [] : [[0, 1], [L, -1]]) as [number, number][]) {
    g.strokeRect(Math.min(X(x0), X(x0 + d * 16.5)), Y(cy - 20.16), 16.5 * s, 40.32 * s);
    g.strokeRect(Math.min(X(x0), X(x0 + d * 5.5)), Y(cy - 9.16), 5.5 * s, 18.32 * s);
    g.beginPath(); g.arc(X(x0 + d * 11), Y(cy), 9.15 * s, d > 0 ? -0.927 : Math.PI - 0.927, d > 0 ? 0.927 : Math.PI + 0.927); g.stroke();
    for (const yy of [0, W]) { g.beginPath(); g.arc(X(x0), Y(yy), 1 * s, 0, Math.PI * 2); g.stroke(); }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 16; tex.colorSpace = THREE.SRGBColorSpace;
  return { tex, pad };
}

function skyTexture() {
  const c = document.createElement('canvas'); c.width = 4; c.height = 256;
  const g = c.getContext('2d')!; const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, '#8fb3d6'); grd.addColorStop(0.55, '#d9e4ea'); grd.addColorStop(1, '#eef0ea');
  g.fillStyle = grd; g.fillRect(0, 0, 4, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function netTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d')!; g.strokeStyle = 'rgba(255,255,255,0.85)'; g.lineWidth = 2;
  g.strokeRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; return t;
}

function ballTexture() {
  const c = document.createElement('canvas'); c.width = 256; c.height = 128;
  const g = c.getContext('2d')!; g.fillStyle = '#fbfaf4'; g.fillRect(0, 0, 256, 128);
  g.fillStyle = '#1b2320';
  for (let i = 0; i < 12; i++) { const x = (i % 6) * 44 + (i > 5 ? 22 : 0), y = i > 5 ? 86 : 30; g.beginPath();
    for (let k = 0; k < 5; k++) { const a = (k / 5) * Math.PI * 2; g.lineTo(x + 11 * Math.cos(a), y + 11 * Math.sin(a)); } g.fill(); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function numberSprite(text: string, colour: string) {
  const c = document.createElement('canvas'); c.width = 128; c.height = 64;
  const g = c.getContext('2d')!; g.fillStyle = colour; g.beginPath(); g.roundRect(14, 8, 100, 48, 12); g.fill();
  g.fillStyle = '#fff'; g.font = '700 34px Archivo, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, 64, 33);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  // fixed size on screen: readable far away, not a wall of colour up close
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true, sizeAttenuation: false }));
  sp.scale.set(0.05, 0.025, 1); sp.renderOrder = 10; return sp;
}

/** Paper label with ink text; redrawn only when its text changes. */
function textSprite() {
  const c = document.createElement('canvas'); c.width = 512; c.height = 96;
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  // constant size on screen, so a label reads the same from any camera distance
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true, sizeAttenuation: false }));
  sp.renderOrder = 11;
  let last = '';
  const set = (text: string) => {
    if (text === last) return; last = text;
    const g = c.getContext('2d')!; g.clearRect(0, 0, c.width, c.height);
    g.font = '600 40px Archivo, sans-serif';
    const w = Math.min(c.width - 8, g.measureText(text).width + 48);
    g.fillStyle = 'rgba(251,249,244,0.96)'; g.strokeStyle = 'rgba(27,35,32,0.35)'; g.lineWidth = 3;
    g.beginPath(); g.roundRect((c.width - w) / 2, 12, w, 72, 36); g.fill(); g.stroke();
    g.fillStyle = '#1b2320'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(text, c.width / 2, 50);
    t.needsUpdate = true;
  };
  sp.scale.set(0.24, 0.045, 1);
  return { sp, set };
}

/* ---------------- figures ---------------- */
interface Limb { hip: THREE.Group; knee: THREE.Group; ankle: THREE.Group }
interface Arm { shoulder: THREE.Group; elbow: THREE.Group }
interface Figure {
  root: THREE.Group; body: THREE.Group; chest: THREE.Group; pelvis: THREE.Group; legL: Limb; legR: Limb; armL: Arm; armR: Arm;
  ring: THREE.Mesh; heading: number; keeper: boolean; phase: number; last?: THREE.Vector3; lastT?: number; lean: number; bank: number;
}

const SEG = { cap: 6, radial: 18 };
/** A smooth limb or torso from a side profile [radius, height] turned about its axis. */
function lathe(profile: [number, number][], segments = SEG.radial) {
  const g = new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), segments);
  g.computeVertexNormals();
  return g;
}

function makeFigure(id: number, team: number, keeper: boolean, label = String(id)): Figure {
  const kitColour = keeper ? '#e2c53a' : TEAM_HEX[team] ?? TEAM_HEX[0];
  const kit = new THREE.MeshStandardMaterial({ color: kitColour, roughness: 0.62, metalness: 0.02 });
  const shortsM = new THREE.MeshStandardMaterial({ color: team === REF ? '#111' : keeper ? '#222' : '#f1efe7', roughness: 0.7 });
  const skin = new THREE.MeshStandardMaterial({ color: ['#b98762', '#8d5f3e', '#d7a985', '#6b4630'][id % 4], roughness: 0.68 });
  const sock = new THREE.MeshStandardMaterial({ color: kitColour, roughness: 0.78 });
  const boot = new THREE.MeshStandardMaterial({ color: '#16181a', roughness: 0.35, metalness: 0.1 });
  const hair = new THREE.MeshStandardMaterial({ color: ['#1d140e', '#3b2618', '#0e0c0b', '#5a3b22'][id % 4], roughness: 0.92 });
  const mesh = (g: THREE.BufferGeometry, m: THREE.Material) => { const o = new THREE.Mesh(g, m); o.castShadow = true; return o; };
  const root = new THREE.Group();
  const body = new THREE.Group(); root.add(body);
  // pelvis and chest are separate so they can counter-rotate while running
  const pelvis = new THREE.Group(); pelvis.position.y = 0.92; body.add(pelvis);
  const chest = new THREE.Group(); chest.position.y = 1.02; body.add(chest);
  const shorts = mesh(lathe([[0.001, -0.16], [0.16, -0.15], [0.175, -0.02], [0.165, 0.1], [0.15, 0.14], [0.001, 0.15]]), shortsM);
  shorts.scale.set(1.12, 1, 0.78); pelvis.add(shorts);
  // shirt: waist, ribs, chest and shoulders as one turned profile, flattened front to back
  const shirt = mesh(lathe([[0.001, -0.02], [0.15, 0], [0.16, 0.12], [0.185, 0.28], [0.2, 0.4], [0.17, 0.5], [0.09, 0.56], [0.001, 0.57]]), kit);
  shirt.scale.set(1.2, 1, 0.72); chest.add(shirt);
  const neck = mesh(new THREE.CylinderGeometry(0.05, 0.058, 0.12, 12), skin); neck.position.y = 0.58; chest.add(neck);
  const head = mesh(new THREE.SphereGeometry(0.112, 20, 14), skin); head.position.set(0, 0.72, 0.008); head.scale.set(0.92, 1.08, 1); chest.add(head);
  const jaw = mesh(new THREE.SphereGeometry(0.085, 16, 10), skin); jaw.position.set(0, 0.665, 0.03); jaw.scale.set(1, 0.8, 1); chest.add(jaw);
  for (const x of [-0.105, 0.105]) { const ear = mesh(new THREE.SphereGeometry(0.022, 10, 8), skin); ear.position.set(x, 0.72, 0); ear.scale.set(0.5, 1, 0.8); chest.add(ear); }
  const cap = mesh(new THREE.SphereGeometry(0.118, 20, 9, 0, Math.PI * 2, 0, Math.PI * 0.52), hair); cap.position.set(0, 0.74, -0.01); cap.rotation.x = -0.28; chest.add(cap);
  const leg = (x: number): Limb => {
    const hip = new THREE.Group(); hip.position.set(x, 0.88, 0); body.add(hip);
    // thigh: wide at the hip, tapering to the knee, with the quadriceps bulge
    const thigh = mesh(lathe([[0.05, -0.43], [0.062, -0.36], [0.078, -0.2], [0.085, -0.08], [0.08, 0], [0.001, 0.02]]), shortsM);
    thigh.material = shortsM; hip.add(thigh);
    const skinThigh = mesh(lathe([[0.05, -0.43], [0.061, -0.36], [0.071, -0.26], [0.072, -0.22], [0.001, -0.21]]), skin); hip.add(skinThigh);
    const knee = new THREE.Group(); knee.position.y = -0.43; hip.add(knee);
    const kneecap = mesh(new THREE.SphereGeometry(0.052, 12, 8), skin); kneecap.position.set(0, 0, 0.012); knee.add(kneecap);
    // calf: muscle belly high, slim ankle; sock and shin pad over it
    const calf = mesh(lathe([[0.03, -0.42], [0.036, -0.34], [0.058, -0.16], [0.06, -0.1], [0.05, -0.02], [0.001, 0]]), sock); knee.add(calf);
    const ankle = new THREE.Group(); ankle.position.y = -0.43; knee.add(ankle);
    const shoe = mesh(new THREE.CapsuleGeometry(0.045, 0.16, SEG.cap, 20), boot); shoe.rotation.x = Math.PI / 2; shoe.position.set(0, -0.025, 0.06); shoe.scale.set(1.05, 1, 0.72); ankle.add(shoe);
    return { hip, knee, ankle };
  };
  const arm = (x: number): Arm => {
    const shoulder = new THREE.Group(); shoulder.position.set(x, 1.43, 0); body.add(shoulder);
    const delt = mesh(new THREE.SphereGeometry(0.068, 14, 9), kit); delt.scale.set(1, 0.9, 0.95); shoulder.add(delt);
    const upper = mesh(lathe([[0.04, -0.29], [0.047, -0.2], [0.052, -0.08], [0.05, 0], [0.001, 0.01]], 24), keeper ? kit : skin); shoulder.add(upper);
    const sleeve = mesh(lathe([[0.055, -0.13], [0.06, -0.06], [0.058, 0.02], [0.001, 0.03]], 24), kit); shoulder.add(sleeve);
    const elbow = new THREE.Group(); elbow.position.y = -0.29; shoulder.add(elbow);
    const fore = mesh(lathe([[0.03, -0.25], [0.034, -0.2], [0.043, -0.08], [0.038, 0], [0.001, 0.01]], 24),
      keeper ? new THREE.MeshStandardMaterial({ color: '#f2f2f2', roughness: 0.6 }) : skin); elbow.add(fore);
    const hand = mesh(new THREE.SphereGeometry(0.04, 10, 8), keeper ? new THREE.MeshStandardMaterial({ color: '#f2f2f2' }) : skin);
    hand.position.y = -0.28; hand.scale.set(0.8, 1.15, 0.55); elbow.add(hand);
    return { shoulder, elbow };
  };
  const legL = leg(-0.095), legR = leg(0.095), armL = arm(-0.255), armR = arm(0.255);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.64, 64), new THREE.MeshBasicMaterial({ color: '#f2b705', transparent: true, opacity: 0, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02; root.add(ring);
  if (team !== REF) { const tag = numberSprite(label, kitColour); tag.position.y = 2.25; root.add(tag); }
  return { root, body, chest, pelvis, legL, legR, armL, armR, ring, heading: 0, keeper, phase: (id * 0.37) % (Math.PI * 2), lean: 0, bank: 0 };
}

/**
 * Gait driven by distance travelled, so feet do not slide: stride length grows
 * with speed (about 1.3 m walking to 4 m sprinting, two steps per stride), and
 * the cycle advances by distance / stride. Walking and running blend by speed:
 * knee lift, heel kick, arm swing and the double bounce of a run grow with it.
 * Pelvis and shoulders counter-rotate, the trunk leans into acceleration and
 * banks into turns.
 */
function pose(f: Figure, sp: number, moved: number, accel: number, turnRate: number, dt: number) {
  const stride = 1.2 + 0.32 * sp;
  f.phase = (f.phase + (moved / stride) * Math.PI * 2) % (Math.PI * 2);
  const run = THREE.MathUtils.smoothstep(sp, 1.8, 3.6);             // 0 walking, 1 running
  const amp = Math.min(1, sp / 7);
  const ph = f.phase, s1 = Math.sin(ph), s2 = Math.sin(ph + Math.PI);
  const hipAmp = 0.35 + 0.55 * amp;
  f.legL.hip.rotation.x = s1 * hipAmp - 0.05 * run; f.legR.hip.rotation.x = s2 * hipAmp - 0.05 * run;
  // knee flexes on the swing (leg moving forward), more when running
  const swingL = Math.max(0, Math.cos(ph)), swingR = Math.max(0, Math.cos(ph + Math.PI));
  const kneeMax = 0.35 + 1.25 * run * (0.4 + 0.6 * amp);
  f.legL.knee.rotation.x = 0.08 + swingL * kneeMax; f.legR.knee.rotation.x = 0.08 + swingR * kneeMax;   // shin folds back
  f.legL.ankle.rotation.x = 0.3 * swingL - 0.15 * Math.max(0, -s1); f.legR.ankle.rotation.x = 0.3 * swingR - 0.15 * Math.max(0, -s2);
  const armAmp = 0.25 + 0.7 * amp;
  f.armL.shoulder.rotation.set(-s1 * armAmp, 0, -0.1); f.armR.shoulder.rotation.set(-s2 * armAmp, 0, 0.1);   // opposite to the same-side leg
  f.armL.elbow.rotation.x = f.armR.elbow.rotation.x = -(0.25 + 1.05 * run);
  f.pelvis.rotation.y = 0.12 * s1 * amp; f.chest.rotation.y = -0.16 * s1 * amp;
  // lean: forward with speed and acceleration, sideways into turns (eased)
  const k = 1 - Math.exp(-dt * 6);
  f.lean += (Math.min(0.32, 0.03 * sp + 0.05 * Math.max(-2, Math.min(3, accel))) - f.lean) * k;
  f.bank += (Math.max(-0.3, Math.min(0.3, -turnRate * sp * 0.05)) - f.bank) * k;
  f.body.rotation.set(f.lean, 0, f.bank);
  // double bounce per stride when running, a gentle one walking; breathing when still
  const bounce = Math.abs(Math.sin(ph)) * (0.012 + 0.045 * run * amp);
  const breathe = sp < 0.3 ? Math.sin(performance.now() / 700 + f.phase) * 0.004 : 0;
  f.body.position.set(0, bounce + breathe - 0.03 * run, 0);
  f.chest.scale.set(1 + breathe * 2, 1, 1 + breathe * 2);
}

/** Kick: the striking leg draws back with the knee folded, then swings
 *  through, straightening at contact, into a follow-through. */
function kick(f: Figure, u: number, right: boolean) {
  const leg = right ? f.legR : f.legL, plant = right ? f.legL : f.legR;
  const back = u < 0.5 ? u / 0.5 : 0, through = u < 0.5 ? 0 : (u - 0.5) / 0.5;
  leg.hip.rotation.x = 0.85 * back + (0.85 - 2.2 * through) * (u >= 0.5 ? 1 : 0);
  leg.knee.rotation.x = u < 0.5 ? 1.35 * back : 1.35 * Math.max(0, 1 - through * 1.6);
  plant.knee.rotation.x = 0.35 * Math.sin(Math.PI * u);
  f.body.rotation.x = 0.1 - 0.18 * Math.sin(Math.PI * u);
  const arm = right ? f.armL : f.armR; arm.shoulder.rotation.z = (right ? -1 : 1) * 0.95 * Math.sin(Math.PI * u);
}

/** Keeper: set, then dive towards the ball side or gather it centrally. */
function save(f: Figure, u: number, lateral: number) {
  const dive = Math.abs(lateral) > 0.8;
  if (dive) {
    const side = Math.sign(lateral), k = Math.min(1, u * 1.6);
    f.body.rotation.z = -side * 1.25 * k; f.body.position.set(side * Math.min(Math.abs(lateral), 2.2) * k, 0.25 * Math.sin(Math.PI * k), 0);
    for (const a of [f.armL, f.armR]) { a.shoulder.rotation.set(0, 0, side * -2.6 * k); a.elbow.rotation.x = -0.1; }
    f.legL.hip.rotation.x = 0.2 * k; f.legR.hip.rotation.x = -0.3 * k;
  } else {
    const k = Math.min(1, u * 2);
    for (const a of [f.armL, f.armR]) { a.shoulder.rotation.set(-1.3 * k, 0, 0); a.elbow.rotation.x = -0.4 * k; }
    f.legL.knee.rotation.x = f.legR.knee.rotation.x = -0.5 * k; f.body.position.y = -0.12 * k;
  }
}

interface Props { analysis: Analysis; t: number; selected?: number; shot?: AnalysisEvent }

export function Replay3D({ analysis, t, selected, shot }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ update: (t: number) => void; setView: (v: View) => void } | null>(null);
  const [view, setView] = useState<View>(shot ? 'director' : 'broadcast');
  const L = analysis.match.pitch_length_m, W = analysis.match.pitch_width_m;
  const futsal = analysis.match.pitch_kind === 'futsal';
  const GW = futsal ? 3 : 7.32, GH = futsal ? 2 : 2.44, GD = futsal ? 1.0 : 1.8;     // goal width, height, net depth
  const activeShot = useMemo(() => shot ?? analysis.events.find(e => e.type === 'shot' && e.t != null && t >= e.t - 0.5 && t <= e.t + 3), [analysis, shot, t]);
  const activeShotRef = useRef<AnalysisEvent | undefined>(activeShot);
  activeShotRef.current = activeShot;

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(el.clientWidth, el.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = skyTexture();
    scene.fog = new THREE.Fog('#e3e8e6', 140, 320);
    const camera = new THREE.PerspectiveCamera(36, el.clientWidth / el.clientHeight, 0.1, 800);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.maxPolarAngle = Math.PI / 2.04; controls.minDistance = 3; controls.maxDistance = 220;

    scene.add(new THREE.HemisphereLight('#f5f7ff', '#4c6a43', 0.85));
    const sun = new THREE.DirectionalLight('#fff4e0', 2.4);
    sun.position.set(-40, 70, 30); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
    Object.assign(sun.shadow.camera, { left: -75, right: 75, top: 55, bottom: -55, near: 1, far: 220 });
    scene.add(sun);

    const { tex, pad } = pitchTexture(L, W, futsal);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(L + 2 * pad, W + 2 * pad), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.96 }));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);

    // stands and advertising boards: a quiet stadium shell for depth cues
    const standMat = new THREE.MeshStandardMaterial({ color: '#8a8f8a', roughness: 0.9 });
    const seatMat = new THREE.MeshStandardMaterial({ color: '#56606a', roughness: 0.8 });
    const boardMat = new THREE.MeshStandardMaterial({ color: '#e9e6dc', roughness: 0.5 });
    const addStand = (len: number, x: number, z: number, rotY: number) => {
      const g = new THREE.Group();
      for (let k = 0; k < 6; k++) {
        const step = new THREE.Mesh(new THREE.BoxGeometry(len, 0.9, 1.4), k % 2 ? seatMat : standMat);
        step.position.set(0, 0.45 + k * 0.9, -k * 1.4); step.receiveShadow = true; g.add(step);
      }
      const board = new THREE.Mesh(new THREE.BoxGeometry(len * 0.9, 0.9, 0.1), boardMat);
      board.position.set(0, 0.45, 3.2); board.castShadow = true; g.add(board);
      g.position.set(x, 0, z); g.rotation.y = rotY; scene.add(g);
    };
    addStand(L + 20, 0, -(W / 2 + pad + 4), 0);
    addStand(L + 20, 0, W / 2 + pad + 4, Math.PI);
    addStand(W + 10, -(L / 2 + pad + 4), 0, Math.PI / 2);
    addStand(W + 10, L / 2 + pad + 4, 0, -Math.PI / 2);

    // goals with nets
    const postMat = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.3, metalness: 0.1 });
    const netTex = netTexture();
    const goal = (gx: number, dir: number) => {
      const g = new THREE.Group();
      const post = new THREE.CylinderGeometry(0.06, 0.06, GH, 16);
      for (const z of [-GW / 2, GW / 2]) { const p = new THREE.Mesh(post, postMat); p.position.set(gx, GH / 2, z); p.castShadow = true; g.add(p); }
      const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, GW + 0.12, 16), postMat); bar.rotation.x = Math.PI / 2; bar.position.set(gx, GH, 0); bar.castShadow = true; g.add(bar);
      const net = (w: number, h: number, repX: number, repY: number) => {
        const t = netTex.clone(); t.needsUpdate = true; t.repeat.set(repX, repY);
        return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: t, transparent: true, alphaTest: 0.2, side: THREE.DoubleSide, roughness: 1 }));
      };
      const back = net(GW, GH, GW * 6.3, GH * 6.3); back.position.set(gx - dir * GD, GH / 2, 0); back.rotation.y = Math.PI / 2; back.rotation.x = 0; g.add(back);
      const top = net(GW, GD, GW * 6.3, GD * 6.3); top.rotation.set(-Math.PI / 2, 0, Math.PI / 2); top.position.set(gx - dir * GD / 2, GH, 0); g.add(top);
      for (const z of [-GW / 2, GW / 2]) { const side = net(GD, GH, GD * 6.3, GH * 6.3); side.position.set(gx - dir * GD / 2, GH / 2, z); g.add(side); }
      scene.add(g);
    };
    goal(-L / 2, -1); goal(L / 2, 1);

    // ball
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.11, 32, 24),          // FIFA size 5: 22 cm across
       new THREE.MeshStandardMaterial({ map: ballTexture(), roughness: 0.45 }));
    ball.castShadow = true; scene.add(ball);
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.32, 24), new THREE.MeshBasicMaterial({ color: '#000', transparent: true, opacity: 0.32, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.012; scene.add(shadow);
    const arcMat = new THREE.LineBasicMaterial({ color: '#f2b705', linewidth: 2 });
    const ARC_MAX = 512;                          // 1.6 s of ball track at any sampling rate we record
    const arcGeo = new THREE.BufferGeometry();
    arcGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARC_MAX * 3), 3).setUsage(THREE.DynamicDrawUsage));
    arcGeo.setDrawRange(0, 0);
    const arc = new THREE.Line(arcGeo, arcMat); arc.frustumCulled = false; scene.add(arc);
    const dropGeo = new THREE.BufferGeometry();
    dropGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3).setUsage(THREE.DynamicDrawUsage));
    const drop = new THREE.Line(dropGeo, new THREE.LineDashedMaterial({ color: '#1b2320', dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.55 }));
    drop.frustumCulled = false;
    scene.add(drop);
    const wedge = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: '#f2b705', transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    wedge.position.y = 0.02; scene.add(wedge);

    // last defender: the offside reference, the second-last opponent from his goal line
    const lineMat = new THREE.MeshBasicMaterial({ color: '#2251a3', transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false });
    const lastLine = new THREE.Mesh(new THREE.PlaneGeometry(W, 3.2), lineMat);
    lastLine.rotation.y = Math.PI / 2; lastLine.position.y = 1.6; lastLine.visible = false; scene.add(lastLine);
    const lineEdge = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.03, -W / 2), new THREE.Vector3(0, 0.03, W / 2)]),
      new THREE.LineBasicMaterial({ color: '#2251a3', transparent: true, opacity: 0.8 }));
    lineEdge.visible = false; scene.add(lineEdge);
    const lineTag = textSprite(); lineTag.sp.visible = false; scene.add(lineTag.sp);
    const distTag = textSprite(); distTag.sp.visible = false; scene.add(distTag.sp);

    const figures = new Map<number, Figure>();
    const keepers = new Set(analysis.players.filter(pl => pl.role === 'goalkeeper').map(pl => pl.id));
    const shirts = new Map(analysis.players.filter(pl => pl.number != null).map(pl => [pl.id, String(pl.number)]));
    const actions = analysis.events.filter(ev => (ev.type === 'pass' || ev.type === 'shot') && ev.t != null && ev.status !== 'dismissed');
    const frames = analysis.timeline.frames;
    let current: View = 'broadcast', ballPos = new THREE.Vector3(), centre = new THREE.Vector3(), lastBall = new THREE.Vector3();
    let camGoal: { pos: THREE.Vector3; target: THREE.Vector3 } | null = null;
    let directing = false, shot: View | '' = '', cut = false;
    const shotGoalX = () => {
      const e = activeShotRef.current; if (!e) return L / 2;
      const dir = analysis.teams[String(e.team) as '1' | '2']?.attack_dir ?? 1; return dir > 0 ? L / 2 : -L / 2;
    };
    const place = (v: View) => {
      current = v;
      const gx = shotGoalX(), s = Math.sign(gx) || 1;
      const presets: Record<View, [THREE.Vector3, THREE.Vector3]> = {
        broadcast: [new THREE.Vector3(centre.x, 11, W / 2 + 15), centre.clone()],
        behind: [new THREE.Vector3(gx + s * 14, 5.5, 0), new THREE.Vector3(gx - s * 22, 0, 0)],
        keeper: [new THREE.Vector3(gx - s * 0.6, 1.7, 0), new THREE.Vector3(gx - s * 30, 0.8, 0)],
        above: [new THREE.Vector3(centre.x, 85, 0.1), new THREE.Vector3(centre.x, 0, 0)],
        ball: [new THREE.Vector3(ballPos.x - 10, 7, ballPos.z + 12), ballPos.clone()],
        // behind the ball, looking along the shot
        shooter: (() => { const d = new THREE.Vector3(gx - ballPos.x, 0, -ballPos.z).normalize();
          return [ballPos.clone().addScaledVector(d, -9).setY(3.2), new THREE.Vector3(gx, 1, 0)] as [THREE.Vector3, THREE.Vector3]; })(),
        director: [new THREE.Vector3(centre.x, 11, W / 2 + 15), centre.clone()],
      };
      const [pos, target] = presets[v];
      camGoal = { pos, target };
    };

    const update = (time: number) => {
      const s = interpolateAt(frames, time);
      const seen = new Set<number>();
      if (s?.players.length) {
        const own = s.players.filter(p => p.team !== REF);
        if (own.length) centre = new THREE.Vector3(own.reduce((a, p) => a + p.x, 0) / own.length - L / 2, 0, own.reduce((a, p) => a + p.y, 0) / own.length - W / 2);
      }
      const inWay = new Set<number>();
      const e = activeShotRef.current;
      if (e?.x != null && e.y != null && s) {
        const gx = shotGoalX() + L / 2;
        const tri = [[e.x, e.y], [gx, W / 2 - GW / 2], [gx, W / 2 + GW / 2]];
        const sign = (p: number[], a: number[], b: number[]) => (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1]);
        for (const p of s.players) {
          if (p.team === e.team || p.team === REF) continue;
          const q = [p.x, p.y], d1 = sign(q, tri[0], tri[1]), d2 = sign(q, tri[1], tri[2]), d3 = sign(q, tri[2], tri[0]);
          if (!((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))) inWay.add(p.id);
        }
        if (!wedge.geometry.attributes.position) {
          wedge.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3).setUsage(THREE.DynamicDrawUsage));
          wedge.geometry.setIndex([0, 1, 2]); wedge.frustumCulled = false;
        }
        const wp = wedge.geometry.attributes.position as THREE.BufferAttribute;
        tri.forEach(([x, y], i) => wp.setXYZ(i, x - L / 2, 0.02, y - W / 2)); wp.needsUpdate = true; wedge.visible = true;
        // second-last defender from the defended goal line (keepers count)
        const defenders = s.players.filter(p => p.team !== e.team && p.team !== REF && p.team !== 0)
          .map(p => ({ id: p.id, d: Math.abs(p.x - gx), x: p.x })).sort((a, b) => a.d - b.d);
        const last = defenders.length >= 2 ? defenders[1] : undefined;
        lastLine.visible = lineEdge.visible = lineTag.sp.visible = Boolean(last);
        if (last) {
          lastLine.position.x = lineEdge.position.x = last.x - L / 2;
          lineTag.set(`Last defender ${last.id}`); lineTag.sp.position.set(last.x - L / 2, 3.9, -W / 2 + 6);
        }
        const dist = Math.hypot(gx - e.x, W / 2 - e.y);
        distTag.set(`${dist.toFixed(1)} m to goal`); distTag.sp.visible = true;
        distTag.sp.position.set((e.x + gx) / 2 - L / 2, 1.1, (e.y + W / 2) / 2 - W / 2);
      } else { wedge.visible = lastLine.visible = lineEdge.visible = lineTag.sp.visible = distTag.sp.visible = false; }
      if (s) for (const p of s.players) {
        let f = figures.get(p.id);
        if (!f) { f = makeFigure(p.id, p.team, keepers.has(p.id), shirts.get(p.id) ?? String(p.id)); figures.set(p.id, f); scene.add(f.root); }
        seen.add(p.id); f.root.visible = true;
        const here = new THREE.Vector3(p.x - L / 2, 0, p.y - W / 2);
        const dt = f.lastT == null ? 0 : Math.max(0, Math.min(0.5, time - f.lastT));
        const jump = f.last == null || Math.abs(time - (f.lastT ?? time)) > 0.6;      // a seek, not motion
        const moved = jump ? 0 : here.distanceTo(f.last!);
        f.root.position.copy(here);
        let turnRate = 0;
        if (Math.hypot(p.dx, p.dy) > 0.02) {
          const target = Math.atan2(p.dx, p.dy);
          let d = target - f.heading; d = Math.atan2(Math.sin(d), Math.cos(d));
          const step = d * (1 - Math.exp(-(dt || 0.05) * 9));                          // critically damped turn
          turnRate = dt > 0 ? step / dt : 0; f.heading += jump ? d : step;
        }
        f.root.rotation.y = f.heading;
        const sp = Math.min(p.sp ?? 0, 9);
        const prevSp = (f as Figure & { sp?: number }).sp ?? sp;
        (f as Figure & { sp?: number }).sp = sp;
        pose(f, sp, moved, dt > 0 ? (sp - prevSp) / dt : 0, turnRate, dt || 0.05);
        f.last = here; f.lastT = time;
        // event-driven actions near this moment
        for (const ev of actions) {
          const u = (time - (ev.t! - 0.35)) / 0.55;
          if (u < 0 || u > 1) continue;
          const kicker = ev.type === 'pass' ? ev.players[0] : (s.holder ?? -1);
          if (kicker === p.id) kick(f, u, (s.ball?.[1] ?? p.y) >= p.y);
        }
        if (f.keeper && activeShotRef.current?.t != null) {
          const ev = activeShotRef.current, u = (time - ev.t!) / 0.9;
          if (u >= 0 && u <= 1.4 && s.ball) save(f, Math.min(u, 1), (s.ball[1] - p.y) * (f.heading > 0 ? 1 : -1));
        }
        const mat = f.ring.material as THREE.MeshBasicMaterial;
        mat.color.set(inWay.has(p.id) ? '#b23a2e' : p.id === s.holder ? '#f2b705' : '#1b2320');
        mat.opacity = inWay.has(p.id) || p.id === s.holder ? 0.95 : p.id === selected ? 0.6 : 0;
      }
      for (const [id, f] of figures) if (!seen.has(id)) f.root.visible = false;
      const track = analysis.ball_track ?? [];
      const bAt = track.length ? ballAt(track, time) : (s?.ball ? [s.ball[0], s.ball[1], s.ball[2]] as [number, number, number] : null);
      if (bAt) {
        ballPos = new THREE.Vector3(bAt[0] - L / 2, Math.max(0.11, bAt[2] + 0.11), bAt[1] - W / 2);
        ball.visible = shadow.visible = true;
        const moved = ballPos.distanceTo(lastBall); lastBall = ballPos.clone();
        if (moved < 3) { const dirv = new THREE.Vector3(ballPos.z - ball.position.z, 0, ball.position.x - ballPos.x); ball.rotateOnWorldAxis(dirv.lengthSq() > 1e-8 ? dirv.normalize() : new THREE.Vector3(1, 0, 0), moved / 0.11); }
        ball.position.copy(ballPos); shadow.position.set(ballPos.x, 0.012, ballPos.z);
        // the shadow spreads and fades as the ball rises
        shadow.scale.setScalar(0.45 * (1 + bAt[2] * 0.08)); (shadow.material as THREE.MeshBasicMaterial).opacity = 0.32 / (1 + bAt[2] * 0.15);
        const pts = (track.length ? trackWindow(track, time - 1.6, time).map(r => [r[1], r[2], r[3]])
          : frames.filter(fr => fr.t > time - 1.6 && fr.t <= time && fr.b).map(fr => fr.b!))
          .map(b => new THREE.Vector3(b[0] - L / 2, Math.max(0.11, b[2] + 0.11), b[1] - W / 2));
        const ap = arcGeo.attributes.position as THREE.BufferAttribute, from = Math.max(0, pts.length - ARC_MAX);
        for (let i = from; i < pts.length; i++) ap.setXYZ(i - from, pts[i].x, pts[i].y, pts[i].z);
        ap.needsUpdate = true; arcGeo.setDrawRange(0, pts.length - from);
        const dp = dropGeo.attributes.position as THREE.BufferAttribute;
        dp.setXYZ(0, ballPos.x, ballPos.y, ballPos.z); dp.setXYZ(1, ballPos.x, 0, ballPos.z); dp.needsUpdate = true;
        drop.computeLineDistances(); drop.visible = bAt[2] > 0.4;
      } else ball.visible = shadow.visible = drop.visible = false;
      if (current === 'ball' || current === 'broadcast') place(current);
      if (current === 'director' || directing) {
        // cut through a shot: wide, behind the shooter, the keeper's view, then above
        const ev = activeShotRef.current;
        const phase: View = !ev?.t ? 'broadcast' : time < ev.t - 1.0 ? 'broadcast' : time < ev.t + 0.25 ? 'shooter'
          : time < ev.t + 1.6 ? 'keeper' : 'above';
        directing = true;
        if (phase !== shot) { shot = phase; place(phase); current = 'director'; cut = phase !== 'broadcast'; }
        else if (phase === 'broadcast') { place('broadcast'); current = 'director'; }
      }
    };
    let dirty = true, onScreen = true;
    const draw = (time: number) => { update(time); dirty = true; };
    api.current = { update: draw, setView: (v: View) => { directing = v === 'director'; shot = ''; place(v === 'director' ? 'broadcast' : v); if (v === 'director') current = 'director'; dirty = true; } };
    update(t); api.current.setView(view);
    camera.position.copy(camGoal!.pos); controls.target.copy(camGoal!.target);

    // Draw only when something changed: the moment (update), the camera (easing
    // or the user orbiting) or the size. A paused replay then costs nothing, and
    // an off-screen or hidden tab stops drawing altogether.
    let raf = 0, lastFrame = performance.now();
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now(), dtc = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
      if (!onScreen || document.hidden) return;
      if (camGoal) {
        if (cut) { camera.position.copy(camGoal.pos); controls.target.copy(camGoal.target); cut = false; }
        const kp = 1 - Math.exp(-dtc * 3.2), kt = 1 - Math.exp(-dtc * 4.5);          // eased, same speed at any frame rate
        const before = camera.position.distanceToSquared(camGoal.pos) + controls.target.distanceToSquared(camGoal.target);
        camera.position.lerp(camGoal.pos, kp); controls.target.lerp(camGoal.target, kt);
        if (before > 1e-6) dirty = true;
        if (camera.position.distanceTo(camGoal.pos) < 0.05 && current !== 'broadcast' && current !== 'ball' && current !== 'director') camGoal = null;
      }
      if (controls.update()) dirty = true;
      if (!dirty) return;
      dirty = false;
      renderer.render(scene, camera);
    };
    loop();
    controls.addEventListener('start', () => { camGoal = null; });
    controls.addEventListener('change', () => { dirty = true; });
    const io = new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; dirty = true; });
    io.observe(el);
    const ro = new ResizeObserver(() => { camera.aspect = el.clientWidth / Math.max(1, el.clientHeight); camera.updateProjectionMatrix(); renderer.setSize(el.clientWidth, el.clientHeight); dirty = true; });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); controls.dispose();
      scene.traverse(o => { const m = o as THREE.Mesh; m.geometry?.dispose?.(); });
      renderer.dispose(); el.removeChild(renderer.domElement); api.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis]);

  useEffect(() => { api.current?.update(t); }, [t, selected, activeShot]);
  useEffect(() => { api.current?.setView(view); }, [view]);

  const b = activeShot?.breakdown;
  return <div className="replay3d">
    <div ref={host} className="replay3d-canvas" />
    <div className="replay3d-views"><div className="seg" role="group" aria-label="Camera">
      {VIEWS.map(v => <button key={v.key} aria-pressed={view === v.key} onClick={() => setView(v.key)}>{v.label}</button>)}
    </div></div>
    {b && <div className="shot-hud" aria-label="Shot breakdown">
      <div><Glyph g="ball" size={15} /><b>{b.shot_speed_kmh ?? 'n/a'}</b><small>km/h</small></div>
      <div><Glyph g="target" size={15} /><b>{b.distance_to_goal_m}</b><small>m to goal</small></div>
      <div><Glyph g="tactics" size={15} /><b>{b.goal_angle_deg}°</b><small>goal angle</small></div>
      <div><Glyph g="squad" size={15} /><b>{b.players_in_the_way ?? 'n/a'}</b><small>in the way</small></div>
    </div>}
  </div>;
}
