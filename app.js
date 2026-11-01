import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { N, I, fem, femUnit } from './fem.js';
import { rom, romUnit, romInfo } from './rom.js';

const $ = id => document.getElementById(id);
const materialData = { steel: { label: 'Steel', E: 200e9 }, iron: { label: 'Iron', E: 110e9 }, tungsten: { label: 'Tungsten', E: 400e9 } };
const ui = { load: $('load'), loss: $('loss'), weak: $('weak'), pos: $('pos'), material: $('material') };
let scene, camera, renderer, beamMesh, romLine, baseLine, weakCollar, loadArrow, loadLabel, dimLabel, dust, anim = false, raf = 0, last = 0;
let current = { ref: [], rom: [], err: [] }, lastTable = 0, lastTiming = 0, tFem = null, tRom = null;

function params() { return { load: +ui.load.value * 1000, loss: +ui.loss.value / 100, weak: +ui.weak.value / 100, pos: +ui.pos.value / 100, E: materialData[ui.material.value].E }; }

// ───────────────────────────── 3D scene ─────────────────────────────
function radialTexture(size, stops) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d'), gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  stops.forEach(([o, col]) => gr.addColorStop(o, col)); g.fillStyle = gr; g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function verticalTexture(stops) {
  const c = document.createElement('canvas'); c.width = 8; c.height = 512;
  const g = c.getContext('2d'), gr = g.createLinearGradient(0, 0, 0, 512);
  stops.forEach(([o, col]) => gr.addColorStop(o, col)); g.fillStyle = gr; g.fillRect(0, 0, 8, 512);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
function labelMesh(text, color, w, h) {
  const tc = document.createElement('canvas'); tc.width = 256; tc.height = 64;
  const ctx = tc.getContext('2d'); ctx.fillStyle = color; ctx.font = 'bold 28px monospace'; ctx.textAlign = 'center'; ctx.fillText(text, 128, 40);
  const tex = new THREE.CanvasTexture(tc); tex.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
}

function buildEnvironment() {
  const FLOOR = -0.95;
  // Deep-blue studio backdrop (vertical gradient with a soft horizon glow) — replaces the flat colour.
  scene.background = verticalTexture([[0, '#040b16'], [.34, '#08192e'], [.6, '#0d3050'], [.78, '#0a2640'], [1, '#030a14']]);

  // Soft back-light glow behind the beam.
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture(256, [[0, 'rgba(70,175,235,.60)'], [.35, 'rgba(30,115,175,.22)'], [1, 'rgba(0,0,0,0)']]), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  glow.scale.set(7, 3.4, 1); glow.position.set(0.2, 0.1, -2.4); scene.add(glow);

  // Circular stage with a radial falloff (no square tiles, no hard floor edge).
  const stage = new THREE.Mesh(new THREE.CircleGeometry(3.6, 96), new THREE.MeshBasicMaterial({ map: radialTexture(512, [[0, 'rgba(45,130,190,.62)'], [.3, 'rgba(20,80,130,.38)'], [.65, 'rgba(10,45,80,.14)'], [1, 'rgba(6,16,29,0)']]), transparent: true, depthWrite: false }));
  stage.rotation.x = -Math.PI / 2; stage.position.y = FLOOR; scene.add(stage);

  // Fine concentric guide rings.
  [[.85, .30], [1.45, .22], [2.15, .15], [2.9, .09]].forEach(([r, o]) => {
    const ring = new THREE.Mesh(new THREE.RingGeometry(r - .004, r + .004, 160), new THREE.MeshBasicMaterial({ color: '#33c3ee', transparent: true, opacity: o, depthWrite: false, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = FLOOR + .002; scene.add(ring);
  });

  // Real cast shadow of the beam and wall on the stage.
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.ShadowMaterial({ opacity: .42 }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = FLOOR + .004; shadow.receiveShadow = true; scene.add(shadow);

  // Slow-drifting light particles for depth.
  const count = 170, pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) { pos[3 * i] = (Math.random() - .5) * 6.4; pos[3 * i + 1] = FLOOR + Math.random() * 2.7; pos[3 * i + 2] = (Math.random() - .5) * 6.4; }
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  dust = new THREE.Points(pg, new THREE.PointsMaterial({ map: radialTexture(64, [[0, 'rgba(255,255,255,1)'], [1, 'rgba(255,255,255,0)']]), color: '#8fdcff', size: .07, transparent: true, opacity: .55, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true }));
  scene.add(dust);

  // Fixed support: a wall block standing on the stage with a clamp plate and bolts.
  const wallMat = new THREE.MeshStandardMaterial({ color: '#12304d', roughness: .42, metalness: .5 });
  const wall = new THREE.Mesh(new THREE.BoxGeometry(.16, 1.5, .8), wallMat); wall.position.set(-.58, FLOOR + .75, 0); wall.castShadow = true; wall.receiveShadow = true;
  wall.add(new THREE.LineSegments(new THREE.EdgesGeometry(wall.geometry), new THREE.LineBasicMaterial({ color: '#38bdf8', transparent: true, opacity: .4 }))); scene.add(wall);
  const clamp = new THREE.Mesh(new THREE.BoxGeometry(.035, .24, .34), new THREE.MeshStandardMaterial({ color: '#2a5a80', roughness: .3, metalness: .7 })); clamp.position.set(-.4825, 0, 0); clamp.castShadow = true; scene.add(clamp);
  const boltMat = new THREE.MeshStandardMaterial({ color: '#7fd6ff', roughness: .25, metalness: .8 });
  [[-.28, -.28], [-.28, .28], [.28, -.28], [.28, .28]].forEach(([y, z]) => { const b = new THREE.Mesh(new THREE.CylinderGeometry(.024, .024, .03, 18), boltMat); b.rotation.z = Math.PI / 2; b.position.set(-.485, y, z); scene.add(b); });

  // Engineering dimension line (L = 1 m) in front of the stage.
  const y0 = -.72, z0 = .42, dm = new THREE.LineBasicMaterial({ color: '#4fd2f5', transparent: true, opacity: .55 });
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5, y0, z0), new THREE.Vector3(.5, y0, z0)]), dm));
  [-.5, .5].forEach(x => scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, y0 - .04, z0), new THREE.Vector3(x, y0 + .04, z0)]), dm)));
  dimLabel = labelMesh('L = 1.0 m', '#6fd8f7', .34, .085); dimLabel.position.set(0, y0 - .08, z0); scene.add(dimLabel);
}

function init3d() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, 1, .1, 100); camera.position.set(2.35, 1.8, 3.25);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false }); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap; renderer.outputColorSpace = THREE.SRGBColorSpace; $('scene').appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0x9ecbff, 0x10182a, 1.7));
  const light = new THREE.DirectionalLight(0x8bcfff, 2.4); light.position.set(-2, 4, 3); light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024); Object.assign(light.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: .5, far: 12 }); light.shadow.bias = -.0004; scene.add(light);
  const rim = new THREE.PointLight(0x00e5ff, 1.3, 7); rim.position.set(1, 1.5, -1.5); scene.add(rim);
  buildEnvironment();

  // One solid rectangular beam body. The ROM prediction is an overlay curve, not a second beam.
  beamMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .28, metalness: .32, side: THREE.DoubleSide })); beamMesh.castShadow = true; beamMesh.receiveShadow = true; scene.add(beamMesh);
  romLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: '#ff9c4a', dashSize: .045, gapSize: .028, transparent: true, opacity: .95 })); scene.add(romLine);
  baseLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: '#8da4b8', dashSize: .025, gapSize: .025, transparent: true, opacity: .65 })); scene.add(baseLine);
  weakCollar = new THREE.Mesh(new THREE.BoxGeometry(.075, .075, .15), new THREE.MeshStandardMaterial({ color: '#ff4fa3', emissive: '#6b123c', transparent: true, opacity: .58, roughness: .35, metalness: .2 })); scene.add(weakCollar);
  const arrowMat = new THREE.MeshStandardMaterial({ color: '#ff9c4a', emissive: '#6b2e06', metalness: .3, roughness: .3 });
  loadArrow = new THREE.Group(); const shaft = new THREE.Mesh(new THREE.CylinderGeometry(.012, .012, .32, 12), arrowMat); shaft.position.y = -.16;
  const cone = new THREE.Mesh(new THREE.ConeGeometry(.052, .12, 16), arrowMat); cone.position.y = -.38; loadArrow.add(shaft, cone); scene.add(loadArrow);
  loadLabel = labelMesh('LOAD', '#ffb36c', .35, .09); scene.add(loadLabel);

  const orbit = { drag: false, x: 0, y: 0, theta: Math.atan2(camera.position.x, camera.position.z), phi: Math.acos(camera.position.y / camera.position.length()), dist: camera.position.length() };
  const place = () => { camera.position.set(orbit.dist * Math.sin(orbit.phi) * Math.sin(orbit.theta), orbit.dist * Math.cos(orbit.phi), orbit.dist * Math.sin(orbit.phi) * Math.cos(orbit.theta)); camera.lookAt(0, 0, 0); };
  renderer.domElement.addEventListener('pointerdown', e => { orbit.drag = true; orbit.x = e.clientX; orbit.y = e.clientY; renderer.domElement.setPointerCapture(e.pointerId); });
  renderer.domElement.addEventListener('pointermove', e => { if (!orbit.drag) return; orbit.theta -= (e.clientX - orbit.x) * .006; orbit.phi = Math.max(.2, Math.min(1.45, orbit.phi + (e.clientY - orbit.y) * .005)); orbit.x = e.clientX; orbit.y = e.clientY; place(); });
  renderer.domElement.addEventListener('pointerup', () => orbit.drag = false); renderer.domElement.addEventListener('pointercancel', () => orbit.drag = false);
  renderer.domElement.addEventListener('wheel', e => { orbit.dist = Math.max(2.2, Math.min(7, orbit.dist + e.deltaY * .003)); place(); }, { passive: true });
  camera.lookAt(0, 0, 0); window.addEventListener('resize', resize3d); resize3d(); animateRender();
}
function closedBeamGeometry(arr, depth) {
  const pos = [], col = [], idx = [], hy = .035, hz = depth / 2;
  for (let i = 0; i <= N; i++) { const x = -.5 + i / N, y = arr[i], t = i / N, c = new THREE.Color().setHSL(.56 - .42 * t, .92, .53); for (const [dy, dz] of [[-hy, -hz], [hy, -hz], [hy, hz], [-hy, hz]]) { pos.push(x, y + dy, dz); col.push(c.r, c.g, c.b); } }
  for (let i = 0; i < N; i++) for (let j = 0; j < 4; j++) { const a = 4 * i + j, b = 4 * i + (j + 1) % 4, c = 4 * (i + 1) + j, d = 4 * (i + 1) + (j + 1) % 4; idx.push(a, c, b, b, c, d); }
  idx.push(0, 2, 1, 0, 3, 2); const k = 4 * N; idx.push(k, k + 1, k + 2, k, k + 2, k + 3);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g.setIndex(idx); g.computeVertexNormals(); return g;
}
function lineGeometry(arr, z) { const pts = []; for (let i = 0; i <= N; i++) pts.push(new THREE.Vector3(-.5 + i / N, arr[i], z)); return new THREE.BufferGeometry().setFromPoints(pts); }

// ───────────────────────────── metrics ─────────────────────────────
function compare(ref, pred) {
  let n2 = 0, d2 = 0, mx = 0;
  for (let i = 0; i < ref.length; i++) { const e = ref[i] - pred[i]; n2 += e * e; d2 += ref[i] * ref[i]; mx = Math.max(mx, Math.abs(e)); }
  return { rel: 100 * Math.sqrt(n2 / (d2 || 1e-30)), rmse: Math.sqrt(n2 / ref.length), max: mx, tip: Math.abs(ref[ref.length - 1] - pred[pred.length - 1]) };
}
const mm = v => v * 1000;
const fmtMM = v => { const a = Math.abs(mm(v)); return mm(v).toFixed(a >= 10 ? 2 : a >= 1 ? 3 : 4) + ' mm'; };
const matName = E => Object.keys(materialData).find(k => materialData[k].E === E);
const T = romInfo.metrics.test;

function context(rel) {
  const q = T.rel_l2_pct;
  if (rel <= q.median) return 'Better than the median held-out test case (' + q.median.toFixed(2) + '%).';
  if (rel <= q.p95) return 'Typical: below the 95th percentile of held-out test cases (' + q.p95.toFixed(2) + '%).';
  return 'Harder than 95% of held-out test cases (> ' + q.p95.toFixed(2) + '%) — expect larger ROM error here.';
}
function timeIt(fn, reps) { const t0 = performance.now(); for (let i = 0; i < reps; i++) fn(); return (performance.now() - t0) / reps; }
const ema = (old, v) => old == null ? v : .7 * old + .3 * v;

function update() {
  const p = params(), now = performance.now();
  $('loadVal').textContent = (p.load / 1000).toFixed(0) + ' kN'; $('lossVal').textContent = (p.loss * 100).toFixed(0) + '%'; $('weakVal').textContent = (p.weak * 100).toFixed(0) + '%'; $('posVal').textContent = (p.pos * 100).toFixed(0) + '%';
  $('materialVal').textContent = materialData[ui.material.value].label; $('materialNote').textContent = 'Young’s modulus: ' + (p.E / 1e9).toFixed(0) + ' GPa';

  const ref = fem(p), pred = rom(p), m = compare(ref, pred), max = Math.max(...ref.map(Math.abs), 1e-9);
  current = { ref, rom: pred, err: ref.map((v, i) => pred[i] - v) };

  // Keep the physical shape readable without making tiny displacements look like a giant spike.
  const scale = Math.min(5.0, .42 / max), scaled = ref.map(v => v * scale), scaledROM = pred.map(v => v * scale);
  beamMesh.geometry.dispose(); beamMesh.geometry = closedBeamGeometry(scaled, .105);
  romLine.geometry.dispose(); romLine.geometry = lineGeometry(scaledROM, .062); romLine.computeLineDistances(); romLine.visible = $('showROM').checked;
  baseLine.geometry.dispose(); baseLine.geometry = lineGeometry(Array(N + 1).fill(0), -.062); baseLine.visible = $('showUndeformed').checked;
  weakCollar.visible = $('showDamage').checked && p.loss > 0; weakCollar.position.set(-.5 + p.weak, scaled[Math.round(p.weak * N)], 0);
  const y = scaled[Math.round(p.pos * N)]; loadArrow.position.set(-.5 + p.pos, y + .22, 0); loadLabel.position.set(-.5 + p.pos, y + .40, .04);

  $('femMetric').textContent = (max * 1000).toFixed(3) + ' mm'; $('romMetric').textContent = (Math.max(...pred.map(Math.abs)) * 1000).toFixed(3) + ' mm'; $('errMetric').textContent = m.rel.toFixed(2) + '%';
  $('relError').textContent = m.rel.toFixed(3) + '%'; $('rmseError').textContent = fmtMM(m.rmse); $('maxError').textContent = fmtMM(m.max); $('tipError').textContent = fmtMM(m.tip);
  $('errContext').textContent = context(m.rel);

  // Live timing (throttled; uncached FEM solve vs ROM forward pass).
  if (now - lastTiming > 400) {
    lastTiming = now; tFem = ema(tFem, timeIt(() => femUnit(p.pos, p.weak, p.loss), 1)); tRom = ema(tRom, timeIt(() => romUnit(p.pos, p.weak, p.loss), 40));
    $('timeFem').textContent = tFem.toFixed(2) + ' ms'; $('timeRom').textContent = tRom < .1 ? tRom.toFixed(3) + ' ms' : tRom.toFixed(2) + ' ms'; $('speedup').textContent = '≈ ' + Math.max(1, Math.round(tFem / tRom)) + '× faster';
  }

  if (!anim || now - lastTable > 150) { lastTable = now; renderTable(p); }
  drawPlot(); drawErrPlot(); $('sceneHint').textContent = 'VISUAL SCALE AUTO';
}

function renderTable(p) {
  const cases = [
    ['Current controls', p], ['No local stiffness loss', { ...p, loss: 0 }], ['Half applied load', { ...p, load: p.load * .5 }], ['Double applied load', { ...p, load: p.load * 2 }],
    ['Iron', { ...p, E: materialData.iron.E }], ['Steel', { ...p, E: materialData.steel.E }], ['Tungsten', { ...p, E: materialData.tungsten.E }],
    ['Weak zone moved to 25%', { ...p, weak: .25 }], ['Load moved to 90%', { ...p, pos: .9 }]
  ];
  $('scenarioRows').innerHTML = cases.map(([name, c], k) => {
    const a = fem(c), b = rom(c), m = compare(a, b);
    return `<tr${k === 0 ? ' class="cur"' : ''}><td>${name}</td><td>${(c.load / 1000).toFixed(1)} kN</td><td>${materialData[matName(c.E)]?.label || (c.E / 1e9).toFixed(0) + ' GPa'}</td><td>${(c.loss * 100).toFixed(0)}% @ ${(c.weak * 100).toFixed(0)}%</td><td>${(c.pos * 100).toFixed(0)}%</td><td>${(mm(a[N])).toFixed(3)} mm</td><td>${(mm(b[N])).toFixed(3)} mm</td><td>${m.rel.toFixed(3)}%</td><td>${mm(m.rmse).toFixed(4)} mm</td><td>${mm(m.max).toFixed(4)} mm</td></tr>`;
  }).join('');
}

function fillStaticStats() {
  const r = T.rel_l2_pct, b = romInfo.metrics.baselines_test_rel_l2_pct, mo = romInfo.metrics.model, f2 = v => v.toFixed(2), f4 = v => v.toFixed(4);
  $('heldoutRows').innerHTML = [
    ['Held-out test cases', T.n.toLocaleString('en-US') + ' (unseen FEM runs)'],
    ['Relative L2 error — mean / median', f2(r.mean) + '% / ' + f2(r.median) + '%'],
    ['Relative L2 error — 95th pct / worst', f2(r.p95) + '% / ' + f2(r.max) + '%'],
    ['RMSE — mean / worst *', f4(T.rmse_mm.mean) + ' / ' + f4(T.rmse_mm.max) + ' mm'],
    ['Max point error — mean / worst *', f4(T.max_abs_err_mm.mean) + ' / ' + f4(T.max_abs_err_mm.max) + ' mm'],
    ['Baseline: cubic polynomial ridge (mean)', f2(b.cubic_poly_ridge.mean) + '%'],
    ['Baseline: mean snapshot, no learning (mean)', b.mean_snapshot.mean.toFixed(0) + '%']
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
  $('modelFacts').textContent = 'Model: MLP 5-96-96-11 (tanh), ' + mo.n_params.toLocaleString('en-US') + ' parameters · 10 POD shape modes · trained on ' + mo.n_train.toLocaleString('en-US') + ' FEM snapshots. * Steel, 12 kN reference case.';
}

// ───────────────────────────── plots ─────────────────────────────
function setupCanvas(id) { const c = $(id), ctx = c.getContext('2d'), r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1; c.width = Math.max(1, r.width * dpr); c.height = Math.max(1, r.height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return { ctx, w: r.width, h: r.height }; }
function drawPlot() {
  const { ctx, w, h } = setupCanvas('plot'), pad = { l: 34, r: 12, t: 12, b: 20 }, a = current.ref, b = current.rom, max = Math.max(...a.map(Math.abs), ...b.map(Math.abs), 1e-10);
  const x = i => pad.l + (w - pad.l - pad.r) * i / N, y = v => pad.t + (h - pad.t - pad.b) * (.5 - v / (2 * max));
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = '#183249'; for (let k = 0; k < 5; k++) { const yy = pad.t + (h - pad.t - pad.b) * k / 4; ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke(); }
  for (const [arr, color, dash] of [[a, '#28c7ff', []], [b, '#ff9c4a', [5, 4]]]) { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath(); arr.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.stroke(); }
  ctx.setLineDash([]); ctx.fillStyle = '#8298ad'; ctx.font = '9px monospace'; ctx.fillText('0', pad.l, h - 5); ctx.fillText('0.5', w / 2, h - 5); ctx.fillText('1.0', w - pad.r - 16, h - 5);
}
function drawErrPlot() {
  const { ctx, w, h } = setupCanvas('errPlot'), pad = { l: 46, r: 12, t: 14, b: 20 }, e = current.err.map(v => v * 1000), m = Math.max(...e.map(Math.abs), 1e-9);
  const x = i => pad.l + (w - pad.l - pad.r) * i / N, y = v => pad.t + (h - pad.t - pad.b) * (.5 - v / (2 * m)), y0 = y(0);
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = '#183249'; ctx.lineWidth = 1; [pad.t, y0, h - pad.b].forEach(yy => { ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke(); });
  ctx.beginPath(); ctx.moveTo(x(0), y0); e.forEach((v, i) => ctx.lineTo(x(i), y(v))); ctx.lineTo(x(N), y0); ctx.closePath(); ctx.fillStyle = 'rgba(255,156,74,.16)'; ctx.fill();
  ctx.strokeStyle = '#ff9c4a'; ctx.lineWidth = 1.8; ctx.beginPath(); e.forEach((v, i) => i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))); ctx.stroke();
  ctx.fillStyle = '#8298ad'; ctx.font = '9px monospace'; const f = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(Math.abs(v) < 1 ? 4 : 3);
  ctx.fillText(f(m), 2, pad.t + 3); ctx.fillText('0', 2, y0 + 3); ctx.fillText(f(-m), 2, h - pad.b + 3); ctx.fillText('0', pad.l, h - 5); ctx.fillText('0.5', w / 2, h - 5); ctx.fillText('1.0', w - pad.r - 16, h - 5);
}

// ───────────────────────────── plumbing ─────────────────────────────
function resize3d() { if (!renderer) return; const r = $('scene').getBoundingClientRect(); camera.aspect = Math.max(.1, r.width) / Math.max(.1, r.height); camera.updateProjectionMatrix(); renderer.setSize(Math.max(1, r.width), Math.max(1, r.height)); }
function animateRender() {
  if (dust) dust.rotation.y += .0005;
  loadLabel.lookAt(camera.position); dimLabel.lookAt(camera.position);
  renderer.render(scene, camera); requestAnimationFrame(animateRender);
}
Object.values(ui).forEach(el => el.addEventListener('input', update)); ui.material.addEventListener('change', update); ['showUndeformed', 'showROM', 'showDamage'].forEach(id => $(id).addEventListener('change', update));
window.addEventListener('resize', () => { drawPlot(); drawErrPlot(); });
$('reset').addEventListener('click', () => { ui.load.value = 12; ui.loss.value = 35; ui.weak.value = 62; ui.pos.value = 74; ui.material.value = 'steel'; stop(); update(); });
function stop() { anim = false; cancelAnimationFrame(raf); $('animIcon').textContent = '▶'; $('animText').textContent = 'Animate moving load'; update(); }
$('animate').addEventListener('click', () => {
  if (anim) { stop(); return; }
  anim = true; $('animIcon').textContent = 'Ⅱ'; $('animText').textContent = 'Pause moving load'; last = 0; let position = +ui.pos.value;
  const tick = t => { if (!anim) return; const dt = Math.min(50, t - last || 16); last = t; position += dt * .035; if (position > 95) position = 5; ui.pos.value = position; update(); raf = requestAnimationFrame(tick); };
  raf = requestAnimationFrame(tick);
});
fillStaticStats(); init3d(); update();
