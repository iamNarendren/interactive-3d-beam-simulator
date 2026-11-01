// Browser inference for the trained reduced-order model (weights exported by train_rom.py).
// Network:  [pos, ln pos, weak-zone start, weak-zone end, loss] → 2×96 tanh → 10 POD shape coefficients + ln(amplitude)
// Prediction = (POD mean + Σ coefficient·mode) · exp(ln amplitude)  — then scaled by P/(E·I) (linear elasticity).
import model from './rom_model.js';
import { I } from './fem.js';

export const romInfo = model;
const { N, weakHalfWidth, featLo, featHi, layers, outMean, outStd, podMean, podBasis, nModes, ranges } = model;

export function weakRegion(weak) {
  let first = -1, last = -1;
  for (let e = 0; e < N; e++) if (Math.abs((e + .5) / N - weak) < weakHalfWidth) { if (first < 0) first = e; last = e; }
  return [first / N, (last + 1) / N];
}
export function inTrainingRange(pos, weak, loss) {
  return pos >= ranges.pos[0] - 1e-9 && pos <= ranges.pos[1] + 1e-9 && weak >= ranges.weak[0] - 1e-9 && weak <= ranges.weak[1] + 1e-9 && loss >= ranges.loss[0] - 1e-9 && loss <= ranges.loss[1] + 1e-9;
}

// Dimensionless ROM prediction (same normalisation as femUnit).
export function romUnit(pos, weak, loss) {
  const [a, b] = weakRegion(weak), f = [pos, Math.log(pos), a, b, loss];
  let h = f.map((v, i) => 2 * (v - featLo[i]) / (featHi[i] - featLo[i]) - 1);
  layers.forEach((layer, li) => {
    h = layer.W.map((row, o) => { let s = layer.b[o]; for (let i = 0; i < row.length; i++) s += row[i] * h[i]; return li < layers.length - 1 ? Math.tanh(s) : s; });
  });
  const o = h.map((v, i) => v * outStd[i] + outMean[i]), amp = Math.exp(o[nModes]);
  const out = new Array(N + 1);
  for (let i = 0; i <= N; i++) { let s = podMean[i]; for (let k = 0; k < nModes; k++) s += o[k] * podBasis[k][i]; out[i] = s * amp; }
  return out;
}

// p = {load [N], E [Pa], pos, weak, loss} → predicted nodal deflections [m]
export function rom(p) { const s = p.load / (p.E * I); return romUnit(p.pos, p.weak, p.loss).map(v => v * s); }
