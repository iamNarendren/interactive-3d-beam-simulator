// Euler–Bernoulli cantilever FEM (48 beam elements) — the reference solver.
// Shared by the app, the ROM training script (train_rom.py mirrors it) and the verification test.
export const N = 48, L = 1, I = 1.2e-6, WEAK_HALF_WIDTH = 0.055;

export function solve(A, b) {
  const n = b.length, M = A.map((r, i) => r.slice().concat(b[i]));
  for (let k = 0; k < n; k++) {
    let p = k;
    for (let i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
    if (Math.abs(M[p][k]) < 1e-19) throw Error('Singular stiffness matrix');
    [M[k], M[p]] = [M[p], M[k]];
    const d = M[k][k];
    for (let j = k; j <= n; j++) M[k][j] /= d;
    for (let i = 0; i < n; i++) if (i !== k) { const f = M[i][k]; for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j]; }
  }
  return M.map(r => r[n]);
}

// Dimensionless response for E·I = 1, P = 1, L = 1 (downward load → negative deflection).
// Physical deflection = unit · P / (E·I)   (exact for a linear-elastic beam).
export function femUnit(pos, weak, loss) {
  const nd = 2 * (N + 1), K = Array.from({ length: nd }, () => Array(nd).fill(0)), F = Array(nd).fill(0), le = L / N;
  for (let e = 0; e < N; e++) {
    const x = (e + .5) / N; let EI = 1;
    if (Math.abs(x - weak) < WEAK_HALF_WIDTH) EI *= 1 - loss;
    const fac = EI / le ** 3,
      ke = [[12, 6 * le, -12, 6 * le], [6 * le, 4 * le * le, -6 * le, 2 * le * le], [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le * le, -6 * le, 4 * le * le]].map(r => r.map(v => v * fac)),
      ids = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) K[ids[i]][ids[j]] += ke[i][j];
  }
  const el = Math.min(N - 1, Math.floor(pos * N)), r = pos * N - el;
  F[2 * el] -= (1 - r); F[2 * (el + 1)] -= r;
  const free = Array.from({ length: nd - 2 }, (_, i) => i + 2),
    u = solve(free.map(i => free.map(j => K[i][j])), free.map(i => F[i])), full = [0, 0, ...u];
  return Array.from({ length: N + 1 }, (_, i) => full[2 * i]);
}

// The unit solution depends only on (pos, weak, loss); load and material just scale it.
const cache = new Map();
export function femUnitCached(pos, weak, loss) {
  const key = pos.toFixed(6) + '|' + weak.toFixed(6) + '|' + loss.toFixed(6);
  let v = cache.get(key);
  if (!v) { if (cache.size > 300) cache.clear(); v = femUnit(pos, weak, loss); cache.set(key, v); }
  return v;
}

// p = {load [N], E [Pa], pos, weak, loss} → nodal deflections [m]
export function fem(p) { const s = p.load / (p.E * I); return femUnitCached(p.pos, p.weak, p.loss).map(v => v * s); }
