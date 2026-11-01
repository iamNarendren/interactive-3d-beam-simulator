"""
Train the reduced-order ML model used by the browser app.

Pipeline
--------
1. Generate FEM snapshots with the SAME Euler-Bernoulli beam FEM as app.js / fem.js
   (48 elements, cantilever, local EI reduction in a window |x - weak| < 0.055).
2. Physics-guided normalisation: for a linear-elastic beam the displacement scales
   exactly with P / (E*I). We therefore train on the dimensionless response
        yhat = u * E * I / (P * L^3)
   so load magnitude and material (Young's modulus) enter analytically in the app,
   and the network only has to learn the dependence on the load position, the
   weak-zone location/extent and the stiffness loss.
3. Reduce each snapshot to (a) a unit-norm SHAPE compressed with POD/SVD and
   (b) its log-amplitude. A small MLP (tanh, 2 x 96 hidden units) is trained to map
        [load position, log(load position), weak-zone start, weak-zone end, stiffness loss]
   to [POD shape coefficients, log-amplitude].
4. Evaluate on a held-out test set that is never used for training,
   model selection or hyper-parameter choice.
5. Export weights + held-out metrics to rom_model.js for in-browser inference.

Run:  python train_rom.py          (about 3-4 minutes on a laptop CPU)
Requires: numpy, scipy, scikit-learn
"""
import json
import time
import warnings

import numpy as np
from scipy.stats import qmc
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- FEM (matches fem.js)
N = 48                       # beam elements
WEAK_HALF_WIDTH = 0.055      # weak zone: |element midpoint - weak| < 0.055
RANGES = dict(pos=(0.05, 0.95), weak=(0.15, 0.85), loss=(0.0, 0.70))   # = UI slider ranges
LO = np.array([RANGES["pos"][0], RANGES["weak"][0], RANGES["loss"][0]])
HI = np.array([RANGES["pos"][1], RANGES["weak"][1], RANGES["loss"][1]])

# Reference case used to express errors in millimetres (same as the app defaults)
REF_E, REF_I, REF_P = 200e9, 1.2e-6, 12e3


def fem_unit(pos, weak, loss):
    """Dimensionless FEM response (E*I = 1, P = 1, L = 1). Returns N+1 nodal deflections."""
    nd = 2 * (N + 1)
    K = np.zeros((nd, nd))
    F = np.zeros(nd)
    le = 1.0 / N
    base = np.array([[12, 6 * le, -12, 6 * le],
                     [6 * le, 4 * le * le, -6 * le, 2 * le * le],
                     [-12, -6 * le, 12, -6 * le],
                     [6 * le, 2 * le * le, -6 * le, 4 * le * le]]) / le ** 3
    for e in range(N):
        x = (e + 0.5) / N
        s = (1.0 - loss) if abs(x - weak) < WEAK_HALF_WIDTH else 1.0
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += s * base
    el = min(N - 1, int(np.floor(pos * N)))
    r = pos * N - el
    F[2 * el] -= (1 - r)
    F[2 * (el + 1)] -= r
    u = np.zeros(nd)
    u[2:] = np.linalg.solve(K[2:, 2:], F[2:])
    return u[0::2]


def weak_region(weak):
    """First / last+1 weakened element (as fractions of L) -- exactly what the FEM sees."""
    mids = (np.arange(N) + 0.5) / N
    idx = np.where(np.abs(mids - weak) < WEAK_HALF_WIDTH)[0]
    return idx[0] / N, (idx[-1] + 1) / N


def features(X):
    out = []
    for pos, weak, loss in X:
        a, b = weak_region(weak)
        out.append([pos, np.log(pos), a, b, loss])
    return np.array(out)


def sample(n, seed, pos_warp=1.0):
    s = qmc.Sobol(3, scramble=True, seed=seed).random(n)
    s[:, 0] = s[:, 0] ** pos_warp
    return LO + s * (HI - LO)


def snapshots(X):
    return np.array([fem_unit(*x) for x in X])


def rel_l2(Yp, Y):
    return 100 * np.linalg.norm(Yp - Y, axis=1) / np.linalg.norm(Y, axis=1)


def summarize(v):
    return dict(mean=float(np.mean(v)), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), max=float(np.max(v)))


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    # Training: 8192 snapshots, denser near the support where the response changes fastest.
    # Validation / test: uniform over the full slider range, independent Sobol sequences.
    Xtr, Xva, Xte = sample(8192, 11, pos_warp=1.6), sample(512, 2), sample(1024, 3)
    Ytr, Yva, Yte = snapshots(Xtr), snapshots(Xva), snapshots(Xte)
    Ftr, Fva, Fte = features(Xtr), features(Xva), features(Xte)
    print(f"FEM snapshots: train {len(Xtr)}, validation {len(Xva)}, test {len(Xte)}  ({time.time() - t0:.1f}s)")

    flo, fhi = Ftr.min(0), Ftr.max(0)
    norm_in = lambda F: 2 * (F - flo) / (fhi - flo) - 1

    # POD of unit-norm shapes + log amplitude
    amp = np.linalg.norm(Ytr, axis=1)
    shapes = Ytr / amp[:, None]
    mean = shapes.mean(0)
    _, s, Vt = np.linalg.svd(shapes - mean, full_matrices=False)
    K_MODES = 10
    energy = 1 - np.cumsum(s ** 2)[K_MODES - 1] / np.sum(s ** 2)
    basis = Vt[:K_MODES]                                  # (K, N+1)
    T = np.c_[(shapes - mean) @ basis.T, np.log(amp)]
    tm, ts = T.mean(0), T.std(0)
    print(f"POD modes kept: {K_MODES}  (residual shape energy {energy:.2e})")

    def decode(o):                                        # network output -> snapshots
        o = o * ts + tm
        return (o[:, :K_MODES] @ basis + mean) * np.exp(o[:, K_MODES])[:, None]

    mlp = MLPRegressor(hidden_layer_sizes=(96, 96), activation="tanh", solver="lbfgs",
                       max_iter=5000, tol=1e-12, alpha=1e-8, random_state=0)
    mlp.fit(norm_in(Ftr), (T - tm) / ts)
    predict = lambda F: decode(mlp.predict(norm_in(F)))
    print(f"MLP trained ({time.time() - t0:.0f}s total)")

    # Baseline for context: cubic polynomial ridge on the same inputs/targets
    poly = PolynomialFeatures(3)
    ridge = Ridge(alpha=1e-6).fit(poly.fit_transform(norm_in(Ftr)), (T - tm) / ts)
    predict_poly = lambda F: decode(ridge.predict(poly.transform(norm_in(F))))
    # Trivial baseline: mean training snapshot (no learning of parameter dependence)
    mean_snapshot = Ytr.mean(0)

    # ------------------------------------------------ held-out evaluation
    scale_mm = 1000 * REF_P / (REF_E * REF_I)             # dimensionless -> mm for the reference case
    res = {}
    for name, split_F, split_Y in [("train", Ftr, Ytr), ("validation", Fva, Yva), ("test", Fte, Yte)]:
        Yp = predict(split_F)
        r = rel_l2(Yp, split_Y)
        rmse = np.sqrt(np.mean((Yp - split_Y) ** 2, axis=1)) * scale_mm
        mx = np.max(np.abs(Yp - split_Y), axis=1) * scale_mm
        tip = np.abs(Yp[:, -1] - split_Y[:, -1]) * scale_mm
        res[name] = dict(n=len(split_Y), rel_l2_pct=summarize(r), rmse_mm=summarize(rmse),
                         max_abs_err_mm=summarize(mx), tip_err_mm=summarize(tip))
    res["baselines_test_rel_l2_pct"] = dict(
        cubic_poly_ridge=summarize(rel_l2(predict_poly(Fte), Yte)),
        mean_snapshot=summarize(rel_l2(np.tile(mean_snapshot, (len(Yte), 1)), Yte)))
    res["model"] = dict(type="MLP 5-96-96-11 (tanh) + POD shape basis", pod_modes=K_MODES,
                        pod_residual_energy=float(energy), n_train=len(Xtr),
                        n_params=int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                        reference_case="steel, 12 kN, I = 1.2e-6 m^4, L = 1 m")
    t = res["test"]
    print("\nHELD-OUT TEST (never used for training or model selection)")
    print(f"  relative L2 error [%]   mean {t['rel_l2_pct']['mean']:.3f}  median {t['rel_l2_pct']['median']:.3f}  "
          f"p95 {t['rel_l2_pct']['p95']:.3f}  max {t['rel_l2_pct']['max']:.3f}")
    print(f"  RMSE [mm, ref. case]    mean {t['rmse_mm']['mean']:.4f}  max {t['rmse_mm']['max']:.4f}")
    print(f"  max |error| [mm]        mean {t['max_abs_err_mm']['mean']:.4f}  max {t['max_abs_err_mm']['max']:.4f}")
    print(f"  baselines (rel L2 %): cubic ridge mean {res['baselines_test_rel_l2_pct']['cubic_poly_ridge']['mean']:.3f}, "
          f"mean-snapshot {res['baselines_test_rel_l2_pct']['mean_snapshot']['mean']:.1f}")

    # ------------------------------------------------ export for the browser
    r10 = lambda a: [float(f"{v:.10g}") for v in np.ravel(a)]
    model = dict(
        N=N, weakHalfWidth=WEAK_HALF_WIDTH, nModes=K_MODES,
        ranges=RANGES, featLo=r10(flo), featHi=r10(fhi),
        layers=[dict(W=[r10(w.T[i]) for i in range(w.shape[1])], b=r10(b))
                for w, b in zip(mlp.coefs_, mlp.intercepts_)],       # W[out][in]
        outMean=r10(tm), outStd=r10(ts), podMean=r10(mean), podBasis=[r10(row) for row in basis],
        metrics=res)
    with open("rom_model.js", "w") as f:
        f.write("// AUTO-GENERATED by train_rom.py -- do not edit by hand.\n")
        f.write("export default " + json.dumps(model, separators=(",", ":")) + ";\n")
    with open("rom_validation.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved rom_model.js and rom_validation.json  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
