#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt  


try:
    from sklearn.isotonic import IsotonicRegression

    SK_OK = True
except Exception:  # pragma: no cover
    SK_OK = False

try:
    from sklearn.linear_model import HuberRegressor

    SKL_HUBER_OK = True
except Exception:  # pragma: no cover
    SKL_HUBER_OK = False

try:
    from scipy.optimize import minimize

    SCIPY_OK = True
except Exception:  # pragma: no cover
    SCIPY_OK = False


STAGE1_ALPHA_DB_RANGE = (0.3, 12.0, 28)  # (lo, hi, num) 
STAGE1_BETA_DB_RANGE = (0.3, 30.0, 36)

LOCAL_ALPHA_STAGE2 = dict(span_lo=1.8, span_hi=1.8, num=17)
LOCAL_BETA_STAGE2 = dict(span_lo=1.8, span_hi=1.8, num=20)

LOCAL_ALPHA_STAGE3 = dict(span_lo=1.4, span_hi=1.4, num=19)  
LOCAL_BETA_STAGE3 = dict(span_lo=1.4, span_hi=1.4, num=23)

TOPK_STAGE1 = 30
TOPK_STAGE2 = 15
TOPK_STAGE3 = 10

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
})



def parse_float_array_from_string(s: object) -> np.ndarray:
    if s is None:
        return np.array([], dtype=float)
    if isinstance(s, (list, tuple, np.ndarray)):
        try:
            return np.asarray(s, dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)
    if not isinstance(s, str):
        try:
            return np.array(s, dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)

    t = s.strip()
    t = t.replace("[", "").replace("]", "")
    if not t:
        return np.array([], dtype=float)

    parts = re.split(r"[,\s\t]+", t)
    vals: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            vals.append(float(p))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def eesm_alpha_beta(sinr_sc: np.ndarray, alpha: float, beta: float) -> float:
    s = np.asarray(sinr_sc, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.nan
    x = -s / max(beta, 1e-12)
    m = float(np.max(x))
    lme = m + np.log(np.mean(np.exp(x - m)))  # log(mean(exp(x)))
    return -alpha * float(lme)


def compute_xeff_list(sinr_list: List[np.ndarray], alpha: float, beta: float) -> np.ndarray:
    return np.asarray([eesm_alpha_beta(s, alpha, beta) for s in sinr_list], dtype=float)




def spearman_objective(xeff: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(xeff) & np.isfinite(y)
    if not np.any(m):
        return float("inf")
    rho = spearmanr(xeff[m], y[m]).correlation
    rho = float(rho) if np.isfinite(rho) else 0.0
    return 1.0 - abs(rho)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if yt.size == 0:
        return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, Spearman=np.nan)
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) + 1e-18
    r2 = 1.0 - ss_res / ss_tot
    sp = float(spearmanr(yt, yp).correlation) if yt.size >= 3 else np.nan
    return dict(MAE=mae, RMSE=rmse, R2=r2, Spearman=sp)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return np.nan
    a = np.rint(y_true[m]).astype(int)
    b = np.rint(y_pred[m]).astype(int)
    mn, mx = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
    n = mx - mn + 1
    if n <= 0:
        return np.nan
    O = np.zeros((n, n), dtype=float)
    for i, j in zip(a, b):
        O[i - mn, j - mn] += 1
    act = np.array([(a == (i + mn)).sum() for i in range(n)], dtype=float)
    pred = np.array([(b == (i + mn)).sum() for i in range(n)], dtype=float)
    E = np.outer(act, pred) / max(len(a), 1)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((n - 1) ** 2) if n > 1 else 0.0
    num = float((W * O).sum())
    den = float((W * E).sum()) if float((W * E).sum()) != 0 else np.nan
    if not np.isfinite(den) or den == 0:
        return np.nan
    return 1.0 - num / den




def _sparse_ticklabels(vals: List[float], max_ticks: int = 10) -> Tuple[List[int], List[str]]:
    vals = list(sorted(vals))
    n = len(vals)
    if n <= max_ticks:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int).tolist()
    labels = [f"{vals[i]:.3g}" for i in idx]
    return idx, labels


def plot_loss_curve(df: pd.DataFrame, title: str, outpath: Path, topk: int = 12) -> None:
    sub = df[["alpha", "beta", "loss"]].dropna().sort_values("loss", ascending=True).reset_index(drop=True)
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(sub)), sub["loss"].values, lw=2)
    ax.set_xlabel("Grid rank (sorted by loss)")
    ax.set_ylabel("Loss = 1 - |Spearman ρ|")
    ax.set_title(title)
    if len(sub) and 0 < topk < len(sub):
        thr = float(sub.loc[topk - 1, "loss"])
        ax.axhline(thr, ls="--")
        ax.text(0.02, 0.94, f"Top-{topk} loss ≤ {thr:.4g}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_param_distribution_heatmap(
    df: pd.DataFrame,
    outpath: Path,
    mode: str = "softmin",
    tau: float = 0.12,
    topk: int = 20,
    title: str = "Good-parameter distribution",
    max_ticks_axis: int = 10,
) -> None:
    xs = sorted(df["beta"].dropna().unique())
    ys = sorted(df["alpha"].dropna().unique())
    X = {v: j for j, v in enumerate(xs)}
    Y = {v: i for i, v in enumerate(ys)}
    Z = np.zeros((len(ys), len(xs)), dtype=float)

    sub = df[["alpha", "beta", "loss"]].dropna().copy().sort_values("loss", ascending=True).reset_index(drop=True)

    if mode == "softmin":
        L = sub["loss"].to_numpy()
        if not np.isfinite(L).any():
            W = np.zeros_like(L)
        else:
            L0 = float(np.nanmin(L))
            scale = float(np.nanstd(L) + 1e-12)
            W = np.exp(-(L - L0) / max(tau * scale, 1e-9))
            W = W / (np.nansum(W) + 1e-18)
    else:  # top-k 均分权
        k = min(topk, len(sub))
        W = np.zeros(len(sub), dtype=float)
        if k > 0:
            W[:k] = 1.0 / k

    for w, (_, row) in zip(W, sub.iterrows()):
        i = Y[row["alpha"]]
        j = X[row["beta"]]
        Z[i, j] += float(w)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, aspect="auto", origin="lower")
    xi, xl = _sparse_ticklabels(xs, max_ticks_axis)
    yi, yl = _sparse_ticklabels(ys, max_ticks_axis)
    ax.set_xticks(xi)
    ax.set_xticklabels(xl, rotation=35, ha="right")
    ax.set_yticks(yi)
    ax.set_yticklabels(yl)
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    ax.set_title(f"{title} ({mode})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("weight")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nm_history(history_df: Optional[pd.DataFrame], outpath: Path, title: str = "Nelder–Mead (rank) loss") -> None:
    if history_df is None or history_df.empty:
        return
    idx_best = int(history_df["loss"].astype(float).idxmin())
    best_row = history_df.loc[idx_best]
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.plot(history_df["iter"].values, history_df["loss"].values, lw=2)
    ax.axvline(best_row["iter"], ls="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss = 1 - |Spearman ρ|")
    ax.set_title(f"{title} (best@iter={int(best_row['iter'])}, loss={best_row['loss']:.6g})")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_iso(xeff: np.ndarray, y: np.ndarray, iso: "IsotonicRegression", outpath: Path) -> None:
    m = np.isfinite(xeff) & np.isfinite(y)
    xs, ys = xeff[m], y[m]
    if xs.size == 0:
        return
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    nb = min(30, max(8, len(xs) // 100))
    bins = np.array_split(np.arange(len(xs)), nb)
    xb = np.array([np.mean(xs[b]) for b in bins])
    yb = np.array([np.mean(ys[b]) for b in bins])
    yhatb = iso.predict(xb)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.plot(xb, yb, marker="o", lw=1.5, label="Empirical (bin means)")
    ax.plot(xb, yhatb, marker="s", lw=1.5, label="Isotonic fit")
    ax.set_xlabel("EESM effective (x_eff)")
    ax.set_ylabel("Label")
    ax.set_title("Monotone calibration (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_scatter(xeff: np.ndarray, y: np.ndarray, yhat: np.ndarray, outpath: Path, title: str) -> None:
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.scatter(xeff, y, s=10, alpha=0.45, label="train")
    order = np.argsort(xeff)
    ax.plot(xeff[order], yhat[order], lw=2, label="fit")
    ax.set_xlabel("EESM effective (x_eff)")
    ax.set_ylabel("Label")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path, title: str = "Residuals (train)") -> None:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    res = (y_pred[m] - y_true[m]).ravel()
    if res.size == 0:
        return
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111)
    ax.hist(res, bins=40)
    ax.set_title(title)
    ax.set_xlabel("pred - true")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)




def fit_calibrator(x: np.ndarray, y: np.ndarray, kind: str = "isotonic"):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(float)
    y = y[m].astype(float)
    if x.size < 2:
        c = float(np.nanmean(y))
        return ("constant", dict(c=c))

    if kind == "isotonic":
        if not SK_OK:
            raise RuntimeError("scikit-learn 不可用：无法使用 IsotonicRegression")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        return ("isotonic", dict(model=iso))

    if kind == "affine_pos":
        xm, ym = x.mean(), y.mean()
        xc, yc = x - xm, y - ym
        a = max(0.0, float(np.sum(xc * yc) / (np.sum(xc * xc) + 1e-18)))
        b = float(ym - a * xm)
        return ("affine_pos", dict(a=a, b=b))

    if kind == "huber_lin":
        if SKL_HUBER_OK:
            mdl = HuberRegressor(alpha=0.0, fit_intercept=True)
            mdl.fit(x.reshape(-1, 1), y)
            return ("huber_lin", dict(model=mdl))
        # fallback: OLS
        X = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        return ("ols", dict(a=float(a), b=float(b)))

    if kind == "poly3":
        X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return ("poly3", dict(w=w.astype(float)))

    raise ValueError(f"Unknown calibrator kind: {kind}")


def predict_calibrator(kind_payload, x: np.ndarray) -> np.ndarray:
    kind, payload = kind_payload
    x = np.asarray(x, dtype=float)
    if kind == "constant":
        return np.full_like(x, payload["c"], dtype=float)
    if kind == "isotonic":
        return payload["model"].predict(x)
    if kind == "affine_pos":
        return payload["a"] * x + payload["b"]
    if kind == "huber_lin":
        mdl = payload["model"]
        try:
            return mdl.predict(x.reshape(-1, 1))
        except Exception:
            a = float(getattr(mdl, "coef_", [0.0])[0])
            b = float(getattr(mdl, "intercept_", 0.0))
            return a * x + b
    if kind == "ols":
        return payload["a"] * x + payload["b"]
    if kind == "poly3":
        w = payload["w"]
        X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
        return X @ w
    return x



def local_geom_grid(center: float, span_lo: float, span_hi: float, num: int) -> np.ndarray:
    lo = center / max(span_lo, 1.00001)
    hi = center * max(span_hi, 1.00001)
    lo = max(lo, 1e-6)
    hi = max(hi, 1.00002 * lo)
    return np.geomspace(lo, hi, num=int(num))


def _fit_linear_for_record(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return 0.0, float(np.nanmean(y))
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def grid_scan_alpha_beta_rank(
    sinr_list: List[np.ndarray],
    y: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    y = np.asarray(y, dtype=float)
    total = len(alpha_grid) * len(beta_grid)
    with tqdm(total=total, desc="Grid α–β (Spearman)", ncols=80) as pbar:
        for a in alpha_grid:
            for b in beta_grid:
                xeff = compute_xeff_list(sinr_list, float(a), float(b))
                loss = spearman_objective(xeff, y)
                aa, bb = _fit_linear_for_record(xeff, y)
                yhat = aa * xeff + bb
                m = np.isfinite(yhat) & np.isfinite(y)
                metrics = (
                    regression_metrics(y[m], yhat[m])
                    if np.any(m)
                    else dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, Spearman=np.nan)
                )
                rows.append(dict(alpha=float(a), beta=float(b), loss=float(loss), a_lin=aa, b_lin=bb, **metrics))
                pbar.update(1)
    return pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)


def multi_stage_grid_search_rank(
    sinr_list: List[np.ndarray],
    y: np.ndarray,
    stages: List[dict],
    outdir: Path,
    heatmap_mode: str = "softmin",
    heatmap_tau: float = 0.12,
    topk_curve: int = 12,
    max_ticks_axis: int = 10,
):
    stage_results: Dict[str, pd.DataFrame] = {}
    last_df: Optional[pd.DataFrame] = None

    def build_axis(st: dict, spec, axis_name: str) -> np.ndarray:
        if not isinstance(spec, (tuple, list)):
            return np.array(spec, dtype=float)
        kind = spec[0]
        if kind == "geom":
            _, lo, hi, num = spec
            return np.geomspace(float(lo), float(hi), int(num))
        if kind == "array":
            return np.array(spec[1], dtype=float)
        if kind == "local":
            ref = st.get("refine", {}).get("from", None)
            if ref is None or ref not in stage_results:
                raise RuntimeError(f"{st.get('name','stage')}: local {axis_name} 需要 refine.from 指定来源阶段")
            prev_df = stage_results[ref]
            centers = prev_df.sort_values("loss").head(int(st.get("topk", 20)))[axis_name].values
            p = spec[1]
            grids = [
                local_geom_grid(float(c), p.get("span_lo", 1.8), p.get("span_hi", 1.8), p.get("num", 15))
                for c in centers
            ]
            axis = np.unique(np.concatenate(grids))
            return axis
        raise ValueError(f"Unknown axis kind: {kind}")

    for st in stages:
        name = st.get("name", "stage")
        a_spec = st["alpha_grid"]
        b_spec = st["beta_grid"]
        alpha_axis = build_axis(st, a_spec, "alpha")
        beta_axis = build_axis(st, b_spec, "beta")

        df = grid_scan_alpha_beta_rank(sinr_list, y, alpha_axis, beta_axis)
        (outdir / f"{name}_grid_metrics.csv").write_text(df.to_csv(index=False))

        plot_loss_curve(df, title=f"{name}: loss (sorted)", outpath=outdir / f"{name}_curve_loss.png", topk=topk_curve)
        plot_param_distribution_heatmap(
            df,
            outpath=outdir / f"{name}_heatmap_param_distribution.png",
            mode=heatmap_mode,
            tau=heatmap_tau,
            title=f"{name}: good-parameter distribution",
            max_ticks_axis=max_ticks_axis,
        )

        stage_results[name] = df.copy()
        last_df = df

    last_sorted = last_df.sort_values("loss", ascending=True).reset_index(drop=True)  # type: ignore[arg-type]
    alpha0, beta0 = float(last_sorted.loc[0, "alpha"]), float(last_sorted.loc[0, "beta"])  # type: ignore[index]
    return last_sorted, (alpha0, beta0), stage_results



def nm_rank_alpha_beta(
    sinr_list: List[np.ndarray],
    y: np.ndarray,
    alpha0: float,
    beta0: float,
    restarts: int = 12,
    maxiter: int = 4500,
    jitter_alpha: float = 0.4,
    jitter_beta: float = 0.4,
    verbose: bool = True,
) -> Dict[str, object]:
    history: List[Dict[str, float]] = []
    eval_cnt = {"n": 0}

    def obj(u: np.ndarray) -> float:
        eval_cnt["n"] += 1
        alpha = float(np.exp(u[0]))
        beta = float(np.exp(u[1]))
        xeff = compute_xeff_list(sinr_list, alpha, beta)
        loss = spearman_objective(xeff, y)
        history.append(dict(iter=float(eval_cnt["n"]), loss=float(loss), alpha=alpha, beta=beta))
        if verbose and eval_cnt["n"] % 100 == 0:
            print(f"[NM-rank {eval_cnt['n']:>5}] loss={loss:.6g} α={alpha:.4g} β={beta:.4g}")
        return float(loss)

    u0 = np.array([np.log(max(alpha0, 1e-6)), np.log(max(beta0, 1e-6))], dtype=float)
    best_u, best_f = u0.copy(), obj(u0)

    if not SCIPY_OK:
        warnings.warn("SciPy 不可用：跳过 Nelder–Mead，仅使用网格最优点。")
        return dict(alpha=float(np.exp(best_u[0])), beta=float(np.exp(best_u[1])), loss=best_f, history_df=pd.DataFrame(history), method="grid_only")

    rng = np.random.default_rng(2025)
    starts = [u0] + [u0 + rng.normal(scale=[jitter_alpha, jitter_beta]) for _ in range(restarts)]

    for us in starts:
        res = minimize(
            obj,
            us,
            method="Nelder-Mead",
            options=dict(maxiter=maxiter, xatol=1e-7, fatol=1e-7, disp=False),
        )
        if res.success and float(res.fun) < float(best_f):
            best_f = float(res.fun)
            best_u = np.asarray(res.x, dtype=float)

    alpha = float(np.exp(best_u[0]))
    beta = float(np.exp(best_u[1]))
    hist_df = pd.DataFrame(history)
    if verbose:
        print(f"[NM-rank BEST] loss={best_f:.6g} α={alpha:.6g} β={beta:.6g}")
    return dict(alpha=alpha, beta=beta, loss=float(best_f), history_df=hist_df, method="nelder-mead-rank")



def run_pipeline(
    train_path: Path,
    valid_path: Optional[Path],
    per_sc_col: str,
    label_col: str,
    outdir: Path,
    calibrator_kind: str = "isotonic",
    discrete_mcs: bool = False,
    nm_restarts: int = 2,
    nm_maxiter: int = 100,
    nm_jitter_alpha: float = 0.4,
    nm_jitter_beta: float = 0.4,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path) if (valid_path and valid_path.exists()) else None

    y_train = pd.to_numeric(train_df[label_col], errors="coerce").values.astype(float)
    sinr_train = [parse_float_array_from_string(s) for s in train_df[per_sc_col].astype(str)]

    stages = [
        dict(
            name="stage1",
            alpha_grid=("geom", *STAGE1_ALPHA_DB_RANGE),
            beta_grid=("geom", *STAGE1_BETA_DB_RANGE),
            topk=TOPK_STAGE1,
        ),
        dict(
            name="stage2",
            alpha_grid=("local", LOCAL_ALPHA_STAGE2),
            beta_grid=("local", LOCAL_BETA_STAGE2),
            topk=TOPK_STAGE2,
            refine={"from": "stage1"},
        ),

    ]

    last_sorted, (alpha0, beta0), _ = multi_stage_grid_search_rank(
        sinr_list=sinr_train,
        y=y_train,
        stages=stages,
        outdir=outdir,
        heatmap_mode="softmin",
        heatmap_tau=0.12,
        topk_curve=12,
        max_ticks_axis=10,
    )


    fit_rank = nm_rank_alpha_beta(
        sinr_list=sinr_train,
        y=y_train,
        alpha0=alpha0,
        beta0=beta0,
        restarts=nm_restarts,
        maxiter=nm_maxiter,
        jitter_alpha=nm_jitter_alpha,
        jitter_beta=nm_jitter_beta,
        verbose=True,
    )

    alpha, beta = float(fit_rank["alpha"]), float(fit_rank["beta"])
    hist_df: pd.DataFrame = fit_rank.get("history_df", pd.DataFrame())  # type: ignore[assignment]
    if hist_df is not None and not hist_df.empty:
        hist_df.to_csv(outdir / "nelder_mead_rank_history.csv", index=False)
        plot_nm_history(hist_df, outdir / "nelder_mead_rank_loss_curve.png")
        best_row = hist_df.loc[hist_df["loss"].astype(float).idxmin()]
        print(f"[Info] NM-rank 最低 loss 出现在第 {int(best_row['iter'])} 轮（loss={best_row['loss']:.6g}）。")


    xeff_tr = compute_xeff_list(sinr_train, alpha, beta)
    calib = fit_calibrator(xeff_tr, y_train, kind=calibrator_kind)
    yhat_tr = predict_calibrator(calib, xeff_tr)

    metrics_train = regression_metrics(y_train, yhat_tr)
    out_metrics = {
        **metrics_train,
        "calibrator": calibrator_kind,
        "alpha": alpha,
        "beta": beta,
        "rank_loss": float(fit_rank["loss"]),
    }

    if discrete_mcs:
        y_true_idx = np.rint(y_train)
        y_pred_idx = np.rint(yhat_tr)
        qwk = quadratic_weighted_kappa(y_true_idx, y_pred_idx)
        out_metrics["QWK"] = float(qwk)

    with open(outdir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)


    if calibrator_kind == "isotonic":
        assert calib[0] == "isotonic"
        plot_calibration_iso(xeff_tr, y_train, calib[1]["model"], outdir / "calibration_train.png")
    else:
        plot_calibration_scatter(
            xeff_tr, y_train, yhat_tr, outdir / "calibration_train.png", title=f"Calibration (fit={calibrator_kind})"
        )
    plot_residual_hist(y_train, yhat_tr, outdir / "residuals_train.png")


    train_df["eesm_rank_xeff"] = xeff_tr
    train_df["pred_label_fit"] = yhat_tr
    if discrete_mcs:
        train_df["pred_mcs_index"] = np.rint(yhat_tr)
    train_df.to_csv(outdir / "train_with_rank_fit.csv", index=False)

    if valid_df is not None:
        sinr_valid = [parse_float_array_from_string(s) for s in valid_df[per_sc_col].astype(str)]
        xeff_va = compute_xeff_list(sinr_valid, alpha, beta)
        yhat_va = predict_calibrator(calib, xeff_va)
        valid_df["eesm_rank_xeff"] = xeff_va
        valid_df["pred_label_fit"] = yhat_va
        if discrete_mcs:
            valid_df["pred_mcs_index"] = np.rint(yhat_va)
        valid_df.to_csv(outdir / "valid_with_rank_fit.csv", index=False)


    best_params = dict(
        alpha=alpha,
        beta=beta,
        method=fit_rank["method"],
        rank_loss=float(fit_rank["loss"]),
        calibrator=calibrator_kind,
        **metrics_train,
    )
    with open(outdir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print("Best αβ:", dict(alpha=round(alpha, 6), beta=round(beta, 6)))
    print("Calibrator:", calibrator_kind)
    print("Train metrics:", metrics_train)
    if discrete_mcs:
        print("QWK:", best_params.get("QWK"))
    print("Outputs saved under:", str(outdir))



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EESM α–β rank-objective fitting with multi-stage grids + NM + calibration")
    p.add_argument("--train", required=True, type=Path, help="Path to TRAIN CSV (with labels)")
    p.add_argument("--valid", default=None, type=Path, help="Path to VALID CSV (optional)")
    p.add_argument("--per_sc_col", default="sinr_per_sc_non", help="Per-SC SINR column name (stringified array)")
    p.add_argument("--label_col", default="mcs", help="Label/MCS column name")
    p.add_argument("--outdir", default=Path("./eesm_out"), type=Path, help="Output directory")
    p.add_argument(
        "--calibrator",
        default="isotonic",
        choices=["isotonic", "affine_pos", "huber_lin", "poly3"],
        help="Mapping from x_eff to label/MCS",
    )
    p.add_argument("--discrete_mcs", action="store_true", help="Round predictions to nearest integer and report QWK")
    p.add_argument("--nm_restarts", type=int, default=5)
    p.add_argument("--nm_maxiter", type=int, default=1000)
    p.add_argument("--nm_jitter_alpha", type=float, default=0.4)
    p.add_argument("--nm_jitter_beta", type=float, default=0.4)
    return p


def main(args: argparse.Namespace) -> None:
    run_pipeline(
        train_path=Path(args.train),
        valid_path=Path(args.valid) if args.valid else None,
        per_sc_col=args.per_sc_col,
        label_col=args.label_col,
        outdir=Path(args.outdir),
        calibrator_kind=args.calibrator,
        discrete_mcs=bool(args.discrete_mcs),
        nm_restarts=int(args.nm_restarts),
        nm_maxiter=int(args.nm_maxiter),
        nm_jitter_alpha=float(args.nm_jitter_alpha),
        nm_jitter_beta=float(args.nm_jitter_beta),
    )


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
