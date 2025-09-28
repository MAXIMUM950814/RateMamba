import numpy as np, pandas as pd, ast, json, re
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False
from pretab.preprocessor import Preprocessor

def safe_literal_eval(s):
    if s is None: return None
    if isinstance(s, (list, tuple, np.ndarray, dict)): return s
    if not isinstance(s, str): return s
    t = s.strip()
    t = re.sub(r'^\s*array\s*\(', '(', t)
    try:
        return ast.literal_eval(t)
    except Exception:
        try:
            return json.loads(t)
        except Exception:
            return s

def parse_h_merged_complex(s, expect_shape=(122,2,4,2)):
    obj = safe_literal_eval(s)
    try:
        arr = np.asarray(obj)
        if arr.ndim != 4 or arr.shape[-1] != 2:
            flat = np.asarray(obj).reshape(-1)
            T = int(np.prod(expect_shape))
            if flat.size < T: return None
            arr = flat[:T].reshape(*expect_shape)
        return arr[...,0] + 1j*arr[...,1]
    except Exception:
        return None

def parse_complex_series(s):
    v = safe_literal_eval(s)
    try:
        return np.array(v, dtype=complex).reshape(-1)
    except Exception:
        try:
            a = np.asarray(v)
            if a.ndim >= 1 and a.shape[-1] == 2:
                return (a[...,0] + 1j*a[...,1]).reshape(-1).astype(complex)
        except Exception:
            pass
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))
        if len(nums) == 0: return None
        return np.array([float(x) for x in nums], dtype=float).reshape(-1) + 0j

def feats_h_svd(h_complex, top_singular=2):
    if h_complex is None or h_complex.size == 0: return None
    try:
        K = h_complex.shape[0]
        svs = np.zeros((K, top_singular), dtype=float)
        for k in range(K):
            s = np.linalg.svd(h_complex[k], compute_uv=False)
            s = s[:top_singular]
            if s.size < top_singular:
                s = np.pad(s, (0, top_singular - s.size))
            svs[k] = s
        out = []
        for i in range(top_singular):
            si = svs[:, i]
            out += [float(si.mean()), float(si.std()), float(si.max())]
        mag = np.abs(h_complex).reshape(K, -1)
        phs = np.angle(h_complex).reshape(K, -1)
        out += [float(mag.mean()), float(mag.std()), float(phs.std())]
        return np.array(out, dtype=float)
    except Exception:
        return None

def h_feature_names(top_singular=2):
    names = []
    for i in range(top_singular):
        names += [f"h_sv{i+1}_mean", f"h_sv{i+1}_std", f"h_sv{i+1}_max"]
    names += ["h_mag_mean","h_mag_std","h_phase_std"]
    return names

def feats_complex_series(z):
    if z is None or z.size == 0: return None
    try:
        mag = np.abs(z); phs = np.angle(z)
        p05, p50, p95 = np.percentile(mag, [5,50,95])
        base = [
            float(len(mag)),
            float(mag.mean()), float(mag.std()), float(mag.min()), float(mag.max()),
            float(p05), float(p50), float(p95),
            float(phs.std())
        ]
        nfft = int(2**np.ceil(np.log2(max(8, mag.size))))
        A = np.abs(np.fft.rfft(mag, n=nfft))
        P = A**2; E = float(P.sum())
        if E > 0:
            p = P / E
            H = float(-(p*np.log(p+1e-12)).sum())
            flat = float(np.exp(np.mean(np.log(P+1e-12))) / (P.mean()+1e-12))
            cutoff = float(np.searchsorted(np.cumsum(P), 0.85*E) / len(P))
        else:
            H = flat = cutoff = 0.0
        return np.array(base + [E, H, flat, cutoff], dtype=float)
    except Exception:
        return None

def complex_series_feature_names(prefix):
    return [
        f"{prefix}_len",
        f"{prefix}_mag_mean", f"{prefix}_mag_std", f"{prefix}_mag_min", f"{prefix}_mag_max",
        f"{prefix}_mag_p05", f"{prefix}_mag_p50", f"{prefix}_mag_p95",
        f"{prefix}_phase_std",
        f"{prefix}_spec_energy", f"{prefix}_spec_entropy", f"{prefix}_spec_flatness", f"{prefix}_spec_rolloff",
    ]

CSI_COLS = [
    "csi_matrix_r0_c0","csi_matrix_r0_c1","csi_matrix_r0_c2","csi_matrix_r0_c3",
    "csi_matrix_r1_c0","csi_matrix_r1_c1","csi_matrix_r1_c2","csi_matrix_r1_c3"
]

def make_full_dataframe(df: pd.DataFrame, y_col="mcs", verbose=True) -> pd.DataFrame:
    N = len(df)
    h_names = h_feature_names(2)
    H_block = np.full((N, len(h_names)), np.nan, dtype=float)
    H_ok = np.zeros(N, dtype=bool)
    if "H_merged_complex" in df.columns:
        it = df["H_merged_complex"].tolist()
        rng = tqdm(it) if TQDM and verbose else it
        for i, s in enumerate(rng):
            Hc = parse_h_merged_complex(s, (122,2,4,2))
            feats = feats_h_svd(Hc, top_singular=2)
            if feats is not None:
                H_block[i] = feats; H_ok[i] = True
    H_df = pd.DataFrame(H_block, columns=h_names, index=df.index)
    H_df.loc[~H_ok, :] = np.nan

    csi_frames = []; csi_ok_mat = []
    for col in CSI_COLS:
        if col not in df.columns: continue
        names = complex_series_feature_names(col)
        block = np.full((N, len(names)), np.nan, dtype=float)
        ok = np.zeros(N, dtype=bool)
        it = df[col].tolist()
        rng = tqdm(it) if TQDM and verbose else it
        for i, s in enumerate(rng):
            z = parse_complex_series(s)
            feats = feats_complex_series(z)
            if feats is not None:
                block[i] = feats; ok[i] = True
        csi_frames.append(pd.DataFrame(block, columns=names, index=df.index))
        csi_ok_mat.append(ok)
    CSI_df = pd.concat(csi_frames, axis=1) if csi_frames else pd.DataFrame(index=df.index)
    csi_any_ok = np.any(np.column_stack(csi_ok_mat), axis=1) if csi_ok_mat else np.zeros(N, dtype=bool)

    exclude = set(["H_merged_complex"] + [c for c in CSI_COLS if c in df.columns] + ([y_col] if y_col in df.columns else []))
    num_cols_raw = [c for c in df.columns if (c not in exclude and pd.api.types.is_numeric_dtype(df[c]))]
    cat_cols_raw = [c for c in ["terminal_id"] if c in df.columns]
    num_df = df[num_cols_raw].copy() if num_cols_raw else pd.DataFrame(index=df.index)
    cat_df = df[cat_cols_raw].copy() if cat_cols_raw else pd.DataFrame(index=df.index)

    row_ok = H_ok | csi_any_ok
    H_df = H_df.loc[row_ok]
    CSI_df = CSI_df.loc[row_ok]
    num_df = num_df.loc[row_ok]
    cat_df = cat_df.loc[row_ok]
    y = df.loc[row_ok, y_col].values.reshape(-1, 1) if y_col in df.columns else None

    X_num_in = pd.concat([H_df, CSI_df, num_df], axis=1)

    def _uniq_names(cols):
        seen = {}
        out = []
        for c in cols.astype(str):
            if c not in seen:
                seen[c] = 1
                out.append(c)
            else:
                k = seen[c]; seen[c] += 1
                out.append(f"{c}__dup{k}")
        return out
    X_num_in.columns = _uniq_names(pd.Index(X_num_in.columns))
    X_num_in = X_num_in.replace([np.inf, -np.inf], np.nan).astype("float32")
    all_nan_cols = X_num_in.columns[X_num_in.isna().all(0)]
    X_num_in = X_num_in.drop(columns=list(all_nan_cols))
    nunique = X_num_in.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    X_num_in = X_num_in.drop(columns=const_cols)
    y_1d = y.ravel() if y is not None and y.ndim == 2 and y.shape[1] == 1 else (y if y is not None else None)
    num_cfg = {c: "ple" for c in X_num_in.columns}
    X_cat_in = cat_df.copy()
    if X_cat_in.shape[1] > 0:
        X_cat_in.columns = _uniq_names(pd.Index(X_cat_in.columns))
    cat_cfg = {c: "one-hot" for c in X_cat_in.columns}

    frames = []
    try:
        if X_num_in.shape[1] > 0:
            pre_num = Preprocessor(feature_preprocessing=num_cfg, task="classification")
            X_num_dict = pre_num.fit_transform(X_num_in, y=y_1d)
            for col, arr in X_num_dict.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    frames.append(pd.DataFrame(arr, columns=[f"{col}__0"], index=X_num_in.index))
                else:
                    frames.append(pd.DataFrame(arr,
                        columns=[f"{col}__{i}" for i in range(arr.shape[1])],
                        index=X_num_in.index))
    except Exception:
        frames.append(X_num_in.add_prefix("raw_"))

    try:
        if X_cat_in.shape[1] > 0:
            pre_cat = Preprocessor(feature_preprocessing=cat_cfg, task="classification")
            X_cat_dict = pre_cat.fit_transform(X_cat_in, y=y_1d)
            for col, arr in X_cat_dict.items():
                arr = np.asarray(arr)
                if arr.ndim == 1:
                    frames.append(pd.DataFrame(arr, columns=[f"{col}__0"], index=X_cat_in.index))
                else:
                    frames.append(pd.DataFrame(arr,
                        columns=[f"{col}__{i}" for i in range(arr.shape[1])],
                        index=X_cat_in.index))
    except Exception:
        if X_cat_in.shape[1] > 0:
            frames.append(pd.get_dummies(X_cat_in, dummy_na=True, prefix="rawcat"))

    X_out = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=X_num_in.index)
    X_out.index = X_num_in.index
    X_out[y_col] = y_1d
    return X_out

if __name__ == "__main__":
    df = pd.read_csv("input.csv")
    out = make_full_dataframe(df, y_col="mcs", verbose=True)
    out.to_csv("output.csv", index=False)
